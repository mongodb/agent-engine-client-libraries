"""
Utility functions for Runner SDK.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import time
from enum import Enum
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import httpx


def normalize_content(content: Any) -> str:
    """
    Normalize LLM message content to a string.

    LLM content can be:
    - A string (normal text)
    - A list (multimodal content with text and other parts)
    - None or empty

    This handles cases where LangChain's AIMessageChunk.content is a list
    of content blocks (e.g., from Gemini multimodal responses) rather than
    a simple string.

    Args:
        content: The content to normalize (string, list, or None)

    Returns:
        A string representation of the content
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Handle multimodal content - extract text parts
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict):
                # Common format: {"type": "text", "text": "..."} or {"type": "image", ...}
                if part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                # Skip non-text parts (images, etc.)
            elif hasattr(part, "type") and hasattr(part, "text") and part.type == "text":
                # sdk-core TextBlock or similar typed content blocks
                text_parts.append(part.text)
        return "".join(text_parts)
    # Fallback: convert to string
    return str(content)


def normalize_tool_call_args(args: Any) -> str | None:
    """Normalize streamed tool-call args into the wire-format string payload."""
    if args is None or isinstance(args, str):
        return args
    try:
        return json.dumps(args)
    except (TypeError, ValueError):
        return str(args)


def normalize_optional_str(value: Any) -> str | None:
    """Trim a string tool-metadata value, collapsing blank/non-str to None.

    Shared by the framework SDKs' tool-wrapping code so metadata values written
    as "" or "  " are treated the same as absent
    rather than sent to the OE as a non-empty-looking but meaningless string.
    """
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return None


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def strip_thinking(text: str) -> str:
    """Remove ``<think>...</think>`` blocks and unclosed ``<think>`` tails."""
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def filter_thinking_tokens(token: str, buffer: str, inside: bool) -> tuple[str, str, bool]:
    """Filter ``<think>`` blocks from a stream of token chunks.

    Accumulates text in *buffer* until we can determine whether content
    is inside a thinking block.  Returns ``(streamable, new_buffer,
    inside)`` where *streamable* is the text safe to send to the client.
    """
    buffer += token
    streamable_parts: list[str] = []

    while True:
        if inside:
            close_idx = buffer.find(_THINK_CLOSE)
            if close_idx == -1:
                return "".join(streamable_parts), buffer, True
            buffer = buffer[close_idx + len(_THINK_CLOSE) :]
            inside = False
        else:
            open_idx = buffer.find(_THINK_OPEN)
            if open_idx == -1:
                streamable_parts.append(buffer)
                return "".join(streamable_parts), "", False
            streamable_parts.append(buffer[:open_idx])
            buffer = buffer[open_idx + len(_THINK_OPEN) :]
            inside = True


class RuntimeMode(str, Enum):
    """Runtime mode for the Runner SDK."""

    AER = "aer"
    TOOL = "tool"
    # Tool role, per-call (function) lifecycle: the process boots, runs a
    # single named tool from the registry, reports the result, and exits.
    # TOOL is the same role as a long-running server handling many /execute
    # calls. The value carries the role because the function lifecycle is
    # orthogonal to it (a future per-call AER would be e.g. "aer_function").
    # The invocation is read once from the guest metadata channel, not env.
    TOOL_FUNCTION = "tool_function"


_MODE_MAP: dict[str, RuntimeMode] = {m.value: m for m in RuntimeMode}


def get_runtime_mode() -> RuntimeMode:
    """
    Get the current runtime mode from environment variable.

    Returns:
        RuntimeMode based on RUNNER_MODE environment variable.

    Raises:
        ValueError: If RUNNER_MODE is not set or is not a valid mode.
    """
    raw = os.environ.get("RUNNER_MODE")
    if raw is None:
        raise ValueError(
            f"RUNNER_MODE environment variable is required. Valid values: {', '.join(_MODE_MAP)}"
        )
    mode = raw.lower()
    try:
        return _MODE_MAP[mode]
    except KeyError:
        raise ValueError(
            f"Unknown RUNNER_MODE '{mode}'. Valid values: {', '.join(_MODE_MAP)}"
        ) from None


def get_env(name: str, default: str = "") -> str:
    """Get environment variable with default."""
    return os.environ.get(name, default)


def get_env_int(name: str, default: int) -> int:
    """Get integer environment variable with default."""
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def get_env_float(name: str, default: float) -> float:
    """Get float environment variable with default."""
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def get_env_bool(name: str, default: bool = False) -> bool:
    """Get boolean environment variable with default."""
    value = os.environ.get(name, str(default)).lower()
    return value in ("true", "1", "yes", "on")


# Env vars set by the platform inside the runtime container.
# Filtered OUT of the tenant-visible env mapping passed to
# ``load_runtime_agent_config(env_vars=...)`` so tenant ``agent.yaml`` cannot
# use ``${VAR}`` interpolation to dereference platform secrets or internal
# service addresses (e.g. ``url: https://attacker/${OPENAI_API_KEY}``).
#
# Maintenance: whenever a new platform-owned env var is introduced in
# ``agent-engine-runner-shared`` (or a sibling package whose env reaches the runtime
# container), add its exact name to ``_PLATFORM_ENV_VARS`` or, if it shares a
# common prefix with related platform vars, extend ``_PLATFORM_ENV_VAR_PREFIXES``.
# The prefix list does most of the work -- prefer adding a prefix over
# enumerating individual names.
_PLATFORM_ENV_VAR_PREFIXES: tuple[str, ...] = (
    # ``AGENTIC_*`` is the platform's reserved namespace. Tenants are
    # documented to use unprefixed names for their own vars.
    "AGENTIC_",
    # Component-scoped families
    "MONGOMEM_",  # memory server config
    "MONGODB_",  # MongoDB URI + client tunables
    "VOYAGE_",  # platform-owned embedder config
    "LLM_",  # retry tunables in this module
    "OE_",  # orchestration engine
    "AER_",  # Agent Execution Runtime endpoints
    "TOOL_",  # Tool Pod endpoints
    "ECP_",  # Executor Control Plane
    "RUNNER_",  # runner-internal tunables (mode, stream timeouts, ...)
    "GUARDRAILS_",  # guardrails server config + LLM key
    "FILESYSTEM_",  # Tool Pod sandbox limits
    "SHELL_",  # Tool Pod sandbox limits
    "MAX_",  # Tool Pod sandbox limits (MAX_LS_ENTRIES, MAX_GREP_*, ...)
    # Auth / observability
    "OKTA_",
    "OIDC_",
    "OTEL_",
    # Test infra (must not leak into production tenant interpolation either)
    "E2E_",
    # Container / orchestration infra
    "KUBERNETES_",
    "HELIX_",
    # POSIX
    "LC_",
)

_PLATFORM_ENV_VARS: frozenset[str] = frozenset(
    {
        # Multi-tenant identity
        "ORG_ID",
        "PROJECT_ID",
        "ATLAS_GROUP_ID",
        "TENANT_ID",
        "APP_ID",
        # Distinct from APP_ID above: WORKSPACE_ID is a separate, pre-existing
        # env var read by structured_logging.py for log correlation. It is
        # NOT the source for the agentic_platform.workspace_id resource
        # attribute — tracing/setup.py deliberately reads APP_ID
        # for that, since WORKSPACE_ID is confirmed unset on the AER/Tool Pod
        # (see test_build_resource_attributes_ignores_old_wrong_env_var_names).
        "WORKSPACE_ID",
        "DEPLOYMENT_ID",
        "AGENT_ID",
        # Platform-owned observability env (OE/operator MaaS metering, see
        # maasmetrics/config.go) that tracing/setup.py also reads for the
        # agentic_platform.environment resource attribute.  # open-source-refs:ignore — trace attribute keys are a stored trace-store contract; renaming is tracked separately
        "AGENT_ENGINE_ENVIRONMENT",
        # Platform LLM provider credentials. Read by the standalone memory
        # server for extraction-pipeline provider detection, but potentially
        # present in any runtime container.
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "CEREBRAS_API_KEY",
        "GEMINI_API_KEY",
        # Platform LLM endpoints (not secret, but leaking them via tenant
        # YAML is still pointless and a potential SSRF vector).
        "OPENAI_BASE_URL",
        "ANTHROPIC_BASE_URL",
        # Auth / inter-service secrets
        "A2A_JWT_SECRET",
        # TLS / mTLS certificate paths (operator-mounted for AER→OE HTTPS)
        "TLS_CERT_PATH",
        "TLS_KEY_PATH",
        "TLS_CA_CERT_PATH",
        # TLS / mTLS PEM content (VM mode via SecretKeyRef, kubelet decodes automatically)
        # CRITICAL: Must be excluded to prevent tenant exfiltration via ${TLS_KEY_PEM}
        "TLS_CERT_PEM",
        "TLS_KEY_PEM",
        "TLS_CA_CERT_PEM",
        # Server / network config
        "APP_HOST",
        "APP_PORT",
        "LOG_LEVEL",
        "LOG_DIR",  # operator-injected via render.WorkloadInjectedEnv
        "STRUCTURED_LOGGING",  # operator-injected via render.StructuredLoggingEnv
        "CORS_ALLOWED_ORIGINS",  # OE-injected (forward-defensive on AER)
        "USE_MEMORY_CLIENT",
        "AGENT_STREAM_TERMINAL_DRAIN_TIMEOUT",
        "SHUTDOWN_GRACE_PERIOD_MS",
        # Runner launcher entrypoint baked into the Dockerfile ENV by ECP
        "AGENT_ENTRYPOINT",
        # Platform persistence
        "PLATFORM_DATABASE",
        "MEMORY_DATABASE_NAME",
        "MDB_AGENTIC_STORE_DB",
        "CHECKPOINT_DB_NAME",
        "DB_NAME",
        "CHECKPOINTER_SERVER_SELECTION_TIMEOUT",
        "CHECKPOINTER_CONNECT_TIMEOUT",
        "CHECKPOINTER_SOCKET_TIMEOUT",
        "TEST_MONGO_URI",
        # Tool Pod sandbox (covered partially by prefix, but the no-prefix
        # ``WORKSPACE_DIR`` / ``DOWNLOAD_MAX_BYTES`` names live here).
        "WORKSPACE_DIR",
        "DOWNLOAD_MAX_BYTES",
        # Feature toggles
        "ENABLE_MEMORY",
        "ENABLE_TRACING",
        "ENABLE_VECTOR_SEARCH_TESTS",
        # Dev / test infra
        "SEED_DATA",
        "CI",
        # Container / pod infra
        "POD_NAME",
        "HOSTNAME",
        # Generic POSIX. Substituting these into a URL is never what the
        # tenant means; blocking prevents accidental leakage of process
        # identity / filesystem layout.
        "PATH",
        "HOME",
        "USER",
        "PWD",
        "LANG",
        "TERM",
        "SHELL",
        "SHLVL",
        "_",
    }
)


def is_platform_env_var(name: str) -> bool:
    """Return ``True`` if ``name`` matches a platform-owned env var.

    Public-API view of the same membership check used by ``tenant_env_vars``.
    Used by ``agent_config`` to validate fields that name an env var (e.g.
    ``mcp.servers.*.auth.token_env``) so tenant YAML cannot redirect a
    secret-indirection path at a platform-owned variable.
    """
    return name in _PLATFORM_ENV_VARS or any(
        name.startswith(prefix) for prefix in _PLATFORM_ENV_VAR_PREFIXES
    )


def tenant_env_vars(source: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return the tenant-owned subset of environment variables.

    ``source`` defaults to ``os.environ``. Names listed in ``_PLATFORM_ENV_VARS``
    or matching any prefix in ``_PLATFORM_ENV_VAR_PREFIXES`` are excluded, so
    the result is safe to pass as the substitution mapping to
    ``load_runtime_agent_config(env_vars=...)``.
    """
    if source is None:
        source = os.environ
    return {name: value for name, value in source.items() if not is_platform_env_var(name)}


# LLM retry configuration for rate limit errors
LLM_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "3"))
LLM_INITIAL_BACKOFF = float(os.environ.get("LLM_INITIAL_BACKOFF", "1.0"))  # seconds
LLM_BACKOFF_MULTIPLIER = float(os.environ.get("LLM_BACKOFF_MULTIPLIER", "2.0"))
LLM_MAX_BACKOFF = float(os.environ.get("LLM_MAX_BACKOFF", "30.0"))  # seconds

# Same-step retries of /tool/execute (and the SSE relay) after OE advertises
# retryable=true, and after an SSE transport disconnect. 1 initial + 2 extras;
# lockstep with agent-engine-runner-shared.
OE_RETRYABLE_MAX_ATTEMPTS = 3

# Wait after a dropped SSE connection so the next attempt can land after
# dispatch_heartbeat TTL (30s). Do not reuse RETRY_AFTER_MAX_WAIT_S (10s).
# Lockstep with agent-engine-runner-shared.
OE_DISPATCH_TAKEOVER_RETRY_DELAY_S = 30.0
OE_DISPATCH_RETRY_MAX_WAIT_S = 60.0

# LLM read timeout — LLM generation can take much longer than a typical HTTP request.
# Applies to both streaming (per-chunk wait) and non-streaming (full-response wait) paths.
LLM_READ_TIMEOUT = float(
    os.environ.get(
        "RUNNER_LLM_READ_TIMEOUT",
        os.environ.get("RUNNER_STREAM_READ_TIMEOUT", "300.0"),
    )
)  # seconds

# Tool read timeout — the OE holds /tool/execute open until the tool result comes
# back, so this bounds the tool's own runtime, not the handshake. It matches the
# OE's own tool deadline: a smaller value here abandons a tool the platform is
# still happily running, leaving the SDK with no result to report.
TOOL_READ_TIMEOUT = float(os.environ.get("RUNNER_TOOL_READ_TIMEOUT", "600.0"))  # seconds


def oe_stream_retry_delay_s(retry_after_ms: float | None = None) -> float:
    """Honor a server-provided SSE retry_after_ms, capped.

    Absent/None means retry immediately (reservation-loss). Transport
    disconnects use ``OE_DISPATCH_TAKEOVER_RETRY_DELAY_S`` instead.
    """
    if retry_after_ms is None:
        return 0.0
    delay_s = float(retry_after_ms) / 1000.0
    if delay_s <= 0:
        return 0.0
    return min(delay_s, OE_DISPATCH_RETRY_MAX_WAIT_S)


def sleep_oe_stream_retry(delay_s: float) -> None:
    """Sleep before an SSE same-URL retry. Tests patch this to avoid wall-clock waits."""
    if delay_s > 0:
        time.sleep(delay_s)


def is_retryable_error(error: Exception) -> bool:
    """Classify provider failures before any LLM output has been exposed."""
    seen: set[int] = set()
    retryable = False
    messages: list[str] = []
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, asyncio.CancelledError):
            return False
        status = _exception_http_status(current)
        if status is not None:
            # A structured rejection must not be overridden by error body text.
            if status not in (408, 409, 429) and not 500 <= status < 600:
                return False
            retryable = True
        if isinstance(
            current,
            (
                TimeoutError,
                ConnectionError,
                httpx.TimeoutException,
                httpx.NetworkError,
                httpx.RemoteProtocolError,
            ),
        ):
            retryable = True
        # Provider SDKs wrap transport exceptions, sometimes without a cause.
        if type(current).__name__ in ("APITimeoutError", "APIConnectionError"):
            retryable = True
        if getattr(current, "code", None) in (
            "server_error",
            "rate_limit_exceeded",
            "too_many_requests",
            "overloaded_error",
        ):
            retryable = True
        if getattr(current, "type", None) in ("server_error", "overloaded_error"):
            retryable = True
        messages.append(str(current).lower())
        if current.__cause__ is not None:
            current = current.__cause__
        elif not current.__suppress_context__:
            current = current.__context__
        else:
            current = None

    # Retain support for adapters that expose only an unstructured exception.
    retryable_patterns = [
        "too_many_requests",
        "rate_limit",
        "too many requests",
        "rate limit exceeded",
        "internal server error",
        "service unavailable",
        "gateway timeout",
        "server error",
        "queue_exceeded",
        "high traffic",
        "overloaded",
        "upstream connect error or disconnect/reset before headers",
        "the server had an error processing your request",
    ]
    # Require status context: an arbitrary token count or request ID is not an HTTP failure.
    status_pattern = (
        r"\b(?:http(?: status)?|status(?: code)?|error code)[:=]?\s*(?:408|409|429|5\d\d)\b"
    )
    return retryable or any(
        any(pattern in message for pattern in retryable_patterns)
        or re.search(status_pattern, message) is not None
        for message in messages
    )


def format_llm_error(error: Exception) -> str:
    """Render an LLM provider exception for display, without escaped-JSON text.

    Some provider SDKs (e.g. google-genai) attach the raw API error body as a
    dict on the exception or its ``__cause__`` (``.details``, ``.body``).
    That dict's values are often themselves JSON-encoded strings containing
    real newlines (e.g. a pretty-printed nested error payload), so Python's
    default ``str()`` ends up repr-ing those strings and turning the
    newlines into literal ``\\n`` sequences. Decode any
    JSON-encoded string values first and re-serialize with ``json.dumps`` so
    the result renders as readable, indented JSON instead.
    """
    for candidate in (error, error.__cause__):
        if candidate is None:
            continue
        try:
            body = getattr(candidate, "details", None)
            if not isinstance(body, dict):
                body = getattr(candidate, "body", None)
            if isinstance(body, dict):
                return json.dumps(_decode_nested_json(body), indent=2, default=str)
        except Exception:
            # A provider-supplied .details/.body can be anything -- a plain
            # attribute, a property that raises, a dict with circular refs or
            # exotic values. Never let formatting itself blow up the error
            # path -- fall back to the plain str() below.
            break
    return str(error)


def _decode_nested_json(value: Any) -> Any:
    """Recursively ``json.loads()`` any string value that looks like JSON."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped[0] in "{[":
            try:
                return _decode_nested_json(json.loads(stripped))
            except (json.JSONDecodeError, ValueError):
                return value
        return value
    if isinstance(value, dict):
        return {key: _decode_nested_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_decode_nested_json(item) for item in value]
    return value


def is_llm_credential_rejection(error: BaseException) -> bool:
    """True when an LLM provider exception is an auth rejection (HTTP 401/403).

    Reads only the provider SDK's own status fields — ``status_code``
    (openai/anthropic SDK style), ``response.status_code`` (httpx style), and
    an integer ``code`` (google-genai style) — walking the explicit
    ``__cause__`` chain because LangChain adapters occasionally re-raise with
    ``from``. ``__context__`` is deliberately not walked: an unrelated error
    raised while handling an auth failure would otherwise inherit the
    rejection. Text is never matched: the caller maps a positive result onto
    the wire-level credential error code, and anything unrecognized returns
    False so the failure keeps its existing generic classification.
    """
    seen: set[int] = set()
    stack: list[BaseException | None] = [error]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if _exception_http_status(current) in (401, 403):
            return True
        stack.append(current.__cause__)
    return False


def _exception_http_status(error: BaseException) -> int | None:
    """Extract the HTTP status a provider SDK attached to its exception."""
    for candidate in (
        getattr(error, "status_code", None),
        getattr(getattr(error, "response", None), "status_code", None),
        getattr(error, "code", None),
    ):
        # bool is an int subclass; a True/False attribute is never a status.
        if (
            isinstance(candidate, int)
            and not isinstance(candidate, bool)
            and 100 <= candidate < 600
        ):
            return candidate
    return None


# Configurable HTTP request timeout
def get_request_timeout() -> float:
    """
    Get the HTTP request timeout from environment.

    Uses RUNNER_REQUEST_TIMEOUT env var, defaults to 60.0 seconds.
    """
    return get_env_float("RUNNER_REQUEST_TIMEOUT", 60.0)


# =============================================================================
# Logging Configuration
# =============================================================================


def _sanitize_file_component(part: str, fallback: str) -> str:
    """Keep a log file name inside the log directory.

    ``app_name``/``mode`` are caller-supplied; a path separator or NUL in
    either could otherwise redirect the write outside ``log_dir`` (and in a
    ``dev up --all`` stack the sibling containers share the ``/app`` bind
    mount, so an escape lands on the shared volume). Mirrors the TypeScript
    ``sanitizeFileComponent``.
    """
    cleaned = part.replace("/", "_").replace("\\", "_").replace("\0", "_").strip()
    return cleaned if cleaned else fallback


def _install_file_handler(
    root_logger: logging.Logger,
    app_name: str,
    mode: str,
    log_dir: Optional[str],
    backup_count: int,
    level: int,
    detailed_formatter: logging.Formatter,
) -> Optional[Path]:
    """Attach a ``TimedRotatingFileHandler`` to ``root_logger``.

    Returns the resolved log-file path on success, ``None`` when file
    logging is disabled (unusable log_dir) so callers can report the
    actual destination — or its absence — in the install log line.
    """
    log_dir = log_dir or get_env("LOG_DIR", "./logs")
    try:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        log_file = log_path / (
            f"{_sanitize_file_component(app_name, 'runner')}"
            f"-{_sanitize_file_component(mode, 'aer')}.log"
        )
        file_handler = TimedRotatingFileHandler(
            log_file,
            when="midnight",
            backupCount=backup_count,
            encoding="utf-8",
            utc=True,
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)
        return log_file
    except OSError as e:
        root_logger.warning(
            "File logging disabled (log_dir=%s not writable: %s). "
            "Continuing with console output only.",
            log_dir,
            e,
        )
        return None


def setup_logging(
    *,
    app_name: str = "runner",
    mode: str = "aer",
    log_dir: Optional[str] = None,
    log_level: Optional[str] = None,
    backup_count: int = 5,
) -> logging.Logger:
    """
    Configure logging with both console and file output.

    All arguments are keyword-only. Prior versions took ``log_level`` as the
    first positional argument; keyword-only ensures ``setup_logging("DEBUG")``
    from older call sites fails loudly instead of silently binding ``"DEBUG"``
    to ``app_name`` and taking the default log level.

    When ``STRUCTURED_LOGGING=true`` is set, delegates to
    ``agent_engine_runner_shared.structured_logging.install_structured_logging`` so all
    output emerges as single-line JSON matching the agent-log
    contract. Disk logging is skipped in that mode in production — Fluent
    Bit ships container stdout to S3, so a duplicate on-disk copy adds no
    value and just wastes IO. The one exception is local dev: when
    ``AGENTIC_DEV_MODES`` is set (dev-up compose stacks only), a
    human-readable rotating file sink is attached alongside the structured
    stdout output so the Local Dev UI can read the agent's logs.

    Args:
        app_name: Application name for log file naming
        mode: Runtime mode (aer, tool, tool_function)
        log_dir: Directory for log files (default: ./logs)
        log_level: Log level (default: from LOG_LEVEL env or INFO)
        backup_count: Number of rotated backup files to keep

    Returns:
        Root logger configured with handlers
    """
    if get_env("STRUCTURED_LOGGING", "").lower() == "true":
        # Imported here to keep the structured-logging path opt-in: callers
        # that don't set the env var pay no import-time cost.
        from agent_engine_runner_shared.structured_logging import install_structured_logging

        # Pass mode through so the formatter's ``service`` field reflects
        # the caller's intent even if RUNNER_MODE env happens to be unset
        # — without this, a setup_logging(mode="aer") call with no env
        # would silently produce service="agent-execution-runtime" (the
        # default), lying about which component emitted the line.
        install_structured_logging(level=log_level, mode=mode)
        if os.environ.get("AGENTIC_DEV_MODES"):
            # Normalize before getattr: a caller-passed "debug" would
            # otherwise miss logging.DEBUG and silently fall back to INFO,
            # filtering records that structured stdout still shows.
            level_str = (log_level or get_env("LOG_LEVEL", "INFO")).upper()
            level = getattr(logging, level_str, logging.INFO)
            detailed_formatter = logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            # install_structured_logging owns the stdout handlers; the dev
            # file sink is ours, so drop any handler a previous call left
            # behind — otherwise every repeat call duplicates each record
            # in the file.
            root_logger = logging.getLogger()
            for handler in list(root_logger.handlers):
                if isinstance(handler, TimedRotatingFileHandler):
                    root_logger.removeHandler(handler)
                    try:
                        handler.close()
                    except Exception:
                        pass
            _install_file_handler(
                root_logger,
                app_name,
                mode,
                log_dir,
                backup_count,
                level,
                detailed_formatter,
            )
        return logging.getLogger()

    # Determine log level. Normalize before getattr (see structured branch).
    level_str = (log_level or get_env("LOG_LEVEL", "INFO")).upper()
    level = getattr(logging, level_str, logging.INFO)

    # Formatters
    detailed_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Get root logger. Drop any existing handlers so repeat calls don't
    # accumulate duplicate writers. For file handlers we also close() to
    # release the fd; for stdio-backed StreamHandlers we skip close() —
    # closing sys.stdout/sys.stderr breaks subsequent logging and print()
    # calls in the same process ("I/O operation on closed file").
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    while root_logger.handlers:
        handler = root_logger.handlers.pop()
        stream = getattr(handler, "stream", None)
        if stream in (sys.stdout, sys.stderr):
            continue
        try:
            handler.close()
        except Exception:
            pass

    # Console handler — always attached; disk logging is a convenience,
    # not a correctness requirement, so we install console first.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler with rotation — best effort. A read-only container
    # filesystem or a restricted-mount working directory used to crash
    # TenantRuntime.__init__ here; we now warn via the console handler
    # and continue stdout-only.
    log_file = _install_file_handler(
        root_logger,
        app_name,
        mode,
        log_dir,
        backup_count,
        level,
        detailed_formatter,
    )

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    if log_file is not None:
        root_logger.info(f"Logging initialized: level={level_str}, file={log_file}")
    else:
        root_logger.info(f"Logging initialized: level={level_str}, file=<disabled>")

    return root_logger


# =============================================================================
# Structured Logging Utilities
# =============================================================================

# Logger for this module's utility functions
_log_utils_logger = logging.getLogger(__name__)

# Truncation limits for log readability
MAX_CONTENT_LENGTH = 200

_MAX_LOGGED_PAYLOAD_FIELDS = 20
_MAX_LOGGED_FIELD_NAME_CHARS = 64
_MAX_REDACT_FIELDS = 100


def _is_log_unsafe_codepoint(code: int) -> bool:
    # C0, DEL, and C1 (including CSI U+009B) must not reach the stdout log sink.
    return code < 32 or 127 <= code <= 0x9F


def _bounded_field_name(key: Any) -> str:
    """Return a log-safe, fixed-size field name without stringifying values."""
    if type(key) is not str:
        return "<non-string>"
    prefix = key[:_MAX_LOGGED_FIELD_NAME_CHARS]
    safe = "".join("?" if _is_log_unsafe_codepoint(ord(char)) else char for char in prefix)
    return safe + ("..." if len(key) > _MAX_LOGGED_FIELD_NAME_CHARS else "")


def _shallow_value_summary(value: Any) -> str:
    """Describe one value using only O(1) type/length operations."""
    value_type = type(value)
    if value_type is str:
        return f"string chars={len(value)}"
    if value_type in (bytes, bytearray):
        return f"bytes length={len(value)}"
    if value_type in (list, tuple, set, frozenset):
        return f"array items={len(value)}"
    if value_type is dict:
        return "object"
    if value is None:
        return "null"
    if value_type is bool:
        return "boolean"
    if value_type in (int, float):
        return "number"
    return "object"


def _payload_debug_summary(
    payload: Any,
    fields_to_redact: Optional[List[str]] = None,
) -> str:
    """Describe a payload without serializing or recursively walking it."""
    if type(payload) is not dict:
        return f"type={_shallow_value_summary(payload)} values_omitted=true"

    redact_all = False
    redacted_fields: set[str] = set()
    # isinstance (not `type(...) in`) so the type checker narrows Optional[List]
    # and a malformed catalog value still degrades to no redaction.
    if isinstance(fields_to_redact, (list, tuple)):
        if len(fields_to_redact) > _MAX_REDACT_FIELDS:
            redact_all = True
        else:
            redacted_fields = {field for field in fields_to_redact if type(field) is str}

    fields: list[str] = []
    fields_truncated = False
    for index, (key, value) in enumerate(payload.items()):
        if index >= _MAX_LOGGED_PAYLOAD_FIELDS:
            fields_truncated = True
            break
        name = _bounded_field_name(key)
        summary = (
            "redacted"
            if redact_all or (type(key) is str and key in redacted_fields)
            else _shallow_value_summary(value)
        )
        fields.append(f"{name}=<{summary}>")

    return (
        f"fields=[{', '.join(fields)}] "
        f"fields_truncated={str(fields_truncated).lower()} values_omitted=true"
    )


def _result_debug_summary(result: Any) -> str:
    """Describe a result without exposing dynamic object keys."""
    return f"type={_shallow_value_summary(result)} values_omitted=true"


def log_separator() -> None:
    """Log a minor section separator (for individual operations)."""
    _log_utils_logger.info("-" * 40)


def log_section() -> None:
    """Log a major section separator (for execution boundaries)."""
    _log_utils_logger.info("=" * 60)


def log_llm_messages(
    messages: List[Any],
    prefix: str = "LLM",
    count_only: bool = False,
) -> None:
    """
    Log LLM conversation messages in a consistent format.

    Args:
        messages: List of message dicts or LangChain message objects
        prefix: Log line prefix (e.g., "OE", "LLM", "AER")
        count_only: If True, only log message count (for info level)
    """
    if count_only:
        _log_utils_logger.info(f"{prefix}: {len(messages)} messages")
        return

    _log_utils_logger.debug(f"{prefix}: {len(messages)} messages:")
    for i, msg in enumerate(messages):
        # Handle both dict and LangChain message objects
        if isinstance(msg, dict):
            msg_type = msg.get("type", "unknown")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls")
        else:
            msg_type = getattr(msg, "type", "unknown")
            content = getattr(msg, "content", "")
            tool_calls = getattr(msg, "tool_calls", None)

        content_preview = str(content)[:MAX_CONTENT_LENGTH] if content else "(empty)"

        if tool_calls:
            _log_utils_logger.debug(
                f"{prefix}:   [{i}] {msg_type}: {content_preview}... + {len(tool_calls)} tool_calls"
            )
        else:
            _log_utils_logger.debug(f"{prefix}:   [{i}] {msg_type}: {content_preview}...")


def log_llm_response(
    result: Any,
    step: int,
    prefix: str = "LLM",
) -> None:
    """
    Log an LLM response in a consistent format.

    Args:
        result: LLM response (dict or AIMessage)
        step: Step number
        prefix: Log line prefix
    """
    if result is None:
        return

    if isinstance(result, dict):
        content = result.get("content", "")
        tool_calls = result.get("tool_calls", [])
    else:
        content = getattr(result, "content", "")
        tool_calls = getattr(result, "tool_calls", [])

    content_preview = str(content)[:MAX_CONTENT_LENGTH] if content else "(empty)"
    _log_utils_logger.debug(f"{prefix}: Step {step} - Response: {content_preview}...")

    if tool_calls:
        tool_names = []
        for tc in tool_calls:
            name = tc.get("name", tc) if isinstance(tc, dict) else getattr(tc, "name", str(tc))
            tool_names.append(name)
        _log_utils_logger.info(f"{prefix}: Step {step} - Tool calls: {tool_names}")


def log_tool_request(
    tool_name: str,
    arguments: Dict[str, Any],
    step: int,
    prefix: str = "TOOL",
    fields_to_redact: Optional[List[str]] = None,
) -> None:
    """
    Log a tool execution request.

    Args:
        tool_name: Name of the tool
        arguments: Tool arguments
        step: Step number
        prefix: Log line prefix
        fields_to_redact: Catalog sensitive-field policy. Matching fields omit
            even their type/length metadata; no values are logged.
    """
    log_separator()
    _log_utils_logger.info("%s: Step %s - %s", prefix, step, tool_name)
    if _log_utils_logger.isEnabledFor(logging.DEBUG):
        _log_utils_logger.debug(
            "%s: Step %s - Arguments: %s",
            prefix,
            step,
            _payload_debug_summary(arguments, fields_to_redact),
        )


def log_tool_result(
    tool_name: str,
    step: int,
    status: str,
    result: Any = None,
    error: Optional[str] = None,
    duration_ms: float = 0,
    prefix: str = "TOOL",
) -> None:
    """
    Log a tool execution result.

    Args:
        tool_name: Name of the tool
        step: Step number
        status: Execution status (success, error, suspend, interrupted)
        result: Tool result
        error: Error message if failed
        duration_ms: Execution duration in milliseconds
        prefix: Log line prefix
    """
    _log_utils_logger.info(
        "%s: Step %s - %s %s (%.0fms)", prefix, step, tool_name, status, duration_ms
    )

    if error:
        _log_utils_logger.error("%s: Step %s - Error: %s", prefix, step, error)
    elif result is not None and _log_utils_logger.isEnabledFor(logging.DEBUG):
        # Do not str() the result: that allocates the full body before slice.
        _log_utils_logger.debug(
            "%s: Step %s - Result: %s",
            prefix,
            step,
            _result_debug_summary(result),
        )


def log_cached_result(
    tool_name: str,
    step: int,
    prefix: str = "TOOL",
) -> None:
    """Log that a cached result is being used (replay scenario)."""
    _log_utils_logger.info(f"{prefix}: Step {step} - {tool_name} using CACHED result (replay)")


def log_policy_blocked(
    tool_name: str,
    step: int,
    reason: str,
    prefix: str = "TOOL",
) -> None:
    """Log that a tool call was blocked by policy."""
    _log_utils_logger.warning(f"{prefix}: Step {step} - {tool_name} BLOCKED by policy: {reason}")


def log_execution_start(
    execution_id: str,
    input_keys: List[str],
    prefix: str = "OE",
) -> None:
    """Log the start of an execution."""
    log_section()
    _log_utils_logger.info(f"{prefix}: Starting execution {execution_id[:8]}...")
    _log_utils_logger.debug(f"{prefix}: Input keys: {input_keys}")


def log_execution_callback(
    execution_id: str,
    status: str,
    result: Any = None,
    error: Optional[str] = None,
    suspend_reason: Optional[str] = None,
    prefix: str = "OE",
) -> None:
    """Log an execution callback (completion/suspension/error)."""
    log_section()
    _log_utils_logger.info(f"{prefix}: Execution {execution_id[:8]}... → {status}")

    if suspend_reason:
        _log_utils_logger.info(f"{prefix}: Suspend reason: {suspend_reason}")
    if error:
        _log_utils_logger.error(f"{prefix}: Error: {error}")
    if result:
        result_preview = str(result)[:300]
        _log_utils_logger.debug(f"{prefix}: Result: {result_preview}...")
