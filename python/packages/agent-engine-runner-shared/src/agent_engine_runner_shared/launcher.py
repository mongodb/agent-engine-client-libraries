"""
Launcher module for pipeline-built agent containers.

Reads AGENT_ENTRYPOINT and starts the user's agent code. This module exists so
the generated Dockerfile can use a fixed CMD instruction for every agent image:

    CMD ["/app/.venv/bin/python3", "-m", "agent_engine_runner_shared.launcher"]

Structured logging is installed at the start of ``main()`` (before import)
so import-time and entrypoint failures emit ``ERROR`` records instead of
raw stderr tracebacks. ``TenantRuntime`` re-installs later; that call is
idempotent. When ``STRUCTURED_LOGGING`` is unset, the same failures still
log at ERROR via the legacy text formatter.

Environment variables:
    AGENT_ENTRYPOINT  Startup target in one of these forms:

                      1) ``module.path:function_name`` (e.g. ``my_agent.main:main``)
                         - The function is imported and invoked.
                      2) ``module.path:attribute_name`` (e.g. ``my_agent.agent:app``)
                         - If the attribute is not callable but has a callable
                           ``run()`` method, ``.run()`` is invoked.
                      3) ``module.path`` (e.g. ``my_agent.main``)
                         - Defaults to calling ``module.path:main``.

Security note:
    The AGENT_ENTRYPOINT value is baked into the Dockerfile ENV at build time
    by the CodeBuild pipeline (see buildspec.yml.tmpl). It is not user-supplied
    at container runtime. The importlib.import_module call here serves the same
    role as ``python -m <module>`` — bootstrapping a known, pre-installed module.
    This is an intentional exception to the sec-003 "no dynamic code execution"
    guideline; the user's agent code is already installed in the image and would
    be imported regardless of the mechanism used to start it.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
import traceback
from typing import NoReturn

from agent_engine_runner_shared.error_reporting import _redact_text
from agent_engine_runner_shared.mcp_oauth_secret import materialize_mcp_oauth_secret_cache
from agent_engine_runner_shared.utils import get_runtime_mode, setup_logging

logger = logging.getLogger(__name__)

# Kubernetes' default termination-message path (the "File"
# TerminationMessagePolicy Kubelet always checks first, before falling back to
# log-tail capture). Nothing in the deployed pod spec sets a custom
# terminationMessagePath, so the default applies. Module-level so a test can
# monkeypatch it to a temp file.
_TERMINATION_LOG_PATH = "/dev/termination-log"

# 4 KiB / 40 lines, whichever is reached first — the same bound the deploy
# event pipeline applies (see pkg/logredaction.DefaultMaxBytes/DefaultMaxLines
# in the Go services; kept in sync by convention, not by a shared import,
# since there's no cross-language shared package for this).
_TERMINATION_MESSAGE_MAX_BYTES = 4 * 1024
_TERMINATION_MESSAGE_MAX_LINES = 40


def _bound_text(
    text: str,
    max_bytes: int = _TERMINATION_MESSAGE_MAX_BYTES,
    max_lines: int = _TERMINATION_MESSAGE_MAX_LINES,
) -> str:
    """Truncate ``text`` to at most ``max_lines`` lines or ``max_bytes`` UTF-8
    bytes, whichever limit is hit first, appending a truncation marker when
    either limit was hit. Mirrors pkg/logredaction.Bound's contract."""
    marker = "\n[truncated]"
    truncated = False
    lines = text.split("\n")
    if len(lines) > max_lines:
        text = "\n".join(lines[:max_lines])
        truncated = True

    encoded = text.encode("utf-8")
    if len(encoded) > max_bytes:
        # Trim on a UTF-8 boundary: decode with errors="ignore" after cutting
        # so a multi-byte character split at the boundary is dropped cleanly
        # rather than raising or leaving a mangled trailing byte sequence.
        text = encoded[:max_bytes].decode("utf-8", errors="ignore")
        truncated = True

    if not truncated:
        return text

    # Reserve headroom for `marker` so appending it below can't push the
    # final write past max_bytes — kubelet's own read of
    # /dev/termination-log is a blind byte-level cut that would otherwise
    # mangle the tail, potentially the marker itself.
    budget = max(0, max_bytes - len(marker.encode("utf-8")))
    encoded = text.encode("utf-8")
    if len(encoded) > budget:
        text = encoded[:budget].decode("utf-8", errors="ignore")
    return text + marker


def _write_termination_message(summary: str, exc: BaseException | None = None) -> None:
    """Record a bounded, redacted summary of a fatal startup error to the
    container's termination message before the process exits, so the real
    cause of the crash survives past this process's own stdout into
    ContainerStatus.LastTerminationState.Terminated.Message — the field the
    Agentic Operator's crash diagnostics read, and from there into the
    customer-facing deploy timeline.

    This is customer code (container mode), so unlike the platform's own
    components the design intentionally keeps the full exception type,
    message, and traceback — redacted, not summarized away — since that
    detail is what the customer needs to fix their own agent. Best-effort:
    if the write fails, nothing is lost beyond what
    TerminationMessagePolicy: FallbackToLogsOnError already provides.
    """
    text = summary
    if exc is not None:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        text = f"{summary}\n{tb}"
    text = _redact_text(text)
    text = _bound_text(text)
    try:
        with open(_TERMINATION_LOG_PATH, "w", encoding="utf-8") as f:
            f.write(text)
    except OSError:
        pass


# Startup-failure exit codes. OE's readiness wait can observe a
# workload's real exit code once it exits (via fctr's Wait RPC) but not the
# exception that caused it, so the exit code itself is the only signal that
# reliably survives a startup crash to reach OE. Chosen to avoid every range
# fctr's own const.go already claims: 0/1 (generic), 64-78 (sysexits.h), and
# 128+signal (signal deaths, e.g. 137=SIGKILL, 143=SIGTERM).
#
# Frozen wire contract: OE's classification of a startup failure depends on
# these exact values, and they are duplicated in
# agent-engine-runner-shared/src/launcher.ts. Changing either file breaks OE's ability
# to distinguish failure causes and/or cross-language parity — keep the two
# in lockstep.
EXIT_IMPORT_ERROR = 82
EXIT_NO_ENTRYPOINT = 83
EXIT_STARTUP_CRASH = 84


def _fatal(summary: str, exc: BaseException | None = None, code: int = 1) -> NoReturn:
    """Log an ERROR (with traceback when exc is given), write a bounded,
    redacted termination message, and exit. ``SystemExit`` is not
    ``Exception``, so a surrounding ``except Exception`` in ``main`` will not
    swallow this.

    Routes every deployment-breaking launcher exit through one path — rather
    than instrumenting a hand-picked few — so the container's termination
    message always carries the real cause forward to
    ContainerStatus.LastTerminationState.Terminated.Message.
    """
    if exc is not None:
        logger.error(f"{summary}: {exc}", exc_info=True)
    else:
        logger.error(summary)
    _write_termination_message(summary, exc)
    sys.exit(code)


def _resolve_runner_mode() -> str:
    """Resolve RUNNER_MODE for the early structured-logging install, falling
    back to "aer" when unset or unrecognized. An unrecognized value would reach
    the record's ``service`` field verbatim, and ``/agent-logs`` drops anything
    outside ``{agent-execution-runtime, tool-executor}`` — so the crash log
    would vanish. AGENT_ENTRYPOINT pods only ever run aer/tool.
    """
    try:
        return get_runtime_mode().value
    except ValueError:
        return "aer"


def _resolve_entrypoint() -> tuple[str, str]:
    """Parse AGENT_ENTRYPOINT into (module_path, target_name)."""
    # Surrounding whitespace is stripped because the build pipeline extracts this
    # value from agent.yaml with a shell pipeline: a file authored on Windows
    # leaves a trailing CR that would otherwise become part of the attribute name
    # looked up below.
    raw = os.environ.get("AGENT_ENTRYPOINT", "").strip()
    if not raw:
        _fatal("AGENT_ENTRYPOINT is not set", code=EXIT_NO_ENTRYPOINT)

    if ":" in raw:
        module_path, func_name = raw.rsplit(":", 1)
    else:
        module_path = raw
        func_name = "main"

    module_path = module_path.strip()
    func_name = func_name.strip()

    if not module_path or not func_name:
        _fatal(
            f"Invalid AGENT_ENTRYPOINT '{raw}': both module path and function name must be non-empty",
            code=EXIT_NO_ENTRYPOINT,
        )

    return module_path, func_name


def main() -> None:
    """Entry point invoked by ``python -m agent_engine_runner_shared.launcher``."""
    # Install before any import/entrypoint work. After this, print-to-stderr
    # is captured as WARNING (not ERROR), so failure paths must use the logger.
    setup_logging(app_name="launcher", mode=_resolve_runner_mode())

    try:
        materialize_mcp_oauth_secret_cache()
    except (RuntimeError, OSError) as exc:
        _fatal("Cannot materialize MCP OAuth credentials", exc)

    module_path, target_name = _resolve_entrypoint()

    logger.info("Launcher: importing %s:%s", module_path, target_name)

    try:
        from agent_engine_runner_shared.context import customer_origin_scope

        with customer_origin_scope():
            module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        # This also fires when the agent module imports fine but something
        # *it* imports is missing, which is the common case.
        _fatal(f"Cannot import module '{module_path}'", exc, code=EXIT_IMPORT_ERROR)
    except Exception as exc:
        _fatal(f"Failed to import module '{module_path}'", exc, code=EXIT_IMPORT_ERROR)

    target = getattr(module, target_name, None)
    if target is None:
        _fatal(f"Module '{module_path}' has no attribute '{target_name}'", code=EXIT_NO_ENTRYPOINT)

    # Preferred form: module:function (callable).
    if callable(target):
        try:
            with customer_origin_scope():
                target()
        except Exception as exc:
            _fatal(
                f"Unhandled exception from agent entrypoint '{module_path}:{target_name}'",
                exc,
                code=EXIT_STARTUP_CRASH,
            )
        return

    # Compatibility form: module:app_object where app_object has a run() method.
    run = getattr(target, "run", None)
    if callable(run):
        try:
            with customer_origin_scope():
                run()
        except Exception as exc:
            _fatal(
                f"Unhandled exception from agent entrypoint '{module_path}:{target_name}'",
                exc,
                code=EXIT_STARTUP_CRASH,
            )
        return

    _fatal(
        f"'{module_path}:{target_name}' is not callable and has no callable 'run()' method",
        code=EXIT_NO_ENTRYPOINT,
    )


if __name__ == "__main__":
    main()
