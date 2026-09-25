"""Shared local error reporting helpers for CLI-generated runtime flows."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import sentry_sdk
    from sentry_sdk._types import LogLevelStr
    from sentry_sdk.types import Event, Hint
else:
    try:
        import sentry_sdk
        from sentry_sdk.types import Event, Hint
    except ImportError:
        sentry_sdk = None  # type: ignore[assignment]
        Event = dict[str, Any]  # type: ignore[misc,assignment]
        Hint = dict[str, Any]  # type: ignore[misc,assignment]
    LogLevelStr = str

_ENABLED = False
_MAX_SUMMARY_LINE_LENGTH = 240
_MAX_OUTPUT_TAIL_CHARS = 8 * 1024
# Inline keyword=value / keyword: value. The separator tolerates the key's
# closing quote in EITHER style and whitespace on either side ("token": "v",
# 'token': 'v', token = v) — single-quoted shapes are the Python-repr /
# JS-dump norm for error bodies, and a double-quote-only separator lets them
# through. The value is an escape-aware quoted string in either quote style,
# the same quoted run UNTERMINATED at end of input (length-capped messages),
# or a bare run extending through ';' and through ',' inside an opaque secret
# (password=part1;part2 must not leak its suffix) but stopping at a ',' that
# begins another field — the lookahead treats a following key= / key: token
# as a field boundary, so token=abc,requestId=r1 keeps its siblings readable.
# Same contract as the local-dev-ui scrubbers.
_SECRET_INLINE_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|token|secret|password)"
    r"([\"']?\s*[=:]\s*)"
    r"(\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*$|'(?:\\.|[^'\\])*$|[^\s,]+(?:,(?!\s*[\"']?[A-Za-z0-9_.\-]+[\"']?\s*[=:])[^\s,]+)*)"
)
_BEARER_PATTERN = re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._\-]+")
# Key-name blocklist for structured payloads: a bare secret under a
# secret-named field matches no inline content pattern, so the field name
# itself must trigger masking.
_SECRET_KEY_PATTERN = re.compile(
    r"^(api[_-]?key|access[_-]?token|refresh[_-]?token|id[_-]?token|token|secret|password|credential|authorization|private[_-]?key|client[_-]?secret)$",
    re.IGNORECASE,
)
# Same GREEDY-to-last-'@' semantics as agent_engine_runner_shared.tracing.scrub_credentials

# Generalized to any scheme: the old pattern stopped at the
# first '@' and could not cross whitespace, leaking the password tail of URIs
# whose password carried an unescaped '@' or space into Sentry events. The
# tempered run never crosses a second URL's scheme and may over-redact text
# after the URL that carries its own '@' — fail-closed on purpose.
# Userinfo on any scheme, greedy to the LAST '@' before '/' or whitespace —
# so passwords containing an unescaped '@' are fully masked on every scheme,
# while credential-free URLs followed by prose (an email, a second URI) are
# never touched: the run cannot cross '/' or whitespace, so it stops at the
# authority boundary. Passwords containing a space, newline, or '/' on
# non-mongodb schemes remain uncovered — that shape is indistinguishable
# from prose around a plain URL, and mongodb(+srv) is fully covered by the
# greedy pattern below.
_URL_USERINFO_PATTERN = re.compile(r"(?i)\b([a-z][a-z0-9+.\-]*://)[^/\s]*@")

# Malformed userinfo — passwords carrying an unescaped '@', space, '/', or an
# embedded newline — matched greedily to the LAST '@' before a host-shaped
# token. This engages only for mongodb(+srv)://: Mongo driver errors are the
# documented source of such echoes, and a scheme-bounded greedy run on
# generic text would collapse credential-free URLs and the prose around them
# (host, path, trailing email) on every Sentry event surface. The run is
# tempered so it never crosses a second URI's scheme, and is [^"'] (any char
# except quotes) so it still crosses embedded newlines but stops at a JSON
# string boundary — a credential echoed inside a compact JSON body must not
# fold later siblings into the mask. Same semantics as the tracing scrubbers.
_MONGO_URI_USERINFO_GREEDY_PATTERN = re.compile(
    r"(mongodb(?:\+srv)?://)((?:(?!mongodb(?:\+srv)?://)[^\"'])*)@(?=[^\s@/?#\"'][^\s@?\"']*(?:[/?#:\s\"']|$))",
    re.IGNORECASE,
)


def _resolve_home_dir() -> str:
    try:
        home_dir = str(Path.home())
    except (RuntimeError, KeyError):
        home_dir = os.getenv("HOME", "").strip()
    return home_dir.strip()


_HOME_DIR = _resolve_home_dir()


def init_error_reporting(
    *, surface: str, component: str | None = None, mode: str | None = None
) -> bool:
    global _ENABLED

    dsn = os.getenv("AGENTIC_SENTRY_DSN", "").strip()
    _ENABLED = sentry_sdk is not None and os.getenv("AGENTIC_SENTRY_ENABLED") == "1" and bool(dsn)
    if not _ENABLED:
        return False

    sentry_sdk.init(
        dsn=dsn,
        environment=os.getenv("AGENTIC_SENTRY_ENVIRONMENT", "local-dev"),
        release=os.getenv("AGENTIC_SENTRY_RELEASE") or None,
        traces_sample_rate=0.0,
        debug=os.getenv("AGENTIC_SENTRY_DEBUG") == "1",
        before_send=_before_send,
    )
    sentry_sdk.set_tag("surface", surface)
    if component:
        sentry_sdk.set_tag("component", component)
    if mode:
        sentry_sdk.set_tag("mode", mode)

    return True


def capture_exception(exc: BaseException, *, summary: str | None = None, **extra: Any) -> None:
    if not _ENABLED:
        return
    captured_exc = _exception_for_capture(exc, summary)
    with sentry_sdk.push_scope() as scope:
        if summary:
            scope.set_extra("original_exception_type", type(exc).__name__)
            scope.set_extra("original_exception_message", _redact_text(str(exc)))
        for key, value in extra.items():
            scope.set_extra(key, _redact_value(value, key))
        sentry_sdk.capture_exception(captured_exc)


def capture_message(message: str, level: LogLevelStr = "error", **extra: Any) -> None:
    if not _ENABLED:
        return
    with sentry_sdk.push_scope() as scope:
        for key, value in extra.items():
            scope.set_extra(key, _redact_value(value, key))
        sentry_sdk.capture_message(_redact_text(message), level=level)


def flush(timeout: float = 2.0) -> None:
    if _ENABLED:
        sentry_sdk.flush(timeout=timeout)


def summarize_subprocess_failure(exc: subprocess.CalledProcessError) -> str:
    command = _command_label(exc.cmd)
    headline = _failure_headline(subprocess_output_tail(exc))
    if headline:
        return f"{command} failed: {headline}"
    return f"{command} failed with exit {exc.returncode}"


def subprocess_output_tail(exc: subprocess.CalledProcessError) -> str:
    chunks: list[str] = []
    for value in (getattr(exc, "output", None), getattr(exc, "stderr", None)):
        if value is None:
            continue
        if isinstance(value, bytes):
            chunks.append(value.decode("utf-8", errors="replace"))
        else:
            chunks.append(str(value))
    if not chunks:
        return ""
    return _tail_text("\n".join(chunks))


def _before_send(event: Event, _hint: Hint) -> Event | None:
    event_data = cast(dict[str, Any], event)
    event_data["server_name"] = None
    if "message" in event_data:
        event_data["message"] = _redact_text(str(event_data["message"]))
    for section in (
        "request",
        "user",
        "tags",
        "extra",
        "contexts",
        "exception",
        "breadcrumbs",
        "threads",
        "logentry",
        "modules",
    ):
        payload = event_data.get(section)
        if isinstance(payload, dict):
            event_data[section] = {key: _redact_value(value, key) for key, value in payload.items()}
        elif isinstance(payload, list):
            event_data[section] = [_redact_value(item) for item in payload]
    return event


def _is_usable_home_dir(home_dir: str) -> bool:
    """False when the home is the filesystem root or normalizes to it ("/",
    "//", "/..", "/." — common for passwd-less container UIDs). Replacing
    every "/" would destroy the "://" anchor the credential patterns match
    on and leak secrets into the report — the same degenerate-HOME guard as
    the CLI telemetry and the local-dev Node client.
    """
    return bool(home_dir) and os.path.normpath(home_dir).rstrip("/") != ""


def _redact_value(value: Any, key: str | None = None) -> Any:
    # A secret-named field's value is masked outright: bare secrets match no
    # inline content pattern, so the key name is the only signal.
    if key and _SECRET_KEY_PATTERN.match(key) and isinstance(value, str) and value:
        return "<redacted>"
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, Path):
        return _redact_text(str(value))
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, dict):
        return {k: _redact_value(item, k) for k, item in value.items()}
    return value


def _redact_inline_secret(match: "re.Match[str]") -> str:
    keyword, separator, value = match.group(1), match.group(2), match.group(3)
    # Keep a quoted value quoted in its ORIGINAL style so a redacted body
    # stays parseable; an unterminated quoted run (length-capped input) gets
    # the closing quote back, matching the local-dev scrubbers' contract.
    if value[:1] in ('"', "'"):
        quote = value[0]
        return f"{keyword}{separator}{quote}<redacted>{quote}"
    return f"{keyword}{separator}<redacted>"


def _redact_text(text: str) -> str:
    if not text:
        return text

    redacted = text
    if _is_usable_home_dir(_HOME_DIR):
        redacted = redacted.replace(_HOME_DIR, "<home>")
    redacted = _MONGO_URI_USERINFO_GREEDY_PATTERN.sub(r"\1<redacted>:<redacted>@", redacted)
    redacted = _URL_USERINFO_PATTERN.sub(r"\1<redacted>:<redacted>@", redacted)
    redacted = _SECRET_INLINE_PATTERN.sub(_redact_inline_secret, redacted)
    redacted = _BEARER_PATTERN.sub(r"\1 <redacted>", redacted)
    return redacted


def _exception_for_capture(exc: BaseException, summary: str | None) -> BaseException:
    if not summary:
        return exc
    wrapped = RuntimeError(_redact_text(summary))
    if exc.__traceback__ is not None:
        return wrapped.with_traceback(exc.__traceback__)
    return wrapped


def _command_label(cmd: object) -> str:
    if isinstance(cmd, (list, tuple)):
        parts = [str(part) for part in cmd if str(part).strip()]
        if not parts:
            return "subprocess"
        if len(parts) >= 3 and parts[0] == "uv" and parts[1] == "pip":
            return " ".join(parts[:3])
        if len(parts) >= 2:
            return " ".join(parts[:2])
        return parts[0]
    text = str(cmd).strip()
    return text or "subprocess"


def _failure_headline(output: str) -> str:
    lines = [_normalize_failure_line(line) for line in output.splitlines()]
    normalized = [line for line in lines if line]
    for line in reversed(normalized):
        lower = line.lower()
        if (
            "failed" in lower
            or "error" in lower
            or "not found" in lower
            or "no matching" in lower
            or "exception" in lower
        ):
            return _truncate_summary_line(line)
    if not normalized:
        return ""
    return _truncate_summary_line(normalized[-1])


def _normalize_failure_line(line: str) -> str:
    cleaned = line.strip().lstrip("│>").strip()
    if cleaned.startswith("×"):
        cleaned = cleaned[1:].strip()
    if cleaned.startswith("╰─▶"):
        cleaned = cleaned[3:].strip()
    if cleaned.lower().startswith("error:"):
        cleaned = cleaned[6:].strip()
    return cleaned


def _tail_text(text: str) -> str:
    trimmed = text.strip()
    if len(trimmed) <= _MAX_OUTPUT_TAIL_CHARS:
        return trimmed
    return trimmed[-_MAX_OUTPUT_TAIL_CHARS:]


def _truncate_summary_line(line: str) -> str:
    if len(line) <= _MAX_SUMMARY_LINE_LENGTH:
        return line
    return line[: _MAX_SUMMARY_LINE_LENGTH - 3] + "..."
