import subprocess

import agent_engine_runner_shared.error_reporting as error_reporting
from agent_engine_runner_shared.error_reporting import (
    _before_send,
    _redact_text,
    _redact_value,
    subprocess_output_tail,
    summarize_subprocess_failure,
)


def test_redact_text_scrubs_common_secret_patterns() -> None:
    home_dir = error_reporting._HOME_DIR or "/tmp/example-home"
    redacted = _redact_text(
        f"token=abc123 password:shhh mongodb+srv://user:pass@example.mongodb.net/app {home_dir}/project"
    )

    assert "abc123" not in redacted
    assert "shhh" not in redacted
    assert "token=<redacted>" in redacted
    assert "password:<redacted>" in redacted
    assert "mongodb+srv://<redacted>:<redacted>@example.mongodb.net/app" in redacted
    assert home_dir not in redacted


def test_redact_text_masks_single_quoted_secrets() -> None:
    """Single-quoted keys/values are the Python-repr norm for error bodies —
    a double-quote-only separator lets them through to Sentry."""
    redacted = _redact_text("{'api_key': 'sk-live-9', 'status': 400}")
    assert redacted == "{'api_key': '<redacted>', 'status': 400}"

    assert _redact_text("password='hunter two spaces'") == "password='<redacted>'"
    assert _redact_text("{'password':'p\\'w\\'d'}") == "{'password':'<redacted>'}"
    assert _redact_text("{'token':42,'status':400}") == "{'token':<redacted>,'status':400}"
    assert _redact_text("{'password':'my secret value") == "{'password':'<redacted>'"


def test_redact_text_masks_semicolon_and_comma_inside_bare_secrets() -> None:
    """A bare secret's ';' or ',' is inside the value, not a field boundary;
    a comma followed by a key-shaped token IS a boundary."""
    assert _redact_text("password=part1;part2 rest") == "password=<redacted> rest"
    assert _redact_text("secret=a,b;c,d done") == "secret=<redacted> done"
    assert (
        _redact_text("token=abc,requestId=r1,status=400")
        == "token=<redacted>,requestId=r1,status=400"
    )


def test_redact_text_root_home_still_redacts_credentials() -> None:
    """HOME=/ (passwd-less container UID) must not destroy the credential
    patterns' "://" anchor — the degenerate home is skipped, not replaced."""
    original_home_dir = error_reporting._HOME_DIR
    error_reporting._HOME_DIR = "/"
    try:
        redacted = _redact_text("connect failed: mongodb+srv://user:pass@example.mongodb.net/app")
        assert (
            redacted
            == "connect failed: mongodb+srv://<redacted>:<redacted>@example.mongodb.net/app"
        )
        assert "pass" not in redacted
    finally:
        error_reporting._HOME_DIR = original_home_dir


def test_redact_text_degenerate_home_variants() -> None:
    original_home_dir = error_reporting._HOME_DIR
    try:
        for home in ["/", "//", "/..", "/."]:
            error_reporting._HOME_DIR = home
            redacted = _redact_text("dsn mongodb://user:secret@host/db end")
            assert redacted == "dsn mongodb://<redacted>:<redacted>@host/db end", home
            assert "secret" not in redacted
    finally:
        error_reporting._HOME_DIR = original_home_dir


def test_redact_text_normal_home_still_scrubbed() -> None:
    original_home_dir = error_reporting._HOME_DIR
    error_reporting._HOME_DIR = "/home/testuser"
    try:
        assert (
            _redact_text("/home/testuser/project/main.go failed") == "<home>/project/main.go failed"
        )
    finally:
        error_reporting._HOME_DIR = original_home_dir


def test_redact_value_masks_bare_secrets_under_secret_named_keys() -> None:
    """extra={"apiKey": "sk-..."} must never reach Sentry: a bare secret
    matches no inline text pattern, so the key name is the only signal."""
    redacted = _redact_value(
        {
            "apiKey": "sk-proj-abc123",
            "token": "ghp_secret",
            "nested": {"password": "hunter2", "note": "keep"},
            "normal": "hello",
            "count": 42,
        }
    )

    assert redacted["apiKey"] == "<redacted>"
    assert redacted["token"] == "<redacted>"
    assert redacted["nested"]["password"] == "<redacted>"
    assert redacted["nested"]["note"] == "keep"
    assert redacted["normal"] == "hello"
    assert redacted["count"] == 42


def test_redact_value_blocklist_is_case_insensitive_and_skips_empty() -> None:
    for key in ["apiKey", "API_KEY", "accessToken", "clientSecret", "Authorization"]:
        assert _redact_value("supersecret-value", key) == "<redacted>", key
    assert _redact_value("", "token") == ""
    assert _redact_value(42, "token") == 42


def test_redact_text_masks_quoted_secret_values_in_full() -> None:
    """A quoted secret containing spaces must be masked whole (escape-aware),
    not truncated at the first space — and the quotes stay so JSON parses."""
    redacted = _redact_text('{"password": "hunter 2", "apiKey":"sk-live key", "note": "keep me"}')
    assert "hunter" not in redacted
    assert "sk-live" not in redacted
    assert '"password": "<redacted>"' in redacted
    assert '"apiKey":"<redacted>"' in redacted
    assert '"note": "keep me"' in redacted


def test_redact_text_masks_escaped_quote_inside_quoted_secret() -> None:
    redacted = _redact_text('{"password": "hunt\\"er 2"}')
    assert "hunt" not in redacted
    assert redacted == '{"password": "<redacted>"}'


def test_redact_text_masks_at_bearing_passwords_on_any_scheme() -> None:
    """A password containing an unescaped '@' must be fully masked on generic
    schemes too — not just mongodb — without touching credential-free URLs."""
    redacted = _redact_text("db1 postgres://user:p@ss@db.internal/prod; db2 redis://svc@ss@cache/0")
    assert "p@ss" not in redacted
    assert "svc@ss" not in redacted
    assert "postgres://<redacted>:<redacted>@db.internal/prod" in redacted
    assert "redis://<redacted>:<redacted>@cache/0" in redacted


def test_redact_text_preserves_credential_free_urls_and_prose() -> None:
    """A plain URL followed by prose containing an '@' (an email, a second
    URI) must survive intact — a scheme-bounded greedy run would otherwise
    collapse the host, path, and intervening diagnostic text."""
    text = "GET https://api.example.com/v1/users failed with 503; contact ops@corp.com for help"
    assert _redact_text(text) == text


def test_redact_text_mongo_credential_in_json_body_preserves_siblings() -> None:
    """A mongodb credential echoed inside a compact JSON body must be masked
    without folding later siblings (an email in the next field) into it."""
    redacted = _redact_text('{"error":"mongodb://u:p@h1/db","email":"a@b.com"}')
    assert redacted == '{"error":"mongodb://<redacted>:<redacted>@h1/db","email":"a@b.com"}'


def test_redact_text_masks_uri_userinfo_with_special_char_passwords() -> None:
    """Passwords carrying an unescaped '@' or whitespace defeated the old
    first-'@' pattern and leaked into Sentry events."""
    redacted = _redact_text(
        "err mongodb+srv://user@ssword@example.mongodb.net/app; "
        "err2 mongodb://user:pass word@host/db; "
        "err3 mongodb://user:pass\nword@host/db; "
        "pair mongodb://u:p@h1/db and mongodb://u2:p2@h2/db2"
    )

    assert "ssword" not in redacted
    assert "pass word" not in redacted
    assert "pass\nword" not in redacted
    assert "u:p" not in redacted
    assert "u2:p2" not in redacted
    assert "mongodb+srv://<redacted>:<redacted>@example.mongodb.net/app" in redacted
    assert redacted.count("mongodb://<redacted>:<redacted>@host/db") == 2
    assert "mongodb://<redacted>:<redacted>@h1/db" in redacted
    assert "mongodb://<redacted>:<redacted>@h2/db2" in redacted


def test_summarize_subprocess_failure_prefers_meaningful_output_line() -> None:
    exc = subprocess.CalledProcessError(
        1,
        ["uv", "sync", "--no-dev"],
        output=(
            "Resolved 12 packages\n"
            "  × No solution found when resolving dependencies:\n"
            "  ╰─▶ Because missing-package was not found in the package registry\n"
        ),
    )

    assert (
        summarize_subprocess_failure(exc)
        == "uv sync failed: Because missing-package was not found in the package registry"
    )


def test_subprocess_output_tail_keeps_recent_output() -> None:
    exc = subprocess.CalledProcessError(
        1,
        ["uv", "pip", "install"],
        output="first line\nsecond line\nthird line",
    )

    assert subprocess_output_tail(exc) == "first line\nsecond line\nthird line"


def test_before_send_redacts_exception_values() -> None:
    event = {
        "request": {
            "url": "https://user:pass@example.com/path?token=abc123",
            "headers": {"Authorization": "Bearer abc.def"},
        },
        "user": {"email": "mongodb+srv://user:pass@example.mongodb.net/app"},
        "tags": {"dsn": "mongodb+srv://user:pass@example.mongodb.net/app"},
        "modules": {"pkg": "/tmp/home/project"},
        "exception": {
            "values": [
                {
                    "value": "token=abc123 at /tmp/home/project",
                    "stacktrace": {
                        "frames": [
                            {
                                "filename": "/tmp/home/project/app.py",
                                "vars": {"Authorization": "Bearer abc.def"},
                            }
                        ]
                    },
                }
            ]
        },
    }

    error_reporting._HOME_DIR = "/tmp/home"
    redacted = _before_send(event, {})

    value = redacted["exception"]["values"][0]["value"]
    frame = redacted["exception"]["values"][0]["stacktrace"]["frames"][0]
    assert (
        redacted["request"]["url"]
        == "https://<redacted>:<redacted>@example.com/path?token=<redacted>"
    )
    # The Authorization key matches the secret-key blocklist, so the whole
    # header value is masked (stricter than the bearer-regex scheme-prefix
    # form, and consistent with the Go redactor's credential-header rule).
    assert redacted["request"]["headers"]["Authorization"] == "<redacted>"
    assert (
        redacted["user"]["email"] == "mongodb+srv://<redacted>:<redacted>@example.mongodb.net/app"
    )
    assert redacted["tags"]["dsn"] == "mongodb+srv://<redacted>:<redacted>@example.mongodb.net/app"
    assert redacted["modules"]["pkg"] == "<home>/project"
    assert "abc123" not in value
    assert "<home>" in value
    assert frame["filename"] == "<home>/project/app.py"
    assert frame["vars"]["Authorization"] == "<redacted>"


def test_redact_text_tolerates_missing_home_dir() -> None:
    original_home_dir = error_reporting._HOME_DIR
    error_reporting._HOME_DIR = ""
    try:
        redacted = _redact_text("token=abc123 /tmp/project")
    finally:
        error_reporting._HOME_DIR = original_home_dir

    assert redacted == "token=<redacted> /tmp/project"
