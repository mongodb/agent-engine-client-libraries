"""Executable acceptance boundary for real-world OpenAPI generation."""

from __future__ import annotations

import json
import math
import os
import signal
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pytest
import yaml

from agent_engine_runner_shared.connectors import load_tool_defs
from agent_engine_runner_shared.connectors.generate import (
    CompileResult,
    GenerateError,
    compile_catalog,
    regenerate_catalog,
)

CORPUS_DIR = Path(__file__).parent
MANIFEST = json.loads((CORPUS_DIR / "manifest.json").read_text(encoding="utf-8"))
SPECS_DIR = CORPUS_DIR / "specs"
STRICT = os.environ.get("CONNECTOR_CORPUS_STRICT") == "1"
PRESENTATIONS = {
    "none",
    "bearer",
    "header_api_key",
    "query_api_key",
    "path_api_key",
    "cookie_api_key",
    "basic",
    "compound_headers",
}
SOURCE_AUTH_CASES = [
    ("twilio-com-api", {"basic": {}}),
    ("sendgrid-com", {"api_keys": [{"location": "header", "name": "Authorization"}]}),
    ("nasa-gov-apod", {"api_keys": [{"location": "query", "name": "api_key"}]}),
]


def _auth(api: dict[str, Any]) -> dict[str, Any] | None:
    """The tool.yaml auth block a developer writes for the accepted presentation."""
    presentation = api["auth"]["presentation"]
    if presentation == "none":
        return None
    if presentation == "bearer":
        return {"bearer": "CONNECTOR_TOKEN"}
    if presentation == "basic":
        return {"basic": {"username": "CONNECTOR_USERNAME", "password": "CONNECTOR_PASSWORD"}}
    if presentation.endswith("_api_key"):
        location = presentation.removesuffix("_api_key")
        name = api["auth"].get("name", "api_key")
        return {"api_keys": [{"env": "CONNECTOR_TOKEN", "location": location, "name": name}]}
    return {
        "api_keys": [
            {"env": f"CONNECTOR_SECRET_{index}", "location": "header", "name": f"X-Key-{index}"}
            for index in range(api["auth"]["credential_count"])
        ]
    }


def _spec(api_id: str) -> Path:
    spec = SPECS_DIR / f"{api_id}.json"
    if not spec.is_file():
        message = f"missing corpus fixture {spec.name}; run fetch_corpus.py"
        if STRICT:
            pytest.fail(message)
        pytest.skip(message)
    return spec


def _compile_with_binding(tmp_path: Path, api: dict[str, Any], spec: Path) -> CompileResult:
    """Compile through a hand-written tool.yaml, the route for any auth block."""
    out = tmp_path / api["id"]
    out.mkdir()
    authoring: dict[str, Any] = {
        "name": api["id"],
        "tool_defs": "tool_defs.yaml",
        "source": {
            "type": "openapi",
            "base_url": api["auth"].get("base_url", "https://api.example.com"),
            "spec": os.path.relpath(spec, out),
        },
        "expose": {"allow_all": True},
    }
    auth = _auth(api)
    if auth is not None:
        authoring["auth"] = auth
    (out / "tool.yaml").write_text(yaml.safe_dump(authoring), encoding="utf-8")
    return regenerate_catalog(out / "tool.yaml", skip_unsupported=True)


@contextmanager
def _compile_deadline(seconds: int) -> Iterator[None]:
    def expired(_signum: int, _frame: object) -> None:
        raise TimeoutError(f"corpus compilation exceeded {seconds} seconds")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def test_manifest_locks_the_product_boundary() -> None:
    apis = MANIFEST["apis"]
    assert MANIFEST["source_index_retrieved_at"] == "2026-09-15"
    assert len(MANIFEST["source_commit"]) == 40
    assert MANIFEST["declared_operations"] == 10_059
    assert MANIFEST["aggregate_minimum"] == 9_512
    assert len(apis) == 100
    assert len({api["id"] for api in apis}) == 100
    assert len({api["provider"] for api in apis}) == 100
    assert sum(api["declared"] for api in apis) == MANIFEST["declared_operations"]
    assert sum(api["minimum"] for api in apis) == MANIFEST["aggregate_minimum"]
    assert MANIFEST["aggregate_minimum"] / MANIFEST["declared_operations"] >= 0.94
    for api in apis:
        assert api["minimum"] >= math.ceil(api["declared"] * 0.8)
        assert api["auth"]["presentation"] in PRESENTATIONS
        if api["auth"]["presentation"] == "path_api_key":
            assert "{" + api["auth"]["name"] + "}" in api["auth"]["base_url"]
        assert len(api["sha256"]) == 64
        assert api["path"].startswith("v2/specs/")


def test_strict_fixture_set_is_exact() -> None:
    if not STRICT:
        pytest.skip("strict corpus gate is disabled")
    expected = {f"{api['id']}.json" for api in MANIFEST["apis"]}
    actual = {path.name for path in SPECS_DIR.iterdir()} if SPECS_DIR.is_dir() else set()
    assert actual == expected


@pytest.mark.parametrize(
    ("api_id", "expected"), SOURCE_AUTH_CASES, ids=[case[0] for case in SOURCE_AUTH_CASES]
)
def test_real_source_auth_is_projected(
    tmp_path: Path, api_id: str, expected: dict[str, Any]
) -> None:
    """Keep source-auth parsing covered independently from intentional CLI overrides."""
    spec = _spec(api_id)
    with _compile_deadline(MANIFEST["compile_timeout_seconds"]):
        result = compile_catalog(
            str(spec),
            name=api_id,
            base_url="https://api.example.com",
            skip_unsupported=True,
            out_dir=tmp_path / api_id,
        )

    definitions = load_tool_defs(result.catalog)
    assert definitions.externalize is not None
    assert definitions.externalize["auth"] == expected


def test_real_oauth_source_needs_a_provider_and_proposes_its_scopes(tmp_path: Path) -> None:
    spec = _spec("slack-com")
    with (
        _compile_deadline(MANIFEST["compile_timeout_seconds"]),
        pytest.raises(GenerateError, match="authenticate with OAuth"),
    ):
        compile_catalog(
            str(spec), name="slack", skip_unsupported=True, out_dir=tmp_path / "refused"
        )

    with _compile_deadline(MANIFEST["compile_timeout_seconds"]):
        result = compile_catalog(
            str(spec),
            name="slack",
            provider="slack",
            allow=["slack_chat_postMessage", "slack_users_info"],
            skip_unsupported=True,
            out_dir=tmp_path / "slack",
        )

    assert result.tool_yaml is not None
    assert yaml.safe_load(result.tool_yaml.read_text(encoding="utf-8"))["auth"] == {
        "provider": "slack",
        "scopes": ["chat:write:bot", "chat:write:user", "users:read"],
    }


@pytest.mark.parametrize("api", MANIFEST["apis"], ids=lambda api: api["id"])
def test_api_meets_coverage_floor(tmp_path: Path, api: dict[str, Any]) -> None:
    spec = _spec(api["id"])
    with _compile_deadline(MANIFEST["compile_timeout_seconds"]):
        result = _compile_with_binding(tmp_path, api, spec)
    definitions = load_tool_defs(result.catalog)
    assert len(definitions.tools) >= api["minimum"]
    skipped = (definitions.externalize or {}).get("skipped", [])
    assert skipped == list(result.skipped)
    assert len(definitions.tools) + len(skipped) == api["declared"]
    for operation_id in api.get("required_operation_ids", []):
        suffix = f"_{operation_id}".lower()
        assert any(tool.name.lower().endswith(suffix) for tool in definitions.tools)
