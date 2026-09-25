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

from agent_engine_runner_shared.connectors import load_tool_defs
from agent_engine_runner_shared.connectors.definitions import Auth
from agent_engine_runner_shared.connectors.generate import compile_catalog

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
    ("slack-com", {"type": "bearer"}),
    ("twilio-com-api", {"type": "basic"}),
    ("sendgrid-com", {"type": "api_key", "location": "header", "name": "Authorization"}),
    ("nasa-gov-apod", {"type": "api_key", "location": "query", "name": "api_key"}),
]


def _auth(api: dict[str, Any]) -> Auth | None:
    presentation = api["auth"]["presentation"]
    if presentation == "none":
        return None
    if presentation == "bearer":
        return Auth(type="bearer", env="CONNECTOR_TOKEN")
    if presentation.endswith("_api_key"):
        return Auth(
            type="api_key",
            env="CONNECTOR_TOKEN",
            location=presentation.removesuffix("_api_key"),
            name=api["auth"].get("name", "api_key"),
        )
    if presentation == "basic":
        return Auth(type="basic", username_env="CONNECTOR_USER", password_env="CONNECTOR_PASS")
    return Auth(
        type="api_keys",
        credentials=[
            {"env": f"CONNECTOR_SECRET_{index}", "location": "header", "name": f"X-Key-{index}"}
            for index in range(api["auth"]["credential_count"])
        ],
    )


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
        assert api["url"].startswith("https://api.apis.guru/v2/specs/")


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
    spec = SPECS_DIR / f"{api_id}.json"
    if not spec.is_file():
        message = f"missing corpus fixture {spec.name}; run fetch_corpus.py"
        if STRICT:
            pytest.fail(message)
        pytest.skip(message)
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


@pytest.mark.parametrize("api", MANIFEST["apis"], ids=lambda api: api["id"])
def test_api_meets_coverage_floor(tmp_path: Path, api: dict[str, Any]) -> None:
    spec = SPECS_DIR / f"{api['id']}.json"
    if not spec.is_file():
        message = f"missing corpus fixture {spec.name}; run fetch_corpus.py"
        if STRICT:
            pytest.fail(message)
        pytest.skip(message)
    with _compile_deadline(MANIFEST["compile_timeout_seconds"]):
        result = compile_catalog(
            str(spec),
            name=api["id"],
            base_url=api["auth"].get("base_url", "https://api.example.com"),
            auth=_auth(api),
            skip_unsupported=True,
            out_dir=tmp_path / api["id"],
        )
    definitions = load_tool_defs(result.catalog)
    assert len(definitions.tools) >= api["minimum"]
    skipped = (definitions.externalize or {}).get("skipped", [])
    assert skipped == list(result.skipped)
    assert len(definitions.tools) + len(skipped) == api["declared"]
    for operation_id in api.get("required_operation_ids", []):
        suffix = f"_{operation_id}".lower()
        assert any(tool.name.lower().endswith(suffix) for tool in definitions.tools)
