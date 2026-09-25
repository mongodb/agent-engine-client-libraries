from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agent_engine_runner_shared.connectors.bundle import (
    CONNECTOR_BUNDLE_PATH_ENV,
    ConnectorBundleError,
    load_connector_bundle,
)

FIXTURE = Path(__file__).parent / "fixtures/jira.tool_defs.yaml"


def _write_bundle(tmp_path: Path) -> Path:
    definitions_path = tmp_path / "jiradc.tool_defs.yaml"
    definitions_path.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    index_path = tmp_path / "bundle.yaml"
    index_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "connectors": [
                    {"name": "jiradc", "tool_defs": definitions_path.name},
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return index_path


def test_no_configured_bundle_is_an_io_free_noop(monkeypatch):
    monkeypatch.delenv(CONNECTOR_BUNDLE_PATH_ENV, raising=False)
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected file read")),
    )

    bundle = load_connector_bundle()

    assert bundle.connectors == ()
    assert bundle.tools == ()


def test_loads_all_definitions_from_generated_index(tmp_path):
    bundle = load_connector_bundle(_write_bundle(tmp_path))

    assert [connector.name for connector in bundle.connectors] == ["jiradc"]
    assert [tool.operation.name for tool in bundle.tools] == [
        "jiradc_getComments",
        "jiradc_addComment",
    ]
    assert bundle.tools[0].schema == {
        "name": "jiradc_getComments",
        "description": "Returns comments for an issue.",
        "inputSchema": {
            "type": "object",
            "properties": {"issueIdOrKey": {"type": "string"}},
            "required": ["issueIdOrKey"],
        },
    }
    assert bundle.tools[0].network == ["jira.example"]
    assert bundle.tools[0].registration_metadata == {
        "name": "jiradc_getComments",
        "description": "Returns comments for an issue.",
        "is_local": False,
        "provider_type": None,
        "scopes": [],
        "network": ["jira.example"],
        "timeout_seconds": 30,
        "redact_fields": [],
        "connector": "jiradc",
    }


def test_configured_bundle_path_comes_from_platform_environment(monkeypatch, tmp_path):
    index_path = _write_bundle(tmp_path)
    monkeypatch.setenv(CONNECTOR_BUNDLE_PATH_ENV, str(index_path))

    assert load_connector_bundle().connectors[0].name == "jiradc"


@pytest.mark.parametrize("reference", ["../outside.tool_defs.yaml", "https://example.test/x"])
def test_rejects_nonlocal_or_escaping_definition_reference(tmp_path, reference):
    (tmp_path.parent / "outside.tool_defs.yaml").write_text(
        FIXTURE.read_text(encoding="utf-8"), encoding="utf-8"
    )
    index_path = tmp_path / "bundle.yaml"
    index_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "connectors": [{"name": "jiradc", "tool_defs": reference}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConnectorBundleError, match="local file inside the bundle"):
        load_connector_bundle(index_path)


def test_rejects_definition_symlink_that_escapes_bundle(tmp_path):
    outside_path = tmp_path.parent / "outside.tool_defs.yaml"
    outside_path.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "linked.tool_defs.yaml").symlink_to(outside_path)
    index_path = tmp_path / "bundle.yaml"
    index_path.write_text(
        "version: 1\nconnectors:\n  - name: jiradc\n    tool_defs: linked.tool_defs.yaml\n",
        encoding="utf-8",
    )

    with pytest.raises(ConnectorBundleError, match="local file inside the bundle"):
        load_connector_bundle(index_path)


def test_rejects_index_identity_that_does_not_match_definitions(tmp_path):
    index_path = _write_bundle(tmp_path)
    index_path.write_text(
        "version: 1\nconnectors:\n  - name: other\n    tool_defs: jiradc.tool_defs.yaml\n",
        encoding="utf-8",
    )

    with pytest.raises(ConnectorBundleError, match="identity does not match"):
        load_connector_bundle(index_path)


def test_rejects_duplicate_tool_names_across_connectors(tmp_path):
    index_path = _write_bundle(tmp_path)
    second = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    second["source"]["name"] = "other"
    (tmp_path / "other.tool_defs.yaml").write_text(
        yaml.safe_dump(second, sort_keys=False), encoding="utf-8"
    )
    index_path.write_text(
        "version: 1\nconnectors:\n"
        "  - name: jiradc\n    tool_defs: jiradc.tool_defs.yaml\n"
        "  - name: other\n    tool_defs: other.tool_defs.yaml\n",
        encoding="utf-8",
    )

    with pytest.raises(ConnectorBundleError, match="duplicate connector tool names"):
        load_connector_bundle(index_path)


def test_collision_check_reports_before_registration(tmp_path):
    bundle = load_connector_bundle(_write_bundle(tmp_path))

    with pytest.raises(ConnectorBundleError, match="jiradc_addComment"):
        bundle.require_available_names({"existing", "jiradc_addComment"})
