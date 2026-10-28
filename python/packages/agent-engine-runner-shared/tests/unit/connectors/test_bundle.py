from __future__ import annotations

from pathlib import Path

import pytest

from agent_engine_runner_shared.connectors import load_tool_defs
from agent_engine_runner_shared.connectors.bundle import (
    BundledConnector,
    BundledTool,
    ConnectorBundle,
    ConnectorBundleError,
)

FIXTURE = Path(__file__).parent / "fixtures/jira.tool_defs.yaml"


def _bundle_for(**auth: object) -> ConnectorBundle:
    definitions = load_tool_defs(FIXTURE)
    if auth:
        document = definitions.model_dump()
        document["auth"] = auth
        definitions = type(definitions).model_validate(document)
    connector = BundledConnector(name=definitions.source.name, definitions=definitions)
    return ConnectorBundle(
        connectors=(connector,),
        tools=tuple(
            BundledTool(connector=connector, operation=operation) for operation in definitions.tools
        ),
    )


def test_schema_and_network_are_detached_runtime_views():
    bundle = _bundle_for()

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

    schema = bundle.tools[0].schema
    schema["name"] = "mutated"
    assert bundle.tools[0].schema["name"] == "jiradc_getComments"

    network = bundle.tools[0].network
    network.append("mutated.example")
    assert bundle.tools[0].network == ["jira.example"]


def test_registration_metadata_is_the_shared_aer_and_tool_pod_projection():
    assert _bundle_for().tools[0].registration_metadata == {
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


def test_oauth_metadata_is_explicit_and_not_in_model_schema():
    """Register the provider key and connector-wide scopes outside model arguments."""
    bundle = _bundle_for(type="oauth", provider_type="custom-github", scopes=["repo"])
    for tool in bundle.tools:
        assert tool.registration_metadata["provider_type"] == "custom-github"
        assert tool.registration_metadata["scopes"] == ["repo"]
        assert "provider_type" not in tool.schema
        assert "authorization" not in tool.schema["inputSchema"]["properties"]
    metadata = bundle.tools[0].registration_metadata
    metadata["scopes"].clear()
    assert bundle.tools[0].registration_metadata["scopes"] == ["repo"]


def test_collision_check_reports_before_registration():
    bundle = _bundle_for()

    with pytest.raises(ConnectorBundleError, match="jiradc_addComment"):
        bundle.require_available_names({"existing", "jiradc_addComment"})


def test_empty_bundle_has_no_registration_work():
    bundle = ConnectorBundle()

    assert bundle.connectors == ()
    assert bundle.tools == ()
    bundle.require_available_names({"anything"})
