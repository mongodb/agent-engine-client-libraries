from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agent_engine_runner_shared.agent_config import RuntimeAgentConfig, RuntimeConnectorReference
from agent_engine_runner_shared.connectors import (
    CONNECTOR_BUNDLE_PATH_ENV,
    ConnectorBundle,
    MaterializeError,
    load_connector_bundle,
    materialize_connectors,
    resolve_runtime_bundle,
)

FIXTURE = Path(__file__).parent / "fixtures/jira.tool_defs.yaml"

TOOL_YAML = """
name: jira
tool_defs: jira.tool_defs.yaml
source:
  type: openapi
  base_url: https://jira.corp/rest
expose:
  allow:
    - jiradc_getComments
    - jiradc_addComment
auth:
  type: bearer
  env: JIRA_TOKEN
"""


def _write_authoring(
    tmp_path: Path,
    tool_yaml: str = TOOL_YAML,
    catalog: str | None = None,
) -> Path:
    connectors = tmp_path / "connectors" / "jira"
    connectors.mkdir(parents=True, exist_ok=True)
    (connectors / "jira.tool_defs.yaml").write_text(
        catalog if catalog is not None else FIXTURE.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (connectors / "tool.yaml").write_text(tool_yaml, encoding="utf-8")
    return connectors / "tool.yaml"


def _bundle_for(tmp_path: Path, **kwargs) -> ConnectorBundle:
    tool_yaml_path = _write_authoring(tmp_path, **kwargs)
    return materialize_connectors([tool_yaml_path], root=tmp_path)


class TestMaterializeConnectors:
    def test_materializes_bundle_matching_runtime_contract(self, tmp_path):
        bundle = _bundle_for(tmp_path)

        assert [connector.name for connector in bundle.connectors] == ["jira"]
        assert [tool.operation.name for tool in bundle.tools] == [
            "jiradc_getComments",
            "jiradc_addComment",
        ]
        assert bundle.tools[0].connector.definitions.source.base_url == "https://jira.corp/rest"

    def test_allow_subset_selects_only_listed_operations(self, tmp_path):
        tool_yaml = TOOL_YAML.replace("    - jiradc_addComment\n", "")
        bundle = _bundle_for(tmp_path, tool_yaml=tool_yaml)

        assert [tool.operation.name for tool in bundle.tools] == ["jiradc_getComments"]

    def test_allow_all_with_disallow_subtracts_after(self, tmp_path):
        tool_yaml = TOOL_YAML.replace(
            """expose:
  allow:
    - jiradc_getComments
    - jiradc_addComment
""",
            """expose:
  allow_all: true
  disallow:
    - jiradc_addComment
""",
        )
        bundle = _bundle_for(tmp_path, tool_yaml=tool_yaml)

        assert [tool.operation.name for tool in bundle.tools] == ["jiradc_getComments"]

    def test_expose_is_fail_closed_without_allow_rules(self, tmp_path):
        tool_yaml = TOOL_YAML.replace(
            """expose:
  allow:
    - jiradc_getComments
    - jiradc_addComment
""",
            "expose: {}\n",
        )

        with pytest.raises(MaterializeError, match="fail-closed"):
            _bundle_for(tmp_path, tool_yaml=tool_yaml)

    def test_allow_and_allow_all_are_mutually_exclusive(self, tmp_path):
        tool_yaml = TOOL_YAML.replace("  allow:\n", "  allow_all: true\n  allow:\n")

        with pytest.raises(MaterializeError, match="mutually exclusive"):
            _bundle_for(tmp_path, tool_yaml=tool_yaml)

    def test_unmatched_allow_rule_names_available_operations(self, tmp_path):
        tool_yaml = TOOL_YAML.replace("    - jiradc_addComment\n", "    - jiradc_getIsue\n")

        with pytest.raises(MaterializeError, match="jiradc_getIsue.*jiradc_getComments"):
            _bundle_for(tmp_path, tool_yaml=tool_yaml)

    def test_unmatched_disallow_rule_is_rejected(self, tmp_path):
        tool_yaml = TOOL_YAML.replace("auth:", "  disallow:\n    - jiradc_getTypo\nauth:")

        with pytest.raises(MaterializeError, match="jiradc_getTypo"):
            _bundle_for(tmp_path, tool_yaml=tool_yaml)

    @pytest.mark.parametrize(
        "bad", ["true", "false", "5", "0", '""', "jiradc_getComments", "{a: b}", "{}"]
    )
    def test_non_list_disallow_shape_is_rejected(self, tmp_path, bad):
        """A scalar or string disallow must fail with the scoped error, not a
        raw TypeError, a per-character unmatched-operations list, or silent
        truthiness normalization to no-op."""
        tool_yaml = TOOL_YAML.replace("auth:", f"  disallow: {bad}\nauth:")

        with pytest.raises(MaterializeError, match="disallow must be a list"):
            _bundle_for(tmp_path, tool_yaml=tool_yaml)

    @pytest.mark.parametrize(
        "field, element",
        [
            ("allow", 5),
            ("allow", ["nested"]),
            ("allow", {"a": "value"}),
            ("allow", ""),
            ("allow", None),
            ("disallow", 5),
            ("disallow", ["nested"]),
            ("disallow", {"a": "value"}),
            ("disallow", ""),
            ("disallow", None),
            ("allow", ["jiradc_getComments", 5]),
        ],
    )
    def test_non_string_exposure_elements_are_rejected(self, tmp_path, field, element):
        """Nested lists, mappings, numbers, and empty strings in allow/disallow
        must fail with the connector- and field-scoped error, not a raw
        TypeError from set/sort."""
        rule = yaml.safe_dump([element], default_flow_style=True).strip()
        tool_yaml = TOOL_YAML.replace("auth:", f"  {field}: {rule}\nauth:")

        with pytest.raises(
            MaterializeError, match=rf"connector 'jira': expose\.{field} entries must be"
        ):
            _bundle_for(tmp_path, tool_yaml=tool_yaml)

    def test_fully_disallowed_selection_is_an_error(self, tmp_path):
        tool_yaml = TOOL_YAML.replace(
            "auth:", "  disallow:\n    - jiradc_getComments\n    - jiradc_addComment\nauth:"
        )

        with pytest.raises(MaterializeError, match="select no operations"):
            _bundle_for(tmp_path, tool_yaml=tool_yaml)

    def test_tool_yaml_overrides_catalog_base_url(self, tmp_path):
        bundle = _bundle_for(tmp_path)

        assert bundle.connectors[0].definitions.source.base_url == "https://jira.corp/rest"

    def test_missing_base_url_fails_materialization(self, tmp_path):
        """The endpoint is consumer-owned with no catalog fallback: a
        catalog-selected destination combined with a consumer-selected
        credential would send the tenant secret to the catalog's host."""
        tool_yaml = TOOL_YAML.replace("  base_url: https://jira.corp/rest\n", "")

        with pytest.raises(MaterializeError, match="source.base_url is required"):
            _bundle_for(tmp_path, tool_yaml=tool_yaml)

    def test_catalog_credential_binding_never_survives(self, tmp_path):
        """The catalog carries auth.env (POC format) but must not win: the
        consuming tool.yaml is the only source of credential bindings."""
        tool_yaml = TOOL_YAML.replace(
            """auth:
  type: bearer
  env: JIRA_TOKEN
""",
            "auth:\n  type: none\n",
        )
        bundle = _bundle_for(tmp_path, tool_yaml=tool_yaml)

        assert bundle.connectors[0].definitions.auth is None

    def test_auth_env_without_type_fails_loudly(self, tmp_path):
        """Silently dropping the binding would register unauthenticated tools
        that only fail with remote 401s."""
        tool_yaml = TOOL_YAML.replace(
            """auth:
  type: bearer
  env: JIRA_TOKEN
""",
            "auth:\n  env: JIRA_TOKEN\n",
        )

        with pytest.raises(MaterializeError, match="invalid auth"):
            _bundle_for(tmp_path, tool_yaml=tool_yaml)

    def test_auth_none_with_credentials_fails_loudly(self, tmp_path):
        tool_yaml = TOOL_YAML.replace(
            """auth:
  type: bearer
  env: JIRA_TOKEN
""",
            "auth:\n  type: none\n  env: JIRA_TOKEN\n",
        )

        with pytest.raises(MaterializeError, match="invalid auth"):
            _bundle_for(tmp_path, tool_yaml=tool_yaml)

    def test_credential_is_sent_only_to_the_consumer_origin(self, tmp_path):
        """Execution-level proof of the consumer-owned binding: the credential
        header must reach the tool.yaml base_url host, and no other."""
        import httpx

        from agent_engine_runner_shared.connectors.executor import ConnectorExecutor

        catalog = FIXTURE.read_text(encoding="utf-8").replace(
            "base_url: https://jira.example/rest",
            "base_url: https://catalog-publisher.example/rest",
        )
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={"ok": True})

        bundle = _bundle_for(tmp_path, catalog=catalog)
        tool = bundle.tools[0]
        with ConnectorExecutor(
            tool.connector.definitions, transport=httpx.MockTransport(handler)
        ) as executor:
            executor.execute(
                "jiradc_getComments", {"issueIdOrKey": "DEMO-1"}, env={"JIRA_TOKEN": "secret-token"}
            )

        assert [request.url.host for request in requests] == ["jira.corp"]
        assert requests[0].headers["Authorization"] == "Bearer secret-token"

    def test_tool_yaml_api_key_binding_reaches_artifact(self, tmp_path):
        tool_yaml = TOOL_YAML.replace(
            """auth:
  type: bearer
  env: JIRA_TOKEN
""",
            """auth:
  type: api_key
  env: WEATHER_API_KEY
  header: X-Custom-Key
""",
        )
        bundle = _bundle_for(tmp_path, tool_yaml=tool_yaml)

        definitions = bundle.connectors[0].definitions
        assert definitions.auth is not None
        assert definitions.auth.header_name() == "X-Custom-Key"

    def test_connector_name_becomes_identity_so_traversal_is_rejected(self, tmp_path):
        tool_yaml = TOOL_YAML.replace("name: jira", "name: ../evil")

        with pytest.raises(MaterializeError, match="name must match"):
            _bundle_for(tmp_path, tool_yaml=tool_yaml)

    def test_https_catalog_reference_is_rejected(self, tmp_path):
        tool_yaml = TOOL_YAML.replace(
            "tool_defs: jira.tool_defs.yaml",
            "tool_defs: https://connectors.example.com/jira/tool_defs.yaml",
        )

        with pytest.raises(MaterializeError, match="local catalog path"):
            _bundle_for(tmp_path, tool_yaml=tool_yaml)

    def test_absolute_catalog_reference_is_rejected(self, tmp_path):
        tool_yaml = TOOL_YAML.replace(
            "tool_defs: jira.tool_defs.yaml", "tool_defs: /etc/jira.tool_defs.yaml"
        )

        with pytest.raises(MaterializeError, match="local catalog path"):
            _bundle_for(tmp_path, tool_yaml=tool_yaml)

    def test_symlink_escaping_the_root_is_rejected(self, tmp_path):
        outside = tmp_path.parent / f"{tmp_path.name}-outside"
        outside.mkdir(exist_ok=True)
        (outside / "evil.tool_defs.yaml").write_text(
            FIXTURE.read_text(encoding="utf-8"), encoding="utf-8"
        )
        connectors = tmp_path / "connectors" / "jira"
        connectors.mkdir(parents=True, exist_ok=True)
        (connectors / "jira.tool_defs.yaml").symlink_to(outside / "evil.tool_defs.yaml")
        (connectors / "tool.yaml").write_text(TOOL_YAML, encoding="utf-8")

        with pytest.raises(MaterializeError, match="outside the agent workspace root"):
            materialize_connectors([connectors / "tool.yaml"], root=tmp_path)

    def test_missing_catalog_fails(self, tmp_path):
        connectors = tmp_path / "connectors" / "jira"
        connectors.mkdir(parents=True, exist_ok=True)
        (connectors / "tool.yaml").write_text(TOOL_YAML, encoding="utf-8")

        with pytest.raises(MaterializeError, match="does not exist inside"):
            materialize_connectors([connectors / "tool.yaml"], root=tmp_path)

    def test_duplicate_connector_names_fail(self, tmp_path):
        tool_yaml_path = _write_authoring(tmp_path)

        with pytest.raises(MaterializeError, match="duplicate connector name"):
            materialize_connectors([tool_yaml_path, tool_yaml_path], root=tmp_path)

    def test_cross_connector_tool_name_collision_fails(self, tmp_path):
        first = _write_authoring(tmp_path)
        second_dir = tmp_path / "connectors" / "jira-two"
        second_dir.mkdir(parents=True, exist_ok=True)
        (second_dir / "jira.tool_defs.yaml").write_text(
            FIXTURE.read_text(encoding="utf-8"), encoding="utf-8"
        )
        (second_dir / "tool.yaml").write_text(
            TOOL_YAML.replace("name: jira", "name: jira-two"), encoding="utf-8"
        )

        with pytest.raises(MaterializeError, match="collision"):
            materialize_connectors([first, second_dir / "tool.yaml"], root=tmp_path)

    def test_invalid_tool_yaml_fails_with_path(self, tmp_path):
        tool_yaml_path = _write_authoring(tmp_path, tool_yaml="name: [unclosed\n")

        with pytest.raises(MaterializeError, match="invalid connector authoring file"):
            materialize_connectors([tool_yaml_path], root=tmp_path)

    def test_bundle_round_trips_through_the_runtime_loader(self, tmp_path, tmp_path_factory):
        """The in-memory bundle and the on-disk loader must agree on the same
        authoring inputs — this is the contract a platform-build bundle (which
        IS materialized to files) will have to satisfy."""
        work = tmp_path_factory.mktemp("ondisk")
        definitions = _bundle_for(tmp_path).connectors[0].definitions
        (work / "jira.tool_defs.yaml").write_text(
            yaml.safe_dump(definitions.model_dump(exclude_none=True), sort_keys=False),
            encoding="utf-8",
        )
        index_path = work / "index.yaml"
        index_path.write_text(
            "version: 1\nconnectors:\n  - name: jira\n    tool_defs: jira.tool_defs.yaml\n",
            encoding="utf-8",
        )

        loaded = load_connector_bundle(index_path)

        assert [t.operation.name for t in loaded.tools] == [
            t.operation.name for t in _bundle_for(tmp_path).tools
        ]


class TestResolveRuntimeBundle:
    def test_no_env_and_no_connectors_is_an_empty_noop(self, monkeypatch):
        monkeypatch.delenv(CONNECTOR_BUNDLE_PATH_ENV, raising=False)

        bundle = resolve_runtime_bundle(RuntimeAgentConfig())

        assert bundle == ConnectorBundle()

    def test_pre_materialized_bundle_wins_over_agent_yaml(self, tmp_path, monkeypatch):
        tool_yaml_path = _write_authoring(tmp_path)
        env_bundle_root = tmp_path / "env-bundle"
        env_bundle_root.mkdir()
        (env_bundle_root / "other.tool_defs.yaml").write_text(
            FIXTURE.read_text(encoding="utf-8").replace("name: jiradc", "name: envconn"),
            encoding="utf-8",
        )
        index_path = env_bundle_root / "index.yaml"
        index_path.write_text(
            "version: 1\nconnectors:\n  - name: envconn\n    tool_defs: other.tool_defs.yaml\n",
            encoding="utf-8",
        )
        monkeypatch.setenv(CONNECTOR_BUNDLE_PATH_ENV, str(index_path))

        agent_config = RuntimeAgentConfig(
            path=tmp_path / "agent.yaml",
            connectors=[
                RuntimeConnectorReference(source=str(tool_yaml_path.relative_to(tmp_path)))
            ],
        )
        bundle = resolve_runtime_bundle(agent_config)

        assert [connector.name for connector in bundle.connectors] == ["envconn"]

    def test_traverses_agent_yaml_connectors(self, tmp_path, monkeypatch):
        monkeypatch.delenv(CONNECTOR_BUNDLE_PATH_ENV, raising=False)
        _write_authoring(tmp_path)
        (tmp_path / "agent.yaml").write_text("name: agent\n", encoding="utf-8")

        agent_config = RuntimeAgentConfig(
            path=tmp_path / "agent.yaml",
            connectors=[RuntimeConnectorReference(source="connectors/jira/tool.yaml")],
        )
        bundle = resolve_runtime_bundle(agent_config)

        assert [connector.name for connector in bundle.connectors] == ["jira"]
        assert [tool.operation.name for tool in bundle.tools] == [
            "jiradc_getComments",
            "jiradc_addComment",
        ]

    def test_traversal_source_escaping_agent_root_is_rejected(self, tmp_path, monkeypatch):
        monkeypatch.delenv(CONNECTOR_BUNDLE_PATH_ENV, raising=False)
        outside = tmp_path.parent / f"{tmp_path.name}-outside"
        outside.mkdir(exist_ok=True)
        (outside / "tool.yaml").write_text(TOOL_YAML, encoding="utf-8")

        agent_config = RuntimeAgentConfig(
            path=tmp_path / "agent.yaml",
            connectors=[RuntimeConnectorReference(source="../outside/tool.yaml")],
        )

        with pytest.raises(MaterializeError, match="inside the agent.yaml directory"):
            resolve_runtime_bundle(agent_config)

    def test_traversal_without_located_agent_yaml_fails(self, monkeypatch):
        monkeypatch.delenv(CONNECTOR_BUNDLE_PATH_ENV, raising=False)
        agent_config = RuntimeAgentConfig(
            connectors=[RuntimeConnectorReference(source="connectors/jira/tool.yaml")]
        )

        with pytest.raises(MaterializeError, match="located agent.yaml"):
            resolve_runtime_bundle(agent_config)
