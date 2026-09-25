"""Structured runtime helpers for reading ``agent.yaml``.

Lookup order is intentionally small and explicit:
1. ``config_path`` argument: exact ``agent.yaml`` path, or a directory that contains it
2. ``AGENTIC_AGENT_CONFIG_PATH``: exact in-container file path baked into runtime images
3. ``AGENTIC_AGENT_WORKDIR``: agent working directory used by generated local dev stacks
4. current working directory: supports tests and ad-hoc SDK usage
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated, Any, Literal, Mapping
from urllib.parse import urlparse

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from agent_engine_runner_shared.utils import is_platform_env_var

ConfigPath = str | Path
NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
_FORBIDDEN_MCP_HEADER_KEYS = frozenset(
    {
        "authorization",
        "cookie",
        "proxy-authorization",
    }
)
_MCP_AUTH_FIELD_NAMES = frozenset(
    {
        "token_env",
        "redirect_uri",
        "client_name",
        "scope",
        "token_url",
        "client_id_env",
        "client_secret_env",
    }
)
_MCP_AUTH_REQUIRED_FIELDS = {
    "none": frozenset[str](),
    "bearer_env": frozenset({"token_env"}),
    "oauth": frozenset[str](),
    "client_credentials": frozenset({"client_id_env", "client_secret_env"}),
}
_MCP_AUTH_ALLOWED_FIELDS = {
    "none": frozenset[str](),
    "bearer_env": frozenset({"token_env"}),
    "oauth": frozenset({"redirect_uri", "client_name", "scope"}),
    "client_credentials": frozenset({"token_url", "client_id_env", "client_secret_env", "scope"}),
}

# YAML paths where ``${VAR}`` substitution is performed. Anything outside this
# allowlist (e.g. prompts) passes through unchanged so a literal ``${...}`` is
# never clobbered. ``"*"`` matches any single key segment.
#
# This set is intentionally narrow. Other candidate paths
# (``mcp.servers.*.headers.*``, ``mcp.servers.*.auth.redirect_uri``,
# ``mcp.servers.*.auth.scope``) will be added in dedicated follow-ups when
# concrete use cases land. Adding a path is one frozenset entry plus a
# happy-path test.
_INTERPOLATABLE_PATHS: frozenset[tuple[str, ...]] = frozenset(
    {
        ("mcp", "servers", "*", "url"),
    }
)
# Pattern for an ``${VAR}`` substitution token. Only the braced form is
# recognised -- bare ``$name`` is left untouched so URLs containing real
# ``$`` characters (e.g. OData ``$filter`` / ``$top`` query params) pass
# through unchanged once interpolation is enabled. There is **no escape**
# for the ``${VAR}`` form itself: any well-formed ``${...}`` at an
# interpolatable path always triggers substitution. A user who genuinely
# needs a literal ``${something}`` in an MCP URL has no recourse today; we
# can grow a ``$${...}`` -> ``${...}`` escape later if a real case appears.
#
# Kept in sync with ``mcpURLVarReference`` in
# client-libraries/agent-config/agentconfig/agentconfig.go, which uses the
# same grammar to skip (not reject) interpolated MCP URLs when deriving
# egress hosts at build/validate time.
_VAR_REFERENCE = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)\}")
# Captures any ``${...}`` token, well-formed or not, so we can detect
# malformed markers (``${VAR`` unclosed, ``${}`` empty, ``${1bad}`` invalid
# identifier, ``${VAR with spaces}``) and raise a clear error pointing at
# the YAML path -- instead of letting them flow through to ``urlparse`` and
# surface as the same "must be an absolute http(s) URL" error this PR is
# trying to eliminate.
_ANY_VAR_REFERENCE = re.compile(r"\$\{[^}]*\}?")


def _path_is_interpolatable(path: tuple[str, ...]) -> bool:
    for pattern in _INTERPOLATABLE_PATHS:
        if len(pattern) != len(path):
            continue
        if all(seg == "*" or seg == path[i] for i, seg in enumerate(pattern)):
            return True
    return False


def _interpolate_env_vars(
    data: Any,
    env_vars: Mapping[str, str] | None,
    path: tuple[str, ...] = (),
) -> Any:
    """Substitute ``${VAR}`` at allowlisted YAML paths using ``env_vars``.

    Walks the post-``yaml.safe_load`` structure. Strings outside the allowlist
    are returned unchanged. When ``env_vars`` is ``None`` and an allowlisted
    string contains ``${...}``, fail loudly so a literal interpolation marker
    never reaches downstream validators (which would surface as the confusing
    "must be an absolute http(s) URL" error).
    """
    if isinstance(data, dict):
        return {
            key: _interpolate_env_vars(value, env_vars, path + (str(key),))
            for key, value in data.items()
        }
    if isinstance(data, list):
        return [
            _interpolate_env_vars(item, env_vars, path + (f"[{index}]",))
            for index, item in enumerate(data)
        ]
    if not isinstance(data, str) or not _path_is_interpolatable(path):
        return data

    yaml_path = ".".join(path)

    # Reject malformed markers (``${VAR`` unclosed, ``${}`` empty,
    # ``${1bad}`` invalid identifier, ``${VAR with spaces}``) before the
    # env_vars branch so the error is the same with or without a mapping.
    for marker in _ANY_VAR_REFERENCE.finditer(data):
        if _VAR_REFERENCE.fullmatch(marker.group(0)) is None:
            raise ValueError(
                f"agent.yaml at {yaml_path} has malformed environment "
                f"variable reference {marker.group(0)!r}; expected ${{VAR}} "
                "where VAR is a valid identifier"
            )

    if env_vars is None:
        if _VAR_REFERENCE.search(data) is not None:
            raise ValueError(
                f"agent.yaml at {yaml_path} contains ${{...}} but environment "
                "variable interpolation is not enabled for this load. The "
                "runtime launcher must pass env_vars= to "
                "load_runtime_agent_config()."
            )
        return data

    unset: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        name = match.group("name")
        if name in env_vars:
            return env_vars[name]
        unset.append(name)
        return ""  # the surrounding re.sub() result is discarded when ``unset`` is non-empty

    substituted = _VAR_REFERENCE.sub(_replace, data)
    if unset:
        # Report every unset name in one error so a tenant fixing a
        # multi-var URL doesn't have to redeploy once per missing var.
        # Deduplicate while preserving first-seen order for stable messages.
        seen: dict[str, None] = {}
        for name in unset:
            seen.setdefault(name, None)
        names = ", ".join(f"${{{name}}}" for name in seen)
        raise ValueError(
            f"agent.yaml at {yaml_path} references unset environment variables: {names}"
        )
    return substituted


FeatureName = Literal[
    "memory",
    "guardrails",
    "deep_agent",
    "playground",
    "use_custom_parser",
    "durable_workflow",
]


class AgentFeatureConfig(BaseModel):
    """Runtime feature flags from ``agent.yaml``.

    Keep field names in parity with the TypeScript schema and CLI allowlist.
    Add a field here when introducing a flag, then mirror it in those surfaces.

    ``None`` means the flag was omitted, which lets the runtime fall back to
    legacy environment variables during the transition.
    """

    model_config = ConfigDict(extra="ignore")

    memory: bool | None = None
    guardrails: bool | None = None
    # Opt-in for the Tool-Pod built-in filesystem + shell handlers used by
    # LangChain deepagents' ``SandboxBackendProtocol`` (``filesystem_ls``,
    # ``filesystem_read`` / ``write`` / ``edit`` / ``glob`` / ``grep`` /
    # ``download``, ``shell_execute``). Defaults to off so tenants that don't
    # run deep agents don't get a filesystem/shell attack surface inside
    # their Tool-Pod. Must be true for ``App.deep_agent()`` to work.
    deep_agent: bool | None = None
    # Whether the platform provisions playground UI for the agent (None/True
    # = provisioned, today's behavior). When False — e.g. for non-chat agents
    # with no conversation to preview — no playground is built or served;
    # callers use the invoke API directly. Read at deploy/provisioning time
    # only — no runtime effect.
    playground: bool | None = None
    # Opt-in for author-defined streaming output shaping: when true the adapter
    # runs the registered ``@app.output_parser`` and emits its output as
    # ``custom_event`` frames; off leaves the stream unchanged.
    use_custom_parser: bool | None = None
    # Opt-in declaration for OE-owned durable workflow. Omitted or false means
    # native checkpoints only; only an explicit true opts in. AER advertises
    # this flag with language/framework to OE; session assignment that consumes
    # the record is a follow-up.
    durable_workflow: bool | None = None

    def explicit(self) -> dict[str, bool]:
        """Return only flags present in ``agent.yaml`` (omit unset / ``None``)."""
        return {name: value for name, value in self.model_dump().items() if isinstance(value, bool)}


class A2ASkillConfig(BaseModel):
    """A skill advertised by the agent for A2A discovery."""

    model_config = ConfigDict(extra="forbid")

    name: NonEmptyString
    description: NonEmptyString
    example_input: str | None = None
    example_output: str | None = None


class A2AConfig(BaseModel):
    """Agent-to-agent configuration from ``agent.yaml``.

    When ``enabled`` is true, this agent is discoverable and callable
    by other agents through the Orchestration Engine.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    allowed_callers: list[str] = Field(default_factory=list)
    skills: list[A2ASkillConfig] = Field(default_factory=list)
    input_modes: list[str] = Field(default_factory=list)
    output_modes: list[str] = Field(default_factory=list)


class RuntimeMCPAuthConfig(BaseModel):
    """Authentication settings for a remote MCP server."""

    model_config = ConfigDict(frozen=True, extra="ignore", hide_input_in_errors=True)

    type: Literal["none", "bearer_env", "oauth", "client_credentials"] = Field(
        default="none",
        description="Authentication mode for the MCP server.",
    )
    token_env: NonEmptyString | None = Field(
        default=None,
        description="Environment variable containing the bearer token.",
    )
    redirect_uri: NonEmptyString | None = Field(
        default=None,
        description="Loopback redirect URI used by the interactive OAuth login.",
    )
    client_name: NonEmptyString | None = Field(
        default=None,
        description="OAuth client name shown by the MCP authorization server.",
    )
    scope: NonEmptyString | None = Field(
        default=None,
        description="Optional space-separated OAuth scopes requested for this server.",
    )
    token_url: NonEmptyString | None = Field(
        default=None,
        description="OAuth token endpoint used for client credentials auth.",
    )
    client_id_env: NonEmptyString | None = Field(
        default=None,
        description="Environment variable containing the OAuth client ID.",
    )
    client_secret_env: NonEmptyString | None = Field(
        default=None,
        description="Environment variable containing the OAuth client secret.",
    )

    @model_validator(mode="after")
    def validate_auth(self) -> "RuntimeMCPAuthConfig":
        configured_fields = frozenset(
            field_name
            for field_name in _MCP_AUTH_FIELD_NAMES
            if getattr(self, field_name) is not None
        )
        missing_fields = sorted(_MCP_AUTH_REQUIRED_FIELDS[self.type] - configured_fields)
        if missing_fields:
            required = ", ".join(f"auth.{field_name}" for field_name in missing_fields)
            raise ValueError(f"mcp server auth.type {self.type} requires fields: {required}")

        unsupported_fields = sorted(configured_fields - _MCP_AUTH_ALLOWED_FIELDS[self.type])
        if unsupported_fields:
            unsupported = ", ".join(f"auth.{field_name}" for field_name in unsupported_fields)
            raise ValueError(
                f"mcp server auth.type {self.type} does not support fields: {unsupported}"
            )

        if self.type == "client_credentials" and self.token_url is not None:
            parsed_token_url = urlparse(self.token_url)
            if parsed_token_url.scheme != "https" or parsed_token_url.hostname is None:
                raise ValueError("mcp server auth.token_url must be an absolute https URL")

        for field_name in ("token_env", "client_id_env", "client_secret_env"):
            env_var = getattr(self, field_name)
            if env_var is not None and is_platform_env_var(env_var):
                # These fields are secret-indirection paths read from raw
                # ``os.environ`` and sent to the configured MCP server or token
                # endpoint. Reject platform-owned names so tenant YAML cannot
                # redirect platform secrets to outbound services.
                raise ValueError("Please use a different environment variable name")
        return self


class RuntimeMCPServerConfig(BaseModel):
    """Runtime configuration for one remote MCP server."""

    model_config = ConfigDict(frozen=True, extra="ignore", hide_input_in_errors=True)

    transport: Literal["streamable_http"] = Field(
        default="streamable_http",
        description="MCP transport used to connect to the server.",
    )
    url: NonEmptyString = Field(description="Streamable HTTP endpoint for the MCP server.")
    headers: dict[str, str] = Field(
        default_factory=dict,
        description="Static non-secret HTTP headers sent to the MCP server.",
    )
    auth: RuntimeMCPAuthConfig = Field(
        default_factory=RuntimeMCPAuthConfig,
        description="Authentication settings for the MCP server.",
    )
    allowed_tools: list[NonEmptyString] | None = Field(
        default=None,
        description="Optional fail-closed list of remote MCP tool names to expose.",
    )
    timeout_seconds: int = Field(
        default=30,
        gt=0,
        description="Timeout in seconds for MCP discovery and tool calls.",
    )

    @model_validator(mode="after")
    def validate_server_config(self) -> "RuntimeMCPServerConfig":
        parsed = urlparse(self.url)
        if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
            raise ValueError("mcp server url must be an absolute http(s) URL")
        if parsed.scheme == "http" and self.auth.type != "none":
            raise ValueError("mcp server url must be https when auth.type is set")

        forbidden_headers = sorted(
            header for header in self.headers if header.lower() in _FORBIDDEN_MCP_HEADER_KEYS
        )
        if forbidden_headers:
            raise ValueError(
                "mcp server headers must not include credential headers: "
                f"{forbidden_headers}; use auth.token_env instead"
            )
        return self


class RuntimeMCPConfig(BaseModel):
    """Remote MCP server configuration from ``agent.yaml``."""

    model_config = ConfigDict(frozen=True, extra="ignore", hide_input_in_errors=True)

    servers: dict[NonEmptyString, RuntimeMCPServerConfig] = Field(
        default_factory=dict,
        description="Remote MCP servers keyed by SDK-visible server name.",
    )


class RuntimeConnectorReference(BaseModel):
    """One ``agent.yaml`` connectors entry linking to a connector ``tool.yaml``."""

    model_config = ConfigDict(frozen=True, extra="ignore", hide_input_in_errors=True)

    source: NonEmptyString = Field(
        description="Path to the connector tool.yaml, relative to agent.yaml."
    )


class SecretsConfig(BaseModel):
    # Under `tools`, `invoke_llm` is a reserved key: the platform runs model
    # calls in a tool pod, so the LLM provider key(s) are declared there rather
    # than under a user-defined tool.
    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True)

    # Parsed for compatibility; ignored at enforcement.
    disable_restriction: bool = False
    aer: list[str] = Field(default_factory=list)
    tools: dict[str, list[str]] = Field(default_factory=dict)


class _AgentFileConfig(BaseModel):
    """Subset of the single-agent ``agent.yaml`` shape used by the runtime SDK."""

    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True)

    entrypoint: NonEmptyString | None = None
    language: str | None = None
    framework: str | None = None
    features: AgentFeatureConfig = Field(default_factory=AgentFeatureConfig)
    a2a: A2AConfig = Field(default_factory=A2AConfig)
    mcp: RuntimeMCPConfig = Field(default_factory=RuntimeMCPConfig)
    required_secrets: SecretsConfig = Field(default_factory=SecretsConfig)
    connectors: list[RuntimeConnectorReference] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _null_subsections_to_empty(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for key in ("features", "a2a", "mcp", "required_secrets", "connectors"):
                if data.get(key) is None:
                    data[key] = {}
            if data.get("connectors") == {}:
                data["connectors"] = []
        return data


class RuntimeAgentConfig(BaseModel):
    """Validated runtime view of ``agent.yaml``."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: Path | None = None
    entrypoint: str | None = None
    # Runtime language from agent.yaml; omitted means python (CLI/platform default).
    language: str | None = None
    # Application framework from agent.yaml; omitted means unset at advertise time.
    framework: str | None = None
    features: AgentFeatureConfig = Field(default_factory=AgentFeatureConfig)
    a2a: A2AConfig = Field(default_factory=A2AConfig)
    mcp: RuntimeMCPConfig = Field(default_factory=RuntimeMCPConfig)
    required_secrets: SecretsConfig = Field(default_factory=SecretsConfig)
    # Python/LangGraph only: connector authoring files materialized at startup.
    # The shared Go parser rejects the key for TypeScript agents.
    connectors: list[RuntimeConnectorReference] = Field(default_factory=list)

    def feature_enabled(
        self,
        name: FeatureName,
        default: bool = False,
    ) -> bool:
        """Return a feature flag value, falling back to ``default`` when omitted."""
        value = self.configured_feature(name)
        if value is None:
            return default
        return value

    def configured_feature(
        self,
        name: FeatureName,
    ) -> bool | None:
        """Return the explicit feature value from ``agent.yaml``, if it exists."""
        return getattr(self.features, name)


def _candidate_locations(config_path: ConfigPath | None) -> list[Path]:
    if config_path is not None:
        # Explicit override from the caller. Accept the exact file path or the
        # directory that contains ``agent.yaml``.
        return [Path(config_path)]

    candidates: list[Path] = []
    for raw in (
        # Built/runtime images set the exact in-container ``agent.yaml`` path.
        os.environ.get("AGENTIC_AGENT_CONFIG_PATH"),
        # Local dev stacks set the agent workspace directory.
        os.environ.get("AGENTIC_AGENT_WORKDIR"),
        # Tests and ad-hoc usage often run directly from the agent directory.
        os.getcwd(),
    ):
        if not raw:
            continue
        candidate = Path(raw)
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _config_file_for_location(location: Path) -> Path:
    if location.name == "agent.yaml":
        return location
    return location / "agent.yaml"


def _discover_agent_config_path(config_path: ConfigPath | None = None) -> Path | None:
    for candidate in _candidate_locations(config_path):
        path = _config_file_for_location(candidate)
        if path.is_file():
            return path
    return None


def load_runtime_agent_config(
    config_path: ConfigPath | None = None,
    *,
    env_vars: Mapping[str, str] | None = None,
) -> RuntimeAgentConfig:
    """Load and validate ``agent.yaml`` for runtime use.

    Missing files are treated as an empty config so unit tests and ad-hoc SDK
    usage can still construct ``App`` / ``TenantRuntime`` outside generated
    runtime environments. When a file exists, the runtime validates the fields
    it owns directly (entrypoint/features) while passing the application-owned
    ``config`` block through as raw data.

    ``env_vars`` is the substitution mapping used to resolve ``${VAR}``
    references at allowlisted YAML paths (currently ``mcp.servers.*.url``).
    Pass the tenant-owned subset of the process environment -- never raw
    ``os.environ`` -- so tenant ``agent.yaml`` cannot dereference platform
    secrets. ``agent_engine_runner_shared.utils.tenant_env_vars()`` builds the safe subset.
    When ``env_vars`` is ``None`` and a ``${...}`` reference is present at an
    allowlisted path, the loader raises so the misconfiguration is visible
    instead of falling through to ``urlparse`` with the literal string.
    """

    path = _discover_agent_config_path(config_path)
    if path is None:
        return RuntimeAgentConfig()

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return RuntimeAgentConfig()
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in agent config at {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"agent.yaml at {path} must contain a top-level mapping")

    try:
        data = _interpolate_env_vars(data, env_vars)
    except ValueError as exc:
        raise ValueError(f"Invalid agent.yaml at {path}: {exc}") from exc

    try:
        parsed = _AgentFileConfig.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Invalid agent.yaml at {path}: {exc}") from exc

    language = parsed.language.strip() if isinstance(parsed.language, str) else None
    framework = parsed.framework.strip() if isinstance(parsed.framework, str) else None
    return RuntimeAgentConfig(
        path=path,
        entrypoint=parsed.entrypoint.strip() if isinstance(parsed.entrypoint, str) else None,
        language=language or None,
        framework=framework or None,
        features=parsed.features,
        a2a=parsed.a2a,
        mcp=parsed.mcp,
        required_secrets=parsed.required_secrets,
        connectors=parsed.connectors,
    )
