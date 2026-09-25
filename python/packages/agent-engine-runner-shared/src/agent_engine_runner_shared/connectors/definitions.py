"""Define and validate the stable connector contract consumed at runtime.

The generator writes these models to ``tool_defs.yaml`` and the executor reads
them back. OpenAPI concepts do not cross this boundary: this file describes only
validated tool arguments, HTTP placement, request-body encoding, and materialized
credentials.
"""

from __future__ import annotations

import ipaddress
import json
import re
from pathlib import Path
from typing import Any, Literal, Self
from urllib.parse import unquote, urlsplit

import yaml
from jsonschema import Draft202012Validator, SchemaError
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent_engine_runner_shared.utils import is_platform_env_var

__all__ = ["Auth", "Credential", "StrictSafeLoader", "ToolDefinitions", "load_tool_defs"]

_HEADER = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_RESERVED_HEADERS = {
    "authorization",
    "proxy-authorization",
    "host",
    "cookie",
    "content-length",
    "transfer-encoding",
    "connection",
    "content-type",
}
_TOOL_ARGUMENT_SCHEMA_KEYS = {
    "type",
    "properties",
    "required",
    "$defs",
}
# The parameter types the executor can serialize outside a JSON body. The
# generator mirrors this rule when compiling OpenAPI parameters.
SCALAR_PARAM_TYPES = ("string", "number", "integer", "boolean")
CONNECTOR_NAME_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


def validate_connector_name(value: str) -> str:
    """Connector names key connector identity and appear in bundle indexes."""
    if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value) is None:
        raise ValueError(f"connector name must match {CONNECTOR_NAME_PATTERN}")
    return value


def validate_url(value: str) -> str:
    """Connector transport policy: HTTPS, or HTTP loopback, without userinfo."""
    url = urlsplit(value)
    loopback = url.hostname == "localhost"
    try:
        loopback = loopback or ipaddress.ip_address(url.hostname or "").is_loopback
    except ValueError:
        pass
    if (
        not url.hostname
        or url.username is not None
        or url.password is not None
        or url.fragment
        or "\\" in value
        or any(ord(c) <= 32 for c in value)
        or not (url.scheme == "https" or (url.scheme == "http" and loopback))
    ):
        raise ValueError(
            "connector URL requires HTTPS (HTTP allowed only for loopback), no userinfo or fragment"
        )
    _ = url.port
    return value


def validate_segment(value: str) -> str:
    """Reject traversal and separators, including percent-encoded spellings."""
    # Callers provide raw values and the executor owns URL encoding. An encoded
    # percent is the required marker for another decoding layer, so reject it
    # before checking the once-decoded value.
    if "%25" in value.lower():
        raise ValueError("path parameter must be a single non-traversing segment")
    decoded = unquote(value)
    if decoded in (".", "..") or any(c in decoded for c in "/\\"):
        raise ValueError("path parameter must be a single non-traversing segment")
    return value


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)


class Source(_Model):
    name: str = Field(min_length=1, description="Connector identity")
    base_url: str = Field(description="Configured API base URL")
    openapi: str | None = Field(default=None, description="Authoring source provenance only")
    spec_sha256: str | None = Field(
        default=None, description="Source fingerprint; not a runtime pin"
    )
    generator: str | None = Field(
        default=None, description="Generating tool and version; authoring provenance only"
    )

    @model_validator(mode="after")
    def validate_base(self) -> Self:
        """Keep one safe base URL so operations cannot choose their own origin."""
        validate_url(self.base_url)
        parsed = urlsplit(self.base_url)
        if parsed.query:
            raise ValueError("base URL cannot contain a query")
        all_placeholders = set(re.findall(r"\{([^{}]+)\}", self.base_url))
        path_placeholders = set(re.findall(r"\{([^{}]+)\}", parsed.path))
        if all_placeholders != path_placeholders:
            raise ValueError("base URL placeholders are allowed only in the URL path")
        return self


def _validate_secret_env(value: str | None) -> bool:
    """Accept tenant secret references while reserving platform-owned variables."""
    return bool(
        value and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value) and not is_platform_env_var(value)
    )


def _validate_static_headers(headers: dict[str, str] | None, credentials: set[str]) -> None:
    """Prevent constant headers from taking over transport or credential fields."""
    for name, value in (headers or {}).items():
        if not _HEADER.fullmatch(name) or name.lower() in _RESERVED_HEADERS:
            raise ValueError("static auth headers cannot control transport headers")
        if not value or any(ord(character) < 32 or ord(character) >= 127 for character in value):
            raise ValueError("static auth header values must contain printable ASCII")
    if credentials & {name.lower() for name in (headers or {})}:
        raise ValueError("static auth headers collide with credential headers")


class Credential(_Model):
    env: str = Field(description="Tenant secret environment reference")
    location: Literal["header", "query", "path", "cookie"]
    name: str = Field(min_length=1, description="Credential wire name")
    prefix: str = Field(default="", description="Non-secret value prefix")

    @model_validator(mode="after")
    def validate_credential(self) -> Self:
        """Validate one secret-to-wire binding before the executor can use it."""
        if not _validate_secret_env(self.env):
            raise ValueError("credential requires a tenant-owned environment reference")
        if any(ord(character) < 32 or ord(character) >= 127 for character in self.prefix):
            raise ValueError("credential prefix must contain printable ASCII")
        if self.location in ("header", "cookie") and not _HEADER.fullmatch(self.name):
            raise ValueError("credential header and cookie names must be HTTP tokens")
        if self.location == "header" and self.name.lower() in _RESERVED_HEADERS - {"authorization"}:
            raise ValueError("credential cannot use a transport-controlled header")
        if self.location in ("query", "path") and any(
            ord(character) <= 32 or character in "#&?/{}`\\" for character in self.name
        ):
            raise ValueError("credential query and path names contain unsafe characters")
        return self


class Auth(_Model):
    type: Literal["none", "bearer", "api_key", "basic", "api_keys"] = Field(
        default="none", description="Static auth presentation"
    )
    env: str | None = Field(default=None, description="Tenant secret environment reference")
    header: str | None = Field(default=None, description="API key header name")
    location: Literal["header", "query", "path", "cookie"] | None = None
    name: str | None = None
    prefix: str = ""
    username_env: str | None = None
    password_env: str | None = None
    credentials: list[Credential] | None = Field(default=None, min_length=2)
    headers: dict[str, str] | None = None

    @model_validator(mode="after")
    def validate_auth(self) -> Self:
        """Require exactly the fields belonging to the selected auth presentation."""
        configured = {
            name
            for name in self.__class__.model_fields
            if name == "type" or getattr(self, name) not in (None, "", [], {})
        }
        allowed = {
            "none": {"type"},
            "bearer": {"type", "env", "headers"},
            "api_key": {"type", "env", "header", "location", "name", "prefix", "headers"},
            "basic": {"type", "username_env", "password_env", "headers"},
            "api_keys": {"type", "credentials", "headers"},
        }[self.type]
        if configured - allowed:
            raise ValueError(f"{self.type} auth contains fields for another auth type")

        credential_headers: set[str] = set()
        if self.type == "bearer":
            if not _validate_secret_env(self.env):
                raise ValueError("bearer auth requires one tenant secret env")
            credential_headers.add("authorization")
        elif self.type == "api_key":
            location = self.location or "header"
            name = self.name or self.header or "X-API-Key"
            if self.header is not None and (location != "header" or self.name is not None):
                raise ValueError("legacy auth.header cannot be combined with location or name")
            if not _validate_secret_env(self.env):
                raise ValueError("API key auth requires a tenant-owned environment reference")
            assert self.env is not None
            Credential(env=self.env, location=location, name=name, prefix=self.prefix)
            if location == "header":
                credential_headers.add(name.lower())
        elif self.type == "basic":
            if not _validate_secret_env(self.username_env) or not _validate_secret_env(
                self.password_env
            ):
                raise ValueError("basic auth requires tenant-owned username and password envs")
            if self.username_env == self.password_env:
                raise ValueError("basic auth username and password must use different secrets")
            credential_headers.add("authorization")
        elif self.type == "api_keys":
            if not self.credentials:
                raise ValueError("multi-key auth requires at least two credentials")
            identities = [
                (credential.location, credential.name.lower()) for credential in self.credentials
            ]
            if len(identities) != len(set(identities)):
                raise ValueError("multi-key auth has duplicate credential destinations")
            credential_headers = {
                credential.name.lower()
                for credential in self.credentials
                if credential.location == "header"
            }
        _validate_static_headers(self.headers, credential_headers)
        return self

    def header_name(self) -> str | None:
        """Return the single generated auth header, when this auth shape has one."""
        if self.type == "bearer":
            return "Authorization"
        if self.type == "api_key":
            if (self.location or "header") == "header":
                return self.name or self.header or "X-API-Key"
        return None

    def header_names(self) -> set[str]:
        """List auth-controlled headers so model parameters cannot collide with them."""
        if self.type in ("bearer", "basic"):
            names = {"authorization"}
        elif self.type == "api_key" and (header_name := self.header_name()):
            names = {header_name.lower()}
        elif self.type == "api_keys":
            names = {
                credential.name.lower()
                for credential in self.credentials or []
                if credential.location == "header"
            }
        else:
            names = set()
        return names | {name.lower() for name in (self.headers or {})}

    def credential_bindings(self) -> list[Credential]:
        """Normalize single and compound API keys into the executor's common form."""
        if self.type == "api_key" and self.env:
            return [
                Credential(
                    env=self.env,
                    location=self.location or "header",
                    name=self.name or self.header or "X-API-Key",
                    prefix=self.prefix,
                )
            ]
        return list(self.credentials or [])


class Parameters(_Model):
    path: list[str] = Field(default_factory=list, description="Path parameter names")
    query: list[str] = Field(
        default_factory=list, description="Query parameter names (scalar or array-of-scalar)"
    )
    header: list[str] = Field(
        default_factory=list, description="Header parameter names (scalar or array-of-scalar)"
    )
    cookie: list[str] = Field(default_factory=list, description="Scalar cookie parameter names")
    body: list[str] = Field(default_factory=list, description="Flattened JSON body property names")
    wrapped: bool = Field(
        default=False, description="Use the body argument as the complete request body"
    )
    query_style: dict[str, str] = Field(
        default_factory=dict,
        description="Non-default query serialization per parameter: comma-joined "
        "lists (explode: false), bracketed deepObject lists, or JSON values; "
        "absent entries serialize as repeated keys",
    )
    form_style: dict[str, str] = Field(
        default_factory=dict,
        description="Non-default serialization for wrapped form-body properties",
    )


def _check_schema(schema: Any, local_defs: set[str] | None = None) -> None:
    """Reject schema references that would require runtime document resolution."""
    if isinstance(schema, dict):
        if local_defs is None:
            definitions = schema.get("$defs", {})
            local_defs = set(definitions) if isinstance(definitions, dict) else set()
        reference = schema.get("$ref")
        if reference is not None and (
            not isinstance(reference, str)
            or not reference.startswith("#/$defs/")
            or reference.removeprefix("#/$defs/") not in local_defs
        ):
            raise ValueError("compiled schemas may reference only local definitions")
        if any(key in schema for key in ("$dynamicRef", "$recursiveRef")):
            raise ValueError("compiled schemas may reference only local definitions")
        for value in schema.values():
            _check_schema(value, local_defs)
    elif isinstance(schema, list):
        for value in schema:
            _check_schema(value, local_defs)


def _non_null_type(property_schema: dict[str, Any]) -> Any:
    """Return the one non-null type from an OpenAPI nullable schema."""
    kind = property_schema.get("type")
    if isinstance(kind, list) and "null" in kind:
        members = [member for member in kind if member != "null"]
        if len(members) == 1:
            return members[0]
    return kind


def union_branches(schema: dict[str, Any]) -> list[Any] | None:
    """Return a plain ``anyOf`` or ``oneOf`` union's branches."""
    for keyword in ("anyOf", "oneOf"):
        branches = schema.get(keyword)
        if isinstance(branches, list) and branches:
            if (
                set(schema)
                - {keyword, "description", "title", "default"}
                - {key for key in schema if key.startswith("x-")}
            ):
                return None
            return branches
    return None


def _param_branch_ok(branch: Any, *, allow_object: bool, allow_array: bool) -> bool:
    """Check that one schema branch has a wire spelling for its parameter location."""
    if not isinstance(branch, dict):
        return False
    kind = _non_null_type(branch)
    if kind in SCALAR_PARAM_TYPES:
        return True
    if kind == "array":
        items = branch.get("items")
        return allow_array and isinstance(items, dict) and items.get("type") in SCALAR_PARAM_TYPES
    return allow_object and kind == "object"


def _param_schema_ok(schema: Any, *, allow_object: bool, allow_array: bool) -> bool:
    """Require every possible union branch to be serializable, not just one branch."""
    if _param_branch_ok(schema, allow_object=allow_object, allow_array=allow_array):
        return True
    branches = union_branches(schema) if isinstance(schema, dict) else None
    return bool(branches) and all(
        _param_branch_ok(branch, allow_object=allow_object, allow_array=allow_array)
        for branch in branches
    )


class Operation(_Model):
    name: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$", description="Registered tool name")
    description: str = Field(default="", description="Tool description")
    method: Literal["GET", "HEAD", "OPTIONS", "POST", "PUT", "PATCH", "DELETE"] = Field(
        description="HTTP method"
    )
    path: str = Field(description="API path template relative to the base URL")
    params: Parameters = Field(description="Argument placement")
    inputSchema: dict[str, Any] = Field(description="Compiled JSON input schema")
    body_encoding: Literal["json", "form", "multipart", "text", "binary"] = Field(
        default="json", description="Request body wire encoding"
    )
    content_type: str | None = Field(
        default=None, description="Exact JSON request media type when vendor-specific"
    )

    @model_validator(mode="after")
    def validate_operation(self) -> Self:
        """Keep schema, argument placement, and HTTP serialization mutually consistent."""
        if self.content_type is not None and (
            self.body_encoding != "json"
            or not self.content_type.startswith("application/")
            or not self.content_type.endswith("+json")
            or any(ord(character) <= 32 for character in self.content_type)
        ):
            raise ValueError("content type must be a vendor application +json media type")
        if (
            not self.path.startswith("/")
            or self.path.startswith("//")
            or any(c in self.path for c in "?#\\")
            or any(ord(c) <= 32 for c in self.path)
        ):
            raise ValueError(
                "operation path must be an absolute API path without query or fragment"
            )
        for segment in self.path.split("/"):
            validate_segment(segment)
        placeholders = re.findall(r"\{([^{}]+)\}", self.path)
        if set(placeholders) != set(self.params.path) or re.search(
            r"[{}]", re.sub(r"\{[^{}]+\}", "", self.path)
        ):
            raise ValueError("path placeholders must match path parameters")
        groups = [
            self.params.path,
            self.params.query,
            self.params.header,
            self.params.cookie,
            self.params.body,
            ["body"] if self.params.wrapped else [],
        ]
        names = [name for group in groups for name in group]
        if len(names) != len(set(names)) or (self.params.wrapped and self.params.body):
            raise ValueError("each argument must have exactly one placement")
        if set(self.params.query_style) - set(self.params.query):
            raise ValueError("query serialization styles must name query parameters")
        if any(
            style not in ("comma", "brackets", "json") for style in self.params.query_style.values()
        ):
            raise ValueError("unknown query serialization style")
        if self.params.form_style and self.body_encoding != "form":
            raise ValueError("form serialization styles require a form body")
        if any(style not in ("comma", "brackets") for style in self.params.form_style.values()):
            raise ValueError("unknown form serialization style")
        for name in self.params.header + self.params.cookie:
            if not _HEADER.fullmatch(name):
                raise ValueError("invalid header or cookie parameter name")
        headers = [name.lower() for name in self.params.header]
        if len(headers) != len(set(headers)) or set(headers) & _RESERVED_HEADERS:
            raise ValueError("header parameters cannot override auth or transport headers")
        try:
            json.dumps(self.inputSchema, allow_nan=False)
        except (TypeError, ValueError, RecursionError):
            raise ValueError("compiled input schema must contain JSON values") from None
        try:
            _check_schema(self.inputSchema)
            Draft202012Validator.check_schema(self.inputSchema)
        except RecursionError:
            raise ValueError("compiled input schema is too deeply nested") from None
        except SchemaError:
            raise ValueError("invalid compiled input schema") from None
        if not set(self.inputSchema).issubset(_TOOL_ARGUMENT_SCHEMA_KEYS):
            raise ValueError("tool argument envelope contains unsupported keywords")
        properties = self.inputSchema.get("properties", {})
        if self.body_encoding == "form":
            body_schema = properties.get("body")
            body_properties = (
                body_schema.get("properties") if isinstance(body_schema, dict) else None
            )
            if not self.params.wrapped or self.params.body or not isinstance(body_properties, dict):
                raise ValueError("form bodies must be wrapped objects")
            if set(self.params.form_style) - set(body_properties):
                raise ValueError("form serialization styles must name body properties")
        if self.inputSchema.get("type") != "object" or set(properties) != set(names):
            raise ValueError("input schema properties must exactly match argument placement")
        if not set(self.params.path).issubset(self.inputSchema.get("required", [])):
            raise ValueError("path parameters must be required")
        if not set(self.inputSchema.get("required", [])).issubset(properties):
            raise ValueError("required arguments must have a placement")
        for placement, names in (
            ("path", self.params.path),
            ("query", self.params.query),
            ("header", self.params.header),
            ("cookie", self.params.cookie),
        ):
            allow_object = placement == "query"
            for name in names:
                if not _param_schema_ok(
                    properties[name],
                    allow_object=allow_object,
                    allow_array=placement != "cookie",
                ):
                    raise ValueError(
                        "non-body parameters support scalar, array, object, or union "
                        "serialization only"
                    )
        return self


class ToolDefinitions(_Model):
    source: Source = Field(description="Connector target and provenance")
    auth: Auth | None = Field(default=None, description="Static credential configuration")
    externalize: dict[str, Any] | None = Field(
        default=None, description="Generator provenance and CLI materialization metadata"
    )
    tools: list[Operation] = Field(min_length=1, description="Compiled operations")

    @model_validator(mode="after")
    def validate_tools(self) -> Self:
        """Validate catalog-wide identity and credential ownership invariants."""
        names = [tool.name for tool in self.tools]
        if len(names) != len(set(names)):
            raise ValueError("duplicate tool names")
        auth_headers = self.auth.header_names() if self.auth else set()
        if any(auth_headers & {name.lower() for name in tool.params.header} for tool in self.tools):
            raise ValueError("auth header cannot be a model argument")
        credentials = self.auth.credential_bindings() if self.auth else []
        for credential in credentials:
            if credential.location == "query" and any(
                credential.name in tool.params.query for tool in self.tools
            ):
                raise ValueError("auth query parameter cannot be a model argument")
            if credential.location == "cookie" and any(
                credential.name in tool.params.cookie for tool in self.tools
            ):
                raise ValueError("auth cookie cannot be a model argument")
        path_credentials = {
            credential.name for credential in credentials if credential.location == "path"
        }
        base_placeholders = set(re.findall(r"\{([^{}]+)\}", urlsplit(self.source.base_url).path))
        if path_credentials and path_credentials != base_placeholders:
            raise ValueError("base URL placeholders must exactly match path credential names")
        return self

    def tool_schemas(self) -> list[dict[str, Any]]:
        """Return detached tool descriptions without credentials or transport metadata."""
        return [
            tool.model_dump(include={"name", "description", "inputSchema"}) for tool in self.tools
        ]

    def write(self, path: str | Path) -> None:
        """Serialize a compiled artifact."""
        Path(path).write_text(
            yaml.safe_dump(self.model_dump(exclude_none=True), sort_keys=False), encoding="utf-8"
        )


class StrictSafeLoader(yaml.SafeLoader):
    """Safe loading with unique string mapping keys. Duplicate keys resolve
    last-wins in plain YAML/JSON parsing, which would let a checked-out
    document hide the value that actually drives compilation — so every
    YAML input this package parses (spec, authoring, or compiled artifact)
    goes through this loader."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        """Reject duplicate or non-string keys before they can change document meaning."""
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key in result:
                raise ValueError("YAML requires unique string mapping keys")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def _strict_implicit_resolvers() -> dict:
    """Implicit resolvers without YAML 1.1 bool/timestamp coercion.

    PyYAML resolves ``on``/``off``/``yes``/``no`` to booleans and date-like
    scalars to dates — but real-world OpenAPI documents use both as plain
    strings (GitHub's spec has an ``on:`` key; Stripe versions specs
    ``2022-11-15``; Square embeds out-of-range times that crash timestamp
    construction outright). Only the ``true``/``false`` spellings stay
    boolean, since the generator reads ``required: true``-style flags as
    booleans. Timestamps stay strings; nothing in this package interprets
    dates.
    """
    resolvers: dict = {}
    for first, mappings in yaml.SafeLoader.yaml_implicit_resolvers.items():
        kept = [
            (tag, regexp)
            for tag, regexp in mappings
            if tag not in ("tag:yaml.org,2002:bool", "tag:yaml.org,2002:timestamp")
        ]
        if kept:
            resolvers[first] = kept
    strict_bool = re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$")
    for first in "tTfF":
        resolvers.setdefault(first, []).append(("tag:yaml.org,2002:bool", strict_bool))
    return resolvers


StrictSafeLoader.yaml_implicit_resolvers = _strict_implicit_resolvers()


def _construct_yaml_default_value(loader: yaml.SafeLoader, node: yaml.ScalarNode) -> str:
    """Read YAML's bare ``=`` value as the literal used by published API enums."""
    return "="


StrictSafeLoader.add_constructor("tag:yaml.org,2002:value", _construct_yaml_default_value)


def loads_yaml_document(text: str) -> Any:
    """Parse YAML through the strict loader without repairing invalid input."""
    return yaml.load(text, Loader=StrictSafeLoader)


def load_tool_defs(path: str | Path) -> ToolDefinitions:
    """Read and validate a local compiled artifact; never load an OpenAPI source."""
    return ToolDefinitions.model_validate(
        loads_yaml_document(Path(path).read_text(encoding="utf-8"))
    )
