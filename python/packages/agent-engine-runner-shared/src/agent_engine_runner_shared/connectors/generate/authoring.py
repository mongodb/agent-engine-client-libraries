"""Write the developer-owned ``tool.yaml`` beside a compiled catalog.

This file knows the small authoring format, proposes its auth block, validates
local references and output containment, and refuses overwrites. It does not
parse OpenAPI or compile tools; those responsibilities stay in ``spec.py`` and
``catalog.py``.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from agent_engine_runner_shared.connectors.definitions import (
    _CREDENTIAL_KEYS,
    Auth,
    Source,
    _validate_secret_env,
    validate_connector_name,
)
from agent_engine_runner_shared.connectors.generate.security import _OperationAuth
from agent_engine_runner_shared.connectors.generate.spec import (
    GenerateError,
    _generate_error,
    _quote,
)

_CATALOG_NAME = "tool_defs.yaml"
_TOOL_YAML_NAME = "tool.yaml"
_DEFAULT_PORTS = {"http": 80, "https": 443}
AUTH_CHOICES = ("none", "bearer", "basic", "api_keys")
# The names the workspace Secrets API accepts. The runtime accepts any
# environment name, so a generated binding checks this stricter rule.
_WORKSPACE_SECRET = re.compile(r"[A-Z][A-Z0-9_]{0,127}")


class _AuthoringSource(BaseModel):
    """The ``source`` section of a generated ``tool.yaml``."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["openapi"] = "openapi"
    base_url: str
    spec: str | None = None


class _AuthoringExpose(BaseModel):
    """The fail-closed ``expose`` section of a generated ``tool.yaml``."""

    model_config = ConfigDict(extra="forbid")

    allow: list[str] | None = None
    allow_all: bool | None = None

    @model_validator(mode="after")
    def _require_exposure(self) -> "_AuthoringExpose":
        """Require one explicit exposure policy so generation never grants tools implicitly."""
        if self.allow is not None and self.allow_all:
            raise ValueError("allow and allow_all are mutually exclusive")
        if self.allow is not None and not self.allow:
            # Mirror the materializer: an empty allow list would only produce
            # an authoring file the runtime rejects.
            raise ValueError("expose.allow must be a non-empty list")
        if self.allow is None and self.allow_all is not True:
            raise ValueError("expose requires allow rules or allow_all")
        return self


class _ToolAuthoring(BaseModel):
    """A generated consumer ``tool.yaml``, written through ``safe_dump``."""

    model_config = ConfigDict(extra="forbid")

    name: str
    tool_defs: str
    source: _AuthoringSource
    expose: _AuthoringExpose
    auth: Auth | None = None


def _relative_local(root: Path, reference: str, field: str, must_exist: bool) -> Path:
    """Resolve a portable local reference relative to its authoring file."""
    if Path(reference).is_absolute() or "://" in reference:
        raise GenerateError(f"{field} {_quote(reference)} must be a relative path")
    try:
        resolved = (root / reference).resolve()
    except (OSError, RuntimeError):
        raise GenerateError(f"{field} {_quote(reference)} could not be resolved") from None
    if must_exist and not resolved.is_file():
        raise GenerateError(f"{field} {_quote(reference)} does not exist")
    return resolved


def _ensure_output(root: Path, reference: str) -> Path:
    """Contain generated writes inside the connector folder."""
    root = root.resolve()
    try:
        resolved = (root / reference).resolve(strict=False)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError):
        raise GenerateError(
            f"output {_quote(reference)} resolves outside the connector folder"
        ) from None
    return resolved


def _check_auth_choice(
    auth: str | None, provider: str | None, scopes: Sequence[str] | None
) -> None:
    """Refuse auth choices that contradict each other before anything is read or written."""
    if auth is not None and auth not in AUTH_CHOICES:
        raise GenerateError(f"--auth must be one of {', '.join(AUTH_CHOICES)}")
    if provider is not None and auth is not None:
        raise GenerateError("--provider and --auth cannot be combined; --provider selects OAuth")
    if scopes is not None and provider is None:
        raise GenerateError("--scopes requires --provider")


def _override_record(
    auth: str | None, provider: str | None, records: Iterable[dict[str, Any]]
) -> dict[str, Any] | None:
    """Return the record an explicit choice presents on every operation.

    ``None`` leaves each operation's own security in charge. ``--auth api_keys``
    borrows the placement of the first record that declares keys, because a
    key's location and name cannot come from flags.
    """
    if provider is not None:
        return {"provider": {}}
    if auth is None:
        return None
    if auth == "none":
        return {}
    if auth == "api_keys":
        declared = next((record for record in records if "api_keys" in record), None)
        if declared is None:
            raise GenerateError(
                "--auth api_keys found no declared API key placement; "
                "write auth.api_keys in tool.yaml by hand"
            )
        return {"api_keys": declared["api_keys"]}
    return {auth: {}}


def _catalog_record(externalize: dict[str, Any] | None) -> dict[str, Any] | None:
    """Read a catalog's credential record through an allowlist.

    The catalog is untrusted: only the credential key and each API key's
    location and name are read, never a secret name or header value, and the
    binding model validates them before anything is written. A record from an
    older catalog, which carries ``type`` or is empty for an anonymous override,
    is ignored.
    """
    record = (externalize or {}).get("auth")
    if record is None or record == {} or (isinstance(record, dict) and "type" in record):
        return None
    malformed = GenerateError("catalog externalize.auth is malformed")
    keys = [key for key in record if key != "headers"] if isinstance(record, dict) else []
    if len(keys) != 1 or keys[0] not in _CREDENTIAL_KEYS:
        raise malformed
    if keys[0] != "api_keys":
        return {keys[0]: {}}
    entries = record["api_keys"]
    if not isinstance(entries, list) or not entries:
        raise malformed
    placements = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise malformed
        location, name = entry.get("location"), entry.get("name")
        if not isinstance(location, str) or not isinstance(name, str):
            raise malformed
        placements.append({"location": location, "name": name})
    return {"api_keys": placements}


def _proposed_auth(
    name: str,
    *,
    override: dict[str, Any] | None,
    provider: str | None,
    scopes: Sequence[str] | None,
    declared: Sequence[_OperationAuth],
) -> Auth | None:
    """Return the auth block ``tool.yaml`` starts with, or ``None`` for no credential.

    ``declared`` holds the exposed operations' declared credentials. Without an
    override they propose the key; with ``provider`` they supply the default
    scopes. Secret names always derive from the connector name: the document
    and catalog never choose which workspace secret a binding reads. A derived
    name must be one the workspace Secrets API can create, and each API key
    must derive its own.
    """
    if provider is not None:
        requested = (
            list(scopes)
            if scopes is not None
            else sorted({scope for auth in declared for scope in auth.scopes})
        )
        return _validated_auth({"provider": provider, "scopes": requested or None})
    if override is None:
        # Without a choice, operations whose alternatives conflict were not
        # compiled, so every exposed operation has a record.
        records = [auth.record for auth in declared if auth.record is not None]
        if any("provider" in record for record in records):
            raise GenerateError(
                "the exposed operations authenticate with OAuth; pass --provider "
                "<OAuth app>, or --auth to choose another credential"
            )
        override = records[0] if records else {}
    if not override:
        return None
    prefix = name.upper().replace("-", "_")
    if "bearer" in override:
        binding: dict[str, Any] = {"bearer": f"{prefix}_TOKEN"}
        secrets = [binding["bearer"]]
    elif "basic" in override:
        binding = {"basic": {"username": f"{prefix}_USERNAME", "password": f"{prefix}_PASSWORD"}}
        secrets = list(binding["basic"].values())
    else:
        keys = [
            {
                "env": f"{prefix}_{re.sub(r'[^A-Z0-9]', '_', key['name'].upper())}",
                "location": key["location"],
                "name": key["name"],
            }
            for key in override["api_keys"]
        ]
        binding = {"api_keys": keys}
        secrets = [key["env"] for key in keys]
        shared = [_quote(key["name"]) for key in keys if secrets.count(key["env"]) > 1]
        if shared:
            raise GenerateError(
                f"API keys {', '.join(shared)} derive the same secret name; "
                "pass --auth none and write auth.api_keys in tool.yaml by hand"
            )
    if not all(
        _WORKSPACE_SECRET.fullmatch(secret) and _validate_secret_env(secret) for secret in secrets
    ):
        raise GenerateError(
            f"connector name {_quote(name)} derives a secret name that is not a usable "
            "workspace secret (one that starts with a letter, has at most 128 characters "
            "and is not a platform variable); choose another connector name or write "
            "the auth block by hand"
        )
    return _validated_auth(binding)


def _validated_auth(binding: dict[str, Any]) -> Auth:
    """Validate a generated block with the model the runtime will parse it with."""
    try:
        return Auth.model_validate(binding)
    except ValidationError as exc:
        raise _generate_error(exc, "generated auth block") from exc


def _egress_target(base_url: str) -> tuple[str, int]:
    """Host and port the connector's operations will call."""
    parsed = urlsplit(base_url)
    return parsed.hostname or "", parsed.port or _DEFAULT_PORTS.get(parsed.scheme, 443)


def _guidance_comment(base_url: str, auth: Auth | None) -> str:
    """Report the agent.yaml policy this connector needs, as a file comment.

    The generator never edits agent.yaml; the CLI's create-tool does that after it
    returns. The egress hint is a command so it adapts to whichever egress shape
    the manifest already uses, and every interpolated value passes the generator's
    control-byte sanitizer so no value can escape the comment.
    """
    host, port = _egress_target(base_url)
    envs = auth.secret_names() if auth is not None else []
    target = f"{_quote(host)}:{port}"
    lines = [
        "# Connector operations execute in the tool sandbox. agent.yaml needs these;",
        "# agentengine agent create-tool and agent validate --fix can add them:",
        f"#   egress   allow {target} from the tool sandbox, e.g.",
        f"#            agentengine agent egress add --component tool {shlex.quote(target)}",
    ]
    if envs:
        granted = ", ".join(_quote(env) for env in envs)
        lines.append(f"#   secret   grant {granted} to the tool sandbox under")
        lines.append("#            sandboxes.tool.secrets (sandboxes requires an agent block)")
    elif auth is not None and auth.provider is not None:
        lines.append(
            f"#   auth     OAuth through the project's OAuth app {_quote(auth.provider)}; its grant"
        )
        lines.append(
            "#            decides whether tools call as each user or as the agent's client"
        )
        lines.append("#   secret   none required in the tool sandbox; tokens are request-local")
    else:
        lines.append("#   secret   none required; this connector authenticates anonymously")
    lines.append("#   tools    list this connector's tools under sandboxes.tool.tools when")
    lines.append("#            agent.yaml has a sandboxes block")
    return "\n".join(lines) + "\n"


def _write_tool_yaml(
    out_dir: Path,
    *,
    name: str,
    catalog_ref: str,
    base_url: str,
    allow: Sequence[str] | None,
    auth: Auth | None,
    spec_ref: str | None,
) -> Path:
    """Validate and serialize the complete consumer-owned authoring file."""
    try:
        authoring = _ToolAuthoring(
            name=name,
            tool_defs=catalog_ref,
            source=_AuthoringSource(base_url=base_url, spec=spec_ref),
            expose=(
                _AuthoringExpose(allow=list(allow))
                if allow is not None
                else _AuthoringExpose(allow_all=True)
            ),
            auth=auth,
        )
    except ValidationError as exc:
        raise _generate_error(exc, "generated authoring file") from exc
    target = _ensure_output(out_dir, _TOOL_YAML_NAME)
    document = authoring.model_dump(exclude_none=True)
    if auth is not None:
        document["auth"] = auth.model_dump(exclude_none=True, exclude_defaults=True)
    target.write_text(
        _guidance_comment(base_url, auth) + yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )
    return target


def _declared_spec_ref(spec: str, out_dir: Path) -> str | None:
    """The spec reference to record for regeneration, relative to the
    connector folder; specs outside the folder are not locally owned."""
    try:
        return str(Path(spec).resolve().relative_to(out_dir.resolve()))
    except (OSError, ValueError):
        return None


def _validate_binding(name: str, base_url: str) -> None:
    """Validate a consumer binding through the runtime models the
    materializer will parse it with, so scaffolding cannot produce an
    authoring file the build subsequently rejects."""
    try:
        validate_connector_name(name)
    except ValueError as exc:
        raise GenerateError(str(exc)) from None
    try:
        Source(name=name, base_url=base_url)
    except ValidationError as exc:
        raise _generate_error(exc, "consumer binding") from exc
