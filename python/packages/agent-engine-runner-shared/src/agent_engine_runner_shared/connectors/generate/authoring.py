"""Write the developer-owned ``tool.yaml`` beside a compiled catalog.

This file knows the small authoring format, validates local references and output
containment, and refuses overwrites. It does not parse OpenAPI or compile tools;
those responsibilities stay in ``spec.py`` and ``catalog.py``.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Literal, Sequence
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from agent_engine_runner_shared.connectors.definitions import Auth, Source, validate_connector_name
from agent_engine_runner_shared.connectors.generate.spec import (
    GenerateError,
    _generate_error,
    _quote,
)

_CATALOG_NAME = "tool_defs.yaml"
_TOOL_YAML_NAME = "tool.yaml"
_DEFAULT_PORTS = {"http": 80, "https": 443}


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


def _credential_envs(auth: Auth | None) -> list[str]:
    """Tenant secret references the effective auth presentation reads."""
    if auth is None or auth.type == "none":
        return []
    if auth.type in ("bearer", "api_key"):
        return [env for env in (auth.env,) if env]
    if auth.type == "basic":
        return [env for env in (auth.username_env, auth.password_env) if env]
    return [credential.env for credential in auth.credentials or []]


def _egress_target(base_url: str) -> tuple[str, int]:
    """Host and port the connector's operations will call."""
    parsed = urlsplit(base_url)
    return parsed.hostname or "", parsed.port or _DEFAULT_PORTS.get(parsed.scheme, 443)


def _guidance_comment(base_url: str, auth: Auth | None) -> str:
    """Report the agent.yaml policy this connector needs, as a file comment.

    Platform policy stays explicit and developer-owned: nothing here writes or
    derives agent.yaml. The egress hint is a command so it adapts to whichever
    egress shape the manifest already uses, and every interpolated value passes
    the generator's control-byte sanitizer so no value can escape the comment.
    """
    host, port = _egress_target(base_url)
    envs = _credential_envs(auth)
    target = f"{_quote(host)}:{port}"
    lines = [
        "# Connector operations execute in the tool sandbox; this file enables no",
        "# platform policy. Declare the connector's requirements in agent.yaml:",
        f"#   egress   allow {target} from the tool sandbox, e.g.",
        f"#            agentengine agent egress add --component tool {shlex.quote(target)}",
    ]
    if envs:
        granted = ", ".join(_quote(env) for env in envs)
        lines.append(f"#   secret   grant {granted} to the tool sandbox under")
        lines.append("#            sandboxes.tool.secrets (sandboxes requires an agent block)")
    else:
        lines.append("#   secret   none required; this connector authenticates anonymously")
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
