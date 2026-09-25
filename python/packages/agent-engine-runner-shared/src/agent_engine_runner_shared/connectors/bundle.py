"""Load build-materialized connector definitions for runtime registration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from agent_engine_runner_shared.connectors.definitions import (
    Operation,
    ToolDefinitions,
    load_tool_defs,
)
from agent_engine_runner_shared.connectors.executor import DEFAULT_EXECUTION_TIMEOUT_SECONDS

__all__ = [
    "CONNECTOR_BUNDLE_PATH_ENV",
    "BundledConnector",
    "BundledTool",
    "ConnectorBundle",
    "ConnectorBundleError",
    "load_connector_bundle",
]

CONNECTOR_BUNDLE_PATH_ENV = "AGENTIC_CONNECTOR_BUNDLE_PATH"


class ConnectorBundleError(ValueError):
    """Raised when a materialized connector bundle cannot be registered safely."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)


class _ConnectorIndexEntry(_Model):
    name: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    tool_defs: str = Field(min_length=1, max_length=1024)


class _ConnectorBundleIndex(_Model):
    version: Literal[1]
    connectors: list[_ConnectorIndexEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_connector_names(self) -> Self:
        names = [connector.name for connector in self.connectors]
        if len(names) != len(set(names)):
            raise ValueError("duplicate connector identities")
        return self


class _IndexLoader(yaml.SafeLoader):
    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key in result:
                raise ValueError("connector bundle index requires unique string mapping keys")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


@dataclass(frozen=True)
class BundledConnector:
    """One validated compiled connector from the generated bundle index."""

    name: str
    definitions: ToolDefinitions


@dataclass(frozen=True)
class BundledTool:
    """One operation and the connector definitions needed to execute it."""

    connector: BundledConnector
    operation: Operation

    @property
    def schema(self) -> dict[str, Any]:
        """Return a detached SDK-visible schema."""
        return self.operation.model_dump(include={"name", "description", "inputSchema"})

    @property
    def network(self) -> list[str]:
        """Return the single outbound host declared by this connector."""
        hostname = urlsplit(self.connector.definitions.source.base_url).hostname
        assert hostname is not None  # ToolDefinitions validated the URL.
        return [hostname]

    @property
    def registration_metadata(self) -> dict[str, Any]:
        """Return the shared AER and Tool Pod registration policy."""
        operation = self.operation
        return {
            "name": operation.name,
            "description": operation.description.strip() or f"REST connector tool {operation.name}",
            "is_local": False,
            "provider_type": None,
            "scopes": [],
            "network": self.network,
            "timeout_seconds": DEFAULT_EXECUTION_TIMEOUT_SECONDS,
            "redact_fields": [],
            "connector": self.connector.name,
        }


@dataclass(frozen=True)
class ConnectorBundle:
    """Fully validated runtime bundle shared by AER and Tool Pod startup."""

    connectors: tuple[BundledConnector, ...] = ()
    tools: tuple[BundledTool, ...] = ()

    def require_available_names(self, registered_names: set[str]) -> None:
        """Reject all name collisions before a caller mutates its registry."""
        collisions = sorted({tool.operation.name for tool in self.tools} & registered_names)
        if collisions:
            raise ConnectorBundleError(
                "connector tool names collide with registered or reserved tools: "
                + ", ".join(collisions)
            )


def _load_index(path: Path) -> _ConnectorBundleIndex:
    try:
        document = yaml.load(path.read_text(encoding="utf-8"), Loader=_IndexLoader)
        return _ConnectorBundleIndex.model_validate(document)
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError, ValueError) as exc:
        raise ConnectorBundleError("invalid connector bundle index") from exc


def _resolve_member(bundle_root: Path, reference: str) -> Path:
    if Path(reference).is_absolute() or urlsplit(reference).scheme:
        raise ConnectorBundleError("connector definitions must be a local file inside the bundle")
    try:
        resolved = (bundle_root / reference).resolve(strict=True)
        resolved.relative_to(bundle_root)
    except (OSError, RuntimeError, ValueError):
        raise ConnectorBundleError(
            "connector definitions must be a local file inside the bundle"
        ) from None
    if not resolved.is_file():
        raise ConnectorBundleError("connector definitions must be a local file inside the bundle")
    return resolved


def load_connector_bundle(path: str | Path | None = None) -> ConnectorBundle:
    """Load the generated bundle index and every referenced compiled definition.

    When the platform has not configured a bundle path, return immediately without
    touching the filesystem. Runtime loading accepts only local definition files
    contained by the directory that holds the generated index.
    """
    configured_path = path
    if configured_path is None:
        configured_path = os.environ.get(CONNECTOR_BUNDLE_PATH_ENV)
        if not configured_path:
            return ConnectorBundle()

    try:
        index_path = Path(configured_path).resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        raise ConnectorBundleError("configured connector bundle index does not exist") from None
    if not index_path.is_file():
        raise ConnectorBundleError("configured connector bundle index is not a file")

    index = _load_index(index_path)
    bundle_root = index_path.parent
    connectors: list[BundledConnector] = []
    tools: list[BundledTool] = []
    for entry in index.connectors:
        definition_path = _resolve_member(bundle_root, entry.tool_defs)
        try:
            definitions = load_tool_defs(definition_path)
        except (OSError, UnicodeError, yaml.YAMLError, ValidationError, ValueError) as exc:
            raise ConnectorBundleError("invalid compiled connector definitions") from exc
        if definitions.source.name != entry.name:
            raise ConnectorBundleError("connector index identity does not match its definitions")
        connector = BundledConnector(name=entry.name, definitions=definitions)
        connectors.append(connector)
        tools.extend(BundledTool(connector=connector, operation=tool) for tool in definitions.tools)

    names = [tool.operation.name for tool in tools]
    if len(names) != len(set(names)):
        raise ConnectorBundleError("duplicate connector tool names across the bundle")
    return ConnectorBundle(connectors=tuple(connectors), tools=tuple(tools))
