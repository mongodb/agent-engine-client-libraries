"""Validated runtime views shared by AER and Tool Pod connector registration.

The authoring materializer builds these objects from ``agent.yaml``,
``tool.yaml``, and the local compiled catalog. AER consumes each tool's schema
and registration metadata; Tool Pod consumes the same views to bind executor
callables. Nothing here reads files or performs I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from agent_engine_runner_shared.connectors.definitions import (
    Operation,
    ToolDefinitions,
)
from agent_engine_runner_shared.connectors.executor import DEFAULT_EXECUTION_TIMEOUT_SECONDS

__all__ = [
    "BundledConnector",
    "BundledTool",
    "ConnectorBundle",
    "ConnectorBundleError",
]


class ConnectorBundleError(ValueError):
    """Raised when a connector bundle cannot be registered safely."""


@dataclass(frozen=True)
class BundledConnector:
    """One validated compiled connector from the materialized runtime view."""

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
        """Return detached AER/Tool Pod policy, with delegated binding outside the schema."""
        operation = self.operation
        auth = self.connector.definitions.auth
        provider = auth.provider if auth is not None else None
        return {
            "name": operation.name,
            "description": operation.description.strip() or f"REST connector tool {operation.name}",
            "is_local": False,
            "provider": provider,
            "scopes": list(auth.scopes or []) if auth is not None and provider else [],
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
