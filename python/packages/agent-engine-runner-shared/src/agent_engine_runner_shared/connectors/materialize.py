"""In-memory materialization of connector authoring files into a runtime bundle.

This is the bridge between the authoring surface (``tool.yaml`` + a local
compiled catalog) and the merged runtime registration paths. Like MCP tool
discovery, each process derives its own view from the same agent.yaml
configuration: AER registers schemas as ordinary remote tools, the Tool Pod
registers executor callables. Nothing is written or shared between pods;
every pod holds its own copy of the authoring files and materializes in
memory. No network I/O: catalogs are local files resolved relative to their
``tool.yaml``. HTTPS catalog references are a post-preview expansion.

``tool.yaml`` is author-owned (CLI-generated in a later slice, hand-edited
after), so it is parsed leniently; the contract is enforced where it is free
and load-bearing:

- the fail-closed ``expose`` decision (no allow rules expose nothing),
- the connector name, which keys the connector identity,
- workspace containment for local catalog paths (no escape, no network),
- the shared definition loader revalidating every effective definition, so a
  malformed auth binding or endpoint fails with the loader's error rather
  than a silent one.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import yaml

from agent_engine_runner_shared.connectors.bundle import (
    CONNECTOR_BUNDLE_PATH_ENV,
    BundledConnector,
    BundledTool,
    ConnectorBundle,
    load_connector_bundle,
)
from agent_engine_runner_shared.connectors.definitions import (
    Auth,
    ToolDefinitions,
    load_tool_defs,
    validate_connector_name,
)

if TYPE_CHECKING:
    from agent_engine_runner_shared.agent_config import RuntimeAgentConfig

__all__ = [
    "MaterializeError",
    "materialize_connectors",
    "resolve_runtime_bundle",
]


class MaterializeError(ValueError):
    """Raised when connector authoring inputs cannot produce a runtime bundle."""


def _load_tool_yaml(path: Path) -> dict[str, Any]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise MaterializeError(f"invalid connector authoring file {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise MaterializeError(f"connector authoring file {path} must contain a mapping")
    return document


def _require_string(document: dict[str, Any], key: str, path: Path) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value:
        raise MaterializeError(
            f"connector authoring file {path} requires a non-empty string {key!r}"
        )
    return value


def _resolve_contained(root: Path, base: Path, reference: str, field: str) -> Path:
    """Resolve ``reference`` against ``base`` and require it to stay inside ``root``."""
    if Path(reference).is_absolute() or "://" in reference:
        raise MaterializeError(
            f"{field} must be a local catalog path relative to this file; "
            "HTTPS catalog references are not supported in the preview"
        )
    try:
        resolved = (base / reference).resolve(strict=True)
    except (OSError, RuntimeError):
        raise MaterializeError(
            f"{field} {reference!r} does not exist inside the agent workspace root"
        ) from None
    try:
        resolved.relative_to(root)
    except ValueError:
        # resolve() follows symlinks, so an escaping symlink lands outside root.
        raise MaterializeError(
            f"{field} {reference!r} resolves outside the agent workspace root"
        ) from None
    if not resolved.is_file():
        raise MaterializeError(f"{field} {reference!r} is not a file")
    return resolved


def _operation_names(name: str, field: str, values: list[Any]) -> list[str]:
    """Validate exposure entries as operation-name strings before set use —
    non-string elements (nested lists, mappings, numbers) would otherwise
    escape as raw TypeError from set/sort operations."""
    for value in values:
        if not isinstance(value, str) or not value:
            raise MaterializeError(
                f"connector {name!r}: expose.{field} entries must be non-empty operation names"
            )
    return values


def _select_operations(
    name: str, expose: dict[str, Any], definitions: ToolDefinitions
) -> list[str]:
    """Apply the consumer's fail-closed exposure policy to the catalog."""
    available = [tool.name for tool in definitions.tools]
    allow = expose.get("allow")
    allow_all = expose.get("allow_all") is True
    disallow = expose.get("disallow")
    if disallow is None:
        disallow = []
    if not isinstance(disallow, list):
        # Presence with a non-list value is an error — including falsy shapes
        # (false, 0, "", {}), which must not silently normalize to no-op.
        raise MaterializeError(
            f"connector {name!r}: expose.disallow must be a list of operation names"
        )
    _operation_names(name, "disallow", disallow)

    if allow_all:
        if allow is not None:
            raise MaterializeError(
                f"connector {name!r}: expose.allow and expose.allow_all are mutually exclusive"
            )
        selected = set(available)
    else:
        if not isinstance(allow, list) or not allow:
            raise MaterializeError(
                f"connector {name!r}: expose is fail-closed — declare at least one "
                "allow rule or set allow_all: true"
            )
        _operation_names(name, "allow", allow)
        unmatched = sorted(set(allow) - set(available))
        if unmatched:
            raise MaterializeError(
                f"connector {name!r}: allow rules match no catalog operation: "
                f"{unmatched}; available operations: {sorted(available)}"
            )
        selected = set(allow)

    unmatched_disallow = sorted(set(disallow) - set(available))
    if unmatched_disallow:
        raise MaterializeError(
            f"connector {name!r}: disallow rules match no catalog operation: {unmatched_disallow}"
        )
    selected -= set(disallow)

    if not selected:
        raise MaterializeError(
            f"connector {name!r}: exposure filters select no operations; "
            "an agent must expose at least one connector tool"
        )
    return [tool.name for tool in definitions.tools if tool.name in selected]


def _apply_connector_bindings(
    name: str,
    base_url: str,
    auth: dict[str, Any] | None,
    definitions: ToolDefinitions,
    selected: list[str],
) -> ToolDefinitions:
    """Resolve the credential binding and endpoint every selected tool uses.

    Reads the connector's authoring settings and stamps them onto each
    selected operation: the credential environment reference (which workspace
    secret the executor presents) and the API endpoint. The endpoint is
    consumer-owned with no fallback: a catalog-chosen destination combined
    with a consumer-chosen credential would send the tenant secret to
    whoever published the catalog. The rebuild re-runs the shared definition
    validators, which own auth shape, env naming and URL policy.
    """
    document: dict[str, Any] = definitions.model_dump()
    document["source"] = {**document["source"], "name": name, "base_url": base_url}
    document["auth"] = auth
    document["tools"] = [
        tool.model_dump() for tool in definitions.tools if tool.name in set(selected)
    ]
    try:
        return ToolDefinitions.model_validate(document)
    except ValueError as exc:
        raise MaterializeError(f"connector {name!r}: invalid effective definition: {exc}") from exc


def _materialize_connector(tool_yaml_path: Path, root: Path) -> ToolDefinitions:
    """Cross-reference one ``tool.yaml`` with its local catalog and populate
    the effective definitions for exactly the operations it exposes."""
    document = _load_tool_yaml(tool_yaml_path)
    name = _require_string(document, "name", tool_yaml_path)
    try:
        validate_connector_name(name)
    except ValueError as exc:
        raise MaterializeError(f"connector {name!r}: {exc}") from None

    reference = _require_string(document, "tool_defs", tool_yaml_path)
    catalog_path = _resolve_contained(root, tool_yaml_path.parent, reference, "tool_defs")
    try:
        definitions = load_tool_defs(catalog_path)
    except (OSError, UnicodeError, yaml.YAMLError, ValueError) as exc:
        raise MaterializeError(
            f"connector {name!r}: invalid catalog {catalog_path.name}: {exc}"
        ) from exc

    source = document.get("source") or {}
    if not isinstance(source, dict):
        raise MaterializeError(f"connector {name!r}: source must be a mapping")
    base_url = source.get("base_url")
    if not isinstance(base_url, str) or not base_url:
        # Consumer-owned endpoint, no catalog fallback: a catalog-selected
        # destination combined with a consumer-selected credential would send
        # the tenant secret to the catalog publisher's host.
        raise MaterializeError(
            f"connector {name!r}: source.base_url is required in tool.yaml "
            "(the catalog's base URL is never used as a fallback)"
        )
    auth = document.get("auth")
    if auth is not None and not isinstance(auth, dict):
        raise MaterializeError(f"connector {name!r}: auth must be a mapping")
    if auth:
        # Validate the credential binding through the shared model before
        # normalizing, so malformed intent (env without type, credentials on
        # none) fails loudly instead of silently degrading to no auth.
        try:
            Auth.model_validate(auth)
        except ValueError as exc:
            raise MaterializeError(f"connector {name!r}: invalid auth: {exc}") from exc
        if auth.get("type", "none") == "none":
            auth = None
    expose = document.get("expose") or {}
    if not isinstance(expose, dict):
        raise MaterializeError(f"connector {name!r}: expose must be a mapping")

    selected = _select_operations(name, expose, definitions)
    return _apply_connector_bindings(name, base_url, auth, definitions, selected)


def materialize_connectors(connector_paths: Sequence[str | Path], *, root: Path) -> ConnectorBundle:
    """Materialize connector authoring files into an in-memory runtime bundle.

    ``root`` is the containment boundary (typically the ``agent.yaml``
    directory): every local path inside each ``tool.yaml`` must resolve inside
    it. Pure function of the input files — no filesystem writes, no network.
    """
    if not connector_paths:
        raise MaterializeError("no connector authoring files configured")
    root = root.resolve()
    if not root.is_dir():
        raise MaterializeError(f"agent root {root} does not exist")

    tool_yaml_paths: list[Path] = []
    for provided in connector_paths:
        if Path(provided).is_absolute():
            # Callers may pass an absolute path, but it must still resolve
            # inside the containment root (symlinks included).
            try:
                resolved = Path(provided).resolve(strict=True)
                resolved.relative_to(root)
            except (OSError, RuntimeError, ValueError):
                raise MaterializeError(
                    f"connector authoring file {provided} does not exist inside {root}"
                ) from None
            tool_yaml_paths.append(resolved)
        else:
            tool_yaml_paths.append(
                _resolve_contained(root, root, str(provided), "connectors source")
            )

    connectors: list[BundledConnector] = []
    seen_connectors: set[str] = set()
    seen_tools: dict[str, str] = {}
    tools: list[BundledTool] = []
    for tool_yaml_path in tool_yaml_paths:
        definitions = _materialize_connector(tool_yaml_path, root)
        name = definitions.source.name
        if name in seen_connectors:
            raise MaterializeError(f"duplicate connector name {name!r}")
        seen_connectors.add(name)

        for operation in definitions.tools:
            if operation.name in seen_tools:
                raise MaterializeError(
                    f"connector tool name collision between {seen_tools[operation.name]!r} "
                    f"and {name!r}: {operation.name}"
                )
            seen_tools[operation.name] = name
            tools.append(
                BundledTool(
                    connector=BundledConnector(name=name, definitions=definitions),
                    operation=operation,
                )
            )
        connectors.append(BundledConnector(name=name, definitions=definitions))

    return ConnectorBundle(connectors=tuple(connectors), tools=tuple(tools))


def resolve_runtime_bundle(agent_config: RuntimeAgentConfig) -> ConnectorBundle:
    """Resolve the connector bundle for this process.

    A pre-materialized bundle published by the platform build (via
    ``AGENTIC_CONNECTOR_BUNDLE_PATH``) wins; otherwise the ``agent.yaml``
    ``connectors`` entries are materialized in memory from local files. With
    neither configured this is an immediate no-op without file or network
    I/O, matching the no-connector contract.
    """
    configured = os.environ.get(CONNECTOR_BUNDLE_PATH_ENV)
    if configured:
        return load_connector_bundle(configured)

    # Shape check, not truthiness: mocked runtime configs auto-create
    # truthy attributes, and only a real list/tuple of references counts.
    references = getattr(agent_config, "connectors", None)
    if not isinstance(references, (list, tuple)) or not references:
        return ConnectorBundle()

    config_path = getattr(agent_config, "path", None)
    if config_path is None:
        raise MaterializeError(
            "connector configuration requires a located agent.yaml to resolve against"
        )
    root = Path(config_path).parent
    try:
        for entry in references:
            resolved = (root / entry.source).resolve(strict=True)
            resolved.relative_to(root.resolve())
    except (OSError, RuntimeError, ValueError):
        raise MaterializeError(
            "connectors source must be a path inside the agent.yaml directory"
        ) from None
    return materialize_connectors([entry.source for entry in references], root=root)
