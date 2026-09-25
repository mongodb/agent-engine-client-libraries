"""Build one stable connector catalog from an OpenAPI document.

This module owns the operation-level pipeline: discover HTTP operations, determine
one catalog-wide credential presentation, compile each supported operation, and
record any operation that was skipped. It deliberately ignores path- and
operation-level ``servers`` because the connector has one authoritative base URL.
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal, cast

from pydantic import ValidationError

from agent_engine_runner_shared.connectors.definitions import Operation, ToolDefinitions
from agent_engine_runner_shared.connectors.generate.naming import (
    _assign_tool_names,
    _operation_raw_name,
)
from agent_engine_runner_shared.connectors.generate.parameters import _collect_parameters
from agent_engine_runner_shared.connectors.generate.security import (
    _operation_auth,
    _resolve_schemes,
)
from agent_engine_runner_shared.connectors.generate.spec import (
    GenerateError,
    _generate_error,
    _generator_version,
    _quote,
    _resolve_reference_object,
    _SpecDocument,
)

_METHODS = ("GET", "HEAD", "OPTIONS", "POST", "PUT", "PATCH", "DELETE")
_UNSUPPORTED_METHODS = ("TRACE",)


def _default_base_url(document: _SpecDocument, provenance: str) -> str | None:
    """Choose a usable document server only when the caller supplied no override."""
    for server in document.servers:
        if "{" not in server.url:
            return server.url
    if document.servers:
        raise GenerateError(
            f"spec {_quote(provenance)} declares only templated servers; pass an explicit base URL"
        )
    return None


def _discover_operations(
    document: _SpecDocument,
    root: dict[str, Any],
    provenance: str,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any], str, str]], list[str]]:
    """Return HTTP operations, recording only operation-local structural failures.

    OpenAPI permits nested server declarations, but a connector intentionally has
    one base URL selected by ``_build_catalog``. Nested servers are therefore
    ignored rather than becoming per-operation routing behavior.
    """
    operations = []
    skipped = []
    for path, raw_item in document.paths.items():
        if not isinstance(raw_item, dict):
            raise GenerateError(f"spec {_quote(provenance)}: path {_quote(path)} is not a mapping")
        item = (
            _resolve_reference_object(
                raw_item, root, f"spec {_quote(provenance)}: path {_quote(path)}"
            )
            if "$ref" in raw_item
            else raw_item
        )
        if "$ref" in raw_item and not any(
            item.get(method.lower()) for method in _METHODS + _UNSUPPORTED_METHODS
        ):
            raise GenerateError(
                f"spec {_quote(provenance)}: path {_quote(path)} reference is not a Path Item"
            )

        for method in _METHODS + _UNSUPPORTED_METHODS:
            operation = item.get(method.lower())
            if operation is None:
                continue
            where = f"operation {method} {_quote(path)}"
            if not isinstance(operation, dict):
                skipped.append(f"{where} is not a mapping")
            elif method in _UNSUPPORTED_METHODS:
                skipped.append(f"{where} uses an unsupported HTTP method")
            else:
                operations.append((item, operation, method, path))
    return operations, skipped


def _credential_parameters(auth: dict[str, Any] | None) -> set[tuple[str, str]]:
    """Identify wire fields supplied by auth so they are not exposed to the model."""
    if not auth:
        return set()
    kind = auth.get("type")
    if kind in ("bearer", "basic"):
        credentials = [{"location": "header", "name": "authorization"}]
    elif kind == "api_key":
        credentials = [
            {
                "location": auth.get("location", "header"),
                "name": auth.get("name") or auth.get("header") or "X-API-Key",
            }
        ]
    elif kind == "api_keys":
        credentials = auth.get("credentials", [])
    else:
        credentials = []
    result = {
        (
            str(credential.get("location")),
            str(credential.get("name", "")).lower()
            if credential.get("location") == "header"
            else str(credential.get("name", "")),
        )
        for credential in credentials
        if isinstance(credential, dict)
    }
    result.update(("header", str(name).lower()) for name in auth.get("headers", {}))
    return result


def _operation_credentials_conflict(operation: Operation, auth: dict[str, Any] | None) -> bool:
    """Check whether catalog auth would overwrite an anonymous operation argument."""
    parameters = {("query", name) for name in operation.params.query}
    parameters.update(("header", name.lower()) for name in operation.params.header)
    parameters.update(("cookie", name) for name in operation.params.cookie)
    return bool(parameters & _credential_parameters(auth))


def _build_operation(
    root: dict[str, Any],
    item: dict[str, Any],
    operation: dict[str, Any],
    method: str,
    path: str,
    tool_name: str,
    auth: dict[str, Any] | None,
) -> Operation:
    """Compile one discovered operation into the runtime ``Operation`` contract."""
    raw_name = _operation_raw_name(operation, method, path)
    where = f"operation {_quote(raw_name)} ({method} {_quote(path)})"
    description = operation.get("summary") or operation.get("description") or ""
    if not isinstance(description, str):
        description = ""

    omitted = _credential_parameters(auth)
    if operation.get("requestBody") is not None:
        omitted.add(("header", "content-type"))
    placement, arguments, required, body_encoding, schema_defs, content_type = _collect_parameters(
        root, item, operation, path, where, omitted
    )
    input_schema: dict[str, Any] = {"type": "object", "properties": arguments}
    if required:
        input_schema["required"] = list(dict.fromkeys(required))
    if schema_defs:
        input_schema["$defs"] = schema_defs

    try:
        return Operation(
            name=tool_name,
            description=description,
            method=cast(
                Literal["GET", "HEAD", "OPTIONS", "POST", "PUT", "PATCH", "DELETE"],
                method,
            ),
            path=path,
            params=placement,
            inputSchema=input_schema,
            body_encoding=cast(
                Literal["json", "form", "multipart", "text", "binary"], body_encoding
            ),
            content_type=content_type,
        )
    except ValidationError as exc:
        raise _generate_error(exc, where) from exc
    except RecursionError:
        raise GenerateError(f"{where}: schema is too deeply nested") from None


def _build_catalog(
    document: _SpecDocument,
    root: dict[str, Any],
    raw: bytes,
    provenance: str,
    name: str,
    base_url: str | None,
    auth_override: dict[str, Any] | None = None,
    skip_unsupported: bool = False,
) -> tuple[ToolDefinitions, tuple[str, ...]]:
    """Compile a document using one base URL and one credential presentation.

    Invalid global state rejects the document. Unsupported operation-local state
    is collected so compatibility mode can produce a useful partial catalog with
    explicit provenance instead of silently changing request semantics.
    """
    resolved_base = base_url or _default_base_url(document, provenance)
    if not resolved_base:
        raise GenerateError(
            f"spec {_quote(provenance)} declares no servers; pass an explicit base URL"
        )
    operations, skipped = _discover_operations(document, root, provenance)

    if auth_override is not None:
        authenticated = [(entry, auth_override) for entry in operations]
    else:
        schemes = _resolve_schemes(document, root)
        authenticated = []
        for item, operation, method, path in operations:
            where = f"operation {method} {_quote(path)}"
            try:
                auth = _operation_auth(schemes, document, operation, where)
            except GenerateError as exc:
                skipped.append(str(exc))
                continue
            authenticated.append(((item, operation, method, path), auth))

    raw_names = [
        _operation_raw_name(operation, method, path)
        for (_, operation, method, path), _ in authenticated
    ]
    identities = [
        f"{name}_{method} {path} {raw_name}"
        for raw_name, ((_, _, method, path), _) in zip(raw_names, authenticated)
    ]
    names = _assign_tool_names(name, raw_names, identities)
    compiled = []
    for ((item, operation, method, path), operation_auth), tool_name in zip(authenticated, names):
        try:
            tool = _build_operation(root, item, operation, method, path, tool_name, operation_auth)
        except GenerateError as exc:
            skipped.append(str(exc))
            continue
        compiled.append((operation_auth, tool, method, path))

    catalog_auth = auth_override
    if auth_override is None:
        catalog_auth = next((auth for auth, _, _, _ in compiled if auth is not None), None)

    tools = []
    for operation_auth, tool, method, path in compiled:
        if operation_auth is not None and operation_auth != catalog_auth:
            skipped.append(
                f"operation {method} {_quote(path)} requires a different credential than the selected catalog"
            )
        elif operation_auth is None and _operation_credentials_conflict(tool, catalog_auth):
            skipped.append(
                f"operation {method} {_quote(path)} has a parameter reserved by catalog auth"
            )
        else:
            tools.append(tool)

    if skipped and not skip_unsupported:
        raise GenerateError(
            f"the document declares {len(skipped)} unsupported operations: {'; '.join(skipped)}"
        )
    if not tools:
        raise GenerateError("the document declares no supported operations")

    externalize: dict[str, Any] = {}
    if catalog_auth is not None:
        externalize["auth"] = catalog_auth
    if skipped:
        externalize["skipped"] = skipped
    try:
        definitions = ToolDefinitions.model_validate(
            {
                "source": {
                    "name": name,
                    "base_url": resolved_base,
                    "openapi": provenance,
                    "spec_sha256": hashlib.sha256(raw).hexdigest(),
                    "generator": _generator_version(),
                },
                "externalize": externalize or None,
                "tools": [tool.model_dump(exclude_none=True) for tool in tools],
            }
        )
    except ValidationError as exc:
        raise _generate_error(exc, "compiled catalog") from exc
    return definitions, tuple(skipped)
