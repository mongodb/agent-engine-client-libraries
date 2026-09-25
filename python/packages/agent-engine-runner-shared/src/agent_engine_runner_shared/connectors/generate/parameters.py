"""Project OpenAPI inputs onto the connector argument and wire contract.

Parameters become a model-visible JSON Schema plus their HTTP location and the few
non-default styles the executor implements. Request bodies keep their schema when
flattening would lose presence or validation semantics. Every accepted shape must
have one deterministic wire spelling; everything else raises ``GenerateError`` so
compatibility mode can record the skipped operation.
"""

from __future__ import annotations

import re
from typing import Any

from agent_engine_runner_shared.connectors.definitions import (
    SCALAR_PARAM_TYPES,
    Parameters,
    union_branches,
)
from agent_engine_runner_shared.connectors.generate.spec import (
    GenerateError,
    _quote,
    _resolve_reference_object,
    _resolve_schema,
    _resolve_within,
)

_DEFAULT_PARAM_STYLES = {
    "path": "simple",
    "query": "form",
    "header": "simple",
    "cookie": "form",
}
_SCALAR_CONSTRAINT_KEYS = (
    "enum",
    "const",
    "pattern",
    "format",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minLength",
    "maxLength",
    "multipleOf",
)
_ARRAY_CONSTRAINT_KEYS = ("minItems", "maxItems", "uniqueItems")
_BODY_METADATA_KEYS = {
    "type",
    "properties",
    "required",
    "description",
    "title",
    "deprecated",
    "default",
    "examples",
    "example",
    "externalDocs",
    "readOnly",
    "writeOnly",
    "$schema",
}


def _effective_kind(schema: Any) -> Any:
    """Return the one non-null type from an OpenAPI nullable schema."""
    kind = schema.get("type") if isinstance(schema, dict) else None
    if isinstance(kind, list) and "null" in kind:
        members = [member for member in kind if member != "null"]
        if len(members) == 1:
            return members[0]
    return kind


def _translate_exclusive_bounds(schema: dict[str, Any], where: str) -> dict[str, Any]:
    """Translate OpenAPI 3.0 boolean bounds into the JSON Schema form we validate."""
    for exclusive, inclusive in (
        ("exclusiveMinimum", "minimum"),
        ("exclusiveMaximum", "maximum"),
    ):
        flag = schema.get(exclusive)
        if flag is True:
            if schema.get(inclusive) is None:
                raise GenerateError(f"{where}: {exclusive} requires {inclusive}")
            schema = {**schema, exclusive: schema[inclusive]}
            schema.pop(inclusive)
        elif flag is False:
            schema = {key: value for key, value in schema.items() if key != exclusive}
    return schema


def _scalar_property(schema: dict[str, Any], where: str) -> dict[str, Any]:
    """Keep the constraints the executor's scalar and array encodings preserve."""
    result: dict[str, Any] = {"type": schema.get("type")}
    if _effective_kind(schema) == "array":
        items = _translate_exclusive_bounds(schema.get("items") or {}, where)
        result["items"] = {
            key: value
            for key, value in items.items()
            if key == "type" or key in _SCALAR_CONSTRAINT_KEYS
        }
        result.update((key, schema[key]) for key in _ARRAY_CONSTRAINT_KEYS if key in schema)
    result.update((key, schema[key]) for key in _SCALAR_CONSTRAINT_KEYS if key in schema)
    if isinstance(schema.get("description"), str) and schema["description"]:
        result["description"] = schema["description"]
    return _translate_exclusive_bounds(result, where)


def _check_array_items(items: Any, where: str, parameter: str) -> None:
    """Reject nullable array items because form-style parameters cannot spell null."""
    if (
        isinstance(items, dict)
        and _effective_kind(items) in SCALAR_PARAM_TYPES
        and items.get("type") not in SCALAR_PARAM_TYPES
    ):
        raise GenerateError(f"{where}: {parameter} has nullable array items")


def _union_property(
    schema: dict[str, Any], branches: list[dict[str, Any]], where: str
) -> dict[str, Any]:
    """Preserve a union after reducing each serializable scalar or array branch."""
    keyword = "anyOf" if "anyOf" in schema else "oneOf"
    result: dict[str, Any] = {
        keyword: [
            _scalar_property(branch, where)
            if _effective_kind(branch) in SCALAR_PARAM_TYPES or _effective_kind(branch) == "array"
            else branch
            for branch in branches
        ]
    }
    result.update(
        (key, schema[key]) for key in ("description", "title", "default") if key in schema
    )
    return result


def _declared_parameters(
    root: dict[str, Any],
    item: dict[str, Any],
    operation: dict[str, Any],
    path: str,
    where: str,
    omitted_parameters: set[tuple[str, str]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Resolve path-level parameters and operation-level overrides."""
    declared: dict[tuple[str, str], dict[str, Any]] = {}
    for source in (item.get("parameters"), operation.get("parameters")):
        for parameter in source or []:
            if not isinstance(parameter, dict):
                raise GenerateError(f"{where}: parameters must be mappings")
            if "$ref" in parameter:
                parameter = _resolve_reference_object(parameter, root, where)
            placement = parameter.get("in")
            name = parameter.get("name")
            if placement not in _DEFAULT_PARAM_STYLES or not isinstance(name, str):
                raise GenerateError(
                    f"{where}: parameter {_quote(name)} has an unsupported placement"
                )
            identity = (placement, name.lower() if placement == "header" else name)
            if identity in omitted_parameters:
                continue
            schema = parameter.get("schema")
            if not isinstance(schema, dict):
                content = parameter.get("content")
                media = content.get("application/json") if isinstance(content, dict) else None
                schema = media.get("schema") if isinstance(media, dict) else None
                if placement != "query" or not isinstance(schema, dict):
                    raise GenerateError(f"{where}: parameter {_quote(name)} has no schema")
                parameter = {**parameter, "x-query-json": True}
            declared[(placement, name)] = {**parameter, "schema": schema}

    placeholders = set(re.findall(r"\{([^{}]+)\}", path))
    declared_paths = {name for placement, name in declared if placement == "path"}
    missing = placeholders - declared_paths
    extra = declared_paths - placeholders
    if missing:
        raise GenerateError(
            f"{where}: path placeholder {_quote(sorted(missing)[0])} has no parameter declaration"
        )
    if extra:
        raise GenerateError(
            f"{where}: path parameter {_quote(sorted(extra)[0])} has no matching placeholder"
        )
    return declared


def _required_names(schema: dict[str, Any], where: str) -> list[str]:
    """Read required property names before they are promoted to tool arguments."""
    names = schema.get("required", [])
    if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
        raise GenerateError(f"{where}: required must be a list of property names")
    return names


def _allows_null(schema: Any) -> bool:
    """Return whether a compiled schema explicitly accepts JSON null."""
    if not isinstance(schema, dict):
        return False
    kind = schema.get("type")
    if kind == "null" or isinstance(kind, list) and "null" in kind:
        return True
    if isinstance(schema.get("enum"), list) and None in schema["enum"]:
        return True
    branches = union_branches(schema)
    return bool(branches) and any(_allows_null(branch) for branch in branches)


def _required_nullable_property(
    schema: Any, where: str, *, recursive: bool, prefix: str = ""
) -> str | None:
    """Find a required property whose null value would disappear on the wire."""
    if not isinstance(schema, dict):
        return None
    properties = schema.get("properties")
    if isinstance(properties, dict):
        for name in _required_names(schema, where):
            property_schema = properties.get(name)
            if _allows_null(property_schema):
                return f"{prefix}.{name}" if prefix else name
        if recursive:
            for name, property_schema in properties.items():
                path = f"{prefix}.{name}" if prefix else name
                if found := _required_nullable_property(
                    property_schema, where, recursive=True, prefix=path
                ):
                    return found
    for branch in union_branches(schema) or []:
        if found := _required_nullable_property(branch, where, recursive=recursive, prefix=prefix):
            return found
    return None


def _reject_nullable_body(schema: dict[str, Any], where: str, encoding: str) -> None:
    """Reject null when the selected body encoding has no null spelling."""
    if _allows_null(schema):
        raise GenerateError(f"{where}: nullable {encoding} body is not supported")


def _json_body_arguments(
    media: Any,
    root: dict[str, Any],
    where: str,
    claim: Any,
    placement: Parameters,
    required: list[str],
    arguments: dict[str, dict[str, Any]],
    body_required: bool,
    schema_defs: dict[str, Any],
) -> str:
    """Flatten simple JSON objects and wrap bodies whose semantics need the envelope."""
    if not isinstance(media, dict):
        raise GenerateError(f"{where}: application/json request body entry is malformed")
    schema, resolved_defs = _resolve_schema(media.get("schema") or {}, root, where)
    schema_defs.update(resolved_defs)
    flattens = (
        _effective_kind(schema) == "object"
        and isinstance(schema.get("properties"), dict)
        and schema["properties"]
        and body_required == bool(schema.get("required"))
        and not _allows_null(schema)
        and not (set(schema) - _BODY_METADATA_KEYS)
        and not (set(schema["properties"]) & set(arguments))
    )
    if flattens:
        for name, property_schema in schema["properties"].items():
            claim(name, property_schema)
            placement.body.append(name)
        required.extend(_required_names(schema, where))
    else:
        claim("body", schema)
        placement.wrapped = True
        if body_required:
            required.append("body")
    return "json"


def _form_branch_kind(branch: Any, root: dict[str, Any], where: str) -> str:
    """Classify a form value so its declared style maps to one wire spelling.

    The schema itself stays intact in the generated ``body`` argument. We inspect
    only the value shape because arrays, objects, and scalars use different
    URL-encoded field names and separators.
    """
    resolved = _resolve_within(branch, root, where)
    if not isinstance(resolved, dict):
        raise GenerateError(f"{where}: form property schema is malformed")
    nested = union_branches(resolved)
    if nested is not None:
        kinds = {_form_branch_kind(item, root, where) for item in nested}
        return "object" if "object" in kinds else "array" if "array" in kinds else "scalar"
    kind = _effective_kind(resolved)
    if kind in SCALAR_PARAM_TYPES:
        return "scalar"
    if kind == "array":
        items = _resolve_within(resolved.get("items") or {}, root, where)
        _check_array_items(items, where, "form property")
        if _effective_kind(items) not in SCALAR_PARAM_TYPES:
            raise GenerateError(f"{where}: form arrays must contain scalar values")
        return "array"
    if kind == "object":
        return "object"
    raise GenerateError(f"{where}: form property type {_quote(kind)} is not supported")


def _form_body_arguments(
    media: Any,
    root: dict[str, Any],
    where: str,
    claim: Any,
    placement: Parameters,
    required: list[str],
    body_required: bool,
) -> str:
    """Keep the form schema intact and record only its wire-style differences."""
    if not isinstance(media, dict):
        raise GenerateError(
            f"{where}: application/x-www-form-urlencoded request body entry is malformed"
        )
    schema = _resolve_within(media.get("schema") or {}, root, where)
    _reject_nullable_body(schema, where, "form")
    properties = schema.get("properties") if isinstance(schema, dict) else None
    if _effective_kind(schema) != "object" or not isinstance(properties, dict):
        raise GenerateError(f"{where}: form request body must be an object")
    if name := _required_nullable_property(schema, where, recursive=True):
        raise GenerateError(f"{where}: form body has required nullable property {_quote(name)}")
    encoding = media.get("encoding", {})
    if not isinstance(encoding, dict):
        raise GenerateError(f"{where}: form body encoding map is malformed")

    styles: dict[str, str] = {}
    for name, property_schema in properties.items():
        entry = encoding.get(name, {})
        if not isinstance(entry, dict):
            raise GenerateError(f"{where}: form encoding for {_quote(name)} is malformed")
        if "allowReserved" in entry and entry["allowReserved"] is not False:
            raise GenerateError(
                f"{where}: form property {_quote(name)} uses unsupported allowReserved"
            )
        style = entry.get("style", "form")
        explode = entry.get("explode")
        kind = _form_branch_kind(property_schema, root, where)
        if style == "deepObject" and explode is not False:
            styles[name] = "brackets"
        elif style == "form" and explode is False and kind == "array":
            styles[name] = "comma"
        elif style != "form" or kind == "object":
            raise GenerateError(
                f"{where}: form property {_quote(name)} requires a supported encoding"
            )

    claim("body", schema)
    placement.wrapped = True
    placement.form_style = styles
    if body_required:
        required.append("body")
    return "form"


def _contains_binary(schema: Any) -> bool:
    """Detect file-like multipart parts that the JSON-only tool input cannot supply."""
    if isinstance(schema, dict):
        return schema.get("format") in ("binary", "base64") or any(
            _contains_binary(value) for value in schema.values()
        )
    if isinstance(schema, list):
        return any(_contains_binary(value) for value in schema)
    return False


def _multipart_body_arguments(
    media: Any,
    root: dict[str, Any],
    where: str,
    claim: Any,
    placement: Parameters,
    required: list[str],
    body_required: bool,
    schema_defs: dict[str, Any],
) -> str:
    """Keep a multipart object intact while rejecting unsupported file parts."""
    if not isinstance(media, dict):
        raise GenerateError(f"{where}: multipart/form-data request body entry is malformed")
    if media.get("encoding"):
        raise GenerateError(f"{where}: custom multipart part encodings are not supported")
    schema, resolved_defs = _resolve_schema(media.get("schema") or {}, root, where)
    schema_defs.update(resolved_defs)
    _reject_nullable_body(schema, where, "multipart")
    if _effective_kind(schema) != "object" or not isinstance(schema.get("properties"), dict):
        raise GenerateError(f"{where}: multipart request body must be an object")
    if name := _required_nullable_property(schema, where, recursive=False):
        raise GenerateError(
            f"{where}: multipart body has required nullable property {_quote(name)}"
        )
    if _contains_binary(schema):
        raise GenerateError(f"{where}: multipart file uploads are not supported")
    claim("body", schema)
    placement.wrapped = True
    if body_required:
        required.append("body")
    return "multipart"


def _request_body_arguments(
    request_body: Any,
    root: dict[str, Any],
    where: str,
    claim: Any,
    placement: Parameters,
    required: list[str],
    arguments: dict[str, dict[str, Any]],
    schema_defs: dict[str, Any],
) -> tuple[str, str | None]:
    """Select one supported request media type and compile its body argument."""
    if not isinstance(request_body, dict):
        raise GenerateError(f"{where}: requestBody must be a mapping")
    if "$ref" in request_body:
        request_body = _resolve_reference_object(request_body, root, where)
    content = request_body.get("content")
    if not isinstance(content, dict):
        raise GenerateError(f"{where}: requestBody must declare content media types")
    body_required = request_body.get("required") is True

    if "application/json" in content:
        return _json_body_arguments(
            content["application/json"],
            root,
            where,
            claim,
            placement,
            required,
            arguments,
            body_required,
            schema_defs,
        ), None
    if "application/x-www-form-urlencoded" in content:
        return _form_body_arguments(
            content["application/x-www-form-urlencoded"],
            root,
            where,
            claim,
            placement,
            required,
            body_required,
        ), None
    if "multipart/form-data" in content:
        return _multipart_body_arguments(
            content["multipart/form-data"],
            root,
            where,
            claim,
            placement,
            required,
            body_required,
            schema_defs,
        ), None
    if "text/plain" in content:
        media = content["text/plain"]
        if not isinstance(media, dict):
            raise GenerateError(f"{where}: text/plain request body entry is malformed")
        schema, resolved_defs = _resolve_schema(
            media.get("schema") or {"type": "string"}, root, where
        )
        schema_defs.update(resolved_defs)
        _reject_nullable_body(schema, where, "text")
        if _effective_kind(schema) != "string":
            raise GenerateError(f"{where}: text/plain request body must be a string")
        claim("body", schema)
        placement.wrapped = True
        if body_required:
            required.append("body")
        return "text", None
    if "application/octet-stream" in content:
        media = content["application/octet-stream"]
        if not isinstance(media, dict):
            raise GenerateError(
                f"{where}: application/octet-stream request body entry is malformed"
            )
        schema, _ = _resolve_schema(media.get("schema") or {}, root, where)
        if schema and (
            not isinstance(schema, dict)
            or schema.get("type") != "string"
            or schema.get("format") not in (None, "binary", "byte")
        ):
            raise GenerateError(f"{where}: binary request body must use a string schema")
        claim(
            "body",
            {
                "type": "string",
                "contentEncoding": "base64",
                "description": "Base64-encoded binary request body",
            },
        )
        placement.wrapped = True
        if body_required:
            required.append("body")
        return "binary", None

    json_media = sorted(kind for kind in content if kind.endswith("+json"))
    if json_media:
        selected = json_media[0]
        return _json_body_arguments(
            content[selected],
            root,
            where,
            claim,
            placement,
            required,
            arguments,
            body_required,
            schema_defs,
        ), selected
    raise GenerateError(f"{where}: unsupported media types {[_quote(kind) for kind in content]}")


def _union_parameter(
    schema: dict[str, Any],
    root: dict[str, Any],
    where: str,
    name: str,
    parameter: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    """Require every union branch to share a supported parameter serialization."""
    location = parameter["in"]
    branches = []
    kinds = set()
    for branch in union_branches(schema) or []:
        resolved = _resolve_within(branch, root, where)
        if not isinstance(resolved, dict):
            raise GenerateError(f"{where}: parameter {_quote(name)} has a malformed union")
        kind = _effective_kind(resolved)
        if kind == "array":
            _check_array_items(resolved.get("items"), where, f"parameter {_quote(name)}")
            if (
                not isinstance(resolved.get("items"), dict)
                or _effective_kind(resolved["items"]) not in SCALAR_PARAM_TYPES
            ):
                raise GenerateError(f"{where}: parameter {_quote(name)} has an unsupported union")
        elif kind == "object" and location != "query":
            raise GenerateError(f"{where}: parameter {_quote(name)} has an unsupported union")
        elif kind not in SCALAR_PARAM_TYPES and kind != "object":
            raise GenerateError(f"{where}: parameter {_quote(name)} has an unsupported union")
        kinds.add(kind)
        branches.append(resolved)

    style = parameter.get("style")
    explode = parameter.get("explode")
    query_style = None
    if location == "query":
        if style == "deepObject" and explode is not False:
            query_style = "brackets" if "array" in kinds else None
        elif "object" in kinds:
            raise GenerateError(f"{where}: object query unions require deepObject encoding")
        elif style in (None, "form") and explode is False and "array" in kinds:
            query_style = "comma"
        elif style not in (None, "form"):
            raise GenerateError(f"{where}: query style {_quote(style)} is not supported")
    elif style not in (None, _DEFAULT_PARAM_STYLES[location]):
        raise GenerateError(f"{where}: {location} style {_quote(style)} is not supported")
    return _union_property(schema, branches, where), query_style


def _collect_parameters(
    root: dict[str, Any],
    item: dict[str, Any],
    operation: dict[str, Any],
    path: str,
    where: str,
    omitted_parameters: set[tuple[str, str]] | None = None,
) -> tuple[Parameters, dict[str, dict[str, Any]], list[str], str, dict[str, Any], str | None]:
    """Compile all operation inputs into one schema and one placement map."""
    declared = _declared_parameters(root, item, operation, path, where, omitted_parameters or set())
    placement = Parameters()
    arguments: dict[str, dict[str, Any]] = {}
    required: list[str] = []
    schema_defs: dict[str, Any] = {}

    def claim(name: str, schema: dict[str, Any]) -> None:
        """Prevent two OpenAPI inputs from becoming the same model argument."""
        if name in arguments:
            raise GenerateError(f"{where}: argument {_quote(name)} has multiple placements")
        arguments[name] = schema

    for (location, name), parameter in declared.items():
        schema, resolved_defs = _resolve_schema(parameter["schema"], root, where)
        if resolved_defs:
            raise GenerateError(
                f"{where}: recursive {location} parameter {_quote(name)} is not supported"
            )
        style: str | None = None
        required_parameter = parameter.get("required") is True or location == "path"
        if parameter.get("x-query-json") is not True:
            if required_parameter and _allows_null(schema):
                raise GenerateError(
                    f"{where}: required nullable {location} parameter {_quote(name)} "
                    "has no supported wire representation"
                )
            if location == "query" and (
                nested := _required_nullable_property(schema, where, recursive=True)
            ):
                raise GenerateError(
                    f"{where}: query parameter {_quote(name)} has required nullable "
                    f"property {_quote(nested)}"
                )
        if "allowReserved" in parameter and parameter["allowReserved"] is not False:
            raise GenerateError(
                f"{where}: {location} parameter {_quote(name)} uses unsupported allowReserved"
            )
        if parameter.get("x-query-json") is True:
            compiled = schema
            style = "json"
        elif isinstance(schema, dict) and union_branches(schema) is not None:
            compiled, style = _union_parameter(schema, root, where, name, parameter)
        else:
            kind = _effective_kind(schema)
            is_array = (
                kind == "array"
                and isinstance(schema.get("items"), dict)
                and _effective_kind(schema["items"]) in SCALAR_PARAM_TYPES
            )
            is_object = location == "query" and kind == "object"
            if kind == "array":
                _check_array_items(schema.get("items"), where, f"parameter {_quote(name)}")
            if kind not in SCALAR_PARAM_TYPES and not is_array and not is_object:
                raise GenerateError(
                    f"{where}: {location} parameter {_quote(name)} is not serializable"
                )

            declared_style = parameter.get("style")
            explode = parameter.get("explode")
            if is_object:
                if declared_style != "deepObject" or explode is False:
                    raise GenerateError(
                        f"{where}: object query parameter {_quote(name)} requires deepObject"
                    )
                compiled = schema
            else:
                if location == "query" and is_array:
                    if declared_style == "deepObject" and explode is not False:
                        style = "brackets"
                    elif declared_style in (None, "form") and explode is False:
                        style = "comma"
                    elif declared_style not in (None, "form"):
                        raise GenerateError(
                            f"{where}: query style {_quote(declared_style)} is not supported"
                        )
                elif declared_style not in (None, _DEFAULT_PARAM_STYLES[location]):
                    raise GenerateError(
                        f"{where}: {location} style {_quote(declared_style)} is not supported"
                    )
                compiled = _scalar_property(schema, where)

        claim(name, compiled)
        getattr(placement, location).append(name)
        if style:
            placement.query_style[name] = style
        if required_parameter:
            required.append(name)

    body_encoding = "json"
    content_type = None
    if operation.get("requestBody") is not None:
        body_encoding, content_type = _request_body_arguments(
            operation["requestBody"],
            root,
            where,
            claim,
            placement,
            required,
            arguments,
            schema_defs,
        )
    return placement, arguments, required, body_encoding, schema_defs, content_type
