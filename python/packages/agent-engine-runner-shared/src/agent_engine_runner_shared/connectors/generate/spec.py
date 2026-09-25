"""Load and normalize the small OpenAPI surface needed by the compiler.

This module owns strict local JSON/YAML loading, the top-level OpenAPI 3 envelope,
safe error rendering, and local-reference resolution. It is not a complete OpenAPI
object model: parameter, body, security, naming, and output policy belong to their
focused modules, while compiled schemas remain self-contained for runtime use.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator

from agent_engine_runner_shared.connectors.definitions import loads_yaml_document


class GenerateError(ValueError):
    """Raised when an OpenAPI document or recipe cannot produce a catalog.

    Messages may quote spec content, so they are sanitized and truncated and
    must never be surfaced where control bytes are meaningful.
    """


class _RecursiveSchema(GenerateError):
    pass


@dataclass(frozen=True)
class _Server(BaseModel):
    """One ``servers`` entry."""

    model_config = ConfigDict(extra="allow")

    url: str


class _SecurityScheme(BaseModel):
    """The fields of an OpenAPI security scheme the generator reads."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    type: str
    scheme: str | None = None
    location: str | None = Field(default=None, alias="in")
    name: str | None = None


class _SpecDocument(BaseModel):
    """The parts of an OpenAPI 3.0 document the generator reads."""

    model_config = ConfigDict(extra="allow")

    openapi: str
    paths: dict[str, Any]
    servers: list[_Server] = Field(default_factory=list)
    security: list[Any] | None = None
    components: dict[str, Any] | None = None

    @field_validator("openapi")
    @classmethod
    def _require_v3(cls, value: str) -> str:
        """Reject other OpenAPI generations before projection begins."""
        if not value.startswith("3."):
            raise ValueError("spec is not an OpenAPI 3.x document")
        return value


_SCHEMES = TypeAdapter(dict[str, _SecurityScheme])


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """JSON object hook rejecting duplicate keys: a spec with two ``paths`` or
    ``security`` mappings must fail, not silently resolve to the last value."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GenerateError(f"duplicate key {_quote(key)} in the JSON document")
        result[key] = value
    return result


def _generator_version() -> str:
    """Record the installed producer version as artifact provenance."""
    try:
        return (
            f"agent-engine-runner-shared/{importlib.metadata.version('agent-engine-runner-shared')}"
        )
    except importlib.metadata.PackageNotFoundError:
        return "agent-engine-runner-shared/unknown"


def _quote(value: Any, limit: int = 120) -> str:
    """Render untrusted spec content safe for terminal and log surfaces.

    Every control-category character — ASCII C0, DEL, the C1 range (which
    includes the single-character CSI U+009B), and the bidi/format controls
    that reorder display text — becomes ``?``, matching the CLI's termsafe
    rule. Spaces and printable Unicode survive for readability."""
    text = "".join("?" if unicodedata.category(ch)[0] == "C" else ch for ch in str(value))
    if len(text) > limit:
        text = text[:limit] + "..."
    return text


def _generate_error(exc: ValidationError, context: str) -> GenerateError:
    """Convert Pydantic failures into one sanitized generator error contract."""
    error = exc.errors()[0]
    # Validation locations can contain untrusted spec keys (e.g. a paths
    # mapping key); they must pass the same sanitization as quoted content.
    location = _quote(" ".join(str(part) for part in error.get("loc", ())))
    message = str(error.get("msg", exc))
    detail = f"{location}: {message}" if location else message
    return GenerateError(f"{context}: {detail}")


def _resolve_within(node: Any, root: dict[str, Any], where: str) -> Any:
    """Resolve references within one operation's scope so every failure
    carries the operation context and stays inside the GenerateError contract."""
    try:
        return _inline(node, root)
    except GenerateError as exc:
        raise GenerateError(f"{where}: {exc}") from None
    except RecursionError:
        raise GenerateError(f"{where}: schema is too deeply nested") from None


def _load_spec(spec: str) -> tuple[_SpecDocument, dict[str, Any], bytes]:
    """Read a local OpenAPI 3.0 document; return (envelope, raw document, bytes)."""
    path = Path(spec)
    if re.match(r"^https?://", spec):
        raise GenerateError(
            f"spec {_quote(spec)} is a URL; downloading is the CLI's responsibility — "
            "pass a local file"
        )
    if not path.is_file():
        raise GenerateError(f"spec {_quote(spec)} does not exist")
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeError:
        raise GenerateError(f"spec {_quote(spec)} is not valid UTF-8") from None
    try:
        # Some published JSON specifications contain raw control characters in
        # example strings. JSON's non-strict string mode preserves those values;
        # duplicate keys and all structural errors still fail closed.
        document = json.loads(text, object_pairs_hook=_unique_object, strict=False)
    except GenerateError:
        raise
    except ValueError:
        try:
            document = loads_yaml_document(text)
        except (ValueError, yaml.YAMLError) as exc:
            raise GenerateError(
                f"spec {_quote(spec)} is not valid JSON or YAML: {_quote(exc)}"
            ) from None
    if not isinstance(document, dict):
        raise GenerateError(f"spec {_quote(spec)} must contain a mapping")
    try:
        return _SpecDocument.model_validate(document), document, raw
    except ValidationError as exc:
        raise _generate_error(
            exc, f"spec {_quote(spec)} is not a supported OpenAPI 3.0 document"
        ) from exc


def _translate_nullable(node: dict[str, Any]) -> dict[str, Any]:
    """Translate OpenAPI 3.0 ``nullable`` into JSON Schema."""
    nullable = node["nullable"]
    if nullable is True:
        rest = {k: v for k, v in node.items() if k != "nullable"}
        if isinstance(rest.get("type"), str):
            rest = {**rest, "type": [rest["type"], "null"]}
        if isinstance(rest.get("enum"), list) and None not in rest["enum"]:
            rest = {**rest, "enum": [*rest["enum"], None]}
        return rest
    if nullable is False:
        return {k: v for k, v in node.items() if k != "nullable"}
    raise GenerateError(f"nullable must be a boolean: {_quote(nullable)}")


def _resolve_pointer(ref: Any, root: dict[str, Any]) -> dict[str, Any]:
    """Resolve one local JSON Pointer without fetching another document."""
    if not isinstance(ref, str) or not ref:
        raise GenerateError(f"schema reference must be a non-empty string: {_quote(ref)}")
    if not ref.startswith("#/"):
        raise GenerateError(f"external schema references are not supported: {_quote(ref)}")
    target: Any = root
    # A JSON Pointer in a URI fragment is first URI-decoded as a whole. Only
    # then do decoded slashes delimit tokens and ~1/~0 escape token contents.
    pointer = unquote(ref[1:])
    for raw_token in pointer[1:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(target, list):
            if not token.isdigit() or int(token) >= len(target):
                raise GenerateError(f"unresolvable schema reference {_quote(ref)}")
            target = target[int(token)]
            continue
        if not isinstance(target, dict):
            raise GenerateError(f"unresolvable schema reference {_quote(ref)}")
        if token not in target:
            raise GenerateError(f"unresolvable schema reference {_quote(ref)}")
        target = target[token]
    if not isinstance(target, dict):
        raise GenerateError(f"schema reference {_quote(ref)} resolves to a non-object value")
    return target


def _inline(node: Any, root: dict[str, Any], depth: int = 0) -> Any:
    """Inline local references, translating OpenAPI-only schema keywords."""
    if depth > 32:
        raise _RecursiveSchema("schema references are nested or recursive beyond the limit")
    if isinstance(node, dict):
        if "nullable" in node:
            node = _translate_nullable(node)
        if "$ref" in node:
            ref = node["$ref"]
            target = _resolve_pointer(ref, root)
            siblings = {k: v for k, v in node.items() if k != "$ref"}
            return _inline({**target, **siblings}, root, depth + 1)
        return {k: _inline(v, root, depth) for k, v in node.items()}
    if isinstance(node, list):
        return [_inline(v, root, depth) for v in node]
    return node


class _SchemaResolver:
    """Rewrite recursive OpenAPI schema references into local JSON Schema defs."""

    def __init__(self, root: dict[str, Any]) -> None:
        """Create one operation-local resolver and its self-contained definitions."""
        self.root = root
        self.defs: dict[str, Any] = {}
        self._names: dict[str, str] = {}

    def resolve(self, node: Any) -> Any:
        """Rewrite local references to stable ``$defs`` entries, including cycles."""
        if isinstance(node, dict):
            if "nullable" in node:
                node = _translate_nullable(node)
            if "$ref" in node:
                ref = node["$ref"]
                target = _resolve_pointer(ref, self.root)
                assert isinstance(ref, str)
                name = self._names.setdefault(
                    ref, f"ref_{hashlib.sha256(ref.encode()).hexdigest()[:12]}"
                )
                if name not in self.defs:
                    self.defs[name] = {}
                    self.defs[name] = self.resolve(target)
                siblings = {
                    key: self.resolve(value) for key, value in node.items() if key != "$ref"
                }
                return {"$ref": f"#/$defs/{name}", **siblings}
            return {key: self.resolve(value) for key, value in node.items()}
        if isinstance(node, list):
            return [self.resolve(value) for value in node]
        return node


def _resolve_schema(node: Any, root: dict[str, Any], where: str) -> tuple[Any, dict[str, Any]]:
    """Inline ordinary schemas and preserve recursive graphs faithfully."""
    try:
        return _inline(node, root), {}
    except _RecursiveSchema:
        pass
    except GenerateError as exc:
        raise GenerateError(f"{where}: {exc}") from None
    resolver = _SchemaResolver(root)
    try:
        return resolver.resolve(node), resolver.defs
    except GenerateError as exc:
        raise GenerateError(f"{where}: {exc}") from None


def _resolve_reference_object(
    node: dict[str, Any], root: dict[str, Any], where: str
) -> dict[str, Any]:
    """Resolve a Reference Object chain without traversing its contents."""
    resolved = node
    seen: set[str] = set()
    while "$ref" in resolved:
        ref = resolved.get("$ref")
        if not isinstance(ref, str) or ref in seen:
            raise GenerateError(f"{where}: reference chain is malformed or recursive")
        seen.add(ref)
        try:
            target = _resolve_pointer(ref, root)
        except GenerateError as exc:
            raise GenerateError(f"{where}: {exc}") from None
        resolved = {
            **target,
            **{key: value for key, value in resolved.items() if key != "$ref"},
        }
    return resolved
