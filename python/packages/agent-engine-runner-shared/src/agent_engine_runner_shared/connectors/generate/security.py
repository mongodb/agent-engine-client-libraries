"""Reduce OpenAPI security declarations to supported credential presentations.

Catalogs record where a credential belongs, never its value or tenant environment
name. OAuth 2 and OpenID Connect become Bearer presentation metadata only; token
acquisition and consent are intentionally outside this generator.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from agent_engine_runner_shared.connectors.definitions import Auth
from agent_engine_runner_shared.connectors.generate.spec import (
    _SCHEMES,
    GenerateError,
    _generate_error,
    _quote,
    _resolve_reference_object,
    _SecurityScheme,
    _SpecDocument,
)


def _auth_descriptor(scheme: _SecurityScheme, reference: str) -> dict[str, Any]:
    """Map one security scheme to its credential presentation."""
    if scheme.type == "http" and scheme.scheme == "bearer":
        return {"type": "bearer"}
    if scheme.type == "http" and scheme.scheme == "basic":
        return {"type": "basic"}
    if scheme.type == "http":
        raise GenerateError(
            f"security scheme {_quote(reference)} uses unsupported HTTP scheme "
            f"{_quote(scheme.scheme)}; supported static presentations are API key, "
            "Basic, and Bearer"
        )
    if scheme.type == "apiKey":
        if not scheme.name:
            raise GenerateError(f"security scheme {_quote(reference)} has no wire name")
        if scheme.location not in ("header", "query", "cookie"):
            raise GenerateError(
                f"security scheme {_quote(reference)} has unsupported API key location "
                f"{_quote(scheme.location)}"
            )
        return {"type": "api_key", "location": scheme.location, "name": scheme.name}
    if scheme.type in ("oauth2", "openIdConnect"):
        return {"type": "bearer"}
    raise GenerateError(
        f"security scheme {_quote(reference)} has unsupported type {_quote(scheme.type)}"
    )


def _operation_auth(
    schemes: dict[str, _SecurityScheme] | None,
    document: _SpecDocument,
    operation: dict[str, Any],
    where: str,
) -> dict[str, Any] | None:
    """Resolve an operation to one supported credential presentation."""
    alternatives = operation.get("security") if "security" in operation else document.security
    if not alternatives:
        return None
    if not isinstance(alternatives, list):
        raise GenerateError(f"{where}: security must be a list")
    if any(not isinstance(requirement, dict) for requirement in alternatives):
        raise GenerateError(f"{where}: security requirements must be mappings")
    if any(not requirement for requirement in alternatives):
        # Each entry is an alternative. An empty requirement therefore makes
        # authentication optional, so compile anonymously.
        return None
    descriptor: dict[str, Any] | None = None
    for requirement in alternatives:
        credentials = []
        for reference in requirement:
            scheme = (schemes or {}).get(reference)
            if not isinstance(scheme, _SecurityScheme):
                raise GenerateError(
                    f"{where}: security scheme {_quote(reference)} is not defined in "
                    "components.securitySchemes"
                )
            try:
                item = _auth_descriptor(scheme, reference)
            except GenerateError as exc:
                raise GenerateError(f"{where}: {exc}") from None
            if len(requirement) > 1 and item["type"] != "api_key":
                raise GenerateError(
                    f"{where}: combined security requirements must contain only API keys"
                )
            credentials.append(item)
        if len(credentials) == 1:
            resolved = credentials[0]
        else:
            # One Security Requirement Object means every listed scheme is
            # required. The stable api_keys primitive represents that AND.
            bindings = sorted(
                ({"location": item["location"], "name": item["name"]} for item in credentials),
                key=lambda item: (item["location"], item["name"].lower()),
            )
            identities = {(item["location"], item["name"].lower()) for item in bindings}
            if len(identities) != len(bindings):
                raise GenerateError(
                    f"{where}: combined API keys have duplicate credential destinations"
                )
            resolved = {"type": "api_keys", "credentials": bindings}
        if descriptor is not None and resolved is not None and descriptor != resolved:
            raise GenerateError(
                f"{where}: conflicting authentication alternatives cannot share one credential"
            )
        if resolved is not None:
            descriptor = resolved
    return descriptor


def _catalog_override(auth: Auth | None) -> dict[str, Any] | None:
    """Convert the CLI's auth override to secret-free catalog metadata."""
    if auth is None:
        return None
    if auth.type == "none":
        return {}
    if auth.type == "bearer":
        return {"type": "bearer", **({"headers": auth.headers} if auth.headers else {})}
    if auth.type == "basic":
        return {"type": "basic", **({"headers": auth.headers} if auth.headers else {})}
    if auth.type == "api_key":
        credential = auth.credential_bindings()[0]
        return {
            "type": "api_key",
            "location": credential.location,
            "name": credential.name,
            **({"headers": auth.headers} if auth.headers else {}),
        }
    return {
        "type": "api_keys",
        "credentials": [
            {"location": credential.location, "name": credential.name}
            for credential in auth.credential_bindings()
        ],
        **({"headers": auth.headers} if auth.headers else {}),
    }


def _resolve_schemes(
    document: _SpecDocument, root: dict[str, Any]
) -> dict[str, _SecurityScheme] | None:
    """Parse the document-wide security scheme registry."""
    components = document.components or {}
    raw_schemes = components.get("securitySchemes")
    if raw_schemes is None:
        return None
    if not isinstance(raw_schemes, dict):
        raise GenerateError("components.securitySchemes is malformed")
    resolved_schemes: dict[str, Any] = {}
    for name, scheme in raw_schemes.items():
        if isinstance(scheme, dict) and "$ref" in scheme:
            try:
                scheme = _resolve_reference_object(scheme, root, f"security scheme {_quote(name)}")
            except GenerateError as exc:
                raise GenerateError(f"security scheme {_quote(name)}: {exc}") from None
        resolved_schemes[name] = scheme
    try:
        return _SCHEMES.validate_python(resolved_schemes)
    except ValidationError as exc:
        raise _generate_error(exc, "components.securitySchemes is malformed") from exc
