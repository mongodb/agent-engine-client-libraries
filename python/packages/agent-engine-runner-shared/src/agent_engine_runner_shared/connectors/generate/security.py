"""Reduce OpenAPI security declarations to supported credential records.

Catalogs record where a credential belongs, never its value or tenant
environment name. A record carries the binding key the document implies:
``bearer``, ``basic``, ``api_keys`` with each key's placement, or ``provider``
for OAuth 2 and OpenID Connect. Token acquisition and consent stay outside this
generator.
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class _OperationAuth:
    """One operation's declared credential.

    ``record`` is what every alternative can share, or ``None`` when the
    alternatives present different credentials. ``alternatives`` keeps each
    alternative's record in document order, and ``scopes`` the scopes the
    first OAuth alternative requests, so an explicit choice can still borrow
    the matching alternative.
    """

    record: dict[str, Any] | None
    scopes: tuple[str, ...] = ()
    alternatives: tuple[dict[str, Any], ...] = ()


def _presentation(record: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return what a record puts on the wire.

    ``provider`` and ``bearer`` both send ``Authorization: Bearer``, so an
    operation that accepts OAuth and one that accepts a static token can share
    one catalog.
    """
    if record and "provider" in record:
        return {**{key: value for key, value in record.items() if key != "provider"}, "bearer": {}}
    return record


def _auth_descriptor(scheme: _SecurityScheme, reference: str) -> dict[str, Any]:
    """Map one security scheme to its credential record."""
    if scheme.type == "http" and scheme.scheme == "bearer":
        return {"bearer": {}}
    if scheme.type == "http" and scheme.scheme == "basic":
        return {"basic": {}}
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
        return {"api_keys": [{"location": scheme.location, "name": scheme.name}]}
    if scheme.type in ("oauth2", "openIdConnect"):
        return {"provider": {}}
    raise GenerateError(
        f"security scheme {_quote(reference)} has unsupported type {_quote(scheme.type)}"
    )


def _operation_auth(
    schemes: dict[str, _SecurityScheme] | None,
    document: _SpecDocument,
    operation: dict[str, Any],
    where: str,
) -> _OperationAuth | None:
    """Resolve an operation's security alternatives to supported credential records.

    An operation whose Bearer alternatives are all OAuth records ``provider``:
    only a delegated token satisfies it. One that also accepts HTTP Bearer
    records ``bearer``, because a static token is enough. Alternatives that
    present different credentials leave ``record`` empty for the caller to
    refuse or resolve with an explicit choice.
    """
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
    records: list[dict[str, Any]] = []
    scopes: tuple[str, ...] | None = None
    for requirement in alternatives:
        credentials = []
        for reference, requested in requirement.items():
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
            if len(requirement) > 1 and "api_keys" not in item:
                raise GenerateError(
                    f"{where}: combined security requirements must contain only API keys"
                )
            if "provider" in item and scopes is None and isinstance(requested, list):
                # The first OAuth alternative states the scopes this operation needs.
                scopes = tuple(scope for scope in requested if isinstance(scope, str))
            credentials.append(item)
        if len(credentials) == 1:
            resolved = credentials[0]
        else:
            # One Security Requirement Object means every listed scheme is
            # required. The api_keys list represents that AND.
            bindings = sorted(
                (key for item in credentials for key in item["api_keys"]),
                key=lambda key: (key["location"], key["name"].lower()),
            )
            identities = {(key["location"], key["name"].lower()) for key in bindings}
            if len(identities) != len(bindings):
                raise GenerateError(
                    f"{where}: combined API keys have duplicate credential destinations"
                )
            resolved = {"api_keys": bindings}
        records.append(resolved)
    record: dict[str, Any] | None = records[0]
    if any(_presentation(other) != _presentation(record) for other in records[1:]):
        record = None
    elif "provider" in records[0] and any("bearer" in other for other in records):
        record = {"bearer": {}}
    return _OperationAuth(record=record, scopes=scopes or (), alternatives=tuple(records))


def _binding_record(auth: Auth | None) -> dict[str, Any]:
    """Convert an authored binding to the secret-free record it presents.

    No binding presents no credential, so every operation compiles anonymously.
    """
    if auth is None:
        return {}
    if auth.provider is not None:
        record: dict[str, Any] = {"provider": {}}
    elif auth.bearer is not None:
        record = {"bearer": {}}
    elif auth.basic is not None:
        record = {"basic": {}}
    else:
        record = {
            "api_keys": [
                {"location": key.location, "name": key.name} for key in auth.api_keys or []
            ]
        }
    if auth.headers:
        # Constant headers are reserved like credential headers, so no
        # operation exposes them as model arguments.
        record["headers"] = dict(auth.headers)
    return record


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
