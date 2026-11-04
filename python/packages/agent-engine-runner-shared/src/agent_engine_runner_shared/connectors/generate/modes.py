"""Expose the three supported authoring workflows.

``compile_catalog`` creates both stable files from OpenAPI, ``scaffold_tool_yaml``
binds an existing catalog, and ``regenerate_catalog`` refreshes a locally owned
catalog without rewriting developer choices in ``tool.yaml``. This module only
coordinates the focused parser, compiler, and writer modules.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml
from pydantic import ValidationError

from agent_engine_runner_shared.connectors.definitions import (
    Auth,
    load_tool_defs,
    loads_yaml_document,
    validate_connector_name,
)
from agent_engine_runner_shared.connectors.generate.authoring import (
    _CATALOG_NAME,
    _TOOL_YAML_NAME,
    _catalog_record,
    _check_auth_choice,
    _declared_spec_ref,
    _ensure_output,
    _override_record,
    _proposed_auth,
    _relative_local,
    _validate_binding,
    _write_tool_yaml,
)
from agent_engine_runner_shared.connectors.generate.catalog import (
    _build_catalog,
    _declared_auth,
    _override_operations,
)
from agent_engine_runner_shared.connectors.generate.security import (
    _binding_record,
    _OperationAuth,
)
from agent_engine_runner_shared.connectors.generate.spec import (
    GenerateError,
    _generate_error,
    _load_spec,
    _quote,
)

# Catalogs up to this size list every operation when an allow rule misses.
_LISTED_OPERATIONS = 20


def _unmatched_allow_error(unmatched: list[str], available: set[str]) -> GenerateError:
    """Name the allow rules that match no operation and point at the fix.

    Small catalogs list every operation. Large ones, which can run to
    thousands of operations, report the count and the closest names, so a
    typo does not flood the terminal.
    """
    if len(available) <= _LISTED_OPERATIONS:
        hint = f"available: {sorted(available)}"
    else:
        closest = sorted(
            {match for name in unmatched for match in difflib.get_close_matches(name, available)}
        )
        hint = f"{len(available)} operations available"
        if closest:
            hint += f"; closest: {closest}"
    return GenerateError(f"allow rules match no catalog operation: {unmatched}; {hint}")


@dataclass(frozen=True)
class CompileResult:
    """Generated files and operation-level compatibility failures."""

    catalog: Path
    tool_yaml: Path | None
    skipped: tuple[str, ...] = ()


def compile_catalog(
    spec: str,
    *,
    name: str,
    base_url: str | None = None,
    auth: str | None = None,
    provider: str | None = None,
    scopes: Sequence[str] | None = None,
    allow: Sequence[str] | None = None,
    skip_unsupported: bool = False,
    out_dir: str | Path,
) -> CompileResult:
    """Compile OpenAPI into ``tool_defs.yaml`` and scaffold ``tool.yaml``.

    The auth block is proposed from the exposed operations' security. ``auth``
    (one of ``AUTH_CHOICES``) or ``provider`` overrides the proposal and
    presents that credential on every operation; ``api_keys`` takes its key
    placement from the operations ``allow`` names, and ``scopes`` replaces the
    scopes the document requests. A proposal that needs ``provider`` fails
    before any file is written.
    """
    try:
        validate_connector_name(name)
    except ValueError as exc:
        raise GenerateError(str(exc)) from None
    _check_auth_choice(auth, provider, scopes)
    out = Path(out_dir)
    existing_catalog = (out / _CATALOG_NAME).exists()
    existing_authoring = (out / _TOOL_YAML_NAME).exists()
    if existing_catalog or existing_authoring:
        raise GenerateError(
            f"refusing to overwrite: {_TOOL_YAML_NAME if existing_authoring else _CATALOG_NAME} "
            f"already exists in {_quote(out_dir)}; regenerate with --config instead"
        )
    document, root, raw = _load_spec(spec)
    declared = _declared_auth(document, root, spec)
    candidates = list(declared.values())
    if auth == "api_keys" and allow is not None:
        # An operation --allow excludes must not decide what the exposed
        # operations send. An allow list matching nothing is reported below,
        # with the available operations.
        operations = _override_operations(document, root, spec, name)
        selected = {operations[tool] for tool in allow if tool in operations}
        if selected:
            candidates = [item for operation, item in declared.items() if operation in selected]
    override = _override_record(
        auth, provider, (record for item in candidates for record in item.alternatives)
    )
    catalog, skipped = _build_catalog(
        document,
        root,
        raw,
        spec,
        name,
        base_url,
        override,
        skip_unsupported=skip_unsupported,
    )

    if allow is not None:
        available = {operation.name for operation in catalog.tools}
        unmatched = sorted(set(allow) - available)
        if unmatched:
            raise _unmatched_allow_error(unmatched, available)
    exposed = [
        declared[(operation.method, operation.path)]
        for operation in catalog.tools
        if (allow is None or operation.name in allow)
        and (operation.method, operation.path) in declared
    ]
    binding = _proposed_auth(
        name, override=override, provider=provider, scopes=scopes, declared=exposed
    )

    out.mkdir(parents=True, exist_ok=True)
    catalog_path = _ensure_output(out, _CATALOG_NAME)
    catalog.write(catalog_path)

    tool_yaml_path = _write_tool_yaml(
        out,
        name=name,
        catalog_ref=_CATALOG_NAME,
        base_url=catalog.source.base_url,
        allow=allow,
        auth=binding,
        spec_ref=_declared_spec_ref(spec, out),
    )
    return CompileResult(catalog=catalog_path, tool_yaml=tool_yaml_path, skipped=skipped)


def scaffold_tool_yaml(
    tool_defs: str | Path,
    *,
    name: str | None = None,
    base_url: str,
    auth: str | None = None,
    provider: str | None = None,
    scopes: Sequence[str] | None = None,
    allow: Sequence[str] | None = None,
    out_dir: str | Path,
) -> Path:
    """Scaffold only the consumer ``tool.yaml`` for an existing local catalog.

    The referenced catalog is validated with the runtime loader but never
    rewritten or vendored; the reference stays exactly as declared. No
    OpenAPI source is declared, so regeneration is not implied. ``name``
    defaults to the catalog's ``source.name``. The auth block is proposed from
    the catalog's credential record, as ``compile_catalog`` describes, except
    that scopes come only from ``scopes``.
    """
    _check_auth_choice(auth, provider, scopes)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / _TOOL_YAML_NAME
    if target.exists():
        raise GenerateError(
            f"refusing to overwrite existing authoring file {target}; delete it first"
        )
    reference = str(tool_defs)
    try:
        catalog = load_tool_defs(_relative_local(out, reference, "tool_defs", must_exist=True))
    except GenerateError:
        raise
    except Exception as exc:
        # Malformed catalogs must surface through the GenerateError contract
        # like every other authoring failure, not as raw loader exceptions.
        raise GenerateError(
            f"catalog {_quote(reference)} could not be validated: {_quote(exc)}"
        ) from None
    connector = catalog.source.name if name is None else name
    _validate_binding(connector, base_url)
    available = {operation.name for operation in catalog.tools}
    if allow is not None:
        unmatched = sorted(set(allow) - available)
        if unmatched:
            raise _unmatched_allow_error(unmatched, available)
    record = _catalog_record(catalog.externalize)
    override = _override_record(auth, provider, [record] if record else [])
    binding = _proposed_auth(
        connector,
        override=override,
        provider=provider,
        scopes=scopes,
        declared=[_OperationAuth(record=record, alternatives=(record,))] if record else [],
    )
    return _write_tool_yaml(
        out,
        name=connector,
        catalog_ref=reference,
        base_url=base_url,
        allow=allow,
        auth=binding,
        spec_ref=None,
    )


def _load_authoring(path: Path) -> dict[str, Any]:
    """Load an existing authoring file through the same strict YAML boundary."""
    try:
        document = loads_yaml_document(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
        # The path and parser message can carry content from a checked-out
        # file; both pass the sanitizer before reaching a terminal.
        raise GenerateError(
            f"invalid connector authoring file {_quote(path)}: {_quote(exc)}"
        ) from exc
    if not isinstance(document, dict):
        raise GenerateError(f"connector authoring file {_quote(path)} must contain a mapping")
    return document


def _require_source_field(document: dict[str, Any], key: str, path: Path) -> str:
    """Read a required source string with a useful authoring-file error."""
    source = document.get("source")
    value = source.get(key) if isinstance(source, dict) else None
    if not isinstance(value, str) or not value:
        raise GenerateError(f"connector authoring file {_quote(path)} requires source.{key}")
    return value


def regenerate_catalog(tool_yaml: str | Path, *, skip_unsupported: bool = False) -> CompileResult:
    """Recompile a locally owned catalog from its authoring file.

    Requires the authoring file to declare its local OpenAPI source under
    ``source.spec``; authored exposure, endpoint, and credential settings are
    preserved untouched. Public catalog references need no regeneration.
    """
    path = Path(tool_yaml)
    document = _load_authoring(path)
    name = document.get("name")
    if not isinstance(name, str):
        raise GenerateError(f"connector authoring file {_quote(path)} requires a connector name")
    try:
        validate_connector_name(name)
    except ValueError as exc:
        raise GenerateError(f"connector authoring file {_quote(path)}: {exc}") from None
    reference = document.get("tool_defs")
    if not isinstance(reference, str) or not reference:
        raise GenerateError(
            f"connector authoring file {_quote(path)} requires a tool_defs reference"
        )
    if "://" in reference:
        raise GenerateError(
            f"catalog reference {_quote(reference)} is public; public catalog "
            "references need no regeneration"
        )
    base_url = _require_source_field(document, "base_url", path)
    source = document.get("source")
    spec_ref = source.get("spec") if isinstance(source, dict) else None
    if not isinstance(spec_ref, str) or not spec_ref:
        raise GenerateError(
            f"connector authoring file {_quote(path)} does not declare a local OpenAPI "
            "source under source.spec; regeneration needs the original document"
        )
    _validate_binding(name, base_url)

    # Regeneration presents the authored binding on every operation, and no
    # binding means no credential.
    raw_auth = document.get("auth")
    try:
        override = _binding_record(Auth.model_validate(raw_auth) if raw_auth is not None else None)
    except ValidationError as exc:
        raise _generate_error(
            exc, f"connector authoring file {_quote(path)}: invalid auth"
        ) from exc

    root = path.parent
    spec_path = _relative_local(root, spec_ref, "source.spec", must_exist=True)
    catalog_path = _ensure_output(root, reference)
    spec_document, spec_root, raw = _load_spec(str(spec_path))
    # Provenance records the declared reference, not the resolved absolute
    # path, so regeneration stays byte-stable for unchanged inputs.
    catalog, skipped = _build_catalog(
        spec_document,
        spec_root,
        raw,
        spec_ref,
        name,
        base_url,
        override,
        skip_unsupported=skip_unsupported,
    )
    catalog.write(catalog_path)
    return CompileResult(catalog=catalog_path, tool_yaml=None, skipped=skipped)
