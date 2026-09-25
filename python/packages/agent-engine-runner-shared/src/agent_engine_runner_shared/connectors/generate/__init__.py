"""Authoring-time generation of connector catalogs from OpenAPI 3.0 documents.

The generator is the producer side of the shared catalog contract: every
artifact is built and validated through the same models the runtime loader
enforces, so producer and consumer cannot drift. It runs only on developer
machines and reads local files only — downloading specs and catalogs is the
CLI's responsibility, along with its URL and size policy.

Three library-level modes (re-exported below):

- :func:`compile_catalog` — OpenAPI 3.0 document into a complete
  ``tool_defs.yaml`` catalog, scaffolding ``tool.yaml`` when none exists.
- :func:`scaffold_tool_yaml` — consumer ``tool.yaml`` for an existing local
  catalog; the catalog is validated but never rewritten or vendored.
- :func:`regenerate_catalog` — recompile a locally owned catalog from a
  ``tool.yaml`` that declares its local OpenAPI source, preserving authored
  exposure, endpoint, and credential settings.
"""

from agent_engine_runner_shared.connectors.generate.modes import (
    CompileResult,
    compile_catalog,
    regenerate_catalog,
    scaffold_tool_yaml,
)
from agent_engine_runner_shared.connectors.generate.spec import GenerateError

__all__ = [
    "CompileResult",
    "GenerateError",
    "compile_catalog",
    "regenerate_catalog",
    "scaffold_tool_yaml",
]
