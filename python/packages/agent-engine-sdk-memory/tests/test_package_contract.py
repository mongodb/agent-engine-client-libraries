"""The package's boundary contract: curated public surface and import hygiene.

Three guards that pin what ``agent_engine_sdk_memory`` promises at its edge:
the frozen root ``__all__`` (models + facade names, with the low-level client
hidden), the frozen ``models.__all__``, and the slimness invariant that
importing the package pulls in none of the denylisted heavy/platform deps.
"""

import importlib
import json
import subprocess
import sys

import agent_engine_sdk_memory
import pytest
from agent_engine_sdk_memory import models

# --- curated root surface ---

_FACADE_NAMES = {
    "Memory",
    "AmbientIdentityRuntime",
    "MemoryRequestContext",
    "MemoryRuntime",
    "MemoryClientError",
    "MemoryIdentityError",
    "MemoryAPIError",
    "MemoryAuthError",
    "MemoryBadRequestError",
    "MemoryConnectionError",
    "MemoryNotProvisionedError",
    "MemoryNotSupportedError",
    "MemoryRouteNotFoundError",
    "MemoryServerError",
}


def test_all_is_frozen_to_models_plus_facade():
    expected = set(models.__all__) | _FACADE_NAMES
    assert set(agent_engine_sdk_memory.__all__) == expected


def test_route_not_found_error_subclasses_bad_request():
    # Callers with `except MemoryBadRequestError` must keep catching the route
    # error; pin the inheritance so a refactor of the parent fails loudly here.
    from agent_engine_sdk_memory import (
        MemoryBadRequestError,
        MemoryRouteNotFoundError,
    )

    assert issubclass(MemoryRouteNotFoundError, MemoryBadRequestError)


def test_memory_client_not_in_all():
    assert "MemoryClient" not in agent_engine_sdk_memory.__all__


def test_memory_client_not_reachable_from_root():
    module = importlib.import_module("agent_engine_sdk_memory")
    assert not hasattr(module, "MemoryClient")
    with pytest.raises(AttributeError):
        getattr(module, "MemoryClient")


# --- models surface ---

EXPECTED_MODELS_ALL = [
    "WriteTurnResult",
    "CreateSemanticResult",
    "BulkCreateSemanticResult",
    "CreateEpisodicResult",
    "CreateTaxonomicResult",
    "CreateProceduralResult",
    "CreateUserContextResult",
    "CreateSnapshotResult",
    "PromoteSnapshotResult",
    "DeleteResult",
    "InternalStateResult",
    "ISGenerationResult",
    "MemorySource",
    "SearchSource",
    "RetrievalMode",
    "FormatStyle",
    "SourceSpec",
    "MemoryChunk",
    "SourceOutcome",
    "ContextMetadata",
    "ContextResponse",
    "TagScalar",
    "CustomMemorySaveResult",
    "RetrievedCustomMemory",
    "CustomMemoryRetrieveResult",
]


def test_models_all_matches_expected_exports():
    assert models.__all__ == EXPECTED_MODELS_ALL


def test_every_model_in_all_is_importable():
    for name in EXPECTED_MODELS_ALL:
        assert hasattr(models, name), f"missing model: {name}"


# --- import hygiene (slimness) ---


def _top_level_modules_after_fresh_import() -> set[str]:
    """Top-level module names present after importing the package in a *fresh*
    interpreter.

    A subprocess is required: this test module imports agent_engine_sdk_memory
    at the top (for the surface tests above), so an in-process re-import would be
    a cached no-op and hide a heavy transitive import — the exact regression this
    guard exists to catch.
    """
    code = (
        "import sys, json\n"
        "import agent_engine_sdk_memory\n"
        "import agent_engine_sdk_memory._client\n"
        "import agent_engine_sdk_memory.models\n"
        "print(json.dumps(sorted({n.split('.')[0] for n in sys.modules})))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    return set(json.loads(out.stdout))


def test_import_pulls_in_no_denylisted_module():
    from agent_engine_sdk_memory._denylist import DENYLIST_IMPORT_NAMES

    leaked = _top_level_modules_after_fresh_import() & DENYLIST_IMPORT_NAMES
    assert not leaked, f"agent_engine_sdk_memory leaked heavy imports: {sorted(leaked)}"


def test_denylist_contains_mandated_names():
    from agent_engine_sdk_memory._denylist import DENYLIST_IMPORT_NAMES

    mandated = {"langgraph", "litellm", "fastapi", "pymongo", "numpy", "voyageai"}
    assert mandated <= DENYLIST_IMPORT_NAMES


def test_platform_stack_absent_after_import():
    from agent_engine_sdk_memory._denylist import DENYLIST_IMPORT_NAMES

    platform_stack = {
        "agent_engine_runner_shared",
        "agent_engine_sdk",
        "agent_engine_sdk_langgraph",
        "magenta_sdklanggraph",  # open-source-refs:ignore — retired namespace
    }
    assert platform_stack <= DENYLIST_IMPORT_NAMES

    leaked = _top_level_modules_after_fresh_import() & platform_stack
    assert not leaked, (
        f"agent_engine_sdk_memory imported the platform stack: {sorted(leaked)}"
    )
