"""Explicit do-not-add dependency denylist for agent-engine-sdk-memory.

Single source of truth, checked in alongside the package it guards. This is an
EXPLICIT denylist, not an inverted allowlist: legitimate transitives of
pydantic/httpx (anyio, certifi, h11, ...) are intentionally absent so they never
false-positive. Add obvious heavy offenders here as they appear.
"""

# Top-level import-module names as they appear in sys.modules; checked by
# the import-guard test.
DENYLIST_IMPORT_NAMES: frozenset[str] = frozenset(
    {
        "langgraph",
        "langchain",
        "langchain_core",
        "litellm",
        "fastapi",
        "starlette",
        "uvicorn",
        "pymongo",
        "motor",
        "numpy",
        "pandas",
        "voyageai",
        "openai",
        "anthropic",
        "boto3",
        "deepagents",
        # Platform stack — the slim package must not import these (one-way edge).
        "agent_engine_runner_shared",
        "agent_engine_sdk",
        "agent_engine_sdk_langgraph",
    }
)
