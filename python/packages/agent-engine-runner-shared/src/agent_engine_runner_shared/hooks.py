"""
Framework hook registry for agent-engine-runner-shared.

Hooks are registered once at process startup by the framework SDK
(e.g. agent_engine_sdk_langgraph) before the server starts. Consumers in
agent-engine-runner-shared call get_*() instead of importing framework-specific
modules directly.

Type aliases use sdk-core protocols where possible. The suspend handler
remains loosely typed because its signature is framework-specific
(e.g. langgraph's ``interrupt`` returns whatever the resume caller provides).
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from typing import Any, Callable, Iterator

from agent_engine_sdk.interfaces import BaseLLM

# Native suspend resumes by returning a human decision to the framework.
SuspendHandler = Callable[[dict[str, Any]], dict[str, Any]]
# A durable activity wait always exits through framework control flow.
DurableActivitySuspendHandler = Callable[[dict[str, Any]], None]

# Accepts (raw_llm, tools=...), returns a BaseLLM-compatible adapter.
LLMAdapterFactory = Callable[..., BaseLLM]

# Runs framework-specific OTel instrumentation (e.g. LangChainInstrumentor).
Instrumentor = Callable[[], None]

_suspend_handler: SuspendHandler | None = None
_durable_activity_suspend_handler: DurableActivitySuspendHandler | None = None
_llm_adapter_factory: LLMAdapterFactory | None = None
_instrumentor: Instrumentor | None = None
_llm_registry: dict[str, Any] = {}

# Set while the framework SDK is executing the user's @app.entrypoint
# builder function (see entrypoint_scope()). app.llm() must only be
# called while this is True -- it must be reproducibly re-derivable by
# the Tool Pod's startup registry construction (see docs/runner/README.md
# "Execution lifecycle & secret availability"), so calls from module top
# level, tool bodies, or any other function are rejected at the call site.
#
# A ContextVar (not a plain global) because both the AER and the Tool Pod
# handle requests as concurrent asyncio tasks with no lock serializing
# invocations against each other (unlike the Tool Pod's credential-bearing
# endpoints, which share `_execute_gate`); a plain global would let one
# task's entrypoint_scope() incorrectly gate another concurrently-running
# task's register_llm() call. Mirrors the ContextVar pattern already used
# for per-execution state in context.py.
_entrypoint_active: ContextVar[bool] = ContextVar("entrypoint_active", default=False)


def register_suspend_handler(handler: SuspendHandler) -> None:
    global _suspend_handler
    _suspend_handler = handler


def get_suspend_handler() -> SuspendHandler | None:
    return _suspend_handler


def register_durable_activity_suspend_handler(
    handler: DurableActivitySuspendHandler,
) -> None:
    global _durable_activity_suspend_handler
    _durable_activity_suspend_handler = handler


def get_durable_activity_suspend_handler() -> DurableActivitySuspendHandler | None:
    return _durable_activity_suspend_handler


# Workflow adapter identity registered by the framework SDK (e.g. the
# LangGraph adapter). Absent registration means the framework has no durable
# workflow runtime, so the AER never starts an OE attempt for it.
_workflow_adapter: "tuple[str, str] | None" = None


def register_workflow_adapter(name: str, version: str) -> None:
    global _workflow_adapter
    _workflow_adapter = (name, version)


def clear_workflow_adapter() -> None:
    global _workflow_adapter
    _workflow_adapter = None


def get_workflow_adapter() -> "tuple[str, str] | None":
    return _workflow_adapter


def register_llm_adapter_factory(factory: LLMAdapterFactory) -> None:
    global _llm_adapter_factory
    _llm_adapter_factory = factory


def get_llm_adapter_factory() -> LLMAdapterFactory:
    if _llm_adapter_factory is None:
        raise RuntimeError(
            "No LLM adapter factory registered. "
            "Ensure the framework SDK calls register_llm_adapter_factory() before run()."
        )
    return _llm_adapter_factory


def register_instrumentor(instrumentor: Instrumentor) -> None:
    global _instrumentor
    _instrumentor = instrumentor


def get_instrumentor() -> Instrumentor | None:
    return _instrumentor


@contextlib.contextmanager
def entrypoint_scope() -> Iterator[None]:
    """Mark the dynamic extent of the user's @app.entrypoint call.

    **Framework-internal -- user agent code must never call this.** Users
    only declare an entrypoint (``@app.entrypoint``); the framework SDK
    wraps its own evaluation of that function (in both the AER, once per
    invocation, and the Tool Pod, once per pod lifetime on first
    ``invoke_llm``) in this context manager so that ``register_llm()`` can
    reject ``app.llm()`` calls made outside the entrypoint. Tests that call
    ``app.llm()``/``register_llm()`` directly (bypassing ``@app.entrypoint``
    + ``get_agent()``) use it to simulate that framework evaluation.
    """
    token = _entrypoint_active.set(True)
    try:
        yield
    finally:
        _entrypoint_active.reset(token)


def register_llm(llm_id: str, llm: Any) -> None:
    """Register an LLM by id. Raises ValueError on duplicate id.

    Framework SDKs call this from ``app.llm()`` for every LLM the agent
    uses. The unnamed-LLM convenience case is represented by registering
    under the sentinel id ``"__default__"``; a second unnamed call will
    therefore raise the same duplicate-id error as a second named call
    with the same id.

    Raises:
        RuntimeError: If called outside the dynamic extent of
            ``entrypoint_scope()`` -- i.e. not from inside the function
            decorated with ``@app.entrypoint``. See docs/runner/README.md's
            execution lifecycle contract: every ``app.llm()`` call must be
            reproducible by the Tool Pod's discovery-only entrypoint load,
            which only executes the entrypoint function itself.
    """
    if not _entrypoint_active.get():
        # No call-site naming here: the immediate caller is the SDK's own
        # App.llm() wrapper, not user code, and the exception traceback
        # already shows the user's frame. Matches the TS message.
        raise RuntimeError(
            "app.llm(...) must be called inside the function decorated "
            "with @app.entrypoint. It was called outside the entrypoint's "
            "dynamic extent. Move this call inside your @app.entrypoint "
            "builder function."
        )
    if llm_id in _llm_registry:
        raise ValueError(
            f"llm_id {llm_id!r} is already registered. "
            "Each LLM must have a unique llm_id; pass app.llm(llm, llm_id='...') for each."
        )
    _llm_registry[llm_id] = llm


def get_named_llm(llm_id: str) -> Any:
    if llm_id not in _llm_registry:
        raise KeyError(
            f"llm_id {llm_id!r} not registered. "
            "Call app.llm(llm, llm_id=...) for each LLM before run()."
        )
    return _llm_registry[llm_id]


def has_named_llms() -> bool:
    return bool(_llm_registry)


def reset_llm_registry() -> None:
    global _llm_registry
    _llm_registry = {}


def snapshot_llm_registry() -> dict[str, Any]:
    """Return a shallow copy of the current registry."""
    return dict(_llm_registry)


def reset_hooks() -> None:
    global _suspend_handler, _llm_adapter_factory, _instrumentor, _llm_registry
    global _durable_activity_suspend_handler, _workflow_adapter
    _suspend_handler = None
    _durable_activity_suspend_handler = None
    _llm_adapter_factory = None
    _instrumentor = None
    _llm_registry = {}
    _workflow_adapter = None
    _entrypoint_active.set(False)
