"""Google ADK adapter runtime for Atlas Agent Engine."""

from __future__ import annotations

import functools
import logging
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from agent_engine_sdk import BaseApp, ToolDefinition
from agent_engine_sdk_memory import Memory

from agent_engine_runner_shared import (
    RuntimeAgentConfig,
    RuntimeMode,
    TenantRuntime,
)
from agent_engine_runner_shared.context import (
    customer_origin_scope,
    get_current_wrapper,
)
from agent_engine_runner_shared.context import (
    get_current_user_id as _runner_get_current_user_id,
)
from agent_engine_runner_shared.hooks import register_llm
from agent_engine_runner_shared.memory_appbound import (
    AppBoundCrudClient,
    AppBoundRuntime,
)
from agent_engine_runner_shared.secure_wrapper import (
    create_secure_tool_function,  # pyright: ignore[reportUnknownVariableType]
)
from agent_engine_runner_shared.utils import get_env_bool
from agent_engine_sdk_adk.runner import DurableADKRunner

if TYPE_CHECKING:
    from google.adk.models import BaseLlm

logger = logging.getLogger(__name__)


def _wrap_for_tool_pod(
    fn: Callable[..., Any],
    tool_name: str,
    *,
    allow_direct: bool,
    is_local: bool = True,
    redact_fields: list[str] | None = None,
) -> Callable[..., Any]:
    """Wrap a tool so calls route through the OE.

    Approved local tools execute on this call stack; remote tools execute in
    the Tool Pod.

    Preserves the original signature on the returned wrapper so ADK's
    ``FunctionDeclaration`` introspection picks up the real argument names and
    annotations (``inspect.signature`` honors ``__signature__``). Without this,
    ADK sees a ``(*args, **kwargs)`` wrapper, builds an argument-less schema,
    and the model's tool arguments never reach the function.

    A ``tool_context: ToolContext`` parameter is appended to the exposed
    signature so ADK injects the context at call time and excludes it from the
    declaration sent to the model. The wrapper reads ``function_call_id`` from it
    and forwards it as the stable ``tool_call_id`` so OE's execution-log records
    join to the session message.

    The returned wrapper is a coroutine function: ``inner`` is a
    synchronous call chain that blocks on an HTTP round trip to the OE for
    the full duration of the tool call. Google ADK's ``FunctionTool``
    dispatches on ``inspect.iscoroutinefunction`` and, for a plain sync
    callable, invokes it inline on the AER's event loop — parking it (and
    the health endpoint served on the same loop) for as long as the tool
    runs. ``asyncio.to_thread`` moves that blocking call to a worker thread;
    it captures the current ``contextvars.Context`` before dispatching, so
    ``get_current_wrapper()`` (contextvar-based) still resolves correctly
    inside the thread.
    """
    import asyncio
    import functools
    import inspect

    from google.adk.tools.tool_context import ToolContext

    sig = inspect.signature(fn)
    if "tool_context" in sig.parameters:
        # The user's tool already declares tool_context; ADK injects the context
        # there. Don't append a second one (sig.replace would raise on the
        # duplicate name). wrapped() reads function_call_id from it either way.
        sig_with_ctx = sig
    else:
        ctx_param = inspect.Parameter(
            "tool_context",
            inspect.Parameter.KEYWORD_ONLY,
            annotation=ToolContext,
            default=None,
        )
        # KEYWORD_ONLY params must precede any VAR_KEYWORD (**kwargs) param.
        params = list(sig.parameters.values())
        insert_at = len(params)
        for i, param in enumerate(params):
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                insert_at = i
                break
        params.insert(insert_at, ctx_param)
        sig_with_ctx = sig.replace(parameters=params)

    inner: Callable[..., Any] = create_secure_tool_function(  # type: ignore[no-untyped-call]
        original_tool=fn,
        tool_name=tool_name,
        allow_direct=allow_direct,
        is_local=is_local,
        redact_fields=redact_fields,
    )

    @functools.wraps(fn)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        tool_context = kwargs.pop("tool_context", None)
        if args:
            bound = sig.bind_partial(*args, **kwargs)
            kwargs = dict(bound.arguments)
        tool_call_id = getattr(tool_context, "function_call_id", None)
        return await asyncio.to_thread(inner, tool_call_id=tool_call_id, **kwargs)

    wrapped.__signature__ = sig_with_ctx  # type: ignore[attr-defined]
    return wrapped


def _reject_in_pod_confirmation(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Block native ADK confirmation inside a Tool Pod.

    HITL confirmation must be an AER wait (ADK ``FunctionTool`` with
    ``require_confirmation=True``). A pod-side ``ToolContext.request_confirmation``
    would pause in the worker instead, so this wrapper replaces that method
    when present.
    """

    @functools.wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        tool_context = kwargs.get("tool_context")
        if tool_context is not None:
            requester = getattr(tool_context, "request_confirmation", None)
            if callable(requester):

                def _blocked(*_args: Any, **_kwargs: Any) -> Any:
                    raise RuntimeError(
                        "in-pod ADK request_confirmation is not supported"
                    )

                setattr(tool_context, "request_confirmation", _blocked)
        return fn(*args, **kwargs)

    return wrapped


class App(BaseApp):
    """Google ADK SDK for Atlas Agent Engine.

    Example:

    ```python
    from agent_engine_sdk_adk import App
    from google.adk.agents import LlmAgent
    from google.adk.models.google_llm import Gemini

    app = App(app_name="My Agent")

    @app.tool()
    def my_tool(query: str) -> str:
        \"\"\"Search the database.\"\"\"
        return "result"

    @app.entrypoint
    def build_agent():
        return LlmAgent(
            model=app.llm(Gemini(model="gemini-3-flash-preview")),
            tools=app.tools(),
            name="my_agent",
        )
    ```
    """

    def __init__(
        self,
        app_name: str,
        app_version: str = "1.0.0",
        mongodb_uri: str | None = None,
        database_name: str | None = None,
        enable_tracing: bool | None = None,
        traces_collection_name: str = "traces",
        enable_memory: bool | None = None,
        org_id: str | None = None,
    ) -> None:
        """Initialize the App.

        Args:
            app_name: Application name
            app_version: Application version
            mongodb_uri: MongoDB URI
            database_name: Database name
            enable_tracing: Deprecated. Tracing is always enabled.
            traces_collection_name: Collection name for trace storage
            enable_memory: Deprecated. Use agent.yaml features.memory instead.
            org_id: Deprecated and ignored. The org is taken from the
                ``ORG_ID`` environment variable, which the platform injects.
                Passing this argument raises a DeprecationWarning and will be
                removed in a future release.
        """
        if org_id is not None:
            warnings.warn(
                "App(org_id=...) is deprecated and ignored. "
                "Set the ORG_ID environment variable instead; "
                "this argument will be removed in a future release.",
                DeprecationWarning,
                stacklevel=2,
            )
            org_id = None
        super().__init__(name=app_name)
        self.runner = DurableADKRunner(app_name=app_name)
        self._runtime = TenantRuntime(
            app_name=app_name,
            app_version=app_version,
            mongodb_uri=mongodb_uri,
            database_name=database_name,
            enable_tracing=enable_tracing,
            traces_collection_name=traces_collection_name,
            enable_memory=enable_memory,
            org_id=org_id,
        )
        self._builder_fn: Callable[..., Any] | None = None
        self._tool_defs: list[ToolDefinition] = []
        self._raw_tools: dict[str, Callable[..., Any]] = {}
        self._memory: Memory | None = None

    @property
    def agent_config(self) -> RuntimeAgentConfig:
        return self._runtime.agent_config

    @property
    def memory(self) -> Memory:
        """Return the unified Memory facade over app-bound adapters.

        The return type is ``agent_engine_sdk_memory.Memory``. Methods return
        typed results (``Create*Result``, ``MemoryChunk``, ``ContextResponse``).
        Identity is supplied via per-call args, ``bind``, or ambient
        execution contextvars.
        """
        if self._memory is None:
            runtime = AppBoundRuntime(self._runtime)
            client = AppBoundCrudClient(self._runtime)
            self._memory = Memory(runtime=runtime, client=client)
        return self._memory

    def get_current_user_id(self) -> str | None:
        """Return the user id of the currently-executing request, if any.

        Backed by the request-scoped context agent-engine-runner-shared sets on the AER
        before invoking the agent, so tools can attribute writes to the caller.
        """
        return _runner_get_current_user_id()

    # =========================================================================
    # BaseApp contract
    # =========================================================================

    def get_tool_definitions(self) -> list[ToolDefinition]:
        return list(self._tool_defs)

    def tools(self) -> list[Any]:
        """Return tool callables ready for an ADK agent builder.

        In AER mode, wrap each tool with SecureToolWrapper. Authors
        apply native ADK HITL constructors to these returned callables:
        ``FunctionTool(fn, require_confirmation=True)`` and
        ``LongRunningFunctionTool(fn)``. Do not wrap the raw ``@app.tool``
        function; that has not gone through the secure wrapper yet.
        """
        if self._runtime.mode != RuntimeMode.AER:
            return [_reject_in_pod_confirmation(fn) for fn in self._raw_tools.values()]

        allow_direct = get_env_bool("RUNNER_ALLOW_DIRECT_TOOL_EXECUTION", False)
        wrapped: list[Any] = []
        for name, fn in self._raw_tools.items():
            tool_metadata = self._runtime.get_tool_metadata(name)  # type: ignore[attr-defined]
            is_local = bool(tool_metadata.get("is_local", True))
            raw_redact_fields = tool_metadata.get("redact_fields")
            redact_fields = (
                [
                    field
                    for field in cast(list[object], raw_redact_fields)
                    if isinstance(field, str)
                ]
                if isinstance(raw_redact_fields, list)
                else []
            )
            wrapped.append(
                _wrap_for_tool_pod(
                    fn,
                    name,
                    allow_direct=allow_direct,
                    is_local=is_local,
                    redact_fields=redact_fields,
                )
            )
        return wrapped

    def tool(
        self,
        is_local: bool = True,
        *,
        network: list[str] | None = None,
        timeout: int = 30,
        redact_fields: list[str] | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register a tool function with the platform."""
        app = self

        def wrapper(fn: Callable[..., Any]) -> Callable[..., Any]:
            effective_network = network if network is not None else []
            effective_redact = redact_fields if redact_fields is not None else []

            name = fn.__name__
            if name in app._raw_tools:
                raise ValueError(f"tool {name!r} is already registered")
            metadata: dict[str, Any] = {
                "name": name,
                "description": fn.__doc__ or "",
                "is_local": is_local,
                "network": effective_network,
                "timeout_seconds": timeout,
                "redact_fields": effective_redact,
            }
            app._runtime.register_tool(  # type: ignore[attr-defined]
                name=name, func=fn, metadata=metadata
            )
            app._raw_tools[name] = fn
            app._tool_defs.append(
                ToolDefinition(
                    name=name,
                    description=fn.__doc__ or "",
                    args_schema={},
                    callable=fn,
                    remote=not is_local,
                    network=effective_network,
                    timeout_seconds=timeout,
                    redact_fields=effective_redact,
                )
            )
            return fn

        return wrapper

    def entrypoint(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Mark the ADK agent builder function."""
        self._builder_fn = fn
        return fn

    # =========================================================================
    # LLM API
    # =========================================================================

    def llm(
        self,
        llm: BaseLlm,
        llm_id: str | None = None,
    ) -> BaseLlm:
        """Wrap an ADK LLM for audited I/O through the Orchestration Engine.

        Args:
            llm: ADK BaseLlm instance.
            llm_id: Unique identifier for this LLM.

        Returns:
            SecureLlm in AER mode; the raw *llm* in TOOL mode.
        """
        resolved_id = llm_id if llm_id is not None else "__default__"

        register_llm(resolved_id, llm)

        if self._runtime.mode == RuntimeMode.TOOL:
            return llm

        from agent_engine_sdk_adk.secure_llm import SecureLlm as SecureLlmImpl

        return SecureLlmImpl(
            model=llm.model,
            get_wrapper=get_current_wrapper,
            llm_id=resolved_id,
        )

    # =========================================================================
    # Agent construction
    # =========================================================================

    def get_agent(self, callbacks: list[Any] | None = None) -> Any:
        """Build and return an ADKBaseAgent wrapping the registered entrypoint."""
        if self._builder_fn is None:
            raise RuntimeError(
                "No @app.entrypoint registered. Decorate your ADK agent "
                "builder function with @app.entrypoint before calling run()."
            )

        self._register_hooks()

        from agent_engine_runner_shared.hooks import (
            entrypoint_scope,
            reset_llm_registry,
        )

        reset_llm_registry()
        with entrypoint_scope(), customer_origin_scope():
            adk_agent = self._builder_fn()

        from agent_engine_sdk_adk.agent import ADKBaseAgent

        return ADKBaseAgent(
            adk_agent=adk_agent,
            runner=self.runner,
        )

    def run(self, **kwargs: Any) -> None:
        """Start the agent service."""
        if self._builder_fn is None:
            raise RuntimeError(
                "No @app.entrypoint registered. Decorate your ADK agent "
                "builder function with @app.entrypoint before calling run()."
            )
        self._register_hooks()

        from agent_engine_runner_shared.tracing.setup import (
            _run_instrumentor,  # type: ignore[attr-defined]
        )

        _run_instrumentor()

        self._runtime.register_and_run(self, **kwargs)  # type: ignore[arg-type]

    @staticmethod
    def _register_hooks() -> None:
        """Register ADK-specific hooks with agent-engine-runner-shared."""
        from agent_engine_runner_shared.hooks import (
            register_llm_adapter_factory,
            register_workflow_adapter,
        )

        register_workflow_adapter("google-adk", _adapter_version())

        from agent_engine_sdk_adk.llm_adapter import ADKLLMAdapter

        register_llm_adapter_factory(ADKLLMAdapter)

        # Instrument ADK with OpenInference. This must be active: relying on
        # ADK's native OTEL telemetry instead throws "ContextVar … created in
        # a different Context" errors when run_async is driven across the AER's
        # async boundaries, corrupting the tool-call/response cycle.
        from agent_engine_runner_shared.hooks import register_instrumentor

        try:
            from openinference.instrumentation.google_adk import (  # pyright: ignore[reportMissingImports]
                GoogleADKInstrumentor,
            )
        except ImportError:
            logger.warning(
                "openinference-instrumentation-google-adk not installed; "
                "ADK spans fall back to native telemetry."
            )
            return

        register_instrumentor(lambda: GoogleADKInstrumentor().instrument())


def _adapter_version() -> str:
    """Return the installed adapter package version for OE recovery pinning."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("agent-engine-sdk-adk")
    except PackageNotFoundError:
        return "0.0.0"
