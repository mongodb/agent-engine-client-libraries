"""Google ADK adapter runtime for Atlas Agent Engine."""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, cast, overload

from agent_engine_sdk import AgentInput, BaseApp, RequestContext, ToolDefinition
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
from agent_engine_runner_shared.mcp_tools import (
    MCPConfigError,
    MCPToolBinding,
    discover_mcp_tools,
    make_mcp_tool_pod_callable,
    mcp_server_network_hosts,
)
from agent_engine_runner_shared.memory_appbound import (
    AppBoundCrudClient,
    AppBoundRuntime,
)
from agent_engine_runner_shared.secure_wrapper import (
    create_secure_tool_function,  # pyright: ignore[reportUnknownVariableType]
)
from agent_engine_runner_shared.utils import get_env_bool
from agent_engine_runner_shared.workflow import (
    UnknownOutcomePolicy,
    validate_on_unknown_outcome,
)
from agent_engine_sdk_adk.mcp import RemoteMCPTool, mcp_tool_description
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
    on_unknown_outcome: UnknownOutcomePolicy | None = None,
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
        on_unknown_outcome=on_unknown_outcome,
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


class LLMRegistrationDriftError(RuntimeError):
    """A per-execution build registered a different LLM id set.

    Distinct from an ordinary build failure so the Tool Pod can tell a registry
    that has drifted from one that merely failed to build: a drifted registry is
    not a usable partial discovery. The marker attribute lets runner-shared
    recognise it without importing this SDK, which it is not allowed to do.
    """

    is_llm_registration_drift = True


def _require_builder_arity(builder: Callable[..., Any], load_strategy: str) -> None:
    """Fail fast when the builder cannot accept the strategy's call shape.

    Binds the call the runtime will actually make rather than counting
    parameters, so a builder with optional parameters stays valid and one that
    cannot accept the call is rejected. Runs at decoration time, where a
    mismatch is a coding error the author can see rather than a confusing
    call-site error.
    """
    try:
        signature = inspect.signature(builder)
    except (TypeError, ValueError):
        # An unreadable signature (a builtin, a C callable) is not evidence of
        # a mistake; leave it to the call itself.
        return
    if load_strategy == "per-execution":
        call_args: tuple[Any, ...] = (object(), object())
        requirement = "accept the (input, context) arguments"
    else:
        call_args = ()
        requirement = (
            "take no arguments (the input and context are only passed under "
            "load_strategy='per-execution')"
        )
    try:
        signature.bind(*call_args)
    except TypeError as exc:
        raise TypeError(
            f"@app.entrypoint(load_strategy={load_strategy!r}) builder must "
            f"{requirement}."
        ) from exc


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
        # "global" (default) keeps the existing build behavior; "per-execution"
        # rebuilds the agent on every invocation by running the builder with the
        # invocation's (input, ctx). Read by runner-shared through
        # TenantRuntime.uses_per_invocation_build() to decide whether the Tool
        # Pod keeps partial LLM registrations from a failed discovery.
        self._load_strategy: Literal["global", "per-execution"] = "global"
        # Serializes per-execution builds so their process-global LLM registry
        # reset/register lifetime does not interleave across concurrent
        # invocations. Created lazily on first use, since an asyncio.Lock binds
        # to the running loop.
        self._per_execution_build_lock: asyncio.Lock | None = None
        # Registered LLM ids from the first per-execution build in this process.
        # Every later build must match it, since the Tool Pod resolves models
        # against a registry populated once per pod. See
        # _require_invariant_llm_ids.
        self._llm_id_baseline: frozenset[str] | None = None
        # Whether _llm_id_baseline was recorded by a build that *failed*, and so
        # may be missing ids the builder never reached. A completed build
        # supersedes such a set. See _require_invariant_llm_ids.
        self._llm_id_baseline_provisional = False
        self._tool_defs: list[ToolDefinition] = []
        self._raw_tools: dict[str, Callable[..., Any]] = {}
        self._mcp_tools: dict[str, MCPToolBinding] = {}
        self._memory: Memory | None = None
        if self._runtime.mode == RuntimeMode.AER:
            self._register_mcp_tools_from_config()

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

        Tools discovered from ``agent.yaml`` ``mcp.servers`` entries are
        returned as ready-made ADK tools carrying the server's JSON schema;
        their calls also route through the secure wrapper, with the MCP
        identity OE forwards to the Tool Pod.
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
            binding = self._mcp_tools.get(name)
            if binding is not None:
                invoke: Callable[..., Any] = create_secure_tool_function(  # type: ignore[no-untyped-call]
                    original_tool=fn,
                    tool_name=name,
                    allow_direct=allow_direct,
                    metadata={
                        "mcp_server": binding.server_name,
                        "mcp_tool": binding.tool_name,
                    },
                    is_local=is_local,
                    redact_fields=redact_fields,
                )
                wrapped.append(RemoteMCPTool(binding=binding, invoke=invoke))
                continue
            wrapped.append(
                _wrap_for_tool_pod(
                    fn,
                    name,
                    allow_direct=allow_direct,
                    is_local=is_local,
                    redact_fields=redact_fields,
                    on_unknown_outcome=tool_metadata.get("on_unknown_outcome"),
                )
            )
        return wrapped

    def _register_mcp_tools_from_config(self) -> None:
        """Discover and register the tools of configured remote MCP servers.

        AER-only: Tool Pods skip ``tools/list`` and resolve MCP calls from
        ``agent.yaml`` at execution time, so discovery runs where the model
        schema is built.
        """
        mcp_config = self.agent_config.mcp
        if not mcp_config.servers:
            return

        for binding in discover_mcp_tools(mcp_config):
            self._register_mcp_tool(binding)

    def _register_mcp_tool(self, binding: MCPToolBinding) -> None:
        """Register one discovered MCP tool as a remote SDK tool.

        Mirrors the LangGraph adapter's registration so OE and the Tool Pod
        see the same contract: ``is_local`` false plus the ``mcp_server`` /
        ``mcp_tool`` identity the Tool Pod resolves against ``agent.yaml``.
        """
        name = binding.sdk_tool_name
        if name in self._raw_tools:
            raise MCPConfigError(f"mcp tool {name!r} is already registered")

        description = mcp_tool_description(binding)
        tool_pod_callable = make_mcp_tool_pod_callable(binding)
        network = mcp_server_network_hosts(binding.server_config)
        metadata: dict[str, Any] = {
            "name": name,
            "description": description,
            "is_local": False,
            "network": network,
            "timeout_seconds": binding.server_config.timeout_seconds,
            "redact_fields": [],
            "mcp_server": binding.server_name,
            "mcp_tool": binding.tool_name,
        }
        self._runtime.register_tool(  # type: ignore[attr-defined]
            name=name, func=tool_pod_callable, metadata=metadata
        )
        self._raw_tools[name] = tool_pod_callable
        self._mcp_tools[name] = binding
        self._tool_defs.append(
            ToolDefinition(
                name=name,
                description=description,
                args_schema=binding.input_schema,
                callable=tool_pod_callable,
                remote=True,
                network=network,
                timeout_seconds=binding.server_config.timeout_seconds,
                redact_fields=[],
            )
        )

    def tool(
        self,
        is_local: bool = True,
        *,
        network: list[str] | None = None,
        timeout: int = 30,
        redact_fields: list[str] | None = None,
        on_unknown_outcome: UnknownOutcomePolicy | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register a tool function with the platform.

        Args:
            on_unknown_outcome: What a retry does when the platform lost the
                runner while this tool was running in a durable workflow, so
                nobody knows whether it ran. ``"rerun"`` runs the tool again
                (at-least-once execution; declare it only for a tool that is
                safe to run twice). ``"fail"`` refuses the retry, so the tool
                never runs twice. Leave unset for the platform default.
        """
        app = self
        validate_on_unknown_outcome(on_unknown_outcome)

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
            if on_unknown_outcome is not None:
                metadata["on_unknown_outcome"] = on_unknown_outcome
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

    @overload
    def entrypoint(self, fn: Callable[[], Any], /) -> Callable[[], Any]: ...

    @overload
    def entrypoint(
        self,
        fn: None = None,
        /,
        *,
        load_strategy: Literal["global"] = ...,
    ) -> Callable[[Callable[[], Any]], Callable[[], Any]]: ...

    @overload
    def entrypoint(
        self,
        fn: None = None,
        /,
        *,
        load_strategy: Literal["per-execution"],
    ) -> Callable[
        [Callable[[AgentInput, RequestContext], Any]],
        Callable[[AgentInput, RequestContext], Any],
    ]: ...

    def entrypoint(
        self,
        fn: Callable[..., Any] | None = None,
        /,
        *,
        load_strategy: Literal["global", "per-execution"] = "global",
    ) -> Any:
        """Mark the ADK agent builder function.

        Usable bare (``@app.entrypoint``) for the default behavior, or called
        with a strategy:

        ```python
        @app.entrypoint(load_strategy="per-execution")
        async def build(input: AgentInput, ctx: RequestContext) -> BaseAgent:
            model = app.llm(Gemini(model="gemini-2.5-flash"), llm_id="primary")
            flag = await fetch_flag(ctx.user_id)
            return assemble_agent(model=model, rollout_enabled=flag)
        ```

        ``load_strategy="per-execution"`` runs the builder on every invocation
        with the invocation's ``(input, ctx)``, so it can perform its own
        request-scoped I/O before returning the agent. The builder's arity is
        checked here, at decoration time, against the call the runtime will
        actually make.

        **Registering LLMs under ``per-execution``.** Register every ``llm_id``
        the agent could ever use *before* any construction that depends on the
        invocation context. The Tool Pod discovers an agent's registered ids by
        building it with no invocation context, so a builder that needs context
        cannot be built there; it keeps whatever was registered before such a
        build failed. Reading context is fine, and reading it *before* the
        registrations is what breaks discovery.

        The registered set must also be the same on every invocation, because
        the Tool Pod resolves a model by ``llm_id`` against a registry built
        once per pod. Vary which id the agent *binds* per caller; do not vary
        which ids are *registered*. A per-execution build that registers a
        different set fails at the build, rather than mid-execution when the pod
        cannot resolve the model:

        ```python
        @app.entrypoint(load_strategy="per-execution")
        async def build(input: AgentInput, ctx: RequestContext) -> BaseAgent:
            # Register every selectable model first...
            free = app.llm(Gemini(model="gemini-2.5-flash"), llm_id="free")
            premium = app.llm(Gemini(model="gemini-3-flash-preview"), llm_id="premium")
            # ...then read context, and bind one of them.
            tier = await fetch_tier(ctx.user_id)
            return LlmAgent(
                model=premium if tier == "premium" else free,
                tools=PREMIUM if tier == "premium" else FREE,
                name="my_agent",
            )
        ```
        """
        if load_strategy not in ("global", "per-execution"):
            raise ValueError(
                "load_strategy must be 'global' or 'per-execution', got "
                f"{load_strategy!r}"
            )

        def register(builder: Callable[..., Any]) -> Callable[..., Any]:
            _require_builder_arity(builder, load_strategy)
            self._load_strategy = load_strategy
            self._builder_fn = builder
            return builder

        if fn is not None:
            return register(fn)
        return register

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

    async def build_graph(
        self,
        agent_input: AgentInput | None = None,
        ctx: RequestContext | None = None,
    ) -> Any:
        """Build the agent for this invocation before ``get_agent``.

        Returns ``None`` for a global agent, signalling the caller to use the
        existing build path unchanged. For a per-execution agent, runs the
        builder with ``(agent_input, ctx)`` -- awaiting it when the builder is
        async -- and returns the agent.

        The platform also builds the agent without an invocation to discover its
        registered LLM ids. That call passes ``(None, None)``; a builder that
        needs real context fails there, and the failure names the context-free
        discovery rather than looking like a bug in the builder.
        """
        if self._load_strategy != "per-execution":
            return None
        if self._builder_fn is None:
            raise RuntimeError(
                "No @app.entrypoint registered. Decorate your ADK agent "
                "builder function with @app.entrypoint before calling run()."
            )
        builder_fn = self._builder_fn
        # A context-free build is the platform's discovery path: how the Tool Pod
        # populates its registry, and the only path that can complete a set an
        # earlier failed discovery left provisional. The AER always builds with
        # the invocation's context, so the two never share a process.
        discovery = agent_input is None and ctx is None

        from agent_engine_runner_shared.hooks import (
            entrypoint_scope,
            reset_llm_registry,
        )

        # Serialize the reset/build/register lifetime. The LLM registry is
        # process-global, so two overlapping builds could otherwise interleave
        # their resets and one would reject the other's otherwise-valid
        # app.llm() as a duplicate id. Held across the builder's own I/O.
        if self._per_execution_build_lock is None:
            self._per_execution_build_lock = asyncio.Lock()
        async with self._per_execution_build_lock:
            reset_llm_registry()
            with entrypoint_scope(), customer_origin_scope():
                try:
                    agent = builder_fn(agent_input, ctx)
                    if inspect.isawaitable(agent):
                        agent = await agent
                except Exception as exc:
                    if discovery:
                        # The context-free discovery failed. Validate what it
                        # registered against the baseline if one exists, so a
                        # retry that changed its ids is rejected rather than
                        # retained. A failure that registered nothing is
                        # transient, so no baseline is frozen: the next attempt
                        # may still recover.
                        self._validate_or_record_discovery_baseline()
                        raise RuntimeError(
                            "The agent builder for an agent registered with "
                            "load_strategy='per-execution' failed while building "
                            "without invocation context. A builder that needs "
                            "invocation context cannot be built on the "
                            "context-free discovery path."
                        ) from exc
                    raise
            self._require_invariant_llm_ids(discovery=discovery, completed=True)
            return agent

    def _validate_or_record_discovery_baseline(self) -> None:
        """Validate a failed discovery's registrations, then baseline them.

        A retry that registered a different set is rejected against an existing
        baseline. When no baseline exists yet, one is recorded from the
        registrations the failed discovery did produce, but only if it produced
        any: freezing an empty baseline would block a later successful retry.
        That recorded set is provisional, since the build it came from may have
        been cut short before registering everything.

        A failure that registered nothing is transient whether or not a baseline
        exists: the Tool Pod keeps the registry it already held, so there is no
        set to reject, and reporting one as drift would misdiagnose a transient
        failure as a caller-dependent builder.
        """
        from agent_engine_runner_shared.hooks import snapshot_llm_registry

        registered = frozenset(snapshot_llm_registry())
        if not registered:
            return
        if self._llm_id_baseline is not None:
            self._require_invariant_llm_ids(discovery=True)
            return
        self._llm_id_baseline = registered
        self._llm_id_baseline_provisional = True

    def _require_invariant_llm_ids(
        self, *, discovery: bool = False, completed: bool = False
    ) -> None:
        """Reject a per-execution build that changes the registered LLM ids.

        The Tool Pod resolves a model by ``llm_id`` against a registry built once
        per pod from a context-free build, so every id an invocation could use
        has to be registered there. A builder that registers different ids per
        caller would have the platform ask for a model the pod never registered,
        which fails mid-execution; failing here keeps the failure at the build.

        The first build in this process establishes the baseline. In a Tool Pod
        that is its discovery build; in an AER it is the first invocation. A
        discovery build that fails still records the ids it registered before
        raising, since those are what the Tool Pod retains -- but that set is
        provisional, because the failure may have cut the build short before it
        registered the rest.

        ``discovery`` marks a build on the context-free discovery path, and
        ``completed`` one that ran to the end. A provisional set may still grow,
        but only from a discovery build: the motivating builder registers its
        models and then reads invocation context, so it fails on every
        context-free discovery and never runs to the end, and a model whose
        construction failed once has to be able to join the set when it succeeds.
        Registering *more* is a recovery -- every id a build registered is one an
        invocation may ask the pod for, so the pod only gains by keeping it --
        while dropping or replacing an id stays drift, since the pod would lose a
        model it already served. A discovery that runs to the end is
        authoritative evidence of what the builder registers, so it fixes the set
        from there on.
        """
        from agent_engine_runner_shared.hooks import snapshot_llm_registry

        registered = frozenset(snapshot_llm_registry())
        if self._llm_id_baseline is None:
            self._llm_id_baseline = registered
            return
        if registered != self._llm_id_baseline:
            recovered = (
                self._llm_id_baseline < registered
                and discovery
                and self._llm_id_baseline_provisional
            )
            if not recovered:
                raise LLMRegistrationDriftError(
                    "An agent registered with load_strategy='per-execution' must "
                    "register the same LLM ids on every invocation, because the Tool "
                    "Pod resolves a model by llm_id against a registry built once per "
                    f"pod. Got {sorted(registered)}, expected "
                    f"{sorted(self._llm_id_baseline)}."
                )
            self._llm_id_baseline = registered
        if discovery and completed:
            self._llm_id_baseline_provisional = False

    def get_agent(
        self,
        callbacks: list[Any] | None = None,
        *,
        graph: Any = None,
    ) -> Any:
        """Build and return an ADKBaseAgent wrapping the registered entrypoint.

        ``graph`` is an agent ``build_graph`` already built for this invocation.
        It is only forwarded when present, so a caller using the existing path
        keeps working.
        """
        if self._builder_fn is None:
            raise RuntimeError(
                "No @app.entrypoint registered. Decorate your ADK agent "
                "builder function with @app.entrypoint before calling run()."
            )

        self._register_hooks()

        from agent_engine_sdk_adk.agent import ADKBaseAgent

        if graph is not None:
            # Already built for this invocation. build_graph owns the registry
            # reset on the per-execution path, so resetting here would discard
            # the registrations that build just made and leave the Tool Pod
            # unable to resolve them.
            return ADKBaseAgent(adk_agent=graph, runner=self.runner)

        from agent_engine_runner_shared.hooks import (
            entrypoint_scope,
            reset_llm_registry,
        )

        reset_llm_registry()
        with entrypoint_scope(), customer_origin_scope():
            adk_agent = self._builder_fn()

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
