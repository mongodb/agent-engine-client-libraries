"""LangChain SDK Runtime"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
import warnings
from collections.abc import Callable, Sequence
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

if TYPE_CHECKING:
    from agent_engine_sdk_langgraph.hooks import PrepareAgentInput, ResolveThreadId

from agent_engine_sdk_memory import Memory
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langchain_core.tools import tool as lc_tool
from langgraph.errors import GraphBubbleUp
from langgraph.prebuilt.tool_node import msg_content_output
from agent_engine_sdk import BaseApp, OutputParser, ToolDefinition

from agent_engine_runner_shared.memory_appbound import (
    AppBoundCrudClient,
    AppBoundRuntime,
)
from agent_engine_sdk_langgraph.platform_checkpointer import PlatformCheckpointer
from agent_engine_sdk_langgraph.secure_llm import SecureWrappedLLM
from agent_engine_runner_shared import (
    RuntimeAgentConfig,
    RuntimeMode,
    SessionFinishStatus,
    SuspendPayload,
    TenantRuntime,
    ToolExecutionError,
    request_session_finish,
)
from agent_engine_runner_shared.context import get_current_wrapper
from agent_engine_runner_shared.mcp_tools import (
    MCPConfigError,
    MCPToolBinding,
    discover_mcp_tools,
    make_mcp_tool_pod_callable,
    make_sync_mcp_tool_callable,
    mcp_server_network_hosts,
)
from agent_engine_runner_shared.utils import (
    get_env_bool,
    get_env_float,
    normalize_optional_str,
)
from agent_engine_runner_shared.workflow import current_attempt_context
from agent_engine_runner_shared.workflow.context import current_step_ordinal

logger = logging.getLogger(__name__)

_UNSET = object()  # Sentinel for "use default" -- distinct from None


@contextlib.contextmanager
def _traced_graph_build(*, cache_hit: bool):
    """Open a ``graph.build`` child span around graph materialization.

    No-ops if the optional ``tracing`` extra isn't installed, like
    ``AERServer._traced_step``. Covers only graph materialization, not the
    rest of ``get_agent`` (already covered by AER's ``aer.build_agent``
    span). Opens on a cache hit too (near-zero duration) so a trace can show
    the build was skipped rather than omitting the span.
    """
    try:
        from agent_engine_runner_shared.span_kinds import (
            OPENINFERENCE_SPAN_KIND,
            OpenInferenceSpanKind,
        )
        from agent_engine_runner_shared.span_names import ATTR_CACHE_HIT, GRAPH_BUILD
        from agent_engine_runner_shared.tracing import get_tracer
    except Exception:
        yield None
        return

    tracer = get_tracer("agent-engine-runner-shared.agent-engine-sdk-langgraph")
    with tracer.start_as_current_span(
        GRAPH_BUILD,
        attributes={
            OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKind.CHAIN.value,
            ATTR_CACHE_HIT: cache_hit,
        },
    ) as span:
        yield span


def _durable_tool_message_id(tool_call_id: str) -> str | None:
    attempt = current_attempt_context()
    if attempt is None:
        return None
    return (
        "durable-tool-result:"
        f"{attempt.workflow_identity.execution_id}:"
        f"{current_step_ordinal()}:{tool_call_id}"
    )


def _with_durable_tool_result_identity(
    wrapped_func: Callable[..., Any],
    *,
    tool_name: str,
) -> Callable[..., Any]:
    """Stamp platform-generated tool results before LangGraph's reducer runs."""

    @wraps(wrapped_func)
    def invoke(**kwargs: Any) -> Any:
        result = wrapped_func(**kwargs)
        tool_call_id = kwargs.get("tool_call_id")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            return result
        message_id = _durable_tool_message_id(tool_call_id)
        if message_id is None:
            return result

        content, artifact = result
        message = ToolMessage(
            content=cast(Any, msg_content_output(content)),
            artifact=artifact,
            tool_call_id=tool_call_id,
            name=tool_name,
            id=message_id,
        )
        # The StructuredTool uses content_and_artifact to preserve the existing
        # wire contract. Putting the artifact on the ToolMessage makes LangChain
        # return that message unchanged when it unwraps this tuple.
        return message, None

    return invoke


def _args_schema_with_injected_tool_call_id(args_schema: Any) -> Any:
    """Return a copy of ``args_schema`` extended with an injected ``tool_call_id``.

    LangChain fills an ``InjectedToolCallId`` field with the LLM's stable
    tool-call id at execution time and hides it from the model-facing schema, so
    the secure wrapper can forward the id to OE (joining the execution-log start
    and result records to the session message) without exposing it to the model.
    The model is bound to the original, un-extended tools (``get_tool_schemas``),
    so the injected field is only present on the wrapped tools ToolNode executes.

    Two shapes occur in practice:
    - A pydantic ``BaseModel`` (decorated tools): subclass it and add the field.
    - A raw JSON-schema ``dict`` (MCP tools register ``binding.input_schema``):
      LangChain does not inject into dict schemas at all, so wrap it in a
      permissive model. ``extra="allow"`` lets the real tool arguments pass
      through unchanged — matching the dict path's existing no-validation
      behavior — while still injecting the id.

    Falls back to the original schema for any other shape; no id is injected.
    """
    from langchain_core.tools import InjectedToolCallId
    from pydantic import BaseModel, ConfigDict, create_model

    # Optional type matches the None default so a base schema with
    # validate_default=True (or any path that validates before LangChain injects
    # the id) does not reject the field. LangChain still recognizes the
    # InjectedToolCallId metadata and fills it with the real id at call time.
    injected_field = (Annotated[str | None, InjectedToolCallId], None)

    if isinstance(args_schema, type) and issubclass(args_schema, BaseModel):
        return create_model(
            f"{args_schema.__name__}WithToolCallId",
            __base__=args_schema,
            tool_call_id=injected_field,
        )
    if isinstance(args_schema, dict):
        return create_model(
            "ToolArgsWithToolCallId",
            __config__=ConfigDict(extra="allow"),
            tool_call_id=injected_field,
        )
    return args_schema


def _derive_args_schema(langchain_tool: Any) -> dict[str, Any]:
    """Derive a JSON Schema for a decorated tool's parameters.

    ``lc_tool(fn)`` builds a ``StructuredTool`` whose ``args_schema`` is a
    Pydantic model inferred from the function signature. We surface that as JSON
    Schema so downstream consumers such as API documentation can see the tool's
    parameters. Tools with no parameters yield an object schema with
    no properties. Any unexpected shape degrades to an empty schema.

    Runs at ``@app.tool()`` decoration (i.e. module-import) time, so a Pydantic
    failure here (unresolved forward refs, unsupported/recursive types) must not
    abort import — degrade to ``{}`` and warn.
    """
    from pydantic import BaseModel

    try:
        schema = getattr(langchain_tool, "args_schema", None)
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            return schema.model_json_schema()
        if isinstance(schema, dict):
            return schema
    except Exception as e:
        name = getattr(langchain_tool, "name", "<unknown>")
        logger.warning("Failed to derive args_schema for tool %s: %s", name, e)
    return {}


CHECKPOINTER_SERVER_SELECTION_TIMEOUT_ENV = "CHECKPOINTER_SERVER_SELECTION_TIMEOUT"
CHECKPOINTER_CONNECT_TIMEOUT_ENV = "CHECKPOINTER_CONNECT_TIMEOUT"
CHECKPOINTER_SOCKET_TIMEOUT_ENV = "CHECKPOINTER_SOCKET_TIMEOUT"


def _checkpointer_timeout_ms(name: str, default_s: float) -> int:
    return int(get_env_float(name, default_s) * 1000)


def _adapter_version() -> str:
    """This package's installed version, used as the workflow adapter version."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("agent-engine-sdk-langgraph")
    except PackageNotFoundError:
        return "0.0.0"


def _suspend(payload: Any) -> Any:
    """Route native waits through LangGraph in both execution modes."""
    from langgraph.types import interrupt

    return interrupt(payload)


def _suspend_durable_activity(payload: dict[str, Any]) -> None:
    """Reject the legacy app.suspend() wait on a durable activity.

    Durable waits are framework-native interrupts finalized at the step
    boundary, so ``app.suspend()`` is deliberately unsupported here.
    """
    raise ToolExecutionError(
        "app.suspend() is not supported; use langgraph.types.interrupt() instead"
    )


class App(BaseApp):
    """LangChain SDK for MongoDB Atlas Agent Engine.

    Example:
    ```python
        from agent_engine_sdk_langgraph import App
        from langchain_openai import ChatOpenAI

        app = App(app_name="My Agent")

        @app.tool()
        def my_tool(query: str) -> str:
            return "result"

        @app.entrypoint
        def build_agent():
            llm = app.llm(ChatOpenAI(model="gpt-4"))
            checkpointer = app.checkpointer()
            tools = app.get_tools()
            # Build LangGraph...
            return graph
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
    ):
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
        self._memory: Memory | None = None
        self._builder_fn: Callable[..., Any] | None = None
        self._prepare_input_fn: PrepareAgentInput | None = None
        self._resolve_thread_id_fn: ResolveThreadId | None = None
        # Cached graph.build result, keyed by _adapter_version() so a
        # runtime upgrade invalidates it. Lock dedupes concurrent first
        # invocations to a single build rather than racing the entrypoint
        # function (which has user-visible side effects via app.llm()).
        self._graph_cache: tuple[str, Any] | None = None
        self._graph_cache_lock = threading.Lock()
        self._output_parser_cls: type[OutputParser] | None = None
        self._tool_defs: list[ToolDefinition] = []
        self._lc_tools: dict[str, Any] = {}
        self._mongo_client: Any | None = None
        self._checkpointer: Any | None = None
        if self._runtime.mode == RuntimeMode.AER:
            self._register_mcp_tools_from_config()
            self._register_connector_tools_from_bundle()

    @property
    def agent_config(self) -> RuntimeAgentConfig:
        """Return the runtime SDK's parsed view of agent.yaml."""
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

    # =========================================================================
    # BaseApp contract
    # =========================================================================

    def get_tool_definitions(self) -> list[ToolDefinition]:
        """Return sdk-core ToolDefinitions for all registered tools."""
        return list(self._tool_defs)

    def tools(self) -> list[Any]:
        """Return wrapped, LangGraph-specific tools for ToolNode."""
        return self.get_tools()

    # =========================================================================
    # Decorator API
    # =========================================================================

    def _register_mcp_tools_from_config(self) -> None:
        mcp_config = self.agent_config.mcp
        if not mcp_config.servers:
            return

        for binding in discover_mcp_tools(mcp_config):
            self._register_mcp_tool(binding)

    def _register_mcp_tool(self, binding: MCPToolBinding) -> None:
        tool_pod_callable = make_mcp_tool_pod_callable(binding)
        description = (
            binding.description or f"Remote MCP tool {binding.tool_name}"
        ).strip()
        langchain_tool = StructuredTool.from_function(
            func=make_sync_mcp_tool_callable(binding),
            coroutine=tool_pod_callable,
            name=binding.sdk_tool_name,
            description=description,
            args_schema=binding.input_schema,
        )

        self._register_tool_definition(
            name=binding.sdk_tool_name,
            func=tool_pod_callable,
            description=description,
            args_schema=binding.input_schema,
            is_local=False,
            provider_type=None,
            scopes=[],
            network=mcp_server_network_hosts(binding.server_config),
            timeout_seconds=binding.server_config.timeout_seconds,
            redact_fields=[],
            langchain_tool=langchain_tool,
            additional_metadata={
                "mcp_server": binding.server_name,
                "mcp_tool": binding.tool_name,
            },
            source="mcp",
        )

    def _register_connector_tools_from_bundle(self) -> None:
        # A pre-materialized bundle wins; otherwise materialize the
        # agent.yaml `connectors` authoring files in-process (no network).
        from agent_engine_runner_shared.connectors import resolve_runtime_bundle

        bundle = resolve_runtime_bundle(self._runtime.agent_config)
        if not bundle.tools:
            return

        from agent_engine_runner_shared.toolpod_handlers import BUILTIN_TOOL_NAMES

        bundle.require_available_names(
            set(self._runtime._tools) | set(self._lc_tools) | BUILTIN_TOOL_NAMES
        )
        for bundled_tool in bundle.tools:
            operation = bundled_tool.operation
            metadata = bundled_tool.registration_metadata
            description = metadata["description"]

            def remote_only(**_arguments: Any) -> str:
                raise RuntimeError("connector tools execute only through the Tool Pod")

            remote_only.__name__ = operation.name
            langchain_tool = StructuredTool.from_function(
                func=remote_only,
                name=operation.name,
                description=description,
                args_schema=operation.inputSchema,
            )
            self._register_tool_definition(
                name=operation.name,
                func=remote_only,
                description=description,
                args_schema=operation.inputSchema,
                is_local=metadata["is_local"],
                provider_type=metadata["provider_type"],
                scopes=metadata["scopes"],
                network=metadata["network"],
                timeout_seconds=metadata["timeout_seconds"],
                redact_fields=metadata["redact_fields"],
                langchain_tool=langchain_tool,
                additional_metadata={"connector": metadata["connector"]},
                source="connector",
            )

    def _register_tool_definition(
        self,
        *,
        name: str,
        func: Callable[..., Any],
        description: str,
        args_schema: dict[str, Any],
        is_local: bool,
        provider_type: str | None,
        scopes: list[str],
        network: list[str],
        timeout_seconds: int,
        redact_fields: list[str],
        langchain_tool: Any | None = None,
        additional_metadata: dict[str, Any] | None = None,
        source: Literal["decorator", "mcp", "connector"] = "decorator",
    ) -> None:
        if name in self._runtime._tools or name in self._lc_tools:
            message = f"tool {name!r} is already registered"
            if source == "mcp":
                raise MCPConfigError(f"mcp {message}")
            if source == "connector":
                from agent_engine_runner_shared.connectors import ConnectorBundleError

                raise ConnectorBundleError(f"connector {message}")
            raise ValueError(message)

        description = description.strip()
        metadata = {
            "name": name,
            "description": description,
            "is_local": is_local,
            "provider_type": provider_type,
            "scopes": scopes,
            "network": network,
            "timeout_seconds": timeout_seconds,
            "redact_fields": redact_fields,
        }
        if additional_metadata:
            metadata.update(additional_metadata)

        self._runtime.register_tool(
            name=name,
            func=func,
            metadata=metadata,
        )

        self._lc_tools[name] = (
            langchain_tool if langchain_tool is not None else lc_tool(func)
        )
        self._tool_defs.append(
            ToolDefinition(
                name=name,
                description=description,
                args_schema=args_schema,
                callable=func,
                remote=not is_local,
                provider_type=provider_type,
                scopes=scopes,
                network=network,
                timeout_seconds=timeout_seconds,
                redact_fields=redact_fields,
            )
        )

    def tool(
        self,
        is_local: bool = True,
        *,
        provider_type: str | None = None,
        scopes: list[str] | None = None,
        network: list[str] | None = None,
        timeout: int = 30,
        redact_fields: list[str] | None = None,
        response_format: Literal["content", "content_and_artifact"] = "content",
    ):
        """Register a tool function.

        Creates a LangChain tool and registers the raw function on the runtime
        for Tool Pod execution.

        Args:
            is_local: If True, runs in AER; if False, runs in Tool Pod (default: True)
            provider_type: Credential provider type for delegated auth (e.g. "github")
            scopes: OAuth scopes requested for delegated auth
            network: Allowed network hosts (for tool pod)
            timeout: Execution timeout in seconds
            redact_fields: Fields to redact in logs
            response_format: LangChain tool response format. ``"content"`` (default)
                treats the return value as the ToolMessage content;
                ``"content_and_artifact"`` expects a ``(content, artifact)`` two-tuple,
                where ``artifact`` is data for downstream code that is not sent to the
                model. The artifact must be JSON-serializable so it survives the OE
                round trip. Note it is still recorded in the execution log and is not
                covered by ``redact_fields`` (which redacts inputs only), so do not
                place secrets in it.

        Returns:
            Decorator function
        """
        app = self

        def wrapper(fn: Callable[..., Any]) -> Any:
            effective_network = network or []
            effective_scopes = scopes or []
            effective_redact = redact_fields or []

            langchain_tool = lc_tool(fn, response_format=response_format)
            app._register_tool_definition(
                name=fn.__name__,
                func=fn,
                description=fn.__doc__ or "",
                args_schema=_derive_args_schema(langchain_tool),
                is_local=is_local,
                provider_type=provider_type,
                scopes=effective_scopes,
                network=effective_network,
                timeout_seconds=timeout,
                redact_fields=effective_redact,
                langchain_tool=langchain_tool,
            )

            return fn

        return wrapper

    def entrypoint(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Mark the graph builder function.

        Args:
            fn: Function that builds and returns a CompiledStateGraph

        Returns:
            The original function (unchanged)
        """
        self._builder_fn = fn
        return fn

    def prepare_agent_input(self, fn: PrepareAgentInput) -> PrepareAgentInput:
        """Register a hook that builds graph input from the invocation.

        The hook receives the framework-neutral ``AgentInput`` (the opaque
        caller payload) and ``RequestContext``, and returns a value LangGraph
        can ingest directly — a graph state dict or a ``Command``. It lets an
        agent author control how the request becomes the graph's starting
        input (e.g. fold extra payload context into the prompt, seed a system
        message, or populate custom state keys).

        Only fresh invocations flow through the hook; HITL resume stays
        platform-managed. When no hook is registered, the default
        message-wrapping behavior is used, so existing agents are unaffected.

        Example:

        ```python
        @app.prepare_agent_input
        def prepare(input: AgentInput, ctx: RequestContext) -> GraphInput:
            payload = input.payload
            return {
                "messages": [HumanMessage(content=payload["message"])],
                "extra": payload.get("extra"),
            }
        ```

        Args:
            fn: Callable ``(AgentInput, RequestContext) -> GraphInput``.

        Returns:
            The original function (unchanged), so it can be used as a decorator.
        """
        self._prepare_input_fn = fn
        return fn

    def resolve_thread_id(self, fn: ResolveThreadId) -> ResolveThreadId:
        """Register a hook that builds the LangGraph checkpoint ``thread_id``.

        Callers manage Atlas Agent Engine ``session_id`` (and authenticated ``user_id``).
        The agent owns how those map to the LangGraph checkpoint key. When
        registered, the hook's return value is used verbatim on every
        invocation — fresh and resume — with no workspace suffix appended.
        When no hook is registered, the adapter derives
        ``session_id:workspace_id`` as today.

        Custom keys are invisible to Atlas Agent Engine session-history queries
        (``/query/sessions*``), which still look up only the default
        session/workspace-derived keys. Agents that bypass workspace
        scoping also own collision isolation within the checkpoint
        database (for example when sharing a project store with other
        agents).

        Example:

        ```python
        @app.resolve_thread_id
        def resolve_thread_id(ctx: RequestContext) -> str:
            actor = email_to_actor_id(ctx.user_id or "")
            return f"{ctx.session_id}__{actor}"
        ```

        Args:
            fn: Callable ``(RequestContext) -> str``.

        Returns:
            The original function (unchanged), so it can be used as a decorator.
        """
        self._resolve_thread_id_fn = fn
        return fn

    def output_parser(self, cls: type[OutputParser]) -> type[OutputParser]:
        """Register a custom output parser subclass on this app.

        The class is validated with ``issubclass`` here, at decoration time, so
        a non-``OutputParser`` fails on import. The adapter stores the class
        without instantiating it; because ``parse`` and ``on_stream_error`` are
        abstract, a subclass missing either raises ``TypeError`` the moment it
        is instantiated to stream (once the producer wires that up).

        Example:

        ```python
        @app.output_parser
        class BriefParser(LangGraphOutputParser):
            stream_modes = ("messages", "values")

            async def parse(self, item, ctx): ...
            async def on_stream_error(self, ctx, error): ...
        ```

        Args:
            cls: An ``OutputParser`` subclass.

        Returns:
            The original class (unchanged), so it can be used as a decorator.
        """
        if not (isinstance(cls, type) and issubclass(cls, OutputParser)):
            raise TypeError(
                f"@app.output_parser expects an OutputParser subclass, got {cls!r}"
            )
        self._output_parser_cls = cls
        return cls

    def _get_or_build_graph(self) -> Any:
        """Return the materialized graph, building it at most once.

        The graph carries no per-request state -- ``adapted_callbacks`` and
        the ``LangGraphBaseAgent`` wrapper are rebuilt fresh by every
        ``get_agent()`` call regardless of cache state -- so only the
        expensive part, running ``@app.entrypoint`` and compiling the graph,
        is skipped on a hit. ``reset_llm_registry()`` only runs on a miss:
        a hit leaves the registry exactly as the last build left it.

        Lock-guarded (a plain lock, not ``asyncio.Lock``: this is a sync
        call with no ``await`` between the cache check and the build) so
        concurrent first invocations build exactly once rather than racing
        the entrypoint function, which has user-visible side effects
        (``app.llm()`` raises on a duplicate id).
        """
        if self._builder_fn is None:
            raise RuntimeError(
                "No entrypoint registered. Use @app.entrypoint to mark "
                "the graph builder function."
            )
        builder_fn = self._builder_fn

        from agent_engine_runner_shared.hooks import (
            entrypoint_scope,
            reset_llm_registry,
        )

        current_version = _adapter_version()
        cached = self._graph_cache
        if cached is not None and cached[0] == current_version:
            with _traced_graph_build(cache_hit=True):
                pass
            return cached[1]

        with self._graph_cache_lock:
            # Re-check inside the lock: another thread may have built while
            # this one was waiting, or invalidated a stale version.
            cached = self._graph_cache
            if cached is not None and cached[0] == current_version:
                with _traced_graph_build(cache_hit=True):
                    pass
                return cached[1]

            reset_llm_registry()
            from agent_engine_runner_shared.context import customer_origin_scope

            with (
                entrypoint_scope(),
                customer_origin_scope(),
                _traced_graph_build(cache_hit=False),
            ):
                graph = builder_fn()

            self._graph_cache = (current_version, graph)
            return graph

    def warm_up(self) -> bool:
        """Build and cache the graph ahead of the first ``/execute`` call.

        Never raises: a failure returns ``False`` so the AER records the
        one-shot optimization as incomplete, while the existing lazy build
        remains the fallback for the first real ``get_agent()`` call.
        """
        if self._builder_fn is None:
            return True
        try:
            self._get_or_build_graph()
        except Exception:
            logger.warning(
                "Graph warm-up attempt failed; lazy build remains available "
                "for the first /execute call",
                exc_info=True,
            )
            return False
        return True

    def get_agent(self, callbacks: list[Any] | None = None) -> Any:
        """Build and return a BaseAgent instance.

        Calls the registered entrypoint function to build the graph,
        then wraps it in a LangGraphBaseAgent adapter.

        Args:
            callbacks: Optional list of BaseExecutionCallback instances
                for observability (e.g., NodeExecutionLogger from the AER).
                Wrapped in LangGraphCallbackAdapter before passing to the graph.

        Returns:
            LangGraphBaseAgent instance implementing BaseAgent protocol

        Raises:
            RuntimeError: If no entrypoint function has been registered
        """
        if self._builder_fn is None:
            raise RuntimeError(
                "No entrypoint registered. Use @app.entrypoint to mark "
                "the graph builder function."
            )

        from .agent import LangGraphBaseAgent
        from .durable_subgraphs import DurableSubgraphResolver
        from .node_logger_adapter import LangGraphCallbackAdapter

        # Wrap BaseExecutionCallback instances as LangChain callbacks.
        # The only caller is the AER, which passes NodeExecutionLogger.
        adapted_callbacks = None
        if callbacks:
            adapted_callbacks = []
            for cb in callbacks:
                adapted_callbacks.append(LangGraphCallbackAdapter(cb))

        # The parser is inert unless features.use_custom_parser is true; opting
        # in without a registered parser is a fail-fast misconfiguration.
        use_custom_parser = self.agent_config.feature_enabled("use_custom_parser")
        if use_custom_parser and self._output_parser_cls is None:
            raise RuntimeError(
                "agent.yaml features.use_custom_parser is true but no output "
                "parser is registered; decorate one with @app.output_parser."
            )

        graph = self._get_or_build_graph()

        # Durable eligibility depends on the materialized graph: only the
        # PlatformCheckpointer provides fence-keyed scratch and release, so a
        # graph compiled with a custom saver (or none) must stay native. The
        # adapter version is this package's installed version so the
        # declaration OE records tracks releases.
        #
        # Re-runs on every call, including cache hits: registration is
        # global mutable state, not a property of the graph object, so
        # skipping it on a hit could leave eligibility stale. Cheap (a type
        # check and a dict write) unlike the graph build itself.
        from agent_engine_runner_shared.hooks import (
            clear_workflow_adapter,
            register_workflow_adapter,
        )

        durable_subgraphs = None
        if isinstance(getattr(graph, "checkpointer", None), PlatformCheckpointer):
            register_workflow_adapter("langgraph", _adapter_version())
            resolver = DurableSubgraphResolver(graph)
            if resolver.has_compiled_children:
                durable_subgraphs = resolver
        else:
            # Register-or-clear each materialization so eligibility always
            # reflects the latest graph and cannot drift if builder or
            # checkpointer selection ever becomes dynamic.
            clear_workflow_adapter()
        return LangGraphBaseAgent(
            graph,
            callbacks=adapted_callbacks,
            prepare_input=self._prepare_input_fn,
            output_parser=self._output_parser_cls,
            use_custom_parser=use_custom_parser,
            resolve_thread_id=self._resolve_thread_id_fn,
            durable_subgraphs=durable_subgraphs,
        )

    def run(self, **kwargs) -> None:
        """Start the agent service.

        Registers the builder function and starts the runtime server. This is
        called by the platform entrypoint dispatcher.

        Args:
            **kwargs: Runtime startup options passed to the runtime.
                ``grpc_port`` and ``log_level`` are supported.
                Deprecated ``host`` and ``http_port`` values are still accepted
                for backwards compatibility but are ignored by agent-engine-runner-shared.

        Raises:
            RuntimeError: If no @app.entrypoint has been registered
        """
        if self._builder_fn is None:
            raise RuntimeError(
                "No @app.entrypoint registered. Decorate your graph-builder "
                "function with @app.entrypoint before calling run()."
            )
        # Register framework hooks so agent-engine-runner-shared can call framework
        # code without importing it directly.
        self._register_hooks()

        # Re-run the instrumentor now that the hook is registered.
        # TenantRuntime.__init__ already called setup_tracing() (which sets
        # up the TracerProvider), but _run_instrumentor() ran before hooks
        # were registered and was a no-op. This second call enables the
        # LangChain/OpenInference instrumentation.
        from agent_engine_runner_shared.tracing.setup import _run_instrumentor

        _run_instrumentor()

        # Register the LangGraph query plugin so the AER's /query/sessions*
        # routes can surface framework-specific state. Skipped when no
        # checkpointer is available (MONGODB_URI not configured) or when
        # construction fails (bad URI, auth, etc.) — the routes then
        # return 501 rather than taking down the rest of the AER. Note
        # that this only sees sessions written through app.checkpointer();
        # tenants who compile their graph with a custom checkpointer will
        # get empty /query/sessions* responses.
        if self._runtime.mode == RuntimeMode.AER:
            try:
                checkpointer = self.checkpointer()
            except Exception as exc:  # noqa: BLE001 — query is non-essential
                # Log only the exception class — pymongo's URI-parsing
                # errors can echo the full ``MONGODB_URI`` (credentials
                # and all). Operators who need the underlying detail can
                # reproduce locally with DEBUG logging.
                logger.warning(
                    "Failed to construct checkpointer for query plugin (%s); "
                    "/query/sessions* will return 501",
                    type(exc).__name__,
                )
                checkpointer = None
            # Query reads native Mongo collections only; durable scratch is
            # attempt-local and never a session query source. The native saver
            # is always the MongoDBSaver constructed above when present.
            if isinstance(checkpointer, PlatformCheckpointer):
                checkpointer = cast("Any", checkpointer.native)
            if checkpointer is not None:
                from agent_engine_sdk_langgraph.query import LangGraphQueryPlugin

                # Scope reads through the same resolver the write path uses
                # (APP_ID, with wire workspace_id fallback when APP_ID unset).
                self._runtime.register_query_plugin(
                    LangGraphQueryPlugin(
                        checkpointer,
                        workspace_id_resolver=self._runtime.get_checkpoint_workspace_id,
                    )
                )

        # Pass App instance so AER can call app.get_agent(callbacks).
        # Type ignore: TenantRuntime signature will be updated to accept App.
        self._runtime.register_and_run(self, **kwargs)  # type: ignore[arg-type]

    @staticmethod
    def _register_hooks() -> None:
        """Register framework-specific hooks."""
        from agent_engine_sdk_langgraph.llm_adapter import LangChainLLMAdapter
        from agent_engine_runner_shared.hooks import (
            register_durable_activity_suspend_handler,
            register_instrumentor,
            register_llm_adapter_factory,
            register_suspend_handler,
        )

        register_suspend_handler(_suspend)
        register_durable_activity_suspend_handler(_suspend_durable_activity)
        register_llm_adapter_factory(LangChainLLMAdapter)

        from openinference.instrumentation.langchain import LangChainInstrumentor

        register_instrumentor(lambda: LangChainInstrumentor().instrument())

    # =========================================================================
    # LLM API
    # =========================================================================

    def llm(
        self,
        llm: BaseChatModel,
        llm_id: str | None = None,
    ) -> BaseChatModel:
        """Wrap a LangChain LLM for audited I/O through the Orchestration Engine.

        For agents with a single LLM, call without ``llm_id``::

            llm = app.llm(ChatOpenAI(model="gpt-5.4"))

        For agents with multiple LLMs, every call must supply a unique ``llm_id``::

            llm_a = app.llm(ChatOpenAI(model="gpt-5.4"), llm_id="primary")
            llm_b = app.llm(ChatOpenAI(model="gpt-5.4-mini"), llm_id="fast")

        Unnamed calls register under the sentinel id ``"__default__"``; calling
        ``app.llm()`` twice without an id therefore raises the same duplicate-id
        error as registering the same named id twice.

        Args:
            llm: LangChain LLM instance.
            llm_id: Unique identifier for this LLM. Optional when the agent uses
                a single LLM.

        Returns:
            ``SecureWrappedLLM`` in AER mode; the raw *llm* in TOOL mode.

        Raises:
            ValueError: If ``llm_id`` has already been registered (including a
                second unnamed call which collides on ``"__default__"``).
        """
        from agent_engine_runner_shared.hooks import register_llm

        resolved_id = llm_id if llm_id is not None else "__default__"
        register_llm(resolved_id, llm)

        if self._runtime.mode == RuntimeMode.TOOL:
            return llm
        return SecureWrappedLLM(
            llm=llm, get_wrapper=get_current_wrapper, llm_id=resolved_id
        )

    # =========================================================================
    # Deep Agent API
    # =========================================================================

    def deep_agent(
        self,
        llm: BaseChatModel,
        *,
        tools: Sequence[Any] | None = None,
        subagents: Sequence[Any] | None = None,
        system_prompt: str | None = None,
        middleware: Sequence[Any] = (),
        checkpointer: Any = _UNSET,
        store: Any = None,
        skills: list[str] | None = None,
        backend: Any = None,
    ) -> Any:
        """Create a deep agent graph pre-configured with Atlas Agent Engine secure routing.

        Wraps ``create_agent_engine_deep_agent()`` with:
        - ``SecureWrappedLLM`` as the model (all LLM calls route through OE)
        - Configurable backend (defaults to ``AgentEngineToolPodBackend``)
        - Checkpointer sentinel resolution (``_UNSET`` -> ``app.checkpointer()``,
          ``None`` -> disable, instance -> use directly)
        - SubAgent model validation (string model specs rejected to prevent OE bypass)

        The returned ``CompiledStateGraph`` should be used inside an
        ``@app.entrypoint`` function. ``app.get_agent()`` will then wrap it
        in ``LangGraphBaseAgent`` for AER compatibility.

        Args:
            llm: Base LLM instance to wrap with SecureWrappedLLM.
            tools: Additional tools for the deep agent (merged with built-in tools).
            subagents: SubAgent specs. Each spec with a ``model`` field must pass
                a ``BaseChatModel`` instance, not a string.
            system_prompt: Custom system instructions.
            middleware: Additional middleware, run after the SDK's default
                interrupt-recovery and durable-nesting middleware (see
                create_agent_engine_deep_agent).
            checkpointer: LangGraph checkpointer for state persistence. Defaults
                to ``_UNSET`` which resolves to ``app.checkpointer()`` (MongoDB).
                Pass ``None`` to disable checkpointing. Pass a
                ``BaseCheckpointSaver`` instance to use a custom checkpointer.
            store: LangGraph store for skills and shared data.
            skills: Parent directories (as ``str``) for deepagents skill
                discovery. Each immediate child directory containing ``SKILL.md``
                is one skill. Each path must be a ``str`` — ``pathlib.Path``
                objects are NOT accepted because deepagents' SkillsMiddleware
                calls ``.rstrip("/")`` on the value (use ``str(path)`` at the
                call site if you start from a ``Path``). Relative paths resolve
                against the directory containing ``agent.yaml``;
                ``AGENTIC_SKILLS_DIR`` may set a different skills root. When
                ``None`` (default) no skills are loaded. See the "Skills"
                section below for runtime behavior.
            backend: Backend for filesystem/shell ops. Defaults to
                ``AgentEngineToolPodBackend`` when ``None``. Pass a custom backend
                (e.g. ``StoreBackend``) to override. **Security note:** custom
                backends bypass the default OE-audited I/O path — use only in
                tests or with backends that provide equivalent auditing.

        Skills (progressive disclosure):
            The ``skills`` parameter enables deepagents' progressive-disclosure
            pattern — the LLM sees only skill *metadata* (name + description +
            path) on turn 1 and reads the full SKILL.md body on demand via
            ``read_file(path)``. This keeps the system prompt small while still
            giving the agent access to deep domain knowledge.

        Directory layout and frontmatter:
            Each entry in ``skills=`` is a parent source directory. At runtime,
            ``SkillsMiddleware`` lists that directory through the configured
            backend and treats each immediate child containing ``SKILL.md`` as
            one skill; discovery is not recursive::

                /app/skills/
                  code-review-style/
                    SKILL.md
                    examples/
                      good.py
                  security-checklist/
                    SKILL.md

            deepagents validates each skill's frontmatter itself. It skips
            unreadable or unparsable frontmatter and skills missing ``name`` or
            ``description``; Agent Skills naming or directory-name violations
            are warned about but may still load. This SDK does not validate or
            filter declared paths.

        Subagent non-inheritance:
            Skills attached to the top-level agent DO NOT propagate to
            subagents. If a subagent needs the same skills, pass ``skills=`` on
            its own ``SubAgent`` spec. This is a deepagents-level behavior and
            may change in a future release.

        Runtime state:
            On the first turn, ``SkillsMiddleware.before_agent`` loads the
            metadata into ``state["skills_metadata"]`` and injects a
            ``## Skills System`` block into the system prompt. After turn 1 the
            metadata is cached on the thread; edits to SKILL.md files on disk
            are NOT picked up inside the same thread (start a new thread to
            pick up edits).

        Returns:
            CompiledStateGraph ready for LangGraphBaseAgent wrapping.

        Raises:
            RuntimeError: If any SubAgent spec uses a string model.
        """
        from agent_engine_sdk_langgraph.deep_agent import create_agent_engine_deep_agent

        # Fail fast if the tenant hasn't opted into the Tool-Pod's built-in
        # filesystem / shell handlers — otherwise the deep agent would build
        # successfully here but every ``filesystem_*`` or ``shell_execute``
        # call at runtime would come back as "unknown tool" from the Tool
        # Pod, which is a confusing error surface several layers removed
        # from the real cause. Requiring the flag at construction time
        # gives the developer an immediate, actionable message.
        if not self._runtime.agent_config.feature_enabled("deep_agent", default=False):
            raise RuntimeError(
                "App.deep_agent() requires 'features.deep_agent: true' in "
                "agent.yaml. Without it the Tool-Pod does not register the "
                "built-in filesystem + shell handlers that deep agents rely "
                "on, and every AgentEngineToolPodBackend call would fail at "
                "runtime with 'unknown tool'. Add:\n\n"
                "    features:\n"
                "      deep_agent: true\n\n"
                "to your agent.yaml and redeploy the Tool Pod."
            )

        # No reset_llm_registry() here. get_agent() already clears the registry
        # immediately before running the entrypoint, so a second reset is
        # redundant -- and harmful: Python evaluates call arguments before the
        # callee runs, so a subagent model built inline as
        # ``app.llm(model, "researcher")`` in the ``subagents=[...]`` argument
        # registers *before* this method body executes. Resetting here would
        # wipe that registration, and the Tool Pod's /invoke_llm could no longer
        # resolve the subagent LLM by id.
        secure_llm = self.llm(llm)

        # Checkpointer sentinel resolution:
        # _UNSET -> delegate to app.checkpointer() (returns None outside AER)
        # None   -> disable checkpointing
        # instance -> use directly
        if checkpointer is _UNSET:
            resolved_checkpointer = self.checkpointer()
        else:
            resolved_checkpointer = checkpointer

        # Backend resolution: None -> AgentEngineToolPodBackend default
        if backend is None:
            from agent_engine_sdk_langgraph.backends.toolpod import (
                AgentEngineToolPodBackend,
            )

            resolved_backend = AgentEngineToolPodBackend()
        else:
            resolved_backend = backend

        agent_config_path = self._runtime.agent_config.path
        agent_dir = (
            agent_config_path.parent.resolve()
            if agent_config_path is not None
            else None
        )
        configured_skills_dir = os.environ.get("AGENTIC_SKILLS_DIR")
        if configured_skills_dir:
            skills_base_path = Path(configured_skills_dir)
            if skills_base_path.is_absolute():
                raise ValueError(
                    "AGENTIC_SKILLS_DIR must be relative to the agent source root; "
                    f"got {configured_skills_dir!r}"
                )
            if agent_dir is None:
                raise ValueError(
                    "AGENTIC_SKILLS_DIR requires agent.yaml to determine the "
                    "agent source root"
                )
            skills_base_path = (agent_dir / skills_base_path).resolve()
            agent_dir_str = str(agent_dir)
            skills_base_path_str = str(skills_base_path)
            if (
                os.path.commonpath([agent_dir_str, skills_base_path_str])
                != agent_dir_str
            ):
                raise ValueError(
                    "AGENTIC_SKILLS_DIR must stay within the agent source root; "
                    f"got {configured_skills_dir!r}"
                )
            skills_base_dir = skills_base_path
        else:
            skills_base_dir = agent_dir
        relative_skills = [path for path in skills or [] if not os.path.isabs(path)]
        if skills_base_dir is None and relative_skills:
            logger.warning(
                "Relative skill paths %s cannot be resolved because agent.yaml "
                "was not found and AGENTIC_SKILLS_DIR is unset",
                relative_skills,
            )

        return create_agent_engine_deep_agent(
            secure_llm=secure_llm,
            backend=resolved_backend,
            tools=tools,
            subagents=subagents,
            system_prompt=system_prompt,
            middleware=middleware,
            checkpointer=resolved_checkpointer,
            store=store,
            skills=skills,
            skills_base_dir=skills_base_dir,
        )

    # =========================================================================
    # Checkpointing
    # =========================================================================

    def checkpointer(self) -> Any | None:
        """Get the request-scoped platform checkpointer for LangGraph.

        The returned ``PlatformCheckpointer`` delegates per invocation from
        trusted request context: native sessions use the existing
        MongoDB-backed saver; durable_workflow sessions use attempt-local
        scratch keyed by execution + attempt + fence. It is created lazily on
        first call and cached. The underlying ``MongoClient`` is closed when
        ``App.close()`` is called.

        The database name is read from ``CHECKPOINT_DB_NAME`` when set (exact
        override, no project scoping). Otherwise the existing
        ``MDB_AGENTIC_STORE_DB`` / per-project store resolution is used
        (default: ``"mdb_store"``).

        Returns:
            PlatformCheckpointer instance in AER mode, None otherwise
        """
        if self._runtime.mode == RuntimeMode.AER:
            if self._checkpointer is not None:
                return self._checkpointer

            native: Any | None = None
            if self._runtime.mongodb_uri:
                from langgraph.checkpoint.mongodb import MongoDBSaver
                from pymongo import MongoClient

                from agent_engine_runner_shared.db_config import resolve_store_db_name

                server_selection_timeout_ms = _checkpointer_timeout_ms(
                    CHECKPOINTER_SERVER_SELECTION_TIMEOUT_ENV, 5.0
                )
                connect_timeout_ms = _checkpointer_timeout_ms(
                    CHECKPOINTER_CONNECT_TIMEOUT_ENV, 5.0
                )
                socket_timeout_ms = _checkpointer_timeout_ms(
                    CHECKPOINTER_SOCKET_TIMEOUT_ENV, 15.0
                )
                client = MongoClient(
                    self._runtime.mongodb_uri,
                    serverSelectionTimeoutMS=server_selection_timeout_ms,
                    connectTimeoutMS=connect_timeout_ms,
                    socketTimeoutMS=socket_timeout_ms,
                )
                try:
                    # Exact database override when configured. Must not apply
                    # project scoping or discovery — the value is the final DB
                    # name passed to MongoDBSaver.
                    checkpoint_db_override = os.environ.get(
                        "CHECKPOINT_DB_NAME", ""
                    ).strip()
                    if checkpoint_db_override:
                        checkpoint_db = checkpoint_db_override
                    else:
                        # Resolve the per-project-scoped store DB now that a
                        # client exists, so checkpoints land in the same
                        # database the OE reads. Logged after resolution so
                        # the line shows the real (scoped) database, not the
                        # base name.
                        checkpoint_db = resolve_store_db_name(client)
                    logger.info(
                        "Using MongoDB checkpointer: db=%s "
                        "serverSelectionTimeoutMS=%d connectTimeoutMS=%d "
                        "socketTimeoutMS=%d",
                        checkpoint_db,
                        server_selection_timeout_ms,
                        connect_timeout_ms,
                        socket_timeout_ms,
                    )
                    native = MongoDBSaver(client, db_name=checkpoint_db)
                except Exception:
                    # DB resolution (fail-closed on empty PROJECT_ID) or
                    # MongoDBSaver construction (bad URI, auth) may raise. Close
                    # the freshly-opened client before the exception propagates
                    # so a failed init does not leak a pool connection.
                    client.close()
                    raise
                self._mongo_client = client
            else:
                # Preserve the long-standing no-URI contract: agents compile a
                # stateless graph rather than failing every native invocation
                # on a checkpointer whose native leg cannot exist.
                logger.warning("MongoDB URI not defined, required for checkpointer")
                return None

            self._checkpointer = PlatformCheckpointer(native=native)
            return self._checkpointer

        logger.warning(
            "app.checkpointer() called in %s mode; checkpointing is managed "
            "by the platform in non-AER modes — returning None",
            self._runtime.mode.value,
        )
        return None

    def close(self) -> None:
        """Release resources held by this App instance.

        Closes the MongoDB client opened by :meth:`checkpointer`, if any.
        """
        if self._mongo_client is not None:
            self._mongo_client.close()
            self._mongo_client = None
            self._checkpointer = None

    # =========================================================================
    # Tools
    # =========================================================================

    def get_tools(self) -> list[Any]:
        """Get wrapped tools for ToolNode.

        In AER mode, returns tools wrapped with SecureToolWrapper that route
        all executions through OE for logging and policy enforcement.
        In other modes, returns the original LangChain tools.

        Returns:
            List of tools ready for ToolNode
        """
        tools_list = list(self._lc_tools.values())

        if self._runtime.mode != RuntimeMode.AER:
            return tools_list

        from agent_engine_runner_shared.secure_wrapper import (
            create_secure_tool_function,
        )

        wrapped_tools = []
        allow_direct = get_env_bool("RUNNER_ALLOW_DIRECT_TOOL_EXECUTION", False)

        for tool_obj in tools_list:
            tool_name = tool_obj.name
            tool_metadata = self._runtime.get_tool_metadata(tool_name)
            tool_call_metadata: dict[str, Any] | None = None
            scopes: list[str] = []
            mcp_server_name = tool_metadata.get("mcp_server")
            mcp_tool_name = tool_metadata.get("mcp_tool")
            if isinstance(mcp_server_name, str) and isinstance(mcp_tool_name, str):
                tool_call_metadata = {
                    "mcp_server": mcp_server_name,
                    "mcp_tool": mcp_tool_name,
                }
            provider_type = normalize_optional_str(tool_metadata.get("provider_type"))
            raw_scopes = tool_metadata.get("scopes")
            if isinstance(raw_scopes, list):
                scopes = [
                    scope.strip()
                    for scope in raw_scopes
                    if isinstance(scope, str) and scope.strip()
                ]
            # Always wire the StructuredTool for "content_and_artifact" so an
            # interrupt lands in ToolMessage.artifact rather than colliding with
            # real tool content. tool_declared_format tracks the tool's own
            # original choice, so a "content"-declared tool never has a genuine
            # two-element result silently split and hidden from the model.
            tool_declared_format: Literal["content", "content_and_artifact"] = (
                "content_and_artifact"
                if getattr(tool_obj, "response_format", "content")
                == "content_and_artifact"
                else "content"
            )
            response_format: Literal["content", "content_and_artifact"] = (
                "content_and_artifact"
            )

            # The tool's redact_fields policy must reach the wrapper's debug
            # argument dump (SECBUG-3072).
            raw_redact_fields = tool_metadata.get("redact_fields")
            redact_fields = (
                [f for f in raw_redact_fields if isinstance(f, str)]
                if isinstance(raw_redact_fields, list)
                else []
            )
            wrapper_func = create_secure_tool_function(
                original_tool=tool_obj,
                tool_name=tool_name,
                allow_direct=allow_direct,
                is_local=bool(tool_metadata.get("is_local", True)),
                provider_type=provider_type,
                scopes=scopes,
                metadata=tool_call_metadata,
                response_format=response_format,
                tool_declared_format=tool_declared_format,
                redact_fields=redact_fields,
                is_framework_control_flow=lambda error: isinstance(
                    error, GraphBubbleUp
                ),
            )
            wrapper_func = _with_durable_tool_result_identity(
                wrapper_func,
                tool_name=tool_name,
            )

            wrapped_tool = StructuredTool.from_function(
                func=wrapper_func,
                name=tool_obj.name,
                description=tool_obj.description,
                # Extend the schema with an injected tool_call_id so LangChain
                # hands the LLM's stable id to wrapper_func, which forwards it to
                # OE for the execution-log join key (AP-2838).
                args_schema=_args_schema_with_injected_tool_call_id(
                    tool_obj.args_schema
                ),
                response_format=response_format,
                handle_tool_error=True,
            )
            wrapped_tools.append(wrapped_tool)
            logger.debug("Wrapped tool: %s", tool_name)

        return wrapped_tools

    def get_tool_schemas(self) -> list[Any]:
        """Get tool schemas for llm.bind_tools().

        Returns:
            List of original LangChain tools (not wrapped)
        """
        return list(self._lc_tools.values())

    def suspend(self, reason: str, context: dict[str, Any]) -> str:
        """Generates a suspend command. If a tool should suspend, return the result
        of this function.

        Not supported on durable workflow sessions: use LangGraph's native
        ``interrupt()`` instead. A tool that returns this payload on a durable
        session fails with a clear error before the wait is recorded.
        """
        return SuspendPayload(
            suspend_reason=reason,
            suspend_context=context,
        ).to_json()

    def finish_session(self) -> SessionFinishStatus:
        """Mark this session finished so the platform frees its compute now.

        Call it when the agent is done with the session. The current turn keeps
        running and returns its result normally; once it completes, the platform
        cancels any live sub-agent runs and releases the session's AER and tool
        pods instead of holding them until the idle timeout expires.

        Safe to call more than once: the first call returns REQUESTED, later
        ones ALREADY_REQUESTED. Outside an agent run (local scripts, tool pods)
        there is no session to finish and the call returns UNAVAILABLE without
        raising. Calling it after the turn has already ended - e.g. from
        background work scheduled during the turn but that runs after it -
        also returns UNAVAILABLE: by then nothing is listening for the
        request anymore, so reporting REQUESTED would promise a release that
        will never happen.

        A turn that suspends for human review, or that fails, keeps its
        resources so it stays resumable and diagnosable; the session then falls
        back to the idle timeout.
        """
        return request_session_finish()

    # =========================================================================
    # Guardrails
    # =========================================================================

    def validate_llm_response(self, response: Any) -> Any:
        """Apply guardrails validation to LLM response.

        Checks if the response has content (and no tool calls), runs guardrails
        validation, and returns a new AIMessage with validated content if
        modifications were made.

        Args:
            response: LangChain AIMessage or similar response object

        Returns:
            Original or modified response after validation
        """
        has_content = (
            hasattr(response, "content")
            and isinstance(response.content, str)
            and response.content
        )
        has_tool_calls = hasattr(response, "tool_calls") and response.tool_calls

        if not has_content or has_tool_calls:
            return response

        validated_content = self._runtime.validate_output(response.content)

        if validated_content != response.content:
            from langchain_core.messages import AIMessage

            return AIMessage(
                content=validated_content,
                tool_calls=getattr(response, "tool_calls", None),
                response_metadata=getattr(response, "response_metadata", {}),
            )

        return response

    def validate_output(self, text: str) -> str:
        """Validate text output using configured guardrails.

        Args:
            text: Text to validate

        Returns:
            Validated (possibly modified) text
        """
        return self._runtime.validate_output(text)

    # =========================================================================
    # Agent-to-Agent (A2A)
    # =========================================================================

    def a2a_tools(self) -> list[Any]:
        """Get LangChain tools for A2A agent discovery and invocation.

        Returns StructuredTool instances that execute in the AER process
        (not via the tool pod) because A2A requires the execution context's
        OE URL and A2A JWT.

        Bind them to the LLM and pass to ToolNode alongside any @app.tool
        tools::

            a2a = app.a2a_tools()
            llm_with_tools = llm.bind_tools(app.get_tool_schemas() + a2a)
            ToolNode(app.get_tools() + a2a)

        Returns:
            List of LangChain tools for discover_agents and invoke_agent.
            Empty list when A2A is not enabled in agent.yaml.
        """
        if not self.agent_config.a2a.enabled:
            return []

        from agent_engine_sdk_langgraph.a2a_tools import build_a2a_tools

        return build_a2a_tools(self._runtime)

    # =========================================================================
    # Context
    # =========================================================================

    def get_current_user_id(self) -> str | None:
        """Get current execution context user ID.

        Returns:
            User ID from the current execution context, or None
        """
        return self._runtime.get_current_user_id()
