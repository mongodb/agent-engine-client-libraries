"""
TenantRuntime — framework-agnostic platform runtime for tenant agent applications.

The same application code runs in all supported modes:
- aer: Agent execution with secure wrappers
- tool: Tool function execution

Usage (via a framework SDK, e.g. agent-engine-sdk-langgraph):
    The framework SDK's ``App`` class wraps ``TenantRuntime`` and registers
    hooks so agent-engine-runner-shared can call framework code without importing it.
    See ``agent_engine_runner_shared.hooks`` for the hook registry.
"""

from __future__ import annotations

import asyncio
import logging
import os
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional

from agent_engine_sdk.interfaces import BaseAgent

from agent_engine_runner_shared.agent_config import (
    RuntimeAgentConfig,
    load_runtime_agent_config,
)
from agent_engine_runner_shared.context import get_current_user_id
from agent_engine_runner_shared.tool_memory_ownership import is_memory_read_owned_by_tool
from agent_engine_runner_shared.utils import (
    RuntimeMode,
    get_env,
    get_env_bool,
    get_env_int,
    get_runtime_mode,
    setup_logging,
    tenant_env_vars,
)

if TYPE_CHECKING:
    from agent_engine_sdk_memory.models import (
        ContextResponse,
        CustomMemoryRetrieveResult,
        CustomMemorySaveResult,
        SourceSpec,
    )

logger = logging.getLogger(__name__)


def _resolve_listen_host() -> str:
    """Resolve uvicorn's listen host from ``APP_HOST``, defaulting to
    ``"0.0.0.0"``.

    ECP stamps ``APP_HOST`` on every runner component (AER, Tool) at deploy
    time — ``"::"`` for vm-mode (cell, IPv6-first
    pod network) and ``"0.0.0.0"`` for container-mode (Kind, classic
    helix, v4 pod IPs). The runner-base image also bakes in
    ``APP_HOST=0.0.0.0`` as a baseline. So every production path has the
    var set; this default only matters for ad-hoc Python invocations that
    bypass both the image and ECP, and ``"0.0.0.0"`` is the safe choice
    there.

    Empty string is normalized to the default (a shell that exports
    ``APP_HOST=""`` must not produce a bind to ``""``); any other value
    is passed through unchanged. Callers must set a bind-compatible
    string — a bare IP literal like ``"::"`` or ``"0.0.0.0"``, never the
    bracketed URI form ``"[::]"`` which ``getaddrinfo`` rejects on at
    least macOS. ECP and the runner-base image both honour this.

    History: an earlier version tried to auto-detect dual-stack capability
    by probing the kernel's ``IPV6_V6ONLY`` default, but uvicorn's
    ``socket.create_server`` call forces ``IPV6_V6ONLY=1`` regardless of
    the OS default — making the probe return ``"::"`` on Kind / classic
    helix and breaking kubelet's v4 startup probe. The auto-detection was
    dropped in favour of explicit ECP stamping.
    """
    return os.environ.get("APP_HOST") or "0.0.0.0"


def _maybe_start_debugger() -> None:
    """Local-Kind-only debug hook.

    Called from ``register_and_run`` — the one place every framework
    adapter's ``App.run()`` (LangGraph, ADK, ...) funnels through before
    ``asyncio.run(self._run_async(...))`` binds the real listener — so a
    single hook here covers every current and future SDK adapter without
    needing a change in each one.

    A no-op unless ``MDBAE_LOCAL_MODE=true``, which agentic-operator only sets
    on a container when its local-only ``DebugModeEnabled`` flag is on (see
    ``containerPodTemplate`` in the operator). The ``MDBAE`` prefix is chosen
    to be uncommon: it is not a reserved namespace, but no customer is likely
    to pick an env var of that name, so the stamp will not clash with tenant
    config. ``debugpy`` is a dev dependency, not bundled in production images,
    so importing it here is deliberately deferred until the env var confirms
    it's actually needed.

    Deliberately non-blocking: this calls ``debugpy.listen()`` only, not
    ``wait_for_client()``. The server starts and serves traffic normally
    whether or not anything ever attaches — the Java/JDWP ``suspend=n``
    model, not debugpy's often-demoed "pause until a client connects"
    pattern. A blocking variant was considered and rejected: on a server
    process, the only consequence of blocking at startup is that every
    workspace with ``MDBAE_LOCAL_MODE=true`` hangs indefinitely until someone
    attaches, which makes a cluster-wide debug-mode flag unsafe by
    construction. With the non-blocking model, ``MDBAE_LOCAL_MODE=true`` costs
    nothing on a pod nobody is actively debugging.
    """
    if not get_env_bool("MDBAE_LOCAL_MODE"):
        return

    import debugpy

    port = get_env_int("MDBAE_LOCAL_PORT", 5678)
    # Loopback only: the debugger is reached via `kubectl port-forward`, which
    # connects inside the pod's network namespace and so still reaches
    # 127.0.0.1, while a loopback bind keeps the port off the pod network.
    debugpy.listen(("127.0.0.1", port))
    logger.info(
        "debugpy listening on port %d (non-blocking -- the server starts normally; "
        "attach at any time to set breakpoints)",
        port,
    )


class TenantRuntime:
    """
    Tenant Runtime SDK.

    The runtime discovers its role from RUNNER_MODE environment variable:

    - aer: Full LangGraph execution with SecureToolWrapper
    - tool: Tool function execution
    """

    def __init__(
        self,
        app_name: str = "Agent",
        app_version: str = "1.0.0",
        mongodb_uri: Optional[str] = None,
        database_name: Optional[str] = None,
        enable_tracing: bool | None = None,
        traces_collection_name: str = "traces",
        enable_memory: bool | None = None,
        org_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ):
        """
        Initialize the runner runtime.

        Args:
            app_name: Application name
            app_version: Application version
            mongodb_uri: MongoDB URI (for OE execution logging)
            database_name: Database name
            enable_tracing: Deprecated. Tracing is always enabled.
            traces_collection_name: Collection name for trace storage
            enable_memory: Deprecated. Use agent.yaml features.memory instead.
            org_id: Deprecated and ignored. The org is taken from the
                ``ORG_ID`` environment variable, which the platform injects.
                Passing this argument raises a DeprecationWarning and will be
                removed in a future release.
            project_id: Deprecated and ignored. The project is taken from the
                ``PROJECT_ID`` environment variable, which the platform injects.
                Passing this argument raises a DeprecationWarning and will be
                removed in a future release.
        """
        self.app_name = app_name
        self.app_version = app_version
        self.mode = get_runtime_mode()
        # org_id/project_id come exclusively from the platform-injected env
        # vars. The constructor arguments are deprecated and intentionally
        # ignored — honoring an author-supplied override is what caused the
        # silent tenant-mismatch bug.
        self.org_id = get_env("ORG_ID", None)  # type: ignore[arg-type]  # no default org_id
        self.project_id = get_env("PROJECT_ID", None)  # type: ignore[arg-type]  # no default project_id

        # Setup logging first
        setup_logging(app_name=app_name, mode=self.mode.value)

        self._agent_config = load_runtime_agent_config(env_vars=tenant_env_vars())

        # Config
        self._mongodb_uri = mongodb_uri or get_env("MONGODB_URI")
        self._database_name_override = database_name or None
        self._traces_collection_name = traces_collection_name
        # Deprecation warnings are emitted after setup_logging() so the
        # logger.warning() companion in _warn_if_deprecated_constructor_arg
        # routes through the configured handlers (incl. structured logging).
        self._warn_if_deprecated_constructor_arg(
            message=(
                "TenantRuntime(org_id=...) is deprecated and ignored. "
                "Set the ORG_ID environment variable instead; "
                "this argument will be removed in a future release."
            ),
            used=org_id is not None,
        )
        self._warn_if_deprecated_constructor_arg(
            message=(
                "TenantRuntime(project_id=...) is deprecated and ignored. "
                "Set the PROJECT_ID environment variable instead; "
                "this argument will be removed in a future release."
            ),
            used=project_id is not None,
        )
        self._warn_if_deprecated_constructor_arg(
            message=(
                "TenantRuntime(enable_tracing=...) is deprecated and ignored. "
                "Tracing is always enabled; remove this argument."
            ),
            used=enable_tracing is not None,
        )
        self._enable_memory = self._resolve_feature_toggle(
            "memory",
            "ENABLE_MEMORY",
            deprecated_value=enable_memory,
        )
        # Tool registry
        self._tools: Dict[str, Callable] = {}  # Raw functions (for Tool Pod execution)
        self._tool_definitions: Dict[str, Dict[str, Any]] = {}  # Tool metadata

        # Graph builder (set by register_and_run)
        self._graph_builder: Optional[Callable] = None

        # Memory components (initialized in AER mode only)
        self._memory_engine: Optional[Any] = None
        self._memory_writer: Optional[Any] = None

        # A2A client (lazy-initialized HTTP client for AER mode)
        self._a2a_client: Optional[Any] = None
        self._a2a_oe_url: Optional[str] = None
        self._a2a_token: Optional[str] = None

        # AER query plugin (framework adapter registers an implementation
        # before the server starts; absent on non-AER modes or when the
        # framework has no read-side support).
        self._query_plugin: Optional[Any] = None
        # Wire workspace_id from the most recent /execute — used to scope
        # checkpoint reads when APP_ID is unset (local dev / tests).
        self._checkpoint_wire_workspace_id: Optional[str] = None

        # Tracing is always enabled so every runtime mode emits consistent spans.
        self._setup_tracing()

        # Setup memory if enabled (needed in AER and TOOL modes)
        # - AER mode: for agent memory context and writing
        # - TOOL mode: tools may need memory access
        if self._enable_memory:
            self._setup_memory()

        logger.info(f"TenantRuntime initialized: mode={self.mode.value}, app={app_name}")

    @property
    def mongodb_uri(self) -> str:
        return self._mongodb_uri

    @property
    def agent_config(self) -> RuntimeAgentConfig:
        """Return the parsed runtime view of agent.yaml."""
        return self._agent_config

    @property
    def memory_enabled(self) -> bool:
        """Return the resolved agent.yaml Memory feature setting."""
        return self._enable_memory

    def _resolve_feature_toggle(
        self,
        name: Literal["memory", "deep_agent"],
        env_var: str,
        deprecated_value: bool | None = None,
        default: bool = False,
    ) -> bool:
        configured = self._agent_config.configured_feature(name)
        if configured is not None:
            if deprecated_value is not None:
                self._warn_if_deprecated_constructor_arg(
                    message=(
                        f"TenantRuntime(enable_{name}=...) is deprecated and ignored when "
                        f"agent.yaml features.{name} is set. Remove this argument and keep "
                        "the setting in agent.yaml."
                    ),
                    used=True,
                )
            return configured
        if deprecated_value is not None:
            self._warn_if_deprecated_constructor_arg(
                message=(
                    f"TenantRuntime(enable_{name}=...) is deprecated. Move this setting to "
                    f"agent.yaml features.{name}. The constructor fallback will be removed "
                    "in a future release."
                ),
                used=True,
            )
            return deprecated_value
        if env_var in os.environ:
            return get_env_bool(env_var)
        return default

    def _warn_if_deprecated_constructor_arg(
        self,
        *,
        message: str,
        used: bool,
    ) -> None:
        if not used:
            return
        warnings.warn(message, DeprecationWarning, stacklevel=3)
        logger.warning(message)

    def get_agent(self, callbacks: List[Any] | None = None) -> BaseAgent:
        """Get a BaseAgent instance from the registered App.

        Delegates to the App's get_agent() method, which handles
        framework-specific graph building and callback adaptation.

        Args:
            callbacks: Framework-neutral callbacks for observability
                (e.g., NodeExecutionLogger)

        Returns:
            BaseAgent instance

        Raises:
            RuntimeError: If no App has been registered via register_and_run()
        """
        if self._graph_builder is None:
            raise RuntimeError(
                "No App registered. Call register_and_run() with a BaseApp instance first."
            )

        if not hasattr(self._graph_builder, "get_agent"):
            raise RuntimeError(
                "Graph builder must be a BaseApp instance with get_agent(). "
                "Plain callables are not supported."
            )

        return self._graph_builder.get_agent(callbacks=callbacks)  # type: ignore[union-attr]

    def warm_up_agent(self) -> bool:
        """Best-effort pre-build of the agent graph before the first /execute.

        Delegates to the framework App's warm_up() if it declares one;
        frameworks that don't opt in keep today's lazy build on first
        get_agent(). A ``False`` result reports that the one-shot optimization
        did not complete; ``None`` from an older or third-party implementation
        remains a successful result for compatibility.
        """
        if self._graph_builder is not None and hasattr(self._graph_builder, "warm_up"):
            result = self._graph_builder.warm_up()  # type: ignore[union-attr]
            return result is not False
        return True

    def _setup_tracing(self) -> None:
        """Set up OpenTelemetry tracing.

        If the trace-store database is unreachable at startup, a background
        retry attaches the MongoDB exporter once connectivity recovers —
        without a pod restart. The failure is
        surfaced via /health (DEGRADED) and an ERROR log, not a single warning
        that silently disappears into the pod log stream.
        """

        # Fallback scrub if the tracing import itself fails — in that case no
        # store URI has been touched yet, so the message can't carry one.
        def scrub(message: str) -> str:
            return message

        try:
            from agent_engine_runner_shared.tracing import (
                scrub_credentials,
                setup_tracing,
                start_mongodb_tracing_retry,
            )

            scrub = scrub_credentials

            def connect() -> Any:
                from pymongo import MongoClient

                from agent_engine_runner_shared.db_config import resolve_store_db_name

                client = MongoClient(self._mongodb_uri, serverSelectionTimeoutMS=5000)
                try:
                    client.admin.command("ping")
                    database_name = resolve_store_db_name(client, base=self._database_name_override)
                    return client[database_name][self._traces_collection_name]
                except Exception:
                    try:
                        client.close()
                    except Exception:
                        pass
                    raise

            mongodb_collection = None
            store_unreachable = False
            if self._mongodb_uri:
                try:
                    mongodb_collection = connect()
                except Exception as e:
                    store_unreachable = True
                    logger.error(
                        "Trace store unavailable at startup; will retry in "
                        "the background. Error: %s",
                        scrub(str(e)),
                    )

            setup_tracing(
                service_name=self.app_name,
                mongodb_collection=mongodb_collection,
                # connect() built this client solely for tracing — tracing owns
                # it and closes it at shutdown, like the retry-recovered one.
                take_client_ownership=True,
            )

            if store_unreachable:
                start_mongodb_tracing_retry(connect)

            logger.info("Tracing enabled")
        except ImportError:
            logger.warning(
                "Tracing dependencies not installed. Install: pip install agent-engine-runner-shared[tracing]"
            )
        except Exception as e:
            # Scrub defensively: a setup failure whose message echoes the raw
            # store URI must never write its credentials to the log sink.
            logger.warning(f"Tracing setup failed: {scrub(str(e))}")

    def _setup_memory(self) -> None:
        """Set up memory via MemoryClient (HTTP to OE)."""
        oe_url = os.getenv("OE_URL", "").strip()
        if not oe_url:
            logger.warning(
                "OE_URL not set - memory disabled. "
                "OE_URL should point to the Orchestration Engine "
                "(e.g., http://oe:8000 in local dev, set by ECP in Helix)."
            )
            return

        try:
            from agent_engine_runner_shared.memory import MemoryWriter, create_memory_client

            self._memory_engine = create_memory_client(server_url=oe_url)

            if self._memory_engine:
                self._memory_writer = MemoryWriter(self._memory_engine)
                logger.info(f"Memory integration enabled (HTTP client -> {oe_url})")
            else:
                logger.warning(f"MemoryClient creation failed for {oe_url} - memory disabled")
        except ImportError:
            logger.warning(
                "Memory client dependencies not installed. "
                "Install agent_engine_sdk: pip install agent-engine-sdk"
            )
        except Exception as e:
            logger.warning(f"Memory client setup failed: {e}")

    @property
    def memory_engine(self) -> Optional[Any]:
        """Get the MemoryEngine instance if configured."""
        return self._memory_engine

    @property
    def memory_writer(self) -> Optional[Any]:
        """Get the MemoryWriter instance if configured."""
        return self._memory_writer

    def build_context(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        visibility: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        enabled_sources: Optional[set[str]] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Build memory context for a query.

        Works with both MemoryEngine (direct) and MemoryClient (HTTP) backends.
        Backend choice is controlled by USE_MEMORY_CLIENT environment variable
        at initialization time (_setup_memory).

        Args:
            query: The user's query/message
            user_id: User ID (required for memory isolation)
            session_id: Session ID
            visibility: Visibility scope filter (private, shared, org)
            max_tokens: Optional gross context-construction budget. After
                retrieval and ranking, the server subtracts a 500-token
                formatting reserve, then greedily selects whole memory chunks
                that fit in the remainder. Positive values at or below 500
                leave no budget for memories. Values above 500 can still yield
                empty context when no chunk fits. Omit to use the server
                default.

        Returns:
            Formatted context string, or empty string if memory is disabled
        """
        from agent_engine_runner_shared.workflow import current_attempt_context

        attempt = current_attempt_context()
        if attempt is None and self._memory_engine is None:
            return ""

        if self.org_id is None:
            logger.warning("org_id not set - cannot build memory context")
            return ""

        resolved_user_id = user_id or get_current_user_id()
        if not resolved_user_id:
            logger.warning("user_id not set - cannot build memory context")
            return ""
        if not session_id:
            logger.warning("session_id required for memory context; skipping context build")
            return ""

        from agent_engine_runner_shared.memory import _require_positive_max_tokens

        _require_positive_max_tokens(max_tokens)

        kwargs: Dict[str, Any] = dict(
            memory_engine=self._memory_engine,
            query=query,
            session_id=session_id,
            org_id=self.org_id,
            user_id=resolved_user_id,
            project_id=self.project_id,
            visibility=visibility,
            metadata_filter=metadata_filter,
            enabled_sources=enabled_sources,
        )
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        def _read_memory() -> str:
            from agent_engine_runner_shared.memory import build_context

            try:
                return build_context(**kwargs)
            except Exception as e:
                logger.warning(f"Failed to build memory context: {e}")
                return ""

        if attempt is None or is_memory_read_owned_by_tool():
            return _read_memory()

        from agent_engine_runner_shared.context import get_current_wrapper
        from agent_engine_runner_shared.generated.workflow.v1.activity_pb2 import (
            ACTIVITY_KIND_MEMORY,
        )
        from agent_engine_runner_shared.workflow import (
            allocate_activity_ordinal,
            run_serial_activity,
        )

        wrapper = get_current_wrapper()
        if wrapper is None:
            raise RuntimeError(
                "durable memory context requires the current execution workflow client"
            )

        semantic_input: Dict[str, Any] = dict(
            query=query,
            user_id=resolved_user_id,
            session_id=session_id,
            visibility=visibility,
            metadata_filter=metadata_filter,
            enabled_sources=(sorted(enabled_sources) if enabled_sources is not None else None),
        )
        if max_tokens is not None:
            semantic_input["max_tokens"] = max_tokens

        return run_serial_activity(
            client=wrapper.workflow,
            kind=ACTIVITY_KIND_MEMORY,
            name="memory.build_context",
            activity_ordinal=allocate_activity_ordinal(),
            semantic_input=semantic_input,
            execute=lambda _context: _read_memory(),
            exclusive=True,
        )

    def build_context_from_sources(
        self,
        query: str,
        sources: "list[SourceSpec]",
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        visibility: Optional[str] = None,
        rerank: bool = False,
        max_tokens: Optional[int] = None,
    ) -> "ContextResponse":
        """Build memory context from an explicit, per-source-configured source set.

        The per-source counterpart to :meth:`build_context`. Where that method
        applies one filter and mode to every source and returns a flattened
        string, this preserves the full ``ContextResponse`` so the per-source
        metadata (``ranking_strategy``, ``source_outcomes``) reaches the caller.

        Tenancy (``org_id``, ``project_id``) is stamped from the runtime; the
        caller passes none of it. Unlike :meth:`build_context`, each source's
        ``top_k`` is honored, since it is the caller's explicit per-source intent.
        ``session_id`` is required only when the ``stm`` source is requested;
        omitting it for any other source set is fine.

        Returns an empty ``ContextResponse`` when memory is unavailable or a
        required identity field is unresolved, degrading like
        :meth:`build_context` rather than raising into the agent turn.
        """
        from agent_engine_sdk_memory.models import ContextMetadata, ContextResponse

        from agent_engine_runner_shared.workflow import current_attempt_context

        def _empty() -> "ContextResponse":
            return ContextResponse(formatted_context="", metadata=ContextMetadata())

        attempt = current_attempt_context()
        if attempt is None and self._memory_engine is None:
            return _empty()

        if self.org_id is None:
            logger.warning("org_id not set - cannot build memory context")
            return _empty()

        resolved_user_id = user_id or get_current_user_id()
        if not resolved_user_id:
            logger.warning("user_id not set - cannot build memory context")
            return _empty()

        # STM is the only session-scoped source; every other source reads
        # session-unscoped, so session_id is required only when stm is requested.
        wants_stm = any(getattr(spec, "source", None) == "stm" for spec in sources)
        if wants_stm and not session_id:
            logger.warning("session_id required for the stm source; skipping context build")
            return _empty()

        from agent_engine_runner_shared.memory import _require_positive_max_tokens

        _require_positive_max_tokens(max_tokens)

        kwargs: Dict[str, Any] = dict(
            memory_engine=self._memory_engine,
            query=query,
            sources=sources,
            org_id=self.org_id,
            user_id=resolved_user_id,
            project_id=self.project_id,
            session_id=session_id,
            visibility=visibility,
            rerank=rerank,
        )
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        def _read_memory() -> Dict[str, Any]:
            from agent_engine_runner_shared.memory import build_context_from_sources

            try:
                response = build_context_from_sources(**kwargs)
                return response.model_dump(mode="json")
            except Exception as e:
                logger.warning(f"Failed to build per-source memory context: {e}")
                return _empty().model_dump(mode="json")

        if attempt is None:
            return ContextResponse.model_validate(_read_memory())

        from agent_engine_runner_shared.context import get_current_wrapper
        from agent_engine_runner_shared.generated.workflow.v1.activity_pb2 import (
            ACTIVITY_KIND_MEMORY,
        )
        from agent_engine_runner_shared.workflow import (
            allocate_activity_ordinal,
            run_serial_activity,
        )

        wrapper = get_current_wrapper()
        if wrapper is None:
            raise RuntimeError(
                "durable memory context requires the current execution workflow client"
            )

        # SourceSpec objects are not JSON-shaped; serialize them for the durable
        # activity's checkpointed semantic input.
        semantic_input: Dict[str, Any] = dict(
            query=query,
            user_id=resolved_user_id,
            session_id=session_id,
            visibility=visibility,
            rerank=rerank,
            sources=[spec.model_dump(exclude_none=True) for spec in sources],
        )
        if max_tokens is not None:
            semantic_input["max_tokens"] = max_tokens

        # The activity result is checkpointed as JSON and returns as a plain dict
        # on replay, so round-trip a serialized ContextResponse and rebuild it.
        payload = run_serial_activity(
            client=wrapper.workflow,
            kind=ACTIVITY_KIND_MEMORY,
            name="memory.build_context_from_sources",
            activity_ordinal=allocate_activity_ordinal(),
            semantic_input=semantic_input,
            execute=lambda _context: _read_memory(),
            exclusive=True,
        )
        return ContextResponse.model_validate(payload)

    def write_turn_async(
        self,
        message: str,
        result_messages: List[Any],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        include_user_turn: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        idempotency_key: Optional[str] = None,
    ) -> None:
        """
        Write turn to memory asynchronously (non-blocking).

        Args:
            message: Original user message
            result_messages: All messages from graph execution
            user_id: User ID (defaults to current execution context user)
            session_id: Session ID
            include_user_turn: Record the user message (default True). Pass False
                on resume legs, where there is no new user prompt and the user
                turn was already written when the turn first started.
            metadata: Arbitrary key-value metadata stamped on every turn written.
            idempotency_key: Deduplication key for single-turn writes; ignored
                for multi-message writes (see :func:`write_turn_to_memory`).
        """
        if self._memory_writer is None:
            return

        resolved_user_id = user_id or get_current_user_id()
        if not resolved_user_id:
            logger.warning("user_id not set - skipping turn write")
            return

        self._memory_writer.write_turn_async(
            message=message,
            result_messages=result_messages,
            session_id=session_id,
            org_id=self.org_id,
            user_id=resolved_user_id,
            project_id=self.project_id,
            include_user_turn=include_user_turn,
            metadata=metadata,
            idempotency_key=idempotency_key,
        )

    def save_semantic(
        self,
        text: str,
        label: str,
        user_id: Optional[str] = None,
        source: str = "agent",
        visibility: str = "private",
        metadata: Optional[Dict[str, Any]] = None,
        upsert: bool = True,
        agent_id: Optional[str] = None,
    ) -> bool:
        """
        Save a semantic memory to the memory engine.

        Handles embedding generation and storage automatically.

        Args:
            text: The memory content to store
            label: A unique label/identifier for this memory
            user_id: User ID (required)
            source: Source of the memory (default: "agent")
            visibility: Memory visibility (private, shared, org)
            metadata: Optional caller-supplied metadata dict stored with the memory
            upsert: If True, update existing memory with same label (default: True)

        Returns:
            True if saved successfully, False otherwise
        """
        from agent_engine_runner_shared.memory import create_semantic_memory

        resolved_user_id = user_id or get_current_user_id()
        if not resolved_user_id:
            raise ValueError("user_id is required to save semantic memory")

        result = create_semantic_memory(
            memory_engine=self._memory_engine,
            text=text,
            label=label,
            org_id=self.org_id,
            user_id=resolved_user_id,
            source=source,
            visibility=visibility,
            metadata=metadata,
            upsert=upsert,
            project_id=self.project_id,
            agent_id=agent_id,
        )
        return result is not None

    def search_semantic(
        self,
        query: str,
        user_id: Optional[str] = None,
        visibility: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search semantic memories by query.

        Args:
            query: Search query text
            user_id: Optional user ID filter
            top_k: Maximum number of results

        Returns:
            List of matching memories with label, text, and metadata
        """
        from agent_engine_runner_shared.memory import search_semantic_memory

        return search_semantic_memory(
            memory_engine=self._memory_engine,
            query=query,
            org_id=self.org_id,
            project_id=self.project_id,
            user_id=user_id,
            visibility=visibility,
            top_k=top_k,
        )

    def _require_memory_engine(self) -> Any:
        """Return the memory engine or raise when memory is not enabled."""
        if self._memory_engine is None:
            # Imported lazily: agent-engine-runner-shared only depends on
            # agent_engine_sdk_memory at type-checking time.
            from agent_engine_sdk_memory.errors import MemoryNotSupportedError

            raise MemoryNotSupportedError("memory is not enabled for this agent")
        return self._memory_engine

    def save_custom(
        self,
        memory_type: str,
        content: str,
        tags: Optional[Dict[str, Any]] = None,
        contextual_metadata: Optional[Dict[str, Any]] = None,
    ) -> "CustomMemorySaveResult":
        """Save a memory of a declared custom type.

        Identity is stamped by the platform; the OE memory proxy forwards the
        request to the project's memory server.
        """
        return self._require_memory_engine().create_custom(
            memory_type=memory_type,
            content=content,
            tags=tags,
            contextual_metadata=contextual_metadata,
        )

    def retrieve_custom(
        self,
        memory_type: str,
        query: str,
        tags: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
    ) -> "CustomMemoryRetrieveResult":
        """Retrieve memories of a declared custom type by semantic query."""
        return self._require_memory_engine().retrieve_custom(
            memory_type=memory_type, query=query, tags=tags, top_k=top_k
        )

    def get_semantic(
        self, label: str, user_id: Optional[str] = None, visibility: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get a specific semantic memory by label.

        Args:
            label: Memory label

        Returns:
            Dict with memory details, or None if not found
        """
        from agent_engine_runner_shared.memory import get_semantic_memory

        return get_semantic_memory(
            memory_engine=self._memory_engine,
            org_id=self.org_id,
            project_id=self.project_id,
            label=label,
            user_id=user_id,
            visibility=visibility,
        )

    # =========================================================================
    # Taxonomic Memory API
    # =========================================================================

    def create_taxonomic(
        self,
        domain: str,
        term: str,
        definition: str,
        related_terms: Optional[List[str]] = None,
        visibility: str = "org",
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Create a taxonomic memory entry (domain-specific term with definition).

        Taxonomic memories are used for:
        - Domain terminology definitions (e.g., "deductible", "premium")
        - Query expansion (improving search relevance)
        - Semantic linking via related terms

        Args:
            domain: Domain/category for this term (e.g., "insurance", "coverage_types")
            term: The term being defined (e.g., "deductible")
            definition: Definition of the term
            related_terms: Related terms for semantic linking
            visibility: Visibility scope (default: "org" for organization-wide)
            user_id: User ID (defaults to current execution context user)
            metadata: Arbitrary key-value metadata stored on the entry

        Returns:
            Created document ID, or None if creation fails
        """
        from agent_engine_runner_shared.memory import create_taxonomic_memory

        resolved_user_id = user_id or get_current_user_id()
        if not resolved_user_id:
            raise ValueError("user_id is required to save taxonomic memory")

        return create_taxonomic_memory(
            memory_engine=self._memory_engine,
            domain=domain,
            term=term,
            definition=definition,
            org_id=self.org_id,
            user_id=resolved_user_id,
            project_id=self.project_id,
            related_terms=related_terms,
            visibility=visibility,
            metadata=metadata,
        )

    def search_taxonomic(
        self,
        query: str,
        user_id: Optional[str] = None,
        domain: Optional[str] = None,
        visibility: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search taxonomic memories by query.

        Args:
            query: Search query text (e.g., "what is a deductible")
            domain: Optional domain filter (e.g., "insurance")
            top_k: Maximum number of results

        Returns:
            List of matching terms with term, definition, and related_terms
        """
        from agent_engine_runner_shared.memory import search_taxonomic_memory

        return search_taxonomic_memory(
            memory_engine=self._memory_engine,
            query=query,
            org_id=self.org_id,
            project_id=self.project_id,
            user_id=user_id,
            domain=domain,
            visibility=visibility,
            top_k=top_k,
        )

    def get_taxonomic_term(
        self,
        domain: str,
        term: str,
        user_id: Optional[str] = None,
        visibility: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get a specific taxonomic term by domain and term name.

        Args:
            domain: Domain name (e.g., "insurance")
            term: Term name (e.g., "deductible")

        Returns:
            Dict with term details, or None if not found
        """
        from agent_engine_runner_shared.memory import get_taxonomic_term

        return get_taxonomic_term(
            memory_engine=self._memory_engine,
            org_id=self.org_id,
            project_id=self.project_id,
            domain=domain,
            term=term,
            user_id=user_id,
            visibility=visibility,
        )

    def list_taxonomic_domains(self, visibility: Optional[str] = None) -> List[str]:
        """
        List all distinct taxonomic domains for the organization.

        Returns:
            List of domain names (e.g., ["insurance", "coverage_types", "discounts"])
        """
        from agent_engine_runner_shared.memory import list_taxonomic_domains

        return list_taxonomic_domains(
            memory_engine=self._memory_engine,
            org_id=self.org_id,
            project_id=self.project_id,
            visibility=visibility,
        )

    # =========================================================================
    # Episodic Memory API
    # =========================================================================

    def save_episode(
        self,
        title: str,
        content: str,
        user_id: Optional[str] = None,
        summary: Optional[str] = None,
        session_id: Optional[str] = None,
        participants: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        visibility: str = "private",
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Save an episodic memory (conversation summary or significant event).

        Episodic memories are used for:
        - Storing conversation summaries for future recall
        - Recording significant events/interactions
        - Enabling context retrieval across sessions ("last time we discussed...")

        Args:
            title: Title/label for this episode (e.g., "Customer inquiry about coverage")
            content: Full content of the episode
            user_id: User ID who owns this memory (required)
            summary: Brief summary for search/display (defaults to title)
            session_id: Session ID this episode originated from
            participants: List of participants (e.g., ["Customer", "Alex"])
            tags: Tags for categorization (e.g., ["quote", "auto_insurance"])
            visibility: Visibility scope (default: "private")
            metadata: Optional caller-supplied metadata dictionary stored with
                the episode (mirrors save_semantic's metadata)

        Returns:
            Created document ID, or None if creation fails
        """
        from agent_engine_runner_shared.memory import create_episodic_memory

        resolved_user_id = user_id or get_current_user_id()
        if not resolved_user_id:
            raise ValueError("user_id is required to save episodic memory")

        return create_episodic_memory(
            memory_engine=self._memory_engine,
            title=title,
            content=content,
            org_id=self.org_id,
            user_id=resolved_user_id,
            session_id=session_id,
            summary_text=summary,
            participants=participants,
            tags=tags,
            visibility=visibility,
            project_id=self.project_id,
            agent_id=agent_id,
            metadata=metadata,
        )

    def search_episodes(
        self,
        query: str,
        user_id: Optional[str] = None,
        visibility: Optional[str] = None,
        session_id: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search episodic memories by query.

        Args:
            query: Search query text (e.g., "teenage driver discussion")
            user_id: Optional user ID filter
            session_id: Optional session ID filter
            top_k: Maximum number of results

        Returns:
            List of matching episodes with title, content, summary
        """
        from agent_engine_runner_shared.memory import search_episodic_memory

        return search_episodic_memory(
            memory_engine=self._memory_engine,
            query=query,
            org_id=self.org_id,
            project_id=self.project_id,
            user_id=user_id,
            visibility=visibility,
            session_id=session_id,
            top_k=top_k,
        )

    def list_episodes(
        self,
        user_id: Optional[str] = None,
        visibility: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        List episodic memories with optional filters.

        Args:
            user_id: Optional user ID filter
            session_id: Optional session ID filter
            limit: Maximum number of results

        Returns:
            List of episodic memory entries
        """
        from agent_engine_runner_shared.memory import list_episodic_memories

        return list_episodic_memories(
            memory_engine=self._memory_engine,
            org_id=self.org_id,
            project_id=self.project_id,
            user_id=user_id,
            visibility=visibility,
            session_id=session_id,
            limit=limit,
        )

    # =========================================================================
    # Procedural Memory API
    # =========================================================================

    def discover_procedures(
        self,
        query: str,
        user_id: Optional[str] = None,
        *,
        visibility: Optional[str] = None,
        tags: Optional[List[str]] = None,
        top_k: int = 10,
        similarity_threshold: float = 0.0,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Discover procedures matching a query (lightweight results)."""
        if self._memory_engine is None or not self.org_id:
            return []

        try:
            kwargs: Dict[str, Any] = dict(
                query=query,
                org_id=self.org_id,
                user_id=user_id,
                visibility=visibility,
                project_id=self.project_id,
                tags=tags,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            )
            if metadata_filter is not None:
                kwargs["metadata_filter"] = metadata_filter

            return self._memory_engine.discover_procedures(**kwargs)
        except Exception as e:
            logger.warning(f"Failed to discover procedures: {e}")
            return []

    def get_procedure(
        self,
        procedure_name: str,
        *,
        user_id: Optional[str] = None,
        visibility: Optional[str] = None,
        include_deleted: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Get a full procedural memory by procedure name."""
        if self._memory_engine is None or not self.org_id:
            return None

        try:
            result = self._memory_engine.get_procedural(
                org_id=self.org_id,
                project_id=self.project_id,
                procedure=procedure_name,
                user_id=user_id,
                visibility=visibility,
                include_deleted=include_deleted,
            )
            if result is None:
                return None
            if hasattr(result, "model_dump"):
                return result.model_dump()
            if hasattr(result, "dict"):
                return result.dict()
            return result
        except Exception as e:
            logger.warning(f"Failed to get procedure '{procedure_name}': {e}")
            return None

    def save_procedure(
        self,
        *,
        procedure: str,
        description: str,
        content: str,
        user_id: Optional[str] = None,
        steps: Optional[List[Dict[str, Any]]] = None,
        resources: Optional[List[Dict[str, Any]]] = None,
        allowed_tools: Optional[List[str]] = None,
        compatibility: Optional[str] = None,
        license: Optional[str] = None,
        trigger_conditions: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        visibility: str = "private",
        agent_id: Optional[str] = None,
        extraction_source: Optional[str] = None,
        source_format: Optional[str] = None,
        source_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        update_existing: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Create or update a procedural memory."""
        if self._memory_engine is None or not self.org_id:
            return None

        resolved_user_id = user_id or get_current_user_id()
        if not resolved_user_id:
            raise ValueError("user_id is required to save procedural memory")

        try:
            resolved_project_id = self.project_id
            if update_existing:
                # Identity is org/project/owner/name. Do not filter the
                # existence check by visibility so an update can change it.
                # OE still requires user_id (or visibility) on proxied GETs.
                existing = self._memory_engine.get_procedural(
                    org_id=self.org_id,
                    project_id=resolved_project_id,
                    procedure=procedure,
                    user_id=resolved_user_id,
                )
                if existing is not None:
                    if isinstance(existing, dict):
                        existing_id = existing.get("id") or existing.get("_id")
                    else:
                        existing_id = getattr(existing, "id", None) or getattr(
                            existing, "_id", None
                        )
                    updated = self._memory_engine.update_procedural(
                        org_id=self.org_id,
                        id=str(existing_id) if existing_id is not None else None,
                        procedure=procedure,
                        description=description,
                        content=content,
                        steps=steps,
                        resources=resources,
                        allowed_tools=allowed_tools,
                        trigger_conditions=trigger_conditions,
                        tags=tags,
                        visibility=visibility,
                        project_id=resolved_project_id,
                        user_id=resolved_user_id,
                        metadata=metadata,
                    )
                    if hasattr(updated, "model_dump"):
                        return updated.model_dump()
                    if hasattr(updated, "dict"):
                        return updated.dict()
                    return updated

            created = self._memory_engine.create_procedural(
                procedure=procedure,
                description=description,
                content=content,
                org_id=self.org_id,
                user_id=resolved_user_id,
                steps=steps,
                resources=resources,
                allowed_tools=allowed_tools,
                compatibility=compatibility,
                license=license,
                trigger_conditions=trigger_conditions,
                tags=tags,
                visibility=visibility,
                project_id=resolved_project_id,
                agent_id=agent_id,
                extraction_source=extraction_source,
                source_format=source_format,
                source_path=source_path,
                metadata=metadata,
            )
            create_fallback = {
                "id": str(created.id),
                "procedure": created.procedure,
                "has_embedding": created.has_embedding,
                "acknowledged": created.acknowledged,
            }
            try:
                created_doc = self._memory_engine.get_procedural(
                    org_id=self.org_id,
                    project_id=resolved_project_id,
                    id=str(created.id),
                    user_id=resolved_user_id,
                    visibility=visibility,
                )
            except Exception as fetch_err:
                logger.warning(
                    f"Procedure '{procedure}' created (id={created.id}) but "
                    f"post-create fetch failed: {fetch_err}"
                )
                return create_fallback
            if created_doc is None:
                return create_fallback
            if hasattr(created_doc, "model_dump"):
                return created_doc.model_dump()
            if hasattr(created_doc, "dict"):
                return created_doc.dict()
            return created_doc
        except Exception as e:
            logger.warning(f"Failed to save procedure '{procedure}': {e}")
            return None

    def get_current_user_id(self) -> Optional[str]:
        """
        Get the current user_id from execution context.

        Returns:
            User ID from the current execution context, or None if not available
        """
        from agent_engine_runner_shared.context import get_current_user_id

        return get_current_user_id()

    # =========================================================================
    # Guardrails (stubs — enforcement is now handled by OE; these are kept
    # for backward compatibility with callers that have not yet been updated)
    # =========================================================================

    def has_active_guardrail_rules(self) -> bool:
        """Always returns False. Guardrails enforcement is handled by OE."""
        return False

    def validate_output(self, text: str) -> str:
        """Pass-through. Guardrails enforcement is handled by OE."""
        return text

    # =========================================================================
    # Agent-to-Agent (A2A)
    # =========================================================================

    @property
    def a2a(self) -> Optional[Any]:
        """Get an AgentToAgent client for calling other agents.

        Lazily creates the client using the OE HTTP URL and A2A token
        from the current execution context. Returns None when not in AER mode
        or when the OE did not provide an A2A token.

        The client is cached per OE URL — if the URL changes between requests
        the client is recreated.
        """
        if self.mode != RuntimeMode.AER:
            return None

        from agent_engine_runner_shared.context import get_all_custom_headers, get_current_oe_url

        oe_url = get_current_oe_url()
        if not oe_url:
            logger.debug("No OE URL in execution context — a2a not available")
            return None

        headers = get_all_custom_headers()
        token = headers.get("a2a-token", "")
        if not token:
            logger.debug("No a2a-token in custom headers — a2a not available")
            return None

        if self._a2a_client is None or self._a2a_oe_url != oe_url or self._a2a_token != token:
            from agent_engine_runner_shared.a2a import AgentToAgent

            if self._a2a_client is not None:
                self._a2a_client.close()
            self._a2a_client = AgentToAgent(
                oe_url=oe_url,
                auth_token=token,
            )
            self._a2a_oe_url = oe_url
            self._a2a_token = token
            logger.info(f"A2A client created for OE: {oe_url}")

        return self._a2a_client

    # =========================================================================
    # Tool Registration
    # =========================================================================

    def register_tool(
        self,
        name: str,
        func: Callable,
        metadata: Dict[str, Any],
    ) -> None:
        """Register a raw tool function and its metadata.

        This stores the function for Tool Pod execution and the metadata
        for routing decisions. It does NOT create any framework-specific
        tool objects — that is the responsibility of the framework SDK.

        Args:
            name: Tool name
            func: Raw Python function
            metadata: Tool metadata (is_local, network, timeout, etc.)
        """
        self._tools[name] = func
        self._tool_definitions[name] = metadata
        logger.debug(f"Registered tool: {name}")

    def get_tool_metadata(self, name: str) -> Dict[str, Any]:
        """Get metadata for a registered tool.

        Args:
            name: Tool name

        Returns:
            Tool metadata dict (is_local, network, timeout, etc.)
            or empty dict if tool not found.
        """
        return self._tool_definitions.get(name, {})

    # =========================================================================
    # AER Query Plugin
    # =========================================================================

    def register_query_plugin(self, plugin: Any) -> None:
        """Register the framework adapter's :class:`AERQueryPlugin`.

        Must be called before the AER server starts — i.e. during framework
        ``App`` initialization, before :meth:`register_and_run`, while
        ``mode == RuntimeMode.AER``. Calling more than once replaces the
        previous registration. When no plugin is registered, the AER's
        ``/query/sessions*`` routes return 501.
        """
        self._query_plugin = plugin

    def get_query_plugin(self) -> Optional[Any]:
        """Return the registered :class:`AERQueryPlugin`, or ``None`` if
        none has been registered. Used by the AER routes to decide whether
        to delegate (plugin present) or surface 501 (plugin absent)."""
        return self._query_plugin

    def note_checkpoint_wire_workspace_id(self, wire_workspace_id: str | None) -> None:
        """Remember the wire ``workspace_id`` from ``/execute`` for query reads.

        Query routes carry no workspace identifier; when ``APP_ID`` is unset
        the read path falls back to the most recently observed wire value so
        it scopes checkpoints the same way the write path does.
        """
        if wire_workspace_id:
            self._checkpoint_wire_workspace_id = wire_workspace_id

    def get_checkpoint_workspace_id(self) -> str:
        """Resolved workspace scope for checkpoint reads (matches write path).

        Returns ``""`` only for intentionally unscoped local runtimes.
        Managed AERs carry ``REQUIRE_PROJECT_SCOPED_DB`` and require
        ``APP_ID``; they deliberately reject the wire fallback if it is
        missing.
        """
        from agent_engine_runner_shared.checkpoint_workspace import resolve_checkpoint_workspace_id

        return resolve_checkpoint_workspace_id(self._checkpoint_wire_workspace_id) or ""

    # =========================================================================
    # Run
    # =========================================================================

    def register_and_run(self, graph_builder: Optional[Callable] = None, **kwargs) -> None:
        """
        Register the graph builder and start the runtime.

        Behavior depends on RUNNER_MODE:
        - aer: Stores graph builder, starts AER server
        - tool: Starts Tool server (no graph needed)

        Args:
            graph_builder: Function that returns a compiled LangGraph
            **kwargs: Runtime startup options. Supported key is
                ``log_level``. Deprecated ``host``, ``http_port``, and
                ``grpc_port`` keys are still accepted for backwards
                compatibility but are ignored.
        """
        # Store graph builder (AER will call it when /execute is received)
        if graph_builder:
            self._graph_builder = graph_builder
            logger.info("Graph builder registered")

        _maybe_start_debugger()

        # Run the server
        asyncio.run(self._run_async(**self._normalize_run_kwargs(kwargs)))

    def _normalize_run_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate public run kwargs and drop deprecated listener overrides."""
        supported_kwargs = ("log_level",)
        deprecated_kwargs = ("host", "http_port", "grpc_port")
        unexpected_kwargs = sorted(set(kwargs) - set(supported_kwargs) - set(deprecated_kwargs))
        if unexpected_kwargs:
            if len(unexpected_kwargs) == 1:
                raise TypeError(
                    "register_and_run() got an unexpected keyword argument "
                    f"{unexpected_kwargs[0]!r}"
                )

            formatted_kwargs = ", ".join(repr(key) for key in unexpected_kwargs)
            raise TypeError(
                f"register_and_run() got unexpected keyword arguments: {formatted_kwargs}"
            )

        if kwargs.get("host") is not None:
            logger.warning(
                "register_and_run(host=...) is deprecated and ignored; "
                "the bind address is read from APP_HOST (default 0.0.0.0). "
                "ECP stamps APP_HOST per executor type at deploy time."
            )

        if kwargs.get("http_port") is not None:
            logger.warning(
                "register_and_run(http_port=...) is deprecated and ignored; "
                "agent-engine-runner-shared always binds HTTP to the default port for RUNNER_MODE"
            )

        return {key: kwargs[key] for key in supported_kwargs if key in kwargs}

    async def _run_async(
        self,
        log_level: Optional[str] = None,
    ) -> None:
        """Run the appropriate server based on mode."""
        from agent_engine_runner_shared.server import AERServer, ToolFunctionRunner, ToolServer

        log_level = log_level or get_env("LOG_LEVEL", "info").lower()

        # Function mode is not a server: it runs one tool and returns, binding
        # no listener, so it has no host/port. It reuses tool execution via the
        # ToolExecution mixin, not by being a ToolServer.
        if self.mode == RuntimeMode.TOOL_FUNCTION:
            await ToolFunctionRunner(self).run()
            return

        servers = {
            RuntimeMode.AER: AERServer,
            RuntimeMode.TOOL: ToolServer,
        }

        server_class = servers[self.mode]
        server = server_class(self)

        # APP_PORT lets the platform shift the listener (e.g. into the
        # 32K+ range required for Firecracker microVM ingress). Falls
        # back to the per-mode default when unset.
        port = get_env_int("APP_PORT", server.default_port)

        host = _resolve_listen_host()
        # Log the resolved host so post-deploy verification doesn't have
        # to infer from uvicorn's own log line — vital in vm-mode where
        # exec into the VM is unavailable.
        logger.info("listener bind resolved: host=%r port=%d", host, port)
        await server.run(host, port, log_level)
