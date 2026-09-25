"""
Base server class for Runner SDK components.

Provides common functionality for all server types:
- Health check endpoint
- Startup/shutdown hooks
- FastAPI app setup
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from copy import deepcopy
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from uvicorn.config import LOGGING_CONFIG as UVICORN_LOGGING_CONFIG

from agent_engine_runner_shared.models import HealthResponse, HealthStatus
from agent_engine_runner_shared.server.auth import register_auth_middleware
from agent_engine_runner_shared.server.call_interrupt import register_call_interrupt_route
from agent_engine_runner_shared.server.cors import resolve_cors_policy
from agent_engine_runner_shared.server.drain import DrainRegistry, register_drain_route

if TYPE_CHECKING:
    from agent_engine_runner_shared.runtime import TenantRuntime

logger = logging.getLogger(__name__)

_SHUTDOWN_GRACE_PERIOD_ENV = "SHUTDOWN_GRACE_PERIOD_MS"

_APP_TITLES: dict[str, str] = {
    "aer": "Agent Execution Runtime (AER)",
    "tool": "Tool Executor Pod",
}

_APP_DESCRIPTIONS: dict[str, str] = {
    "aer": (
        "Agent Execution Runtime HTTP API for the MongoDB Atlas Agent Engine. "
        "Executes agents via the BaseAgent protocol with SecureToolWrapper "
        "for audited tool and LLM routing through the Orchestration Engine."
    ),
    "tool": (
        "Tool Executor Pod HTTP API for the MongoDB Atlas Agent Engine. "
        "Runs registered tool functions and LLM invocations in isolated pods "
        "with dedicated network and secret access."
    ),
}


class RunnerAccessLogFilter(logging.Filter):
    """Reduce noise from low-value access logs while keeping normal traffic visible."""

    SUPPRESSED_PATHS = {"/health", "/healthz", "/readyz"}
    POLLING_PATHS = {"/query/execution-logs", "/query/node-executions"}

    @classmethod
    def _extract_path(cls, record: logging.LogRecord) -> Optional[str]:
        args = record.args if isinstance(record.args, tuple) else ()
        if len(args) >= 3:
            # Uvicorn currently stores the raw request target in args[2].
            # Fall back to message parsing below in case that format changes.
            return str(args[2]).split("?", 1)[0]

        message = record.getMessage()
        for path in cls.SUPPRESSED_PATHS | cls.POLLING_PATHS:
            if f'"GET {path} HTTP/' in message or f'"HEAD {path} HTTP/' in message:
                return path

        return None

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != "uvicorn.access":
            return True

        path = self._extract_path(record)
        if path in self.SUPPRESSED_PATHS | self.POLLING_PATHS:
            return False

        return True


class MaxInfoLogFilter(logging.Filter):
    """Drop WARNING and above so the stdout handler carries only informational logs.

    Uvicorn funnels every ``uvicorn``/``uvicorn.error`` record through a single
    handler. Pairing this filter on the stdout handler with a sibling stderr
    handler (level ``WARNING``) keeps informational logs on stdout while leaving
    warnings and errors on stderr.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno < logging.WARNING


def _shutdown_grace_period_seconds() -> Optional[int]:
    """Return the platform-owned Uvicorn drain limit in whole seconds."""
    raw = os.environ.get(_SHUTDOWN_GRACE_PERIOD_ENV)
    if not raw:
        return None
    try:
        milliseconds = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{_SHUTDOWN_GRACE_PERIOD_ENV} must be a positive integer in milliseconds"
        ) from exc
    if milliseconds <= 0:
        raise ValueError(f"{_SHUTDOWN_GRACE_PERIOD_ENV} must be a positive integer in milliseconds")
    return (milliseconds + 999) // 1000


def build_uvicorn_log_config(*, structured: Optional[bool] = None) -> Dict[str, Any]:
    """Clone Uvicorn's logging config and suppress low-value access logs.

    Args:
        structured: Route uvicorn records into the structured JSON pipeline on
            the root logger instead of uvicorn's own plain-text handlers.
            Defaults to the ``STRUCTURED_LOGGING`` check ``setup_logging`` uses.
    """
    if structured is None:
        structured = os.environ.get("STRUCTURED_LOGGING", "").strip().lower() == "true"

    log_config = deepcopy(UVICORN_LOGGING_CONFIG)
    filters = log_config.setdefault("filters", {})
    filters["suppress_runner_access_noise"] = {
        "()": "agent_engine_runner_shared.server.base.RunnerAccessLogFilter",
    }
    loggers = log_config.setdefault("loggers", {})

    if structured:
        # uvicorn's loggers don't propagate and its handler writes plain text
        # into the WARNING-pinned stderr capture, so a route exception arrived
        # as WARNING with no traceback. Propagating to root preserves levelname
        # and exc_info; LOG_LEVEL gating is unaffected.
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
            loggers[name] = {**loggers.get(name, {}), "handlers": [], "propagate": True}
        # The filter normally lives on the "access" handler just removed;
        # logger filters still run before propagation.
        access_logger_cfg = loggers.setdefault("uvicorn.access", {})
        access_logger_cfg["filters"] = [
            *access_logger_cfg.get("filters", []),
            "suppress_runner_access_noise",
        ]
        return log_config

    access_handler = log_config.get("handlers", {}).get("access")
    if access_handler is not None:
        access_handler["filters"] = [
            *access_handler.get("filters", []),
            "suppress_runner_access_noise",
        ]

    # Uvicorn funnels uvicorn + uvicorn.error through one stderr "default"
    # handler, so informational logs like "Application startup complete." are
    # misread as errors. Split that handler by level: informational records go
    # to stdout, while WARNING+ stays on stderr so log-based alerting keeps a
    # clean error signal.
    filters["max_info_level"] = {
        "()": "agent_engine_runner_shared.server.base.MaxInfoLogFilter",
    }
    handlers = log_config.setdefault("handlers", {})
    default_handler = handlers.get("default")
    if default_handler is not None:
        default_handler["stream"] = "ext://sys.stdout"
        default_handler["filters"] = [
            *default_handler.get("filters", []),
            "max_info_level",
        ]
        handlers["default_stderr"] = {
            "formatter": default_handler.get("formatter", "default"),
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
            "level": "WARNING",
        }
        for logger_cfg in log_config.get("loggers", {}).values():
            if "default" in logger_cfg.get("handlers", []):
                logger_cfg["handlers"] = [*logger_cfg["handlers"], "default_stderr"]

    return log_config


class BaseServer(ABC):
    """
    Base server class for all runtime modes.

    Subclasses implement:
    - mode_name: Name of the mode (aer, tool)
    - default_port: Default port for this mode
    - register_routes: Register mode-specific routes
    - on_startup: Optional startup hook
    - on_shutdown: Optional shutdown hook
    - get_health_details: Optional health check details
    """

    # Default ports for each mode
    DEFAULT_PORTS = {
        "aer": 8001,
        "tool": 8002,
    }

    # Backing store for the drain_registry property; a class-level default so
    # servers built without __init__ (test harnesses) still read None first.
    _drain_registry: Optional[DrainRegistry] = None

    def __init__(self, runtime: "TenantRuntime"):
        """
        Initialize the server.

        Args:
            runtime: The TenantRuntime instance
        """
        self.runtime = runtime
        self._app: Optional[FastAPI] = None

    @property
    def drain_registry(self) -> DrainRegistry:
        """Execution-scoped drain state for POST /drain; work routes register
        against it so a cancelled execution's in-flight work can be stopped.

        Lazily created on first access: several test harnesses build servers
        without running ``__init__``, and the registry has no construction-time
        dependencies, so this keeps those paths working.
        """
        if self._drain_registry is None:
            self._drain_registry = DrainRegistry()
        return self._drain_registry

    @property
    @abstractmethod
    def mode_name(self) -> str:
        """Return the mode name (aer, tool)."""
        pass

    @property
    def default_port(self) -> int:
        """Return the default port for this mode."""
        return self.DEFAULT_PORTS.get(self.mode_name, 8000)

    @abstractmethod
    def register_routes(self, app: FastAPI) -> None:
        """Register mode-specific routes on the FastAPI app."""
        pass

    async def on_startup(self) -> None:
        """Called when server starts. Override in subclass."""
        pass

    async def on_shutdown(self) -> None:
        """Called when server shuts down. Override in subclass."""
        pass

    def get_health_details(self) -> Dict[str, Any]:
        """Return additional health check details. Override in subclass."""
        return {}

    def _create_lifespan(self) -> Any:
        """Create a lifespan context manager for the FastAPI app."""
        server = self

        @asynccontextmanager
        async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
            # Startup
            logger.info(f"Starting {server.mode_name} server...")
            await server.on_startup()
            logger.info(f"{server.mode_name.title()} server started")
            yield
            # Shutdown
            logger.info(f"Shutting down {server.mode_name} server...")
            await server.on_shutdown()
            logger.info(f"{server.mode_name.title()} server stopped")

        return lifespan

    def create_app(self) -> FastAPI:
        """Create and configure the FastAPI application."""
        from agent_engine_runner_shared.utils import get_env

        app = FastAPI(
            title=_APP_TITLES.get(self.mode_name, f"Runner SDK - {self.mode_name.title()}"),
            description=_APP_DESCRIPTIONS.get(
                self.mode_name, f"{self.mode_name.title()} server for Runner SDK"
            ),
            version="1.0.0",
            lifespan=self._create_lifespan(),
        )

        # Add CORS middleware (configurable via environment variable).
        # Defaults to deny-all; see agent_engine_runner_shared.server.cors.
        cors_policy = resolve_cors_policy(get_env("CORS_ALLOWED_ORIGINS", ""))
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_policy.allow_origins,
            allow_credentials=cors_policy.allow_credentials,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Registered before the routes so it wraps every one of them,
        # including those added by register_routes below.
        register_auth_middleware(app, self.mode_name)

        # Register common routes
        self._register_common_routes(app)

        # Register mode-specific routes
        self.register_routes(app)

        self._app = app
        return app

    def _register_common_routes(self, app: FastAPI) -> None:
        """Register routes common to all modes."""
        from datetime import datetime, timezone

        from agent_engine_runner_shared.metrics import Metrics

        server = self

        register_drain_route(app, server.drain_registry)
        register_call_interrupt_route(app, server.drain_registry)

        @app.get("/health", response_model=HealthResponse)
        async def health_check() -> HealthResponse:
            """Check component health and readiness.

            Returns the component's health status, runtime mode, version, and
            mode-specific details (e.g. registered tools, graph builder state).
            Used by Kubernetes liveness and readiness probes.

            Probes check the HTTP status code (2xx = pass), not the body, so a
            ``degraded`` status surfaces the state for observability without
            triggering a pod restart — e.g. the trace store being unreachable
            at startup while the pod keeps serving traffic.
            """
            try:
                from agent_engine_runner_shared.tracing import tracing_status

                tracing = tracing_status()
            except ImportError:
                # The tracing extra is optional (TenantRuntime and the AER
                # execute path both tolerate its absence); without it the
                # database exporter is definitionally unconfigured — the pod
                # is healthy, not degraded.
                tracing = {"database_exporter": "disabled"}
            status = (
                HealthStatus.DEGRADED
                if tracing.get("database_exporter") == "degraded"
                else HealthStatus.HEALTHY
            )
            return HealthResponse(
                status=status,
                component=server.runtime.app_name,
                mode=server.mode_name,
                version="1.0.0",
                details={**server.get_health_details(), "tracing": tracing},
            )

        @app.get("/")
        async def root() -> Dict[str, str]:
            """Return basic service identity and status."""
            return {
                "service": server.runtime.app_name,
                "mode": server.mode_name,
                "status": "running",
            }

        @app.get("/metrics")
        async def get_metrics() -> Dict[str, Any]:
            """Return collected execution metrics for this component.

            Includes per-operation timing, error counts, and invocation totals
            aggregated since the last pod restart.
            """
            return {
                "status": "ok",
                "component": server.mode_name,
                "metrics": Metrics.get_all(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def run(
        self,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        log_level: str = "info",
    ) -> None:
        """
        Run the server.

        Args:
            host: Host to bind to. The launch path in
                ``agent_engine_runner_shared.runtime`` auto-detects the host
                (``"::"`` on IPv6 dual-stack, ``"0.0.0.0"`` otherwise)
                and passes it here, so direct callers usually do not
                need to override. Pass the bare IP literal — uvicorn's
                ``getaddrinfo`` does not strip URI brackets on every
                platform.
            port: Port to bind to (defaults to mode default)
            log_level: Logging level
        """
        port = port or self.default_port
        app = self.create_app()

        logger.info(f"Starting {self.mode_name} server on {host}:{port}")

        config = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            log_level=log_level,
            log_config=build_uvicorn_log_config(),
            timeout_graceful_shutdown=(
                _shutdown_grace_period_seconds() if self.mode_name == "aer" else None
            ),
        )
        server = uvicorn.Server(config)
        await server.serve()
