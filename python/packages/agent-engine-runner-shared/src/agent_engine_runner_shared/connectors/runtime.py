"""Tool Pod callables backed by a validated connector bundle."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping

from agent_engine_runner_shared.connectors.bundle import BundledTool, ConnectorBundle
from agent_engine_runner_shared.connectors.executor import ConnectorExecutor

__all__ = ["ConnectorToolRegistration", "ConnectorToolRuntime"]


_current_connector_env: ContextVar[Mapping[str, str] | None] = ContextVar(
    "current_connector_env", default=None
)


@contextmanager
def connector_environment(env: Mapping[str, str]) -> Iterator[None]:
    """Bind the tenant environment captured for one connector invocation."""
    token = _current_connector_env.set(dict(env))
    try:
        yield
    finally:
        _current_connector_env.reset(token)


@dataclass(frozen=True)
class ConnectorToolRegistration:
    """One callable and its ordinary Tool Pod registry metadata."""

    tool: BundledTool
    callable: Callable[..., str]

    @property
    def metadata(self) -> dict[str, Any]:
        return self.tool.registration_metadata


class ConnectorToolRuntime:
    """Own pooled connector executors for one Tool Pod process."""

    def __init__(self, bundle: ConnectorBundle) -> None:
        executors: dict[str, ConnectorExecutor] = {}
        try:
            for connector in bundle.connectors:
                executors[connector.name] = ConnectorExecutor(connector.definitions)
        except Exception:
            for executor in executors.values():
                executor.close()
            raise
        self._executors = executors
        self._registrations = tuple(
            ConnectorToolRegistration(
                tool=tool,
                callable=self._make_callable(executors[tool.connector.name], tool.operation.name),
            )
            for tool in bundle.tools
        )
        self._closed = False

    @property
    def registrations(self) -> tuple[ConnectorToolRegistration, ...]:
        return self._registrations

    @staticmethod
    def _make_callable(executor: ConnectorExecutor, tool_name: str) -> Callable[..., str]:
        def execute(**arguments: Any) -> str:
            env = _current_connector_env.get()
            if env is None:
                raise RuntimeError("connector invocation environment is unavailable")
            return executor.execute(tool_name, arguments, env=env)

        execute.__name__ = tool_name
        return execute

    def close(self) -> None:
        """Release every connector's pooled HTTP transport once."""
        if self._closed:
            return
        self._closed = True
        for executor in self._executors.values():
            executor.close()
