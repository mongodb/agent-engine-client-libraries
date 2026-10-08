"""The application surface for native OpenAI Agents on Atlas Agent Engine."""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from typing import Any, cast

from agent_engine_sdk import BaseApp, ToolDefinition
from agents import Agent, FunctionTool, Model, ModelSettings, function_tool
from agents.function_schema import FuncSchema

from agent_engine_runner_shared import RuntimeAgentConfig, RuntimeMode, TenantRuntime
from agent_engine_runner_shared.context import customer_origin_scope
from agent_engine_runner_shared.context import (
    get_current_user_id as _runner_get_current_user_id,
)
from agent_engine_runner_shared.hooks import (
    entrypoint_scope,
    register_llm,
    register_llm_adapter_factory,
    register_workflow_adapter,
    reset_llm_registry,
    snapshot_llm_registry,
)
from agent_engine_runner_shared.utils import get_env_bool
from agent_engine_sdk_openai_agents.agent import OpenAIAgentsBaseAgent
from agent_engine_sdk_openai_agents.llm_adapter import OpenAIAgentsLLM, RegisteredModel
from agent_engine_sdk_openai_agents.secure_model import SecureModel
from agent_engine_sdk_openai_agents.tools import (
    json_result,
    secure_function_tool,
    tool_schema,
)

__all__ = ["App"]

_FRAMEWORK = "openai-agents"


class App(BaseApp):
    """Run a native OpenAI Agents ``Agent`` with OE as its durable authority.

    Example:

    ```python
    from agents import Agent, ModelSettings
    from agent_engine_sdk_openai_agents import App

    app = App(app_name="support")

    @app.tool()
    def lookup_order(order_id: str) -> str:
        \"\"\"Look up an order.\"\"\"
        ...

    @app.entrypoint
    def build() -> Agent:
        return Agent(
            name="support",
            instructions="Help with orders.",
            model=app.llm("gpt-4.1", settings=ModelSettings(temperature=0)),
            tools=app.tools(),
        )
    ```
    """

    def __init__(
        self,
        app_name: str,
        app_version: str = "1.0.0",
        mongodb_uri: str | None = None,
        database_name: str | None = None,
    ) -> None:
        super().__init__(name=app_name)
        self._runtime = TenantRuntime(
            app_name=app_name,
            app_version=app_version,
            mongodb_uri=mongodb_uri,
            database_name=database_name,
        )
        # User code: nothing enforces that the entrypoint returns an Agent.
        self._builder: Callable[[], object] | None = None
        self._tool_defs: list[ToolDefinition] = []
        self._tool_schemas: dict[str, FuncSchema] = {}
        self._tool_approvals: dict[str, bool] = {}

    @property
    def agent_config(self) -> RuntimeAgentConfig:
        return self._runtime.agent_config

    def get_current_user_id(self) -> str | None:
        """Return the user id of the request currently executing, if any."""
        return _runner_get_current_user_id()

    def get_tool_definitions(self) -> list[ToolDefinition]:
        return list(self._tool_defs)

    def tool(
        self,
        is_local: bool = True,
        *,
        network: list[str] | None = None,
        timeout: int = 30,
        redact_fields: list[str] | None = None,
        needs_approval: bool = False,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register a function tool and its platform execution policy.

        ``needs_approval`` is the SDK's ``function_tool(needs_approval=...)``,
        passed through to the ``FunctionTool`` that ``tools()`` returns: the
        turn pauses before each call until a caller approves or rejects it.
        """

        def register(fn: Callable[..., Any]) -> Callable[..., Any]:
            schema = tool_schema(fn)
            if schema.name in self._tool_schemas:
                raise ValueError(f"tool {schema.name!r} is already registered")
            self._tool_schemas[schema.name] = schema
            self._tool_approvals[schema.name] = needs_approval
            run = json_result(fn)
            description = schema.description or ""
            self._runtime.register_tool(  # type: ignore[attr-defined]
                name=schema.name,
                func=run,
                metadata={
                    "name": schema.name,
                    "description": description,
                    "is_local": is_local,
                    "network": list(network or []),
                    "timeout_seconds": timeout,
                    "redact_fields": list(redact_fields or []),
                },
            )
            self._tool_defs.append(
                ToolDefinition(
                    name=schema.name,
                    description=description,
                    args_schema=schema.params_json_schema,
                    callable=run,
                    remote=not is_local,
                    network=list(network or []),
                    timeout_seconds=timeout,
                    redact_fields=list(redact_fields or []),
                )
            )
            return fn

        return register

    def tools(self) -> list[FunctionTool]:
        """Return the registered tools as native ``FunctionTool`` objects."""
        if self._runtime.mode != RuntimeMode.AER:
            # The Tool Pod builds the agent only to find its models; it runs
            # registered tool functions itself.
            return [function_tool(tool.callable) for tool in self._tool_defs]
        allow_direct = get_env_bool("RUNNER_ALLOW_DIRECT_TOOL_EXECUTION", False)
        return [
            secure_function_tool(
                tool.callable,
                self._tool_schemas[tool.name],
                is_local=not tool.remote,
                redact_fields=tool.redact_fields,
                allow_direct=allow_direct,
                issuer=self,
                needs_approval=self._tool_approvals[tool.name],
            )
            for tool in self._tool_defs
        ]

    def llm(
        self,
        model: str | Model,
        *,
        settings: ModelSettings | None = None,
        llm_id: str | None = None,
    ) -> str | Model:
        """Register the agent's model; call it inside the entrypoint.

        ``settings`` are provider settings such as ``temperature``; the Tool
        Pod applies them where the model runs. Pass a configured
        ``agents.Model`` to use a custom OpenAI client.
        """
        resolved_id = llm_id or "__default__"
        register_llm(resolved_id, RegisteredModel(model, settings or ModelSettings()))
        if self._runtime.mode != RuntimeMode.AER:
            return model
        name = model if isinstance(model, str) else str(getattr(model, "model", ""))
        return SecureModel(llm_id=resolved_id, model_name=name or type(model).__name__)

    def entrypoint(self, fn: Callable[[], object]) -> Callable[[], object]:
        """Mark the function that builds the native ``Agent``."""
        self._builder = fn
        return fn

    def get_agent(self, callbacks: list[Any] | None = None) -> OpenAIAgentsBaseAgent:
        """Build the agent for one AER invocation or for Tool Pod model discovery."""
        # The AER's node-logging callbacks are not attached to OpenAI Agents
        # runs yet; as in the ADK adapter, they are accepted and ignored.
        del callbacks
        builder = self._require_builder()
        self._register_hooks()
        reset_llm_registry()
        with entrypoint_scope(), customer_origin_scope():
            built = builder()
        if not isinstance(built, Agent):
            raise TypeError("@app.entrypoint must return an agents.Agent")
        return OpenAIAgentsBaseAgent(
            cast(Agent[Any], built),
            app_name=self.name,
            registered_llms=snapshot_llm_registry(),
            tool_issuer=self,
        )

    def run(self, **kwargs: Any) -> None:
        """Start the agent service."""
        self._require_builder()
        self._register_hooks()
        self._runtime.register_and_run(self, **kwargs)  # type: ignore[arg-type]

    def _require_builder(self) -> Callable[[], object]:
        if self._builder is None:
            raise RuntimeError(
                "No @app.entrypoint registered. Decorate the function that "
                "builds your agents.Agent."
            )
        return self._builder

    @staticmethod
    def _register_hooks() -> None:
        register_workflow_adapter(_FRAMEWORK, _adapter_version())
        register_llm_adapter_factory(OpenAIAgentsLLM)


def _adapter_version() -> str:
    try:
        return version("agent-engine-sdk-openai-agents")
    except PackageNotFoundError:
        return "0.0.0"
