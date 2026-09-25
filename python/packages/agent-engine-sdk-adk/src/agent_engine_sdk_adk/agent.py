"""Durable Google ADK adapter over the framework-neutral BaseAgent protocol.

``agent.py`` is the orchestrator. Durable setup lives on ``DurableSession``.
Suspend and resume: see the package README.

```
agent.py
├── execution_session.py   durable run (attempt, original input, Runner)
├── platform_session.py    scratch BaseSessionService + OE snapshot
├── stream.py              tokens / suspend / result
├── suspend.py             native ADK wait shapes
└── workflow.py            OE finalize_step / complete
```
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from agent_engine_sdk import (
    AgentInput,
    AgentOutput,
    BaseAgent,
    ExecutionResult,
    RequestContext,
    StreamEvent,
)

from agent_engine_runner_shared.execution import AgentExecutionResult
from agent_engine_sdk_adk.errors import UnsupportedDurableADKError
from agent_engine_sdk_adk.runner import DurableADKRunner
from agent_engine_sdk_adk.stream import stream_invocation, terminal_from_stream_event
from agent_engine_sdk_adk.suspend import invoke_output
from agent_engine_sdk_adk.workflow import complete_durable_execution

if TYPE_CHECKING:
    from google.adk.agents import BaseAgent as ADKAgent
    from google.adk.workflow import BaseNode as ADKNode

__all__ = ["ADKBaseAgent", "UnsupportedDurableADKError"]


class ADKBaseAgent(BaseAgent):
    """Run one ADK 2 agent with OE as its only durable authority."""

    def __init__(
        self,
        adk_agent: ADKAgent | ADKNode,
        app_name: str | None = None,
        *,
        runner: DurableADKRunner | None = None,
    ) -> None:
        self._adk_agent = adk_agent
        if runner is None:
            if not app_name:
                raise ValueError("ADKBaseAgent requires app_name or runner")
            runner = DurableADKRunner(app_name=app_name)
        self.runner = runner

    def execute(self, ctx: RequestContext, input: AgentInput) -> ExecutionResult:
        return AgentExecutionResult(
            invoke=lambda: self.invoke(ctx, input),
            stream=lambda: self.stream(ctx, input),
        )

    async def invoke(self, ctx: RequestContext, input: AgentInput) -> AgentOutput:
        result_text = ""
        suspend = None
        async for event in self.stream(ctx, input):
            result_text, suspend = terminal_from_stream_event(
                event, result_text=result_text, suspend=suspend
            )
        return invoke_output(result_text=result_text, suspend=suspend)

    async def stream(
        self, ctx: RequestContext, input: AgentInput
    ) -> AsyncIterator[StreamEvent]:
        session = self.runner._execution_session(  # pyright: ignore[reportPrivateUsage]
            self._adk_agent,
            ctx=ctx,
        )
        async for event in stream_invocation(
            session=session,
            input=input,
            complete_execution=complete_durable_execution,
        ):
            yield event
