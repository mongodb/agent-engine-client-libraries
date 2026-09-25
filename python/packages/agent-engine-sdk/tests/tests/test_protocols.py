"""Tests for agent_engine_sdk.interfaces."""

from collections.abc import AsyncIterator

from agent_engine_sdk.interfaces import (
    BaseAgent,
    BaseExecutionCallback,
    BaseLLM,
    ExecutionResult,
    NullExecutionCallback,
)
from agent_engine_sdk.models import (
    AgentInput,
    AgentOutput,
    LLMResponse,
    LLMStreamChunk,
    Message,
    RequestContext,
    StreamEvent,
)


class _FakeRunResult:
    """Concrete RunResult satisfying the protocol."""

    def __init__(self, invoke, stream):
        self._invoke = invoke
        self._stream = stream

    def __await__(self):
        return self._invoke().__await__()

    async def __aiter__(self):
        async for event in self._stream():
            yield event


class _FakeAgent:
    """Agent that implements the BaseAgent protocol."""

    def execute(self, ctx: RequestContext, input: AgentInput) -> ExecutionResult:
        async def _invoke():
            return AgentOutput(response={"response": "ok"})

        async def _stream():
            yield StreamEvent(data={"content": "hello"}, event="token")
            yield StreamEvent(data={"content": "done"}, event="done")

        return _FakeRunResult(invoke=_invoke, stream=_stream)


class _FakeLLM:
    def invoke(self, messages: list[Message], **kwargs: object) -> LLMResponse:
        return LLMResponse(content="hi")

    async def ainvoke(self, messages: list[Message], **kwargs: object) -> LLMResponse:
        return LLMResponse(content="hi")

    async def astream(
        self, messages: list[Message], **kwargs: object
    ) -> AsyncIterator[LLMStreamChunk]:
        yield LLMStreamChunk(content="hi")


class _FakeExecutionCallback:
    """Minimal three-method implementation of BaseExecutionCallback.

    Intentionally omits on_node_suspend to verify that the Protocol only
    requires the three core methods — i.e. existing callbacks that predate
    on_node_suspend still satisfy the contract.
    """

    def on_node_start(
        self,
        node_name: str,
        inputs: dict[str, object],
        *,
        run_id: str,
        parent_run_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        pass

    def on_node_end(
        self,
        node_name: str,
        outputs: dict[str, object],
        *,
        run_id: str,
        parent_run_id: str | None = None,
        duration_ms: float | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        pass

    def on_node_error(
        self,
        node_name: str,
        error: str,
        *,
        run_id: str,
        parent_run_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        pass


class _NotAnAgent:
    pass


class _NotAnLLM:
    pass


class TestBaseAgent:
    def test_conforming_instance(self) -> None:
        assert isinstance(_FakeAgent(), BaseAgent)

    def test_non_conforming_instance(self) -> None:
        assert not isinstance(_NotAnAgent(), BaseAgent)


class TestBaseLLM:
    def test_conforming_instance(self) -> None:
        assert isinstance(_FakeLLM(), BaseLLM)

    def test_non_conforming_instance(self) -> None:
        assert not isinstance(_NotAnLLM(), BaseLLM)


class TestBaseExecutionCallback:
    def test_conforming_instance(self) -> None:
        assert isinstance(_FakeExecutionCallback(), BaseExecutionCallback)

    def test_non_conforming_instance(self) -> None:
        assert not isinstance(_NotAnAgent(), BaseExecutionCallback)


class TestNullExecutionCallback:
    def test_satisfies_protocol(self) -> None:
        assert isinstance(NullExecutionCallback(), BaseExecutionCallback)

    def test_subclass_satisfies_protocol(self) -> None:
        class Partial(NullExecutionCallback):
            def on_node_start(self, node_name, inputs, *, run_id, **kwargs) -> None:  # type: ignore[override]
                pass

        assert isinstance(Partial(), BaseExecutionCallback)

    def test_all_methods_are_callable(self) -> None:
        cb = NullExecutionCallback()
        cb.on_node_start("n", {}, run_id="r")
        cb.on_node_end("n", {}, run_id="r")
        cb.on_node_error("n", "err", run_id="r")
        cb.on_node_suspend("n", run_id="r")
