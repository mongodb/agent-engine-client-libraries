"""Native ADK control events and guardrail review continuation."""

from __future__ import annotations

from collections.abc import Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from inspect import isawaitable
from typing import Any, cast

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.events import Event
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.adk.workflow import BaseNode
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field

from agent_engine_runner_shared.models import GuardrailReviewHalt
from agent_engine_runner_shared.secure_llm_proxy import GuardrailReviewPending
from agent_engine_runner_shared.secure_wrapper import PolicyDeniedException
from agent_engine_runner_shared.workflow.context import (
    current_operation_path,
    is_guardrail_review_wait,
)
from agent_engine_sdk_adk.agent_route import configured_sub_agents
from agent_engine_sdk_adk.errors import UnsupportedDurableADKError
from agent_engine_sdk_adk.route import configured_adk_nodes

# ADK excludes this framework-control name before current-turn selection and
# other-agent rendering. A late provider-input filter cannot preserve history.
REVIEW_FUNCTION_NAME = "adk_framework"
REVIEW_METADATA_KEY = "agent_engine_guardrail_review"
_INSTALLED_ATTRIBUTE = "_agent_engine_guardrail_review_installed"
OperationKey = tuple[tuple[str, int], ...]


def operation_key() -> OperationKey:
    """Return the durable operation recording the current model call."""
    return tuple(
        (segment.name, segment.ordinal) for segment in current_operation_path().segments
    )


class ReviewWait(BaseModel):
    """Adapter continuation metadata from a genuine recorded OE halt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    review: GuardrailReviewHalt
    halt_step: int = Field(ge=1, description="OE operational step of the held call")
    operation: OperationKey = Field(description="Durable operation owning the call")

    @property
    def call_id(self) -> str:
        return f"guardrail-review:{self.review.review_id}"


def pending_response(
    pending: GuardrailReviewPending, llm_request: LlmRequest
) -> LlmResponse:
    """Replace withheld model output with ADK's native framework wait."""
    carrier = llm_request.tools_dict.get(REVIEW_FUNCTION_NAME)
    if not isinstance(carrier, _ReviewCarrier) or not carrier.allows_review:
        raise UnsupportedDurableADKError(
            "Google ADK output review is not supported inside parallel_worker children"
        )
    halts = _HALTS.get()
    if halts is None:
        raise UnsupportedDurableADKError("ADK review halt has no execution scope")
    halts[pending.review.review_id] = pending
    wait = ReviewWait(
        review=pending.review, halt_step=pending.step, operation=operation_key()
    )
    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name=REVIEW_FUNCTION_NAME,
                        id=wait.call_id,
                        args={"review_id": wait.review.review_id},
                    )
                )
            ],
        ),
        custom_metadata={REVIEW_METADATA_KEY: wait.model_dump(mode="json")},
        partial=False,
    )


def review_wait_from_event(event: Event) -> ReviewWait | None:
    """Authenticate a native carrier against this attempt's OE halt registry."""
    calls = event.get_function_calls()
    carriers = [call for call in calls if call.name == REVIEW_FUNCTION_NAME]
    metadata = event.custom_metadata or {}
    if REVIEW_METADATA_KEY not in metadata:
        if carriers:
            raise UnsupportedDurableADKError("ADK review carrier has no OE halt marker")
        return None
    try:
        wait = ReviewWait.model_validate(metadata[REVIEW_METADATA_KEY])
    except ValueError as error:
        raise UnsupportedDurableADKError("ADK review halt marker is invalid") from error
    if (
        len(calls) != 1
        or len(carriers) != 1
        or event.content is None
        or len(event.content.parts or []) != 1
        or carriers[0].id != wait.call_id
        or carriers[0].args != {"review_id": wait.review.review_id}
        or not is_guardrail_review_wait(wait.review.review_id)
    ):
        raise UnsupportedDurableADKError("ADK review carrier does not match an OE halt")
    return wait


class _ReviewCarrier(BaseTool):
    def __init__(
        self,
        continuation: ReviewContinuation | None = None,
        *,
        allows_review: bool = True,
    ) -> None:
        self.continuation = continuation
        self.allows_review = allows_review
        super().__init__(
            name=REVIEW_FUNCTION_NAME, description="Internal output review"
        )
        self.is_long_running = True

    async def run_async(self, *, args: dict[str, Any], tool_context: Any) -> None:
        del tool_context
        if not is_guardrail_review_wait(args.get("review_id")):
            raise UnsupportedDurableADKError(
                "ADK review carrier has no recorded OE halt"
            )
        # Returning no result leaves this long-running call unanswered. ADK
        # stops the route; the durable adapter supplies the eventual response.


def _application_tool_callback(callback: Any) -> Any:
    @wraps(callback)
    async def invoke(*args: Any, **kwargs: Any) -> Any:
        tool: BaseTool = kwargs["tool"] if "tool" in kwargs else args[0]
        if isinstance(tool, _ReviewCarrier):
            return None
        result = callback(*args, **kwargs)
        return await result if isawaitable(result) else result

    return invoke


def _application_model_callback(callback: Any) -> Any:
    @wraps(callback)
    async def invoke(*args: Any, **kwargs: Any) -> Any:
        # ADK uses the wrapped signature to adapt callback arguments to either
        # keywords or positional (callback context, LLM request) values.
        llm_request: LlmRequest = (
            kwargs["llm_request"] if "llm_request" in kwargs else args[1]
        )
        carrier = llm_request.tools_dict.get(REVIEW_FUNCTION_NAME)
        if isinstance(carrier, _ReviewCarrier) and carrier.continuation is not None:
            # A replacement response skips SecureLlm and can expose output or
            # execute tools before OE resolves the held review's decision.
            return None
        result = callback(*args, **kwargs)
        return await result if isawaitable(result) else result

    return invoke


def _application_agent_callback(callback: Any) -> Any:
    @wraps(callback)
    async def invoke(*args: Any, **kwargs: Any) -> Any:
        context: CallbackContext = (
            kwargs["callback_context"] if "callback_context" in kwargs else args[0]
        )
        if _continuation_for_context(context) is not None:
            # Agent-level replacement skips the entire model flow, including
            # the OE resolution required before this review can continue.
            return None
        result = callback(*args, **kwargs)
        return await result if isawaitable(result) else result

    return invoke


def install_review_carriers(root: BaseNode) -> None:
    """Install dispatch-only carriers without changing tools or output schemas."""
    nodes = (*configured_adk_nodes(root), *configured_sub_agents(root))
    for node in nodes:
        if not isinstance(node, LlmAgent) or getattr(node, _INSTALLED_ATTRIBUTE, False):
            continue

        async def prepare(
            *,
            callback_context: CallbackContext,
            llm_request: LlmRequest,
            _allows_review: bool = not getattr(
                node, "_agent_engine_parallel_worker", False
            ),
        ) -> None:
            if REVIEW_FUNCTION_NAME in llm_request.tools_dict:
                raise UnsupportedDurableADKError(
                    "adk_framework is reserved for ADK control"
                )
            llm_request.tools_dict[REVIEW_FUNCTION_NAME] = _ReviewCarrier(
                _continuation_for_context(callback_context),
                allows_review=_allows_review,
            )

        for hook in ("before_tool_callback", "after_tool_callback"):
            existing_tools = getattr(node, hook)
            tool_callbacks = (
                cast(list[Any], existing_tools.copy())
                if isinstance(existing_tools, list)
                else [existing_tools]
            )
            object.__setattr__(
                node,
                hook,
                [
                    _application_tool_callback(callback)
                    for callback in tool_callbacks
                    if callback is not None
                ],
            )

        async def protect_control_response(
            *, callback_context: CallbackContext, llm_response: LlmResponse
        ) -> LlmResponse | None:
            del callback_context
            if REVIEW_METADATA_KEY in (llm_response.custom_metadata or {}):
                # User callbacks see genuine model output after approval,
                # never an internal wait in place of the withheld response.
                return llm_response
            return None

        after = node.after_model_callback
        after_callbacks = (
            cast(list[Any], after.copy()) if isinstance(after, list) else [after]
        )
        object.__setattr__(
            node,
            "after_model_callback",
            [
                protect_control_response,
                *(callback for callback in after_callbacks if callback is not None),
            ],
        )

        async def propagate_terminal_error(
            *,
            callback_context: CallbackContext,
            llm_request: LlmRequest,
            error: Exception,
        ) -> None:
            del callback_context, llm_request
            if isinstance(error, (PolicyDeniedException, UnsupportedDurableADKError)):
                # Fallback output or tools must not recover an OE denial or
                # an explicit rejection of unsupported durable execution.
                raise error

        on_error = node.on_model_error_callback
        error_callbacks = (
            cast(list[Any], on_error.copy())
            if isinstance(on_error, list)
            else [on_error]
        )
        object.__setattr__(
            node,
            "on_model_error_callback",
            [
                propagate_terminal_error,
                *(callback for callback in error_callbacks if callback is not None),
            ],
        )

        before_agent = node.before_agent_callback
        agent_callbacks = (
            cast(list[Any], before_agent.copy())
            if isinstance(before_agent, list)
            else [before_agent]
        )
        object.__setattr__(
            node,
            "before_agent_callback",
            [
                _application_agent_callback(callback)
                for callback in agent_callbacks
                if callback is not None
            ],
        )

        existing = node.before_model_callback
        callbacks = (
            cast(list[Any], existing.copy())
            if isinstance(existing, list)
            else [existing]
        )
        object.__setattr__(
            node,
            "before_model_callback",
            [
                prepare,
                *(
                    _application_model_callback(callback)
                    for callback in callbacks
                    if callback is not None
                ),
            ],
        )
        object.__setattr__(node, _INSTALLED_ATTRIBUTE, True)


@dataclass(frozen=True)
class ReviewContinuation:
    wait: ReviewWait
    step: int
    pending: GuardrailReviewPending


_HALTS: ContextVar[dict[str, GuardrailReviewPending] | None] = ContextVar(
    "adk_guardrail_review_halts", default=None
)
_CONTINUATIONS: ContextVar[dict[str, ReviewContinuation] | None] = ContextVar(
    "adk_guardrail_review_continuations", default=None
)


def _continuation_for_context(context: CallbackContext) -> ReviewContinuation | None:
    pending = _CONTINUATIONS.get()
    if not pending:
        return None
    branch = context.branch
    answered_ids = {
        response.id
        for event in context.session.events
        for response in event.get_function_responses()
        if response.name == REVIEW_FUNCTION_NAME
    }
    matches = [
        pending[call.id]
        for event in context.session.events
        if event.author == context.agent_name and event.branch == branch
        for call in event.get_function_calls()
        if call.id in pending and call.id in answered_ids
    ]
    if len(matches) > 1:
        raise UnsupportedDurableADKError("ADK review continuation is ambiguous")
    return matches[0] if matches else None


def take_review_continuation(llm_request: LlmRequest) -> ReviewContinuation | None:
    """Consume the answer bound to this ADK agent and native function-call ID."""
    carrier = llm_request.tools_dict.get(REVIEW_FUNCTION_NAME)
    if not isinstance(carrier, _ReviewCarrier) or carrier.continuation is None:
        return None
    pending = _CONTINUATIONS.get()
    continuation = carrier.continuation
    if (
        pending is None
        or pending.pop(continuation.wait.call_id, None) is not continuation
    ):
        raise UnsupportedDurableADKError("ADK review continuation was already consumed")
    return continuation


@contextmanager
def review_continuation_scope(
    waits: Sequence[ReviewWait],
    *,
    allocation_count: int,
    halts: dict[str, GuardrailReviewPending],
) -> Generator[int, None, None]:
    """Reserve retry-stable resolution steps before parallel routes continue."""
    first_step = max((wait.halt_step for wait in waits), default=0) + allocation_count
    continuations = {
        wait.call_id: ReviewContinuation(
            wait=wait, step=first_step + index, pending=halts[wait.review.review_id]
        )
        for index, wait in enumerate(
            sorted(waits, key=lambda wait: wait.call_id), start=1
        )
    }
    if len(continuations) != len(waits):
        raise UnsupportedDurableADKError("ADK review frontier repeats a review ID")
    halt_token = _HALTS.set(halts)
    token = _CONTINUATIONS.set(continuations)
    try:
        yield first_step + len(waits)
        if continuations:
            raise UnsupportedDurableADKError(
                "ADK did not resolve every answered review"
            )
    finally:
        _CONTINUATIONS.reset(token)
        _HALTS.reset(halt_token)
