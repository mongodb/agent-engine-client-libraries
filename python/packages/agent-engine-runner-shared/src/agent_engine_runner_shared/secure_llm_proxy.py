"""
SecureLLMProxy - Framework-neutral proxy for secure LLM calls through OE.

The proxy packages intercepted invoke_llm requests for the Orchestration Engine
and unwraps the streamed OE relay back into sdk-core models. The OE owns
approval, routing, live SSE relay, and final audit/result recording.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from functools import partial
from typing import Any

import httpx
from agent_engine_sdk.models import (
    LLMInvocationOptions,
    LLMResponse,
    LLMStreamChunk,
    LLMTokenUsage,
    LLMToolCall,
    LLMToolSchema,
    Message,
    ToolCallChunk,
)
from httpx_sse import connect_sse
from pydantic import JsonValue

from agent_engine_runner_shared.context import get_current_user_id
from agent_engine_runner_shared.generated.workflow.v1.activity_pb2 import ACTIVITY_KIND_LLM
from agent_engine_runner_shared.models import (
    GuardrailReviewHalt,
    InvokeLLMRequestArguments,
    LLMPodStreamEvent,
    ToolExecuteResponse,
)
from agent_engine_runner_shared.secure_wrapper import (
    LLMInvocationError,
    OperationalStepAllocator,
    PolicyDeniedException,
    raise_for_oe_rejection,
    request_oe_approval,
    request_oe_approval_retryable,
)
from agent_engine_runner_shared.tls_client import create_httpx_client_with_tls
from agent_engine_runner_shared.utils import (
    LLM_READ_TIMEOUT,
    OE_DISPATCH_TAKEOVER_RETRY_DELAY_S,
    OE_RETRYABLE_MAX_ATTEMPTS,
    get_request_timeout,
    normalize_tool_call_args,
    oe_stream_retry_delay_s,
    sleep_oe_stream_retry,
)
from agent_engine_runner_shared.workflow import current_attempt_context
from agent_engine_runner_shared.workflow.activity import (
    DurableActivityDeniedError,
    run_streaming_activity,
)
from agent_engine_runner_shared.workflow.context import (
    allocate_activity_ordinal,
    is_guardrail_review_wait,
    note_guardrail_review_wait,
    preallocate_activity_ordinals,
    tool_activity_key,
)
from agent_engine_runner_shared.workflow.memory import DurableMemoryState

logger = logging.getLogger(__name__)


def _is_oe_stream_transport_error(exc: BaseException) -> bool:
    """True for a dropped SSE connection that is safe to retry on the same URL.

    User cancellation (GeneratorExit, KeyboardInterrupt, CancelledError) is
    BaseException and never reaches this helper. Idle/read timeouts stay
    terminal — they are not owner-death.
    """
    return isinstance(exc, httpx.TransportError) and not isinstance(exc, httpx.TimeoutException)


GUARDRAIL_REVIEW_PROTOCOL_VERSION = 1

# The halt recorded as an LLM activity's result, and the value a framework
# interrupt carries while the review waits. OE recognizes the wait by this key.
_REVIEW_HALT_RESULT_KEY = "guardrail_review_halt"
_REVIEW_HALT_STEP_KEY = "guardrail_review_halt_step"
GUARDRAIL_REVIEW_WAIT_KEY = "guardrail_review"


class _ReviewHalt:
    """Stands in the chunk stream for a model call OE halted for review.

    It never reaches the proxy's caller: the activity records it as the call's
    result, and the proxy pauses for the review once the activity is recorded.
    """

    def __init__(self, review: GuardrailReviewHalt, step: int):
        self.review = review
        # The step OE bound the review to. A later attempt can count fewer
        # steps before this call than the attempt that halted did, and OE only
        # resolves a review at a step after this one.
        self.step = step


def _model_chunks(chunks: Iterator[LLMStreamChunk | _ReviewHalt]) -> Iterator[LLMStreamChunk]:
    """The chunks of a stream that has no halt to pause for.

    Only the first call of a protocol-following durable activity may halt.
    """
    for chunk in chunks:
        if isinstance(chunk, _ReviewHalt):
            raise LLMInvocationError("OE halted a model call that cannot pause for review")
        yield chunk


def guardrail_review_wait_id(value: Any) -> str | None:
    """The review a pause value names, when this attempt paused for that review.

    The answer to such a pause is the platform's review decision, not part of
    the conversation. An application's own pause is never one, whatever its
    value looks like: only a review the proxy recorded a halt for counts.
    """
    named = value.get(GUARDRAIL_REVIEW_WAIT_KEY) if isinstance(value, dict) else None
    review_id = named.get("review_id") if isinstance(named, dict) else None
    return review_id if is_guardrail_review_wait(review_id) else None


def _review_halt_from_result(result: Any) -> _ReviewHalt | None:
    if not isinstance(result, dict) or _REVIEW_HALT_RESULT_KEY not in result:
        return None
    # A recorded result comes back as JSON numbers, which are floats.
    step = result.get(_REVIEW_HALT_STEP_KEY)
    return _ReviewHalt(
        GuardrailReviewHalt.model_validate(result[_REVIEW_HALT_RESULT_KEY]),
        int(step) if isinstance(step, (int, float)) else 0,
    )


class SecureLLMProxy:
    """Framework-neutral proxy for secure LLM calls through OE.

    Framework callers such as ``SecureWrappedLLM`` own client-side ``llm_call``
    metrics and consume the ``last_*`` fields exposed here after each call.
    """

    def __init__(
        self,
        oe_url: str,
        execution_id: str,
        llm_id: str = "__default__",
        model_name: str = "unknown",
        bound_tools: list[Any] | None = None,
        bound_tool_choice: Any | None = None,
        operational_steps: OperationalStepAllocator | None = None,
        stable_operation_identity: bool = False,
        durable_memory: DurableMemoryState | None = None,
        guardrail_review_protocol: bool = False,
    ):
        self.oe_url = oe_url.rstrip("/")
        self.execution_id = execution_id
        self.model_name = model_name
        self.bound_tools = bound_tools
        # Forwarded to the tool pod's bind_tools call so a forced tool choice
        # (e.g. with_structured_output) survives the OE round-trip.
        self.bound_tool_choice = bound_tool_choice
        self.llm_id = llm_id
        self.stable_operation_identity = stable_operation_identity
        self.durable_memory = durable_memory
        # Set by an adapter whose framework can pause after a model call and
        # re-run the calling code: under a durable attempt the proxy then
        # follows OE's review protocol instead of pausing inside the call.
        self.guardrail_review_protocol = guardrail_review_protocol
        # Prefer the wrapper's allocator so tool + LLM share one sequence.
        self.operational_steps = operational_steps or OperationalStepAllocator()
        self.last_duration_ms: float = 0.0
        self.last_from_cache: bool = False
        self.last_latest_step_number: int | None = None
        self.last_pod_name: str | None = None
        # Optional injected workflow client (tests); when None,
        # run_streaming_activity owns a short-lived client per activity.
        self._workflow: Any | None = None

    @property
    def step_counter(self) -> int:
        """Current operational-step watermark for compatibility readers."""
        return self.operational_steps.current()

    def _allocate_step(self, step: int | None) -> int:
        if step is None:
            return self.operational_steps.next()
        self.operational_steps.observe_at_least(step)
        return step

    def _serialize_bound_tools(self) -> list[LLMToolSchema] | None:
        """Serialize bound tools for transmission to OE/tool pod execution."""
        if not self.bound_tools:
            return None

        serialized: list[LLMToolSchema] = []
        for tool in self.bound_tools:
            if hasattr(tool, "tool_call_schema"):
                tool_call_schema = tool.tool_call_schema
                if isinstance(tool_call_schema, dict):
                    if "name" in tool_call_schema or "function" in tool_call_schema:
                        serialized.append(LLMToolSchema.model_validate(tool_call_schema))
                    else:
                        parameters = {
                            key: value
                            for key, value in tool_call_schema.items()
                            if key != "description"
                        }
                        raw_name = getattr(tool, "name", None)
                        raw_description = getattr(tool, "description", None)
                        serialized.append(
                            LLMToolSchema(
                                name=raw_name
                                if isinstance(raw_name, str)
                                else tool_call_schema.get("title"),
                                description=raw_description
                                if isinstance(raw_description, str)
                                else tool_call_schema.get("description"),
                                parameters=parameters,
                            )
                        )
                elif hasattr(tool_call_schema, "schema"):
                    serialized.append(LLMToolSchema.model_validate(tool_call_schema.schema()))
                else:
                    logger.warning(
                        "SecureLLMProxy: Skipping unrecognized tool_call_schema type %s",
                        type(tool_call_schema).__name__,
                    )
            elif isinstance(tool, (dict, LLMToolSchema)):
                serialized.append(LLMToolSchema.model_validate(tool))
            else:
                logger.warning(
                    "SecureLLMProxy: Skipping unrecognized tool type %s during serialization",
                    type(tool).__name__,
                )
        return serialized or None

    def _build_invoke_request(
        self,
        messages: list[Message],
        stop_sequences: list[str] | None = None,
        options: LLMInvocationOptions | None = None,
        stream: bool = False,
    ) -> InvokeLLMRequestArguments:
        """Build the typed invoke_llm payload OE executes."""
        tools = self._serialize_bound_tools()
        return InvokeLLMRequestArguments(
            messages=messages,
            model=self.model_name,
            llm_id=self.llm_id,
            stop_sequences=stop_sequences,
            tools=tools,
            tool_choice=self.bound_tool_choice,
            options=options,
            stream=stream,
        )

    def _request_oe_execution(
        self,
        invoke_request: InvokeLLMRequestArguments,
        step: int | None = None,
        review_id: str | None = None,
    ) -> ToolExecuteResponse:
        """Ask OE to execute invoke_llm and return the final ToolExecuteResponse."""
        step = self._allocate_step(step)
        review_fields: dict[str, Any] = {}
        if self._follows_review_protocol():
            review_fields["review_protocol"] = GUARDRAIL_REVIEW_PROTOCOL_VERSION
            if review_id is not None:
                review_fields["review_id"] = review_id

        response = request_oe_approval_retryable(
            request_oe_approval,
            oe_url=self.oe_url,
            execution_id=self.execution_id,
            tool_name="invoke_llm",
            arguments=invoke_request.model_dump(mode="json", by_alias=True, exclude_none=True),
            step=step,
            # LLM-output guardrails make /tool/execute wait for the full LLM result.
            timeout=httpx.Timeout(get_request_timeout(), read=LLM_READ_TIMEOUT),
            **review_fields,
        )

        if not response.proceed and response.result is None and response.status != "require_review":
            raise_for_oe_rejection(response.reason, guardrail_meta=response.guardrail_meta)

        self.last_duration_ms = response.duration_ms or 0.0
        self.last_from_cache = response.from_cache or response.cached_result is not None
        self.last_latest_step_number = response.latest_step_number
        self.last_pod_name = response.pod_name

        if response.latest_step_number is not None:
            self.operational_steps.observe_at_least(response.latest_step_number)

        return response

    def _follows_review_protocol(self) -> bool:
        """Whether this call pauses for a review after its activity is recorded.

        Only a durable attempt records the halt and replays it; elsewhere the
        proxy keeps pausing inside the call.
        """
        return self.guardrail_review_protocol and current_attempt_context() is not None

    def _await_review_decision(self, review: GuardrailReviewHalt) -> None:
        """Pause for the review and check the answer names it.

        The pause value is built only from the recorded halt, so it is the
        same on every replay. The answer carries the decision, but OE's record
        is what the resolving call is answered from.
        """
        from agent_engine_runner_shared.hooks import get_suspend_handler

        interrupt = get_suspend_handler()
        if interrupt is None:
            raise LLMInvocationError(
                "Guardrail require_review: no suspend handler registered. "
                "Human review is required but the agent framework has not registered "
                "a suspend handler."
            )
        note_guardrail_review_wait(review.review_id)
        answer = interrupt({GUARDRAIL_REVIEW_WAIT_KEY: {"review_id": review.review_id}})
        named = answer.get(GUARDRAIL_REVIEW_WAIT_KEY) if isinstance(answer, dict) else None
        if not isinstance(named, dict) or named.get("review_id") != review.review_id:
            raise LLMInvocationError("Guardrail review was answered for a different review")

    def _handle_require_review(self, response: ToolExecuteResponse) -> Iterator[LLMStreamChunk]:
        """Handle a guardrail require_review response by suspending via interrupt().

        On first call: interrupt() suspends the LangGraph node; execution halts here.
        On resume: LangGraph re-runs the node; interrupt() returns the OE-dispatched
        resume payload — a dict with a top-level ``guardrail_review`` key containing
        ``decision``, optional ``reviewer_notes``, and (on approve) ``pending_llm_content``.
        The caller (stream()) must ``yield from`` this method.
        """
        from agent_engine_runner_shared.hooks import get_suspend_handler

        interrupt = get_suspend_handler()
        if interrupt is None:
            raise PolicyDeniedException(
                "Guardrail require_review: no suspend handler registered. "
                "Ensure the framework SDK calls register_suspend_handler() before run()."
            )

        suspend_payload = {
            "suspend_reason": "guardrail_require_review",
            "allowed_decisions": ["approve", "deny"],
            "reason": response.reason or "Guardrail required human review",
            "guardrail_meta": response.guardrail_meta.model_dump()
            if response.guardrail_meta
            else None,
        }
        logger.info(
            "Guardrail require_review: suspending for human review",
            extra={"execution_id": self.execution_id, "step": self.step_counter},
        )

        # On first call this suspends via LangGraph interrupt() — never returns.
        # On resume, LangGraph re-runs the node and interrupt() returns the stored decision.
        human_decision = interrupt(suspend_payload)

        if not isinstance(human_decision, dict):
            raise LLMInvocationError(
                f"Guardrail require_review: expected dict from interrupt, got {type(human_decision).__name__}"
            )

        guardrail_review = human_decision.get("guardrail_review")
        if not isinstance(guardrail_review, dict):
            raise LLMInvocationError(
                "Guardrail require_review: resume data missing or invalid 'guardrail_review' key "
                f"(got keys: {list(human_decision.keys())})"
            )

        decision = guardrail_review.get("decision", "")

        if decision == "approve":
            pending_content = guardrail_review.get("pending_llm_content")
            if pending_content is None:
                logger.error(
                    "Guardrail require_review: approved but pending_llm_content missing from resume data",
                    extra={"execution_id": self.execution_id},
                )
                raise LLMInvocationError(
                    "Guardrail require_review: approved but OE did not include pending LLM content"
                )
            logger.info(
                "Guardrail require_review: approved — yielding original LLM content",
                extra={"execution_id": self.execution_id},
            )
            llm_response = self._convert_result_to_response(pending_content)
            tool_call_chunks = self._convert_tool_calls_to_stream_chunks(llm_response.tool_calls)
            if (
                llm_response.content
                or tool_call_chunks
                or llm_response.id is not None
                or llm_response.name is not None
                or llm_response.response_metadata is not None
                or llm_response.additional_kwargs is not None
            ):
                yield LLMStreamChunk(
                    content=llm_response.content or None,
                    tool_calls=tool_call_chunks,
                    id=llm_response.id,
                    name=llm_response.name,
                    response_metadata=llm_response.response_metadata,
                    additional_kwargs=llm_response.additional_kwargs,
                )
            if llm_response.usage is not None:
                yield LLMStreamChunk(usage=llm_response.usage)
        else:
            if decision not in ("deny",):
                logger.warning(
                    "Guardrail require_review: unrecognised decision value — treating as deny",
                    extra={"execution_id": self.execution_id, "decision": decision},
                )
            else:
                logger.info(
                    "Guardrail require_review: denied by reviewer",
                    extra={"execution_id": self.execution_id, "decision": decision},
                )
            raise PolicyDeniedException(
                f"Guardrail require_review: denied by reviewer (decision={decision!r})"
            )

    @staticmethod
    def _convert_result_to_response(result: Any) -> LLMResponse:
        """Convert OE/tool-pod output to sdk-core's LLMResponse shape."""
        if result is None:
            raise LLMInvocationError("OE returned no invoke_llm result")
        try:
            response = LLMResponse.model_validate(result)
        except Exception as exc:  # pragma: no cover - defensive guard
            raise LLMInvocationError(
                f"OE returned unexpected invoke_llm result type: {type(result).__name__}"
            ) from exc
        if response.usage is not None and not response.metadata:
            response.metadata = response.usage.model_dump(exclude_none=True)
        return response

    @staticmethod
    def _convert_tool_calls_to_stream_chunks(
        tool_calls: list[LLMToolCall] | None,
    ) -> list[ToolCallChunk] | None:
        """Convert final tool calls into ToolCallChunk objects for synthetic streaming."""
        if not tool_calls:
            return None

        chunks = []
        for fallback_index, tool_call in enumerate(tool_calls):
            chunk_index = tool_call.index
            if chunk_index is None and tool_call.id is None:
                chunk_index = fallback_index

            chunks.append(
                ToolCallChunk(
                    id=tool_call.id,
                    name=tool_call.name,
                    args=normalize_tool_call_args(tool_call.args),
                    type=tool_call.type,
                    index=chunk_index,
                )
            )
        return chunks or None

    @staticmethod
    def _parse_tool_call_args(raw_args: str) -> Any:
        """Parse a streamed tool-call args payload back to structured JSON when possible."""
        try:
            return json.loads(raw_args)
        except (TypeError, ValueError):
            return raw_args

    @classmethod
    def _convert_stream_chunks_to_tool_calls(
        cls,
        chunks: list[ToolCallChunk],
    ) -> list[LLMToolCall] | None:
        """Accumulate streamed tool-call chunks into final typed tool calls."""
        if not chunks:
            return None

        accumulated: dict[int, dict[str, Any]] = {}
        id_to_index: dict[str, int] = {}
        current_index: int | None = None
        next_index = 0
        for chunk in chunks:
            if chunk.index is not None:
                index = chunk.index
                current_index = index
                next_index = max(next_index, index + 1)
            elif chunk.id is not None:
                index = id_to_index.get(chunk.id)
                if index is None:
                    index = next_index
                    next_index += 1
                    id_to_index[chunk.id] = index
                current_index = index
            elif current_index is not None:
                index = current_index
            else:
                index = next_index
                next_index += 1
                current_index = index

            entry = accumulated.setdefault(index, {})
            if chunk.index is not None:
                entry["index"] = chunk.index
            if chunk.id is not None:
                previous_index = id_to_index.get(chunk.id)
                if previous_index is not None and previous_index != index:
                    logger.warning(
                        "LLM: Tool call id %s changed stream index from %s to %s",
                        chunk.id,
                        previous_index,
                        index,
                    )
                entry["id"] = chunk.id
                id_to_index[chunk.id] = index
            if chunk.name is not None:
                entry["name"] = chunk.name
            if chunk.type is not None:
                entry["type"] = chunk.type
            if chunk.args is not None:
                existing_args = entry.get("args")
                entry["args"] = (
                    f"{existing_args}{chunk.args}" if isinstance(existing_args, str) else chunk.args
                )

        tool_calls: list[LLMToolCall] = []
        for index in sorted(accumulated):
            payload = dict(accumulated[index])
            raw_args = payload.get("args")
            if isinstance(raw_args, str):
                payload["args"] = cls._parse_tool_call_args(raw_args)
            tool_calls.append(LLMToolCall.model_validate(payload))
        return tool_calls or None

    def invoke(
        self,
        messages: list[Message],
        step: int | None = None,
        stop: list[str] | None = None,
        options: LLMInvocationOptions | None = None,
    ) -> LLMResponse:
        """Invoke LLM by collecting the stream-oriented execution path."""
        chunks = list(self.stream(messages, step=step, stop=stop, options=options))
        return self.response_from_stream_chunks(chunks)

    @classmethod
    def response_from_stream_chunks(
        cls,
        chunks: list[LLMStreamChunk],
    ) -> LLMResponse:
        """Collect streamed sdk-core chunks into a final sdk-core LLMResponse."""
        content_parts: list[str] = []
        streamed_tool_calls: list[ToolCallChunk] = []
        usage: LLMTokenUsage | None = None
        message_id: str | None = None
        message_name: str | None = None
        response_metadata: dict[str, JsonValue] | None = None
        additional_kwargs: dict[str, JsonValue] | None = None

        for chunk in chunks:
            if chunk.content:
                content_parts.append(chunk.content)
            if chunk.tool_calls:
                streamed_tool_calls.extend(chunk.tool_calls)
            if chunk.usage is not None:
                usage = chunk.usage
            if chunk.id is not None:
                message_id = chunk.id
            if chunk.name is not None:
                message_name = chunk.name
            if chunk.response_metadata is not None:
                # Metadata maps are overwritten by later chunks on key collisions.
                response_metadata = {
                    **(response_metadata or {}),
                    **chunk.response_metadata,
                }
            if chunk.additional_kwargs is not None:
                # Later chunks win here as well; content/tool calls are accumulated separately.
                additional_kwargs = {
                    **(additional_kwargs or {}),
                    **chunk.additional_kwargs,
                }

        tool_calls = cls._convert_stream_chunks_to_tool_calls(streamed_tool_calls)
        metadata = usage.model_dump(exclude_none=True) if usage is not None else {}
        return LLMResponse(
            content="".join(content_parts),
            tool_calls=tool_calls,
            metadata=metadata,
            usage=usage,
            id=message_id,
            name=message_name,
            response_metadata=response_metadata,
            additional_kwargs=additional_kwargs,
        )

    @staticmethod
    def _stream_chunk_from_event(event: LLMPodStreamEvent) -> LLMStreamChunk | None:
        """Convert one tool-pod SSE payload into an sdk-core stream chunk."""
        content = event.content if event.content else None

        tool_calls: list[ToolCallChunk] = []
        if event.tool_call_chunks:
            tool_calls.extend(event.tool_call_chunks)
        for index, tool_call in enumerate(event.tool_calls or []):
            tool_calls.append(
                ToolCallChunk(
                    id=tool_call.id,
                    name=tool_call.name,
                    args=normalize_tool_call_args(tool_call.args),
                    type=tool_call.type,
                    index=tool_call.index if tool_call.index is not None else index,
                )
            )

        if (
            content is None
            and not tool_calls
            and event.id is None
            and event.name is None
            and event.response_metadata is None
            and event.additional_kwargs is None
            and event.usage is None
        ):
            return None
        return LLMStreamChunk(
            content=content,
            tool_calls=tool_calls or None,
            usage=event.usage,
            id=event.id,
            name=event.name,
            response_metadata=event.response_metadata,
            additional_kwargs=event.additional_kwargs,
        )

    def _stream_from_oe(
        self,
        *,
        stream_url: str,
        step: int,
    ) -> Iterator[LLMStreamChunk]:
        """Stream real-time LLM chunks from OE's audited SSE relay."""
        pod_name: str | None = None
        duration_ms = 0.0
        status = "success"
        error: str | None = None
        error_code: str | None = None
        caught_exception: Exception | None = None
        provider_owned = False
        start_time = time.time()
        exposed_output = False

        for attempt in range(OE_RETRYABLE_MAX_ATTEMPTS):
            retry_same_url = False
            retry_delay_s = 0.0
            done_received = False
            status = "success"
            error = None
            error_code = None
            caught_exception = None
            provider_owned = False
            try:
                with create_httpx_client_with_tls(
                    self.oe_url,
                    httpx.Timeout(get_request_timeout(), read=LLM_READ_TIMEOUT),
                ) as client:
                    with connect_sse(
                        client,
                        "POST",
                        stream_url,
                    ) as event_source:
                        event_source.response.raise_for_status()
                        for sse_event in event_source.iter_sse():
                            if not sse_event.data:
                                continue
                            try:
                                stream_event = LLMPodStreamEvent.model_validate(
                                    json.loads(sse_event.data)
                                )
                            except (json.JSONDecodeError, ValueError):
                                logger.warning(
                                    "LLM: Step %s - Skipping malformed SSE payload: %s",
                                    step,
                                    sse_event.data[:120],
                                )
                                continue

                            if stream_event.interrupted:
                                # OE stopped this call on interrupt: not error, not truncation.
                                # Yield a clean marker so the agent continues instead of raising.
                                done_received = True
                                pod_name = stream_event.pod_name or pod_name
                                if stream_event.duration_ms is not None:
                                    duration_ms = stream_event.duration_ms
                                yield LLMStreamChunk(response_metadata={"interrupted": True})
                                break

                            if stream_event.error:
                                error = stream_event.error
                                if (
                                    stream_event.retryable
                                    and attempt < OE_RETRYABLE_MAX_ATTEMPTS - 1
                                    and not exposed_output
                                ):
                                    retry_same_url = True
                                    retry_delay_s = oe_stream_retry_delay_s(
                                        stream_event.retry_after_ms
                                    )
                                else:
                                    status = "error"
                                    provider_owned = True
                                    error_code = stream_event.error_code
                                break

                            if stream_event.done:
                                done_received = True
                                pod_name = stream_event.pod_name
                                if stream_event.duration_ms is not None:
                                    duration_ms = stream_event.duration_ms
                                if stream_event.usage is not None:
                                    yield LLMStreamChunk(usage=stream_event.usage)
                                break

                            stream_chunk = self._stream_chunk_from_event(stream_event)
                            if stream_chunk is None:
                                continue
                            exposed_output = True
                            yield stream_chunk

            except Exception as exc:
                if status != "error":
                    status = "error"
                    error = str(exc)
                    caught_exception = exc
                if (
                    attempt < OE_RETRYABLE_MAX_ATTEMPTS - 1
                    and not exposed_output
                    and _is_oe_stream_transport_error(exc)
                ):
                    retry_same_url = True
                    retry_delay_s = OE_DISPATCH_TAKEOVER_RETRY_DELAY_S
            except BaseException as exc:
                if status != "error":
                    status = "error"
                    error = (
                        "Stream abandoned by consumer"
                        if isinstance(exc, GeneratorExit)
                        else str(exc)
                    )
                raise
            finally:
                if duration_ms == 0.0:
                    duration_ms = (time.time() - start_time) * 1000
                if not retry_same_url and status == "success" and not done_received:
                    error = "SSE stream ended without done signal (truncated response)"
                    if attempt < OE_RETRYABLE_MAX_ATTEMPTS - 1 and not exposed_output:
                        retry_same_url = True
                        retry_delay_s = OE_DISPATCH_TAKEOVER_RETRY_DELAY_S
                    else:
                        status = "error"

                self.last_duration_ms = duration_ms
                self.last_pod_name = pod_name

            if retry_same_url:
                sleep_oe_stream_retry(retry_delay_s)
                continue

            if status == "error":
                source = "llm" if provider_owned else None
                if caught_exception is not None:
                    raise LLMInvocationError(
                        error or "LLM streaming failed", source=source, error_code=error_code
                    ) from caught_exception
                raise LLMInvocationError(
                    error or "LLM streaming failed", source=source, error_code=error_code
                )
            return

        raise LLMInvocationError(error or "LLM streaming failed")

    def stream(
        self,
        messages: list[Message],
        step: int | None = None,
        stop: list[str] | None = None,
        options: LLMInvocationOptions | None = None,
        activity_key: str | None = None,
    ) -> Iterator[LLMStreamChunk]:
        """Stream invoke_llm chunks through OE approval and OE-owned SSE relay.

        `activity_key` names this call's durable activity. An adapter whose
        framework re-runs code after a pause supplies a key that is the same
        on every run; the default, the operational step, only ever increases
        within an attempt.

        Sessions routed to the platform-owned workflow wrap the call in a
        serial LLM activity: a recorded outcome replays without a model call;
        a fresh dispatch streams the model and records the folded response
        under the OE-issued fence.
        """
        resolved_step = self._allocate_step(step)

        invoke_request = self._build_invoke_request(
            messages=messages,
            stop_sequences=stop,
            options=options,
            stream=True,
        )

        attempt = current_attempt_context()
        if attempt is None:
            yield from _model_chunks(self._stream_llm(invoke_request, resolved_step))
            return

        key = activity_key or f"llm:{resolved_step}"
        halt: _ReviewHalt | None = None
        for chunk in self._run_llm_activity(invoke_request, resolved_step, key, attempt):
            if isinstance(chunk, _ReviewHalt):
                halt = chunk
                continue
            yield chunk
        if halt is None:
            return
        review = halt.review
        # Before the pause, so a sibling call halted in the same superstep has
        # its step counted before either resolves.
        self.operational_steps.observe_at_least(halt.step)

        # The halted call is a recorded activity by now, so the pause is
        # outside it. Once answered, the same request is sent again, naming the
        # review, as its own activity at a later step: OE answers that one from
        # its record of the review.
        self._await_review_decision(review)
        yield from _model_chunks(
            self._run_llm_activity(
                invoke_request,
                self._allocate_step(None),
                f"{key}:review:{review.review_id}",
                attempt,
                review_id=review.review_id,
            )
        )

    def _run_llm_activity(
        self,
        invoke_request: InvokeLLMRequestArguments,
        step: int,
        activity_key: str,
        attempt: Any,
        review_id: str | None = None,
    ) -> Iterator[LLMStreamChunk | _ReviewHalt]:
        """Run one invoke_llm request as a durable LLM activity."""

        def _execute() -> Iterator[LLMStreamChunk | _ReviewHalt]:
            try:
                if review_id is None:
                    yield from self._stream_llm(invoke_request, step)
                else:
                    yield from self._stream_review_resolution(invoke_request, step, review_id)
            except PolicyDeniedException as denial:
                # Record the denial as a DENIED outcome (not FAILED) so a
                # replay reproduces policy-denial semantics.
                raise DurableActivityDeniedError(denial.reason) from denial

        semantic_input = invoke_request.model_dump(mode="json", exclude_none=True)
        if review_id is not None:
            semantic_input["guardrail_review_id"] = review_id
        try:
            yield from run_streaming_activity(
                kind=ACTIVITY_KIND_LLM,
                name=self.model_name,
                activity_ordinal=allocate_activity_ordinal(activity_key),
                semantic_input=semantic_input,
                execute=_execute,
                replay=self._chunks_from_replay_result,
                fold=self._fold_and_preallocate_tools,
                on_activity_resolved=self._memory_hook(),
                client=self._workflow,
                oe_url=self.oe_url,
                attempt=attempt,
                # The activity key is the LLM call's identity, so admission
                # need not be serialized. Frameworks may begin consuming
                # tool-call chunks before the provider stream has fully closed.
                exclusive=False,
            )
        except DurableActivityDeniedError as error:
            # Re-raise the original live denial (with guardrail identity) when
            # this attempt produced it; a replayed denial carries message only.
            if isinstance(error.__cause__, PolicyDeniedException):
                raise error.__cause__ from None
            raise PolicyDeniedException(str(error)) from error

    def _memory_hook(self) -> Any:
        """The hook that settles a resolved LLM activity in durable memory.

        A halt is not a model response. Its activity is acknowledged with no
        conversation content, and the turn's pending input stays pending until
        the review is resolved.
        """
        memory = self.durable_memory
        if memory is None:
            return None
        synchronize = partial(memory.synchronize_llm, user_id=get_current_user_id())

        def _settle(client: Any, context: Any, result: Any) -> None:
            if _review_halt_from_result(result) is not None:
                memory.acknowledge(client, context)
            else:
                synchronize(client, context, result)

        return _settle

    def _chunks_from_replay_result(self, result: Any) -> Iterator[LLMStreamChunk | _ReviewHalt]:
        """Translate a workflow-recorded LLM response back into stream chunks.

        This is the replay leg of the LLM activity: OE returned the outcome
        recorded by an earlier attempt, so no model call happens. ToolCall ids
        in the recorded response are preallocated so sibling tools keep the
        same ordinals as the original turn.
        """
        self.last_from_cache = True
        if isinstance(result, str):
            yield LLMStreamChunk(content=result)
            return
        halt = _review_halt_from_result(result)
        if halt is not None:
            yield halt
            return
        self._preallocate_tool_calls_from_result(result)
        llm_response = self._convert_result_to_response(result)
        tool_call_chunks = self._convert_tool_calls_to_stream_chunks(llm_response.tool_calls)
        yield LLMStreamChunk(
            content=llm_response.content or "",
            tool_calls=tool_call_chunks,
            id=llm_response.id,
            name=llm_response.name,
            response_metadata=llm_response.response_metadata,
            additional_kwargs=llm_response.additional_kwargs,
        )
        if llm_response.usage is not None:
            yield LLMStreamChunk(usage=llm_response.usage)

    def _fold_and_preallocate_tools(
        self, chunks: list[LLMStreamChunk | _ReviewHalt]
    ) -> dict[str, Any]:
        """Fold the stream and preallocate sibling Tool activity ordinals."""
        model_chunks: list[LLMStreamChunk] = []
        for chunk in chunks:
            if isinstance(chunk, _ReviewHalt):
                # The halt is the call's result: it completes the activity and
                # is replayed, so the provider is not called again.
                return {
                    _REVIEW_HALT_RESULT_KEY: chunk.review.model_dump(mode="json"),
                    _REVIEW_HALT_STEP_KEY: chunk.step,
                }
            model_chunks.append(chunk)
        payload = self._response_payload_from_chunks(model_chunks)
        self._preallocate_tool_calls_from_result(payload)
        return payload

    @staticmethod
    def _preallocate_tool_calls_from_result(result: Any) -> None:
        """Assign ordinals to ToolCall ids in framework list order before fan-out."""
        if not isinstance(result, dict):
            return
        tool_calls = result.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            return
        keys: list[str] = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            tool_call_id = tool_call.get("id")
            if isinstance(tool_call_id, str) and tool_call_id:
                keys.append(tool_activity_key(tool_call_id))
        if keys:
            preallocate_activity_ordinals(keys)

    @classmethod
    def _response_payload_from_chunks(cls, chunks: list[LLMStreamChunk]) -> dict[str, Any]:
        """Fold streamed chunks into the terminal payload the workflow records.

        This is the record leg of the LLM activity. It uses the same fold
        consumers apply (`response_from_stream_chunks`) so a later replay —
        including merged tool-call fragments — is equivalent to what the
        original stream's consumer assembled.
        """
        return cls.response_from_stream_chunks(chunks).model_dump(mode="json", exclude_none=True)

    def _stream_llm(
        self,
        invoke_request: InvokeLLMRequestArguments,
        resolved_step: int,
    ) -> Iterator[LLMStreamChunk | _ReviewHalt]:
        """Run the actual invoke_llm call: OE approval, then relay or SSE stream."""
        response = self._request_oe_execution(invoke_request, step=resolved_step)
        yield from self._stream_oe_response(response, resolved_step, resolving=False)

    def _stream_review_resolution(
        self,
        invoke_request: InvokeLLMRequestArguments,
        step: int,
        review_id: str,
    ) -> Iterator[LLMStreamChunk | _ReviewHalt]:
        """Repeat a halted call, naming its review: OE answers from the review."""
        response = self._request_oe_execution(invoke_request, step=step, review_id=review_id)
        yield from self._stream_oe_response(response, step, resolving=True)

    def _stream_oe_response(
        self,
        response: ToolExecuteResponse,
        resolved_step: int,
        *,
        resolving: bool,
    ) -> Iterator[LLMStreamChunk | _ReviewHalt]:
        """Turn OE's answer to an invoke_llm request into stream chunks."""

        # Genuine guardrail block: OE returns the substitute string in result
        # instead of LLM content. Yield it as a normal chunk so the agent
        # receives a clean response rather than an exception. The OE contract
        # is that result is the substitute *string*; any other shape means OE
        # is signalling something we don't understand — treat that as a system
        # error so we don't leak a Python repr into the conversation.
        if not response.proceed:
            if response.status == "require_review":
                if not self._follows_review_protocol():
                    yield from self._handle_require_review(response)
                    return
                # A resolving call is answered from the review's record; a halt
                # in its place, or one that names no review, cannot be resolved.
                if resolving or response.guardrail_review is None:
                    raise LLMInvocationError("Guardrail require_review response named no review")
                yield _ReviewHalt(response.guardrail_review, resolved_step)
                return
            if isinstance(response.result, str):
                yield LLMStreamChunk(content=response.result)
                return
            raise_for_oe_rejection(response.reason, guardrail_meta=response.guardrail_meta)

        result = response.result if response.result is not None else response.cached_result
        status = response.status or (
            "success" if self.last_from_cache and result is not None else None
        )

        if status == "error":
            raise LLMInvocationError(
                response.error or "LLM streaming failed",
                source="llm",
                error_code=response.error_code,
            )
        if status == "interrupted":
            # OE stopped this call on interrupt (inline or replayed from a durable
            # interrupted step): yield the same clean marker as the live SSE path
            # so the agent continues instead of raising on an unexpected status.
            yield LLMStreamChunk(response_metadata={"interrupted": True})
            return
        if response.route_to and response.route_to != "callback" and result is None:
            yield from self._stream_from_oe(stream_url=response.route_to, step=resolved_step)
            return

        if status != "success" and result is None:
            raise LLMInvocationError(f"Unexpected OE invoke_llm status: {status}")

        llm_response = self._convert_result_to_response(result)
        tool_call_chunks = self._convert_tool_calls_to_stream_chunks(llm_response.tool_calls)

        if (
            llm_response.content
            or tool_call_chunks
            or llm_response.id is not None
            or llm_response.name is not None
            or llm_response.response_metadata is not None
            or llm_response.additional_kwargs is not None
        ):
            yield LLMStreamChunk(
                content=llm_response.content or None,
                tool_calls=tool_call_chunks,
                id=llm_response.id,
                name=llm_response.name,
                response_metadata=llm_response.response_metadata,
                additional_kwargs=llm_response.additional_kwargs,
            )

        if llm_response.usage is not None:
            yield LLMStreamChunk(usage=llm_response.usage)
