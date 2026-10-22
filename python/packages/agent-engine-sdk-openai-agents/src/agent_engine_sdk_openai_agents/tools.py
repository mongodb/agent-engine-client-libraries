"""Native OpenAI Agents function tools that execute through OE."""

from __future__ import annotations

import asyncio
import contextvars
import functools
import inspect
import json
import types
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
    Annotated,
    Any,
    Literal,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from agents import AgentsException, FunctionTool, default_tool_error_function
from agents.function_schema import FuncSchema, function_schema
from agents.tool import set_function_tool_failure_error_function
from agents.tool_context import ToolContext
from pydantic_core import to_jsonable_python

from agent_engine_runner_shared.secure_llm_proxy import GuardrailReviewPending
from agent_engine_runner_shared.secure_wrapper import (
    CALL_INTERRUPTED_ARTIFACT_KEY,
    ToolExecutionError,
    ToolPlatformError,
    create_secure_tool_function,  # pyright: ignore[reportUnknownVariableType]
)
from agent_engine_runner_shared.workflow import UnknownOutcomePolicy
from agent_engine_runner_shared.workflow.context import ChildOperationBoundary
from agent_engine_sdk_openai_agents.errors import UnsupportedDurableOpenAIAgentsError

__all__ = [
    "ModelCall",
    "SecureToolInvoker",
    "ToolCallFailed",
    "TurnContext",
    "current_turn",
    "json_result",
    "json_text",
    "secure_function_tool",
    "secure_mcp_tool",
    "tool_schema",
    "turn_scope",
]

_JSON_SCALARS = (str, int, float, bool, type(None))


class ToolCallFailed(AgentsException):
    """Carry the platform tool path's exception through the SDK unchanged.

    The SDK re-raises other tool exceptions as ``UserError``, which would hide
    platform exceptions such as a policy denial that the AER reports by type.
    It passes its own exception family through, so the turn can re-raise the
    original.
    """

    def __init__(self, error: Exception) -> None:
        super().__init__(str(error))
        self.error = error


def tool_schema(fn: Callable[..., Any]) -> FuncSchema:
    """Describe a registered tool, rejecting arguments that are not plain JSON.

    Remote tools receive JSON arguments from OE, so local and remote execution
    can agree only when every parameter's Python type is itself a JSON value:
    a model, dataclass, enum, or datetime would arrive as a different type.
    """
    schema = function_schema(fn, strict_json_schema=True)
    if schema.takes_context:
        raise UnsupportedDurableOpenAIAgentsError(
            f"tool {schema.name!r} cannot take a native context argument"
        )
    hints = get_type_hints(fn, include_extras=True)
    for name, parameter in inspect.signature(fn).parameters.items():
        # OE dispatches a call as named JSON arguments, locally and in the Tool
        # Pod, so a parameter that cannot be passed by name could never be bound.
        if name == "tool_call_id":
            # The platform passes the model's call id under this name.
            raise UnsupportedDurableOpenAIAgentsError(
                f"tool {schema.name!r} cannot have a parameter named 'tool_call_id'; "
                "the platform reserves it for the call's identity"
            )
        # (The SDK's strict schema already refuses **kwargs.)
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.VAR_POSITIONAL):
            raise UnsupportedDurableOpenAIAgentsError(
                f"tool {schema.name!r} parameter {name!r} must be passable by name; "
                "positional-only and *args parameters are not supported"
            )
        if not _is_json_native(hints.get(name, Any)):
            raise UnsupportedDurableOpenAIAgentsError(
                f"tool {schema.name!r} parameter {name!r} must be a JSON value "
                "(str, int, float, bool, None, a Literal of those, or a list of them)"
            )
    return schema


def _is_json_native(annotation: Any) -> bool:
    # Strict tool schemas cannot describe open objects, so the SDK already
    # rejects dict parameters; only closed JSON shapes remain.
    if annotation is Any or annotation in _JSON_SCALARS or annotation is list:
        return True
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is Annotated:
        return _is_json_native(args[0])
    if origin in (Union, types.UnionType, list):
        return all(_is_json_native(arg) for arg in args)
    if origin is Literal:
        return all(isinstance(arg, _JSON_SCALARS) for arg in args)
    return False


def json_value(value: Any) -> Any:
    """Convert a result, however deeply nested, into JSON values.

    Models, dataclasses, enums, and datetimes at any depth become their JSON
    forms; an opaque object raises instead of being stringified.
    """
    return to_jsonable_python(value)


def json_text(value: Any) -> str:
    """Render a result as the text the model and the platform result carry.

    A replacement attempt reads recorded values back from OE's protobuf JSON,
    which keeps neither key order nor the int/float distinction. The text is
    canonical so the replayed history the next model call sees matches the
    original attempt's; otherwise OE rejects that call as nondeterministic.
    """
    if isinstance(value, str):
        return value
    try:
        return json.dumps(
            _canonical(json_value(value)),
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
    except ValueError as error:
        raise UnsupportedDurableOpenAIAgentsError(
            f"{type(value).__name__} result is not JSON-serializable"
        ) from error


# OE records values as protobuf JSON, whose numbers are doubles: an integer
# beyond this magnitude comes back rounded on replay.
_MAX_EXACT_INTEGER = 2**53


def require_exact_integers(value: Any, what: str, *, recorded: bool = False) -> None:
    """Refuse an integer a replacement attempt would read back rounded.

    The replayed text would differ from the original attempt's, and OE would
    reject the next model call as nondeterministic.

    ``recorded`` marks a value OE has already read, so a larger integer may
    have been rounded to 2**53 itself, or arrive as a float: those are refused
    too.
    """
    if isinstance(value, dict):
        for item in cast(dict[str, Any], value).values():
            require_exact_integers(item, what, recorded=recorded)
    elif isinstance(value, list):
        for item in cast(list[Any], value):
            require_exact_integers(item, what, recorded=recorded)
    elif isinstance(value, float):
        if recorded and value.is_integer() and abs(value) >= _MAX_EXACT_INTEGER:
            require_exact_integers(int(value), what, recorded=True)
    elif isinstance(value, int) and not isinstance(value, bool):
        if recorded and abs(value) >= _MAX_EXACT_INTEGER:
            raise UnsupportedDurableOpenAIAgentsError(
                f"{what}: integer {value} reaches the range a durable record "
                "rounds (2**53) and may not be the one the tool returned; "
                "the tool must return it as a string"
            )
        if abs(value) > _MAX_EXACT_INTEGER:
            raise UnsupportedDurableOpenAIAgentsError(
                f"{what}: integer {value} is beyond the range a durable record "
                "keeps exactly (2**53); pass it as a string"
            )


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _canonical(item) for key, item in cast(dict[str, Any], value).items()
        }
    if isinstance(value, list):
        return [_canonical(item) for item in cast(list[Any], value)]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def json_result(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Make a tool return JSON values, whichever pod runs it.

    OE records tool results as JSON, so a model or dataclass result is
    converted where the tool runs rather than rejected as non-JSON.
    """
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def call_async(*args: Any, **kwargs: Any) -> Any:
            return _recordable(fn, json_value(await fn(*args, **kwargs)))

        return call_async

    @functools.wraps(fn)
    def call(*args: Any, **kwargs: Any) -> Any:
        return _recordable(fn, json_value(fn(*args, **kwargs)))

    return call


def _recordable(fn: Callable[..., Any], result: Any) -> Any:
    # Where the tool runs, so the call fails before OE records a result that
    # could not be replayed.
    require_exact_integers(result, f"the result of tool {fn.__name__!r}")
    return result


@dataclass(frozen=True)
class ModelCall:
    """A tool call a model made this turn, and the run it belongs to."""

    # The run's child operations, outermost first; none for the turn's own
    # agents.
    boundaries: tuple[ChildOperationBoundary, ...]
    # Its place among the calls noted this turn. Only compared between calls
    # of one run, which makes them one model response at a time, so it does
    # not matter which of two concurrent runs answered first.
    order: int


class TurnContext:
    """What one turn shares with the tools and nested agents it runs.

    The turn passes this as the Runner's context object, which the SDK hands
    to every tool call and nested run.
    """

    def __init__(self) -> None:
        # A tool call runs in a worker thread that cannot be interrupted, so
        # cancelling the Runner does not stop it. The turn waits for these
        # calls before it ends.
        self._calls: set[asyncio.Future[Any]] = set()
        # Nested agents report progress here while the SDK's own stream is
        # silent waiting for their tool call to return.
        self.events: asyncio.Queue[Any] = asyncio.Queue()
        # An approval is recorded where its call was made, and the SDK reports
        # a pending call by id without saying which run made it.
        self.calls: dict[str, ModelCall] = {}
        # Every parent call of the turn whose nested run stopped for an
        # approval, kept until the turn ends.
        self.stopped_calls: list[Any] = []
        # Model calls OE halted for a guardrail review this attempt, by the id
        # of the review-tool call the Runner stopped on, until each resolves.
        self.reviews: dict[str, GuardrailReviewPending] = {}

    def track(self, call: asyncio.Future[Any]) -> None:
        self._calls.add(call)
        call.add_done_callback(self._calls.discard)

    async def drain(self) -> None:
        """Wait for every started call, even if this wait is itself cancelled.

        A cancellation is re-raised only once the calls have finished, so the
        turn never ends while a tool is still running.
        """
        cancelled = False
        # Until none is left, not one snapshot: a call tracked while this
        # waits, or a second drain, must not let the turn end early.
        while self._calls:
            waiting = asyncio.gather(*self._calls, return_exceptions=True)
            while not waiting.done():
                try:
                    await asyncio.shield(waiting)
                except asyncio.CancelledError:
                    cancelled = True
        if cancelled:
            raise asyncio.CancelledError


_TURN: contextvars.ContextVar[TurnContext | None] = contextvars.ContextVar(
    "agent_engine_openai_agents_turn", default=None
)


def current_turn() -> TurnContext | None:
    """The turn the current task runs for, if any."""
    return _TURN.get()


@contextmanager
def turn_scope(turn: TurnContext) -> Generator[None]:
    """Make ``turn`` current for the tasks started inside the scope.

    The Runner gives its context object to tools but not to models, and a task
    copies the current context when it is created.
    """
    token = _TURN.set(turn)
    try:
        yield
    finally:
        _TURN.reset(token)


class SecureToolInvoker:
    """Run one tool call through OE under the model's stable call id."""

    def __init__(
        self,
        name: str,
        parse: Callable[[str], dict[str, Any]],
        secure: Callable[..., Any],
        issuer: object,
        *,
        mcp: bool = False,
    ) -> None:
        self._name = name
        # The model's argument text as named JSON arguments, or ValueError.
        self._parse = parse
        self._secure = secure
        # Whether the call goes to a remote MCP server instead of a function
        # the app registered.
        self.mcp = mcp
        # The App that issued this tool: its runtime registered the callback
        # and policy the call runs under.
        self.issuer = issuer

    async def __call__(self, context: ToolContext[Any], arguments: str) -> str:
        if not context.tool_call_id:
            raise UnsupportedDurableOpenAIAgentsError(
                f"tool {self._name!r} was called without a call id"
            )
        try:
            arguments_by_name = self._parse(arguments)
        except ValueError as error:
            raise UnsupportedDurableOpenAIAgentsError(
                f"tool {self._name!r} received invalid arguments"
            ) from error
        try:
            # Before the call: a remote tool would receive the rounded value.
            require_exact_integers(
                arguments_by_name, f"the arguments of tool {self._name!r}"
            )
            # As asyncio.to_thread does, but tracked and shielded so a sibling's
            # failure cancels only this task, not the record of the call.
            call: asyncio.Future[tuple[Any, Any]] = (
                asyncio.get_running_loop().run_in_executor(
                    None,
                    functools.partial(
                        contextvars.copy_context().run,
                        self._secure,
                        tool_call_id=context.tool_call_id,
                        **arguments_by_name,
                    ),
                )
            )
            if isinstance(context.context, TurnContext):
                context.context.track(call)
            # If this task is cancelled here, nothing below runs for the
            # call: the turn's drain still waits for it, and its outcome is
            # the one OE recorded.
            content, artifact = await asyncio.shield(call)
            # Only OE can set this artifact marker, so a tool result cannot
            # forge it; a stopped call must not become content the model reads.
            stopped = isinstance(artifact, dict) and bool(
                cast(dict[str, object], artifact).get(CALL_INTERRUPTED_ARTIFACT_KEY)
            )
            if stopped:
                raise UnsupportedDurableOpenAIAgentsError(
                    f"tool call {context.tool_call_id!r} was stopped before completing"
                )
            if self.mcp:
                # The Tool Pod runs an MCP call without this adapter's code,
                # so the result is checked here, after OE has read it.
                require_exact_integers(
                    json_value(content),
                    f"the result of tool {self._name!r}",
                    recorded=True,
                )
            # The SDK would store a non-string result as its Python repr;
            # durable history needs text the next turn can read back.
            return json_text(content)
        except ToolPlatformError as error:
            # The platform failed, perhaps after the tool already ran; told it
            # failed, the model could call it again.
            raise ToolCallFailed(error) from error
        except ToolExecutionError as error:
            # The tool itself failed. Tell the model, as the OpenAI Agents SDK
            # does by default. OE records the failure, so a replay sends the
            # model the same text. Only the tool's message is passed on, not
            # the wrapper's "Tool execution error:" prefix.
            return default_tool_error_function(context, Exception(error.error))
        except Exception as error:
            raise ToolCallFailed(error) from error


def secure_function_tool(
    fn: Callable[..., Any],
    schema: FuncSchema,
    *,
    is_local: bool,
    redact_fields: list[str],
    allow_direct: bool,
    issuer: object,
    needs_approval: bool = False,
    on_unknown_outcome: UnknownOutcomePolicy | None = None,
) -> FunctionTool:
    """Build the native FunctionTool the agent sees for a registered tool."""
    secure: Callable[..., Any] = create_secure_tool_function(  # type: ignore[no-untyped-call]
        original_tool=fn,
        tool_name=schema.name,
        allow_direct=allow_direct,
        is_local=is_local,
        redact_fields=redact_fields,
        # Carries OE's authenticated stopped-call marker beside the content.
        response_format="content_and_artifact",
        tool_declared_format="content",
        on_unknown_outcome=on_unknown_outcome,
    )

    def parse(arguments: str) -> dict[str, Any]:
        # pydantic's ValidationError is a ValueError.
        parsed = schema.params_pydantic_model.model_validate_json(arguments)
        return parsed.model_dump(mode="json")

    tool = FunctionTool(
        name=schema.name,
        description=schema.description or "",
        params_json_schema=schema.params_json_schema,
        on_invoke_tool=SecureToolInvoker(schema.name, parse, secure, issuer),
        strict_json_schema=True,
        needs_approval=needs_approval,
    )
    # The SDK's own handler would also turn a cancelled call and OE control
    # flow into text for the model; SecureToolInvoker reports only a failed
    # tool, and everything else fails the turn.
    return set_function_tool_failure_error_function(tool, None)


def secure_mcp_tool(
    run: Callable[..., Any],
    *,
    name: str,
    description: str,
    input_schema: dict[str, Any],
    metadata: dict[str, Any],
    allow_direct: bool,
    issuer: object,
) -> FunctionTool:
    """Build the native FunctionTool the agent sees for a remote MCP tool.

    ``metadata`` names the server and tool so the Tool Pod can find the
    configured server for the call.
    """
    properties = input_schema.get("properties")
    if isinstance(properties, dict) and "tool_call_id" in properties:
        raise UnsupportedDurableOpenAIAgentsError(
            f"MCP tool {name!r} cannot have a parameter named 'tool_call_id'; "
            "the platform reserves it for the call's identity"
        )
    secure: Callable[..., Any] = create_secure_tool_function(  # type: ignore[no-untyped-call]
        original_tool=run,
        tool_name=name,
        allow_direct=allow_direct,
        metadata=metadata,
        is_local=False,
        response_format="content_and_artifact",
        tool_declared_format="content",
    )

    def parse(arguments: str) -> dict[str, Any]:
        parsed: Any = json.loads(arguments) if arguments else {}
        if not isinstance(parsed, dict):
            raise ValueError("MCP tool arguments must be a JSON object")
        return cast(dict[str, Any], parsed)

    tool = FunctionTool(
        name=name,
        description=description,
        # The server's schema as given: it is not derived from a Python
        # signature, so it need not satisfy the provider's strict mode. The
        # provider does require an object schema that lists its properties.
        params_json_schema={"type": "object", "properties": {}, **input_schema},
        on_invoke_tool=SecureToolInvoker(name, parse, secure, issuer, mcp=True),
        strict_json_schema=False,
    )
    return set_function_tool_failure_error_function(tool, None)
