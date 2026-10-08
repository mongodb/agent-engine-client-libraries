"""Native OpenAI Agents function tools that execute through OE."""

from __future__ import annotations

import asyncio
import contextvars
import functools
import inspect
import json
import types
from collections.abc import Callable
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

from agents import AgentsException, FunctionTool
from agents.function_schema import FuncSchema, function_schema
from agents.tool import set_function_tool_failure_error_function
from agents.tool_context import ToolContext
from pydantic import ValidationError
from pydantic_core import to_jsonable_python

from agent_engine_runner_shared.secure_wrapper import (
    CALL_INTERRUPTED_ARTIFACT_KEY,
    create_secure_tool_function,  # pyright: ignore[reportUnknownVariableType]
)
from agent_engine_sdk_openai_agents.errors import UnsupportedDurableOpenAIAgentsError

__all__ = [
    "SecureToolInvoker",
    "ToolCallFailed",
    "TurnContext",
    "json_result",
    "json_text",
    "secure_function_tool",
    "tool_schema",
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


def require_exact_integers(value: Any, what: str) -> None:
    """Refuse an integer a replacement attempt would read back rounded.

    The replayed text would differ from the original attempt's, and OE would
    reject the next model call as nondeterministic.
    """
    if isinstance(value, dict):
        for item in cast(dict[str, Any], value).values():
            require_exact_integers(item, what)
    elif isinstance(value, list):
        for item in cast(list[Any], value):
            require_exact_integers(item, what)
    elif (
        isinstance(value, int)
        and not isinstance(value, bool)
        and abs(value) > _MAX_EXACT_INTEGER
    ):
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


class SecureToolInvoker:
    """Run one tool call through OE under the model's stable call id."""

    def __init__(
        self, schema: FuncSchema, secure: Callable[..., Any], issuer: object
    ) -> None:
        self._schema = schema
        self._secure = secure
        # The App that issued this tool: its runtime registered the callback
        # and policy the call runs under.
        self.issuer = issuer

    async def __call__(self, context: ToolContext[Any], arguments: str) -> str:
        if not context.tool_call_id:
            raise UnsupportedDurableOpenAIAgentsError(
                f"tool {self._schema.name!r} was called without a call id"
            )
        try:
            parsed = self._schema.params_pydantic_model.model_validate_json(arguments)
        except ValidationError as error:
            raise UnsupportedDurableOpenAIAgentsError(
                f"tool {self._schema.name!r} received invalid arguments"
            ) from error
        arguments_by_name = parsed.model_dump(mode="json")
        try:
            # Before the call: a remote tool would receive the rounded value.
            require_exact_integers(
                arguments_by_name, f"the arguments of tool {self._schema.name!r}"
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
            # The SDK would store a non-string result as its Python repr;
            # durable history needs text the next turn can read back.
            return json_text(content)
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
    )
    tool = FunctionTool(
        name=schema.name,
        description=schema.description or "",
        params_json_schema=schema.params_json_schema,
        on_invoke_tool=SecureToolInvoker(schema, secure, issuer),
        strict_json_schema=True,
        needs_approval=needs_approval,
    )
    # By default the SDK turns a failed or cancelled tool into model-visible
    # text. Durable tool failures and OE control flow must fail the turn instead.
    return set_function_tool_failure_error_function(tool, None)
