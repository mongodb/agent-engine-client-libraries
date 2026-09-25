"""LangGraph output parser and custom-event delivery.

``features.use_custom_parser`` is the opt-in for both: Atlas Agent Engine instantiates
the author ``@app.output_parser``, subscribes to LangGraph ``custom``, and
installs a transport so ``emit_custom_event`` can reach the stream. Without
that flag there is no parser and emit is rejected.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any

from langgraph.config import get_stream_writer
from agent_engine_sdk import OutputParser, RequestContext, StreamEvent
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError

from agent_engine_sdk_langgraph.stream_scope import aclose_stream_iter
from agent_engine_runner_shared.server.chunk_types import CUSTOM_EVENT

__all__ = ["LangGraphOutputParser", "RawStreamItem"]

_logger = logging.getLogger(__name__)

# Bounds tenant-authored parser.parse()/on_stream_error() calls so a hung
# author implementation can't stall the whole stream indefinitely.
PARSER_CALL_TIMEOUT_S = 30.0

# RFC 8259 JSON has no NaN/Infinity; OE chunks are JSON. Same adapter
# ``emit_custom_event`` uses, so parser and raw custom-channel values cannot
# bypass the author-API contract.
_JSON_VALUE_ADAPTER = TypeAdapter(JsonValue, config=ConfigDict(allow_inf_nan=False))


class LangGraphCustomEventTransport:
    """AER transport: resolve ``get_stream_writer`` at emission time and write.

    A missing runnable context while this transport is active means the emit
    happened outside graph execution — that is a platform bug, not a silent drop.
    Unlike Tool Pod HTTP delivery (best-effort / non-fatal), AER does not swallow
    writer failures: failing closed surfaces the wiring bug instead of dropping
    author events without a signal.
    """

    def _write(self, payload: Mapping[str, JsonValue]) -> None:
        try:
            writer = get_stream_writer()
        except Exception as exc:
            raise RuntimeError(
                "emit_custom_event: LangGraph stream writer is unavailable; "
                "called outside a LangGraph runnable context"
            ) from exc
        writer(dict(payload))

    def emit_sync(self, payload: Mapping[str, JsonValue]) -> None:
        self._write(payload)

    async def emit(self, payload: Mapping[str, JsonValue]) -> None:
        self._write(payload)


class DisabledCustomEventTransport:
    """Reject emit when this invocation has no custom-event delivery path.

    ``invoke()`` has no stream. Feature-off streaming uses
    ``FeatureOffCustomEventTransport`` (same reject behavior, different
    author-facing reason).
    """

    _MESSAGE = (
        "emit_custom_event is only available during AER streaming execution; "
        "invoke() does not deliver custom events"
    )

    def emit_sync(self, payload: Mapping[str, JsonValue]) -> None:
        raise RuntimeError(self._MESSAGE)

    async def emit(self, payload: Mapping[str, JsonValue]) -> None:
        raise RuntimeError(self._MESSAGE)


def _json_value(payload: object) -> JsonValue:
    serialized = (
        payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
    )
    return _JSON_VALUE_ADAPTER.validate_python(serialized)


def custom_event(payload: BaseModel | JsonValue) -> StreamEvent:
    """Wrap parser output in a StreamEvent carrying only ``custom_event``.

    ``BaseModel`` is dumped in JSON mode so the payload is JSON-serializable;
    ``data`` is None because a parser event carries no platform payload.
    Fail closed: a parser bug must not push non-JSON onto the OE wire.
    """
    try:
        normalized = _json_value(payload)
    except ValidationError as exc:
        raise ValueError(
            "custom output parser yielded a non-JSON-serializable value"
        ) from exc
    return StreamEvent(data=None, event=CUSTOM_EVENT, custom_event=normalized)


def safe_raw_custom_event(payload: object) -> StreamEvent | None:
    """Convert raw LangGraph custom traffic without failing the agent stream.

    Author ``emit_custom_event`` payloads are already validated. This guard is
    for unrelated graph/library code that writes to LangGraph's shared custom
    channel after Atlas Agent Engine subscribes to it. Skip rather than fail: Atlas Agent Engine did
    not produce the value.

    Only JSON objects become CUSTOM_EVENT frames. ``None`` / scalars / arrays
    are skipped: AER recognizes custom events only when ``custom_event is not
    None``, and a frame with both ``custom_event`` and ``data`` as ``None``
    would fall through to the platform-data path and abort the stream.
    """
    try:
        normalized = _json_value(payload)
    except (ValidationError, TypeError, ValueError) as exc:
        _logger.warning(
            "Ignoring non-JSON LangGraph custom payload of type %s: %s",
            type(payload).__name__,
            exc,
        )
        return None
    if not isinstance(normalized, dict):
        _logger.warning(
            "Ignoring non-object LangGraph custom payload of type %s",
            type(payload).__name__,
        )
        return None
    return StreamEvent(data=None, event=CUSTOM_EVENT, custom_event=normalized)


@dataclass(frozen=True)
class RawStreamItem:
    """One item from LangGraph's ``astream``, normalized.

    ``namespace`` is the subgraph path (``subgraphs=True``); ``()`` at the root.
    ``payload`` is the raw payload for ``stream_mode``, verbatim.
    """

    namespace: tuple[str, ...]
    stream_mode: str
    payload: Any


class LangGraphOutputParser(OutputParser[RawStreamItem]):
    """Base output parser for LangGraph agents.

    Subclasses set ``stream_modes`` and implement ``parse`` / ``on_stream_error``.
    The binding maps ``stream_modes`` onto ``astream(stream_mode=..., subgraphs=
    True)`` and normalizes each 3-tuple into a ``RawStreamItem`` before calling
    ``parse``.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # A bare str is a valid Sequence[str] but char-splits under list(),
        # silently streaming per-character modes. Reject it at definition.
        if isinstance(cls.stream_modes, str):
            raise TypeError(
                f"{cls.__name__}.stream_modes must be a sequence of mode names "
                f"(e.g. a tuple), not a bare str; got {cls.stream_modes!r}"
            )

    def resolved_stream_modes(self) -> list[str]:
        """Return ``stream_modes`` as the list passed to ``astream``."""
        return list(self.stream_modes)


@dataclass(frozen=True)
class OutputParserBinding:
    """Author CUSTOM_EVENT production for one stream: parser plus astream modes.

    Atlas Agent Engine always subscribes to ``messages`` and ``updates``. ``custom`` is
    added when the workspace opted in, even if the parser did not list it, so
    ``emit_custom_event`` still reaches the stream. Any other parser-listed
    modes (e.g. ``values``) are appended.

    ``iter_custom_events`` is the only adapter entry for those frames. Atlas Agent Engine
    tokens / updates stay in the agent loop and do not feed this method.
    """

    parser: OutputParser | None
    parser_modes: frozenset[str]
    stream_modes: tuple[str, ...]

    @classmethod
    def create(
        cls,
        parser_cls: type[OutputParser] | None,
        *,
        use_custom_parser: bool,
    ) -> OutputParserBinding:
        parser = _instantiate_parser(parser_cls, use_custom_parser=use_custom_parser)
        parser_modes = _parser_modes(parser)
        platform_modes = ["messages", "updates"]
        if use_custom_parser:
            platform_modes.append("custom")
        stream_modes = platform_modes + [
            mode for mode in parser_modes if mode not in platform_modes
        ]
        return cls(parser, frozenset(parser_modes), tuple(stream_modes))

    async def iter_custom_events(
        self,
        *,
        stream_mode: str,
        namespace: tuple[str, ...],
        payload: Any,
        ctx: RequestContext,
    ) -> AsyncIterator[StreamEvent]:
        """CUSTOM_EVENT frames for one astream item.

        Two mutually exclusive sources, never both for the same item:

        * The author parser, when it listed this ``stream_mode``. ``parse()``
          yields already-shaped JSON (often from ``messages`` / ``values``).
          That output is wrapped as CUSTOM_EVENT here; it does not go through
          LangGraph's ``custom`` channel.
        * Else LangGraph ``custom`` when Atlas Agent Engine subscribed and the parser
          did not claim it — ``emit_custom_event`` payloads forwarded verbatim.
        """
        if self.parser is not None and stream_mode in self.parser_modes:
            async for event in _emit_parser_events(
                self.parser,
                stream_mode=stream_mode,
                namespace=namespace,
                payload=payload,
                ctx=ctx,
            ):
                yield event
            return
        if stream_mode == "custom" and "custom" in self.stream_modes:
            event = safe_raw_custom_event(payload)
            if event is not None:
                yield event


def _parser_modes(parser: OutputParser | None) -> set[str]:
    if parser is None:
        return set()
    resolver = getattr(parser, "resolved_stream_modes", None)
    return set(resolver() if resolver is not None else parser.stream_modes)


def _instantiate_parser(
    parser_cls: type[OutputParser] | None, *, use_custom_parser: bool
) -> OutputParser | None:
    if not use_custom_parser or parser_cls is None:
        return None
    try:
        return parser_cls()
    except Exception as exc:
        parser_name = getattr(parser_cls, "__name__", repr(parser_cls))
        raise RuntimeError(
            f"Failed to instantiate custom output parser {parser_name}: {exc}"
        ) from exc


async def _emit_parser_events(
    parser: OutputParser,
    *,
    stream_mode: str,
    namespace: tuple[str, ...],
    payload: Any,
    ctx: RequestContext,
) -> AsyncIterator[StreamEvent]:
    parser_iter = parser.parse(
        RawStreamItem(
            namespace=namespace,
            stream_mode=stream_mode,
            payload=payload,
        ),
        ctx,
    ).__aiter__()
    try:
        while True:
            try:
                # Bound each yield individually rather than the whole item, so
                # a parser emitting many events for one item isn't capped by a
                # single deadline.
                parsed = await asyncio.wait_for(
                    parser_iter.__anext__(),
                    timeout=PARSER_CALL_TIMEOUT_S,
                )
            except StopAsyncIteration:
                break
            if parsed is not None:
                yield custom_event(parsed)
    except Exception:
        # Fail the execution rather than silently dropping shaped output.
        # A timeout (asyncio.TimeoutError) is caught here too, so a hung
        # parse() can't stall the stream forever.
        _logger.exception(
            "Custom output parser failed while parsing a "
            "stream item; failing the execution"
        )
        raise
    finally:
        await aclose_stream_iter(parser_iter)


async def emit_parser_stream_error(
    parser: OutputParser, ctx: RequestContext
) -> AsyncIterator[StreamEvent]:
    # agent.py imports this module, so the public type lives there to keep
    # generated API docs on ``agent_engine_sdk_langgraph.agent.InternalStreamError``.
    from agent_engine_sdk_langgraph.agent import InternalStreamError

    try:
        parsed = await asyncio.wait_for(
            parser.on_stream_error(
                ctx,
                # Parsers often stringify this into a customer-visible
                # custom_event. Do not pass LangGraph / LLM / network errors.
                InternalStreamError(),
            ),
            timeout=PARSER_CALL_TIMEOUT_S,
        )
        if parsed is not None:
            yield custom_event(parsed)
    except Exception:
        # Catches asyncio.TimeoutError and JSON validation failures from
        # ``custom_event``, so a bad error-hook payload cannot mask the
        # original graph failure.
        _logger.warning(
            "Custom output parser on_stream_error failed; "
            "suppressing to preserve the original failure",
            exc_info=True,
        )
