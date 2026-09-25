"""Stable identities for messages emitted by durable LangGraph nodes."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from langchain_core.messages import BaseMessage, RemoveMessage, convert_to_messages
from langchain_core.runnables import RunnableLambda
from langgraph.graph import add_messages
from langgraph.types import Command, Overwrite

from agent_engine_runner_shared.workflow import current_attempt_context
from agent_engine_runner_shared.workflow.context import (
    current_operation_path,
    current_step_ordinal,
)

_INSTALLED_ATTRIBUTE = "_agent_engine_durable_message_identity_installed"


def _message_id(node_name: str, output_path: tuple[str | int, ...]) -> str:
    attempt = current_attempt_context()
    if attempt is None:
        raise RuntimeError("durable message identity requires an active attempt")
    execution_id = attempt.workflow_identity.execution_id
    operation_path = [
        [segment.name, segment.ordinal] for segment in current_operation_path().segments
    ]
    producer = json.dumps(
        [operation_path, node_name, output_path],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode()
    producer_hash = hashlib.sha256(producer).hexdigest()
    return f"durable-message:{execution_id}:{current_step_ordinal()}:{producer_hash}"


def _assign_message_id(
    value: Any,
    node_name: str,
    path: tuple[str | int, ...],
) -> BaseMessage:
    message = convert_to_messages([value])[0]
    if message.id is None and not isinstance(message, RemoveMessage):
        message.id = _message_id(node_name, path)
    return message


def _assign_message_update_ids(
    value: Any,
    node_name: str,
    path: tuple[str | int, ...],
) -> Any:
    if isinstance(value, Overwrite):
        return Overwrite(
            _assign_message_update_ids(value.value, node_name, (*path, "__overwrite__"))
        )
    if isinstance(value, dict) and set(value) == {"__overwrite__"}:
        return {
            "__overwrite__": _assign_message_update_ids(
                value["__overwrite__"], node_name, (*path, "__overwrite__")
            )
        }
    if isinstance(value, list):
        return [
            _assign_message_id(item, node_name, (*path, index))
            for index, item in enumerate(value)
        ]
    return _assign_message_id(value, node_name, path)


def _assign_state_update_ids(
    value: Any,
    node_name: str,
    path: tuple[str | int, ...],
    message_channels: frozenset[str],
) -> Any:
    if isinstance(value, dict):
        for key in message_channels & value.keys():
            value[key] = _assign_message_update_ids(
                value[key], node_name, (*path, str(key))
            )
        return value
    return value


def _assign_output_ids(
    value: Any,
    node_name: str,
    message_channels: frozenset[str],
) -> Any:
    if isinstance(value, Command):
        if isinstance(value.update, dict):
            _assign_state_update_ids(
                value.update, node_name, ("update",), message_channels
            )
        # Tuple-pair and implicit-root updates remain LangGraph-owned shapes.
        return value
    return _assign_state_update_ids(value, node_name, (), message_channels)


def _with_durable_message_identity(
    output: Any,
    *,
    node_name: str,
    message_channels: frozenset[str],
) -> Any:
    if current_attempt_context() is not None:
        return _assign_output_ids(output, node_name, message_channels)
    return output


def _identity_runnable(
    node_name: str,
    message_channels: frozenset[str],
) -> RunnableLambda[Any, Any]:
    def assign(output: Any) -> Any:
        return _with_durable_message_identity(
            output,
            node_name=node_name,
            message_channels=message_channels,
        )

    return RunnableLambda(assign)


def install_durable_message_identity(graph: Any) -> None:
    """Stamp missing durable message IDs before named LangGraph reducers run.

    LangGraph's ``add_messages`` reducer assigns a random UUID to every
    ``BaseMessage`` whose id is ``None``. Replacement attempts rebuild node
    outputs, so allowing that reducer to own identity makes otherwise identical
    durable state nondeterministic. This covers ordinary named message-channel
    updates, mapping-form ``Command`` updates, and LangGraph ``Overwrite``
    containers. Wrap every compiled node, including nodes in compiled subgraphs,
    while leaving native execution and explicit IDs alone.
    """
    graphs = (
        [subgraph for _, subgraph in graph.get_subgraphs(recurse=True)]
        if hasattr(graph, "get_subgraphs")
        else []
    )
    graphs.append(graph)
    for compiled in graphs:
        nodes: dict[str, Any] | None = getattr(compiled, "nodes", None)
        if not isinstance(nodes, dict):
            continue
        channels: dict[str, Any] = getattr(compiled, "channels", {})
        message_channels = frozenset(
            name
            for name, channel in channels.items()
            if name != "__root__" and getattr(channel, "operator", None) is add_messages
        )
        if not message_channels:
            continue
        for node_name, node in nodes.items():
            if node_name.startswith("__") or getattr(node, _INSTALLED_ATTRIBUTE, False):
                continue
            node.bound = node.bound | _identity_runnable(node_name, message_channels)
            setattr(node, _INSTALLED_ATTRIBUTE, True)
