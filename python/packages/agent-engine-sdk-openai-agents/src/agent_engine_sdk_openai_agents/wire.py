"""The model-call contract shared by the AER and the Tool Pod.

An OE LLM activity carries the Runner's model request from the AER, where the
native ``Runner`` asks for it, to the Tool Pod, where the native OpenAI model
runs. Each pair of translations lives here so the two ends cannot drift: a
setting the AER sends must be one the Tool Pod applies.
"""

from __future__ import annotations

import json
from dataclasses import fields
from typing import Any, Literal, cast, get_args

from agent_engine_sdk import LLMInvocationOptions
from agents import ModelSettings
from agents.agent_output import AgentOutputSchemaBase
from openai.types.shared import Reasoning
from openai.types.shared.reasoning_effort import ReasoningEffort

from agent_engine_sdk_openai_agents.errors import UnsupportedDurableOpenAIAgentsError

__all__ = [
    "TransportedOutputSchema",
    "invocation_options",
    "native_settings",
    "reasoning_options",
    "tool_choice",
]

# Per-call ModelSettings fields that cross the activity. Provider configuration
# such as temperature belongs to the model registered with app.llm, which the
# Tool Pod applies itself.
_TRANSPORTED = (
    "max_tokens",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "timeout",
    "parallel_tool_calls",
)
_RUNNER_CONTROLLED = frozenset({"tool_choice"})
# Reasoning settings cross the activity as the options contract's flat fields.
_REASONING = frozenset({"reasoning", "verbosity"})
# The SDK's type is an optional Literal today; a flat Literal reads the same.
_EFFORTS = frozenset(
    value
    for arg in get_args(ReasoningEffort)
    for value in (get_args(arg) or (arg,))
    if isinstance(value, str)
)
if not _EFFORTS:
    # Otherwise every effort would be refused with an empty list of choices.
    raise RuntimeError("the OpenAI SDK's ReasoningEffort names no effort levels")
_VERBOSITIES = frozenset(get_args(Literal["low", "medium", "high"]))


def reasoning_options(settings: ModelSettings) -> dict[str, str]:
    """The reasoning effort and verbosity the platform carries, validated.

    Only ``reasoning.effort`` crosses: summaries would be new caller-facing
    output, and a model's hidden reasoning is never carried between calls.
    """
    carried: dict[str, str] = {}
    reasoning = settings.reasoning
    if reasoning is not None:
        unsupported = sorted(
            f"reasoning.{name}"
            for name, value in reasoning
            if name != "effort" and value is not None
        )
        if unsupported:
            raise UnsupportedDurableOpenAIAgentsError(
                f"{', '.join(unsupported)} not supported; durable OpenAI Agents "
                "runs carry reasoning.effort only"
            )
        if reasoning.effort is not None:
            carried["reasoning_effort"] = _one_of(
                reasoning.effort, _EFFORTS, "reasoning.effort"
            )
    if settings.verbosity is not None:
        carried["verbosity"] = _one_of(settings.verbosity, _VERBOSITIES, "verbosity")
    return carried


def _one_of(value: object, allowed: frozenset[str], name: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise UnsupportedDurableOpenAIAgentsError(
            f"{name} must be one of {', '.join(sorted(allowed))}"
        )
    return value


def invocation_options(
    settings: ModelSettings, output_schema: AgentOutputSchemaBase | None
) -> LLMInvocationOptions:
    """AER side: translate the Runner's per-call settings for the activity."""
    untransported = sorted(
        field.name
        for field in fields(settings)
        if getattr(settings, field.name) is not None
        and field.name not in _TRANSPORTED
        and field.name not in _RUNNER_CONTROLLED
        and field.name not in _REASONING
    )
    if untransported:
        raise UnsupportedDurableOpenAIAgentsError(
            "OpenAI model settings cannot cross the platform model activity: "
            f"{', '.join(untransported)}. Set provider settings with "
            "app.llm(..., settings=...)"
        )
    response_format: dict[str, Any] | None = None
    if output_schema is not None and not output_schema.is_plain_text():
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": output_schema.name(),
                "schema": output_schema.json_schema(),
                "strict": output_schema.is_strict_json_schema(),
            },
        }
    carried = reasoning_options(settings)
    return LLMInvocationOptions(
        **{name: getattr(settings, name) for name in _TRANSPORTED},
        reasoning_effort=carried.get("reasoning_effort"),
        verbosity=carried.get("verbosity"),
        response_format=cast(Any, response_format),
    )


def tool_choice(settings: ModelSettings) -> str | None:
    """AER side: the tool choice the activity forwards beside the options."""
    choice = settings.tool_choice
    if choice is not None and not isinstance(choice, str):
        raise UnsupportedDurableOpenAIAgentsError(
            "durable OpenAI Agents runs support only named tool_choice values"
        )
    return choice


def native_settings(
    registered: ModelSettings, options: dict[str, object], choice: object
) -> ModelSettings:
    """Tool Pod side: apply transported options over the registered settings."""
    unknown = sorted(set(options) - {*_TRANSPORTED, "reasoning_effort", "verbosity"})
    if unknown:
        raise UnsupportedDurableOpenAIAgentsError(
            f"unsupported OpenAI model options: {', '.join(unknown)}"
        )
    if choice is not None and not isinstance(choice, str):
        raise UnsupportedDurableOpenAIAgentsError("tool_choice must be a string")
    effort = options.get("reasoning_effort")
    verbosity = options.get("verbosity")
    per_call = ModelSettings(
        **{name: cast(Any, options.get(name)) for name in _TRANSPORTED},
        tool_choice=cast(Any, choice),
        reasoning=None
        if effort is None
        else Reasoning(effort=cast(Any, _one_of(effort, _EFFORTS, "reasoning_effort"))),
        verbosity=None
        if verbosity is None
        else cast(Any, _one_of(verbosity, _VERBOSITIES, "verbosity")),
    )
    return registered.resolve(per_call)


class TransportedOutputSchema(AgentOutputSchemaBase):
    """Tool Pod side: the Runner's structured-output schema, as transported."""

    def __init__(self, response_format: object) -> None:
        fmt = (
            cast(dict[str, Any], response_format)
            if isinstance(response_format, dict)
            else {}
        )
        spec = fmt.get("json_schema") if fmt.get("type") == "json_schema" else None
        if not isinstance(spec, dict):
            raise UnsupportedDurableOpenAIAgentsError(
                "response_format must be a json_schema format"
            )
        spec = cast(dict[str, Any], spec)
        name, schema, strict = spec.get("name"), spec.get("schema"), spec.get("strict")
        if (
            not isinstance(name, str)
            or not isinstance(schema, dict)
            or not isinstance(strict, bool)
        ):
            raise UnsupportedDurableOpenAIAgentsError(
                "response_format json_schema requires name, schema, and strict"
            )
        self._name = name
        self._schema = cast(dict[str, Any], schema)
        self._strict = strict

    def is_plain_text(self) -> bool:
        return False

    def name(self) -> str:
        return self._name

    def json_schema(self) -> dict[str, Any]:
        return self._schema

    def is_strict_json_schema(self) -> bool:
        return self._strict

    def validate_json(self, json_str: str) -> Any:
        # The Runner validates the final output in the AER against the real
        # output type; the Tool Pod only forwards the model's text.
        return json.loads(json_str)
