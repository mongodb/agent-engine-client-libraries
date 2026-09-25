"""Tests for ADKLLMAdapter — platform BaseLLM wrapping ADK BaseLlm."""

from __future__ import annotations

import json
from typing import Any, Literal
from unittest.mock import MagicMock

import pytest
from agent_engine_sdk.models import Message
from google.adk.models import LlmResponse
from google.genai import types
from pydantic import BaseModel

from agent_engine_sdk_adk.messages import tools_dict_to_schemas


class _Coverage(BaseModel):
    level: Literal["full"]


def _quote(coverage: _Coverage) -> str:
    """Quote insurance coverage."""
    return coverage.level


class TestADKLLMAdapter:
    def test_request_preserves_adk_json_schema(self) -> None:
        from google.adk.tools import FunctionTool

        from agent_engine_sdk_adk.llm_adapter import ADKLLMAdapter

        schemas = tools_dict_to_schemas({"quote": FunctionTool(func=_quote)})
        raw_schema = schemas[0].parameters
        assert isinstance(raw_schema, dict)
        assert "$defs" in raw_schema
        assert raw_schema["properties"]["coverage"]["$ref"] == "#/$defs/_Coverage"
        assert raw_schema["$defs"]["_Coverage"]["properties"]["level"]["const"] == (
            "full"
        )

        mock_llm = MagicMock()
        mock_llm.model = "gemini-2.5-flash"
        request = ADKLLMAdapter(llm=mock_llm, tools=schemas)._build_request(
            [Message(role="user", content="quote me")]
        )

        declaration = request.config.tools[0].function_declarations[0]
        assert declaration.parameters_json_schema == raw_schema

    def test_init_accepts_tool_choice_kwarg(self) -> None:
        """The shared adapter factory passes tool_choice. ADK has no
        equivalent forced-tool knob, so it must accept the kwarg without
        error to keep the factory contract."""
        from agent_engine_sdk_adk.llm_adapter import ADKLLMAdapter

        mock_llm = MagicMock()
        mock_llm.model = "gemini-2.5-flash"

        adapter = ADKLLMAdapter(llm=mock_llm, tools=None, tool_choice="Brief")

        assert adapter._llm is mock_llm

    @pytest.mark.asyncio
    async def test_ainvoke_converts_messages_and_returns_response(self) -> None:
        from agent_engine_sdk_adk.llm_adapter import ADKLLMAdapter

        mock_llm = MagicMock()
        mock_llm.model = "gemini-2.5-flash"

        response_content = types.Content(
            role="model",
            parts=[types.Part.from_text(text="Hello from ADK")],
        )

        async def fake_generate(request: Any, stream: bool = False) -> Any:
            yield LlmResponse(content=response_content, partial=False)

        mock_llm.generate_content_async = fake_generate

        adapter = ADKLLMAdapter(llm=mock_llm)
        messages = [Message(role="user", content="hi")]

        result = await adapter.ainvoke(messages)

        assert result.content == "Hello from ADK"

    @pytest.mark.asyncio
    async def test_request_carries_model_name(self) -> None:
        """The Tool Pod calls generate_content_async directly, so the adapter
        must set the model on the LlmRequest — otherwise the provider is hit
        with model=None and returns 404."""
        from agent_engine_sdk_adk.llm_adapter import ADKLLMAdapter

        mock_llm = MagicMock()
        mock_llm.model = "gemini-2.5-flash"
        captured: dict[str, Any] = {}

        async def fake_generate(request: Any, stream: bool = False) -> Any:
            captured["model"] = request.model
            yield LlmResponse(
                content=types.Content(
                    role="model", parts=[types.Part.from_text(text="ok")]
                ),
                partial=False,
            )

        mock_llm.generate_content_async = fake_generate

        adapter = ADKLLMAdapter(llm=mock_llm)
        await adapter.ainvoke([Message(role="user", content="hi")])

        assert captured["model"] == "gemini-2.5-flash"

    @pytest.mark.asyncio
    async def test_astream_emits_one_chunk_with_indexed_tool_calls(self) -> None:
        """astream must drive stream=False and emit one combined chunk whose
        parallel tool calls each get a distinct index with intact args.

        Regression guard for the lost-tool-args bug: ADK's streaming chunks are
        cumulative, so per-chunk emission let the AER accumulator concatenate
        each call's full args into invalid JSON (parsed as empty). Driving
        stream=False and emitting indexed chunks once keeps args intact.
        """
        from agent_engine_sdk_adk.llm_adapter import ADKLLMAdapter

        mock_llm = MagicMock()
        mock_llm.model = "gemini-2.5-flash"
        captured_stream: dict[str, Any] = {}

        content = types.Content(
            role="model",
            parts=[
                types.Part.from_text(text="Working on it."),
                types.Part(
                    function_call=types.FunctionCall(
                        name="save_customer_info", args={"key": "email"}
                    )
                ),
                types.Part(
                    function_call=types.FunctionCall(
                        name="get_quote", args={"vehicle_year": 2023}
                    )
                ),
            ],
        )

        async def fake_generate(request: Any, stream: bool = False) -> Any:
            captured_stream["stream"] = stream
            yield LlmResponse(content=content, partial=False)

        mock_llm.generate_content_async = fake_generate

        adapter = ADKLLMAdapter(llm=mock_llm)
        chunks = [
            c async for c in adapter.astream([Message(role="user", content="hi")])
        ]

        # stream=False is the fix; one combined chunk is emitted.
        assert captured_stream["stream"] is False
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.content == "Working on it."
        assert chunk.tool_calls is not None
        assert len(chunk.tool_calls) == 2
        # Distinct indices so the accumulator keeps them separate.
        assert [tc.index for tc in chunk.tool_calls] == [0, 1]
        assert [tc.name for tc in chunk.tool_calls] == [
            "save_customer_info",
            "get_quote",
        ]
        # Args are intact per call, not concatenated/emptied.
        assert json.loads(chunk.tool_calls[0].args) == {"key": "email"}
        assert json.loads(chunk.tool_calls[1].args) == {"vehicle_year": 2023}
