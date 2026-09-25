"""Tests for bidirectional message conversion between platform and ADK."""

from __future__ import annotations

import json

from agent_engine_sdk.models import Message
from google.genai import types

from agent_engine_sdk_adk.messages import (
    content_to_platform_messages,
    platform_message_to_content,
    tools_dict_to_schemas,
)


class TestPlatformToADK:
    def test_user_message_to_content(self) -> None:
        msg = Message(role="user", content="hello")
        content = platform_message_to_content(msg)

        assert content.role == "user"
        assert len(content.parts) == 1
        assert content.parts[0].text == "hello"


class TestADKToPlatform:
    def test_model_content_to_assistant_message(self) -> None:
        content = types.Content(
            role="model",
            parts=[types.Part.from_text(text="I can help with that.")],
        )
        msgs = content_to_platform_messages(content)

        assert len(msgs) == 1
        assert msgs[0].role == "assistant"
        assert msgs[0].content == "I can help with that."

    def test_function_call_to_tool_calls(self) -> None:
        content = types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name="search", args={"query": "tokyo"}
                    )
                )
            ],
        )
        msgs = content_to_platform_messages(content)

        assert len(msgs) == 1
        assert msgs[0].role == "assistant"
        assert msgs[0].tool_calls is not None
        assert len(msgs[0].tool_calls) == 1
        assert msgs[0].tool_calls[0].name == "search"
        assert msgs[0].tool_calls[0].args == {"query": "tokyo"}

    def test_function_response_to_tool_message(self) -> None:
        content = types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        name="search",
                        response={"items": ["sushi", "ramen"]},
                    )
                )
            ],
        )
        msgs = content_to_platform_messages(content)

        assert len(msgs) == 1
        assert msgs[0].role == "tool"
        assert msgs[0].name == "search"
        assert "sushi" in msgs[0].content  # type: ignore[operator]

    def test_function_response_json_is_stable_after_protobuf_round_trip(self) -> None:
        original = types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        name="load_claim",
                        response={
                            "claim_id": "CLM-1",
                            "coverage": {"limit": 25000, "deductible": 500},
                            "payments": [1200, {"sequence": 2, "amount": 750}],
                            "flagged": False,
                            "note": None,
                        },
                    )
                )
            ],
        )
        replayed = types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        name="load_claim",
                        response={
                            "note": None,
                            "flagged": False,
                            "coverage": {"deductible": 500.0, "limit": 25000.0},
                            "payments": [1200.0, {"amount": 750.0, "sequence": 2.0}],
                            "claim_id": "CLM-1",
                        },
                    )
                )
            ],
        )

        original_message = content_to_platform_messages(original)[0]
        replayed_message = content_to_platform_messages(replayed)[0]

        assert isinstance(original_message.content, str)
        assert original_message.content == replayed_message.content
        assert json.loads(original_message.content) == {
            "claim_id": "CLM-1",
            "coverage": {"deductible": 500, "limit": 25000},
            "flagged": False,
            "note": None,
            "payments": [1200, {"amount": 750, "sequence": 2}],
        }

    def test_function_response_preserves_existing_json_whitespace(self) -> None:
        content = types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        name="save_claim",
                        response={"status": "saved"},
                    )
                )
            ],
        )

        message = content_to_platform_messages(content)[0]

        assert message.content == '{"status": "saved"}'

    def test_function_response_preserves_tool_call_id(self) -> None:
        # The id correlates the result with the assistant tool_call that
        # requested it. STM records it, so dropping it breaks correlation
        # and diverges from the LangGraph adapter's memory parity behavior.
        content = types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id="call-abc", name="search", response={"ok": True}
                    )
                )
            ],
        )
        msgs = content_to_platform_messages(content)

        assert msgs[0].tool_call_id == "call-abc"

    def test_function_response_without_id_is_none(self) -> None:
        content = types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        name="search", response={"ok": True}
                    )
                )
            ],
        )
        msgs = content_to_platform_messages(content)

        assert msgs[0].tool_call_id is None

    def test_user_role_control_frame_is_dropped(self) -> None:
        # ADK's auth/confirmation events are role="user" Content carrying only
        # pseudo-function_calls (adk_request_credential / adk_request_confirmation).
        # Persisting them would write a blank user turn to STM and lose the call
        # detail. Control frames are not
        # conversation; they resume from ADK's session service, so drop them.
        content = types.Content(
            role="user",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        id="adk_xyz",
                        name="adk_request_credential",
                        args={"function_call_id": "orig-1"},
                    )
                )
            ],
        )
        assert content_to_platform_messages(content) == []

    def test_model_issued_request_input_stays_paired_with_tool_response(self) -> None:
        # RequestInput is a model-issued tool. Dropping it leaves the later
        # function response as a tool-role message with no preceding tool_calls,
        # which the LLM provider rejects on continue.
        content = types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        id="manager_approval",
                        name="adk_request_input",
                        args={"message": "review"},
                    )
                )
            ],
        )
        msgs = content_to_platform_messages(content)
        assert len(msgs) == 1
        assert msgs[0].role == "assistant"
        assert msgs[0].tool_calls is not None
        assert msgs[0].tool_calls[0].id == "manager_approval"
        assert msgs[0].tool_calls[0].name == "adk_request_input"

    def test_model_role_confirmation_control_frame_is_dropped(self) -> None:
        content = types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        id="call_bind",
                        name="adk_request_confirmation",
                        args={"hint": "approve bind_policy?"},
                    )
                )
            ],
        )
        assert content_to_platform_messages(content) == []

    def test_confirmation_function_response_is_dropped(self) -> None:
        content = types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id="call_bind",
                        name="adk_request_confirmation",
                        response={"confirmed": True},
                    )
                )
            ],
        )
        assert content_to_platform_messages(content) == []

    def test_parallel_function_responses_each_become_a_message(self) -> None:
        # ADK packs parallel tool results into one Content; every response must
        # survive conversion or the model re-issues the unanswered tool calls.
        content = types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        name="save_customer_info", response={"status": "saved"}
                    )
                ),
                types.Part(
                    function_response=types.FunctionResponse(
                        name="get_quote", response={"premium": "$180"}
                    )
                ),
            ],
        )
        msgs = content_to_platform_messages(content)

        assert [m.name for m in msgs] == ["save_customer_info", "get_quote"]
        assert all(m.role == "tool" for m in msgs)

    def test_parallel_function_responses_keep_distinct_ids(self) -> None:
        content = types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id="c1", name="a", response={"r": 1}
                    )
                ),
                types.Part(
                    function_response=types.FunctionResponse(
                        id="c2", name="b", response={"r": 2}
                    )
                ),
            ],
        )
        msgs = content_to_platform_messages(content)

        assert [m.tool_call_id for m in msgs] == ["c1", "c2"]


class TestPlatformToolToADK:
    def test_tool_message_to_function_response(self) -> None:
        msg = Message(
            role="tool",
            content='{"items": ["sushi"]}',
            name="search",
            tool_call_id="call_123",
        )
        content = platform_message_to_content(msg)

        assert content.role == "user"
        assert content.parts is not None
        assert len(content.parts) == 1
        assert content.parts[0].function_response is not None
        assert content.parts[0].function_response.name == "search"
        assert content.parts[0].function_response.id == "call_123"


class TestToolsSerialization:
    def test_tools_dict_to_schemas(self) -> None:
        from google.adk.tools import FunctionTool

        def greet(name: str) -> str:
            """Say hello."""
            return f"hi {name}"

        tool = FunctionTool(func=greet)
        tools_dict = {"greet": tool}
        schemas = tools_dict_to_schemas(tools_dict)

        assert len(schemas) == 1
        assert schemas[0].name == "greet"
        assert schemas[0].description == "Say hello."
