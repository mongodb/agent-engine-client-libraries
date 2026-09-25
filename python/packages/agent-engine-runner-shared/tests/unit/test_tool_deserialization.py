"""Tests for tool.py typed message deserialization."""

from agent_engine_sdk import LLMToolCall, Message

from agent_engine_runner_shared.server.tool import _deserialize_messages


class TestDeserializeMessages:
    """Tests for _deserialize_messages with typed sdk-core messages."""

    def test_role_user(self):
        """role='user' is preserved correctly."""
        result = _deserialize_messages([Message(role="user", content="hello")])
        assert len(result) == 1
        assert isinstance(result[0], Message)
        assert result[0].role == "user"
        assert result[0].content == "hello"

    def test_role_assistant_with_tool_calls(self):
        """role='assistant' with tool_calls is preserved correctly.

        the OE sends AIMessages as
        {role: "assistant", content: "", tool_calls: [...]}. If deserialized
        as a HumanMessage, tool_calls are lost and gemini loops forever.
        """
        tool_calls = [LLMToolCall(id="tc1", name="list_customer_policies", args={})]
        result = _deserialize_messages(
            [Message(role="assistant", content="", tool_calls=tool_calls)]
        )
        assert result[0].role == "assistant"
        assert result[0].tool_calls == tool_calls

    def test_role_system(self):
        """role='system' is preserved correctly."""
        result = _deserialize_messages([Message(role="system", content="you are a bot")])
        assert result[0].role == "system"

    def test_role_tool(self):
        """role='tool' with tool_call_id is preserved correctly."""
        result = _deserialize_messages(
            [Message(role="tool", content='{"count": 0}', tool_call_id="tc1", name="lookup")]
        )
        assert result[0].role == "tool"
        assert result[0].tool_call_id == "tc1"
        assert result[0].name == "lookup"

    def test_preserves_tool_calls(self):
        """Preserves tool_calls list on assistant messages."""
        tool_calls = [LLMToolCall(id="tc1", name="search", args={"q": "test"})]
        result = _deserialize_messages(
            [Message(role="assistant", content="", tool_calls=tool_calls)]
        )
        assert result[0].tool_calls == tool_calls

    def test_preserves_name(self):
        """Preserves name field on tool messages."""
        result = _deserialize_messages(
            [Message(role="tool", content="result", name="my_tool", tool_call_id="tc1")]
        )
        assert result[0].name == "my_tool"

    def test_full_conversation(self):
        """Full conversation as produced by Message.model_dump()."""
        msgs = [
            Message(role="system", content="You are a helper."),
            Message(role="user", content="What policies do I have?"),
            Message(
                role="assistant",
                content="",
                tool_calls=[LLMToolCall(id="tc1", name="list_policies", args={})],
            ),
            Message(
                role="tool",
                content='{"count": 0}',
                tool_call_id="tc1",
                name="list_policies",
            ),
        ]
        result = _deserialize_messages(msgs)
        assert [m.role for m in result] == ["system", "user", "assistant", "tool"]
        assert result[2].tool_calls == [LLMToolCall(id="tc1", name="list_policies", args={})]
        assert result[3].tool_call_id == "tc1"
