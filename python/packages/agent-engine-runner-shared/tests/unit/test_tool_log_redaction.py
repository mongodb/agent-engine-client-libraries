"""
Unit tests for tool request/result logging.

Mirrors the TypeScript tests/unit/tool_log_redaction.test.ts.

Platform debug logs must never emit argument or result bodies:
serializing them on the invoke path stalls large-payload tools, and the
catalog redact_fields contract is satisfied by omitting values entirely.
"""

import logging
from unittest.mock import Mock, patch

import pytest

from agent_engine_runner_shared.utils import log_tool_request, log_tool_result

_UTILS_LOGGER = "agent_engine_runner_shared.utils"


def _debug_output(caplog) -> str:
    return "\n".join(r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG)


class TestLogToolRequest:
    def test_debug_lists_keys_and_omits_values(self, caplog):
        with caplog.at_level(logging.DEBUG, logger=_UTILS_LOGGER):
            log_tool_request(
                "stripe_charge",
                {"card_number": "4111111111111111", "cvv": "123", "amount": 42},
                1,
                fields_to_redact=["card_number", "cvv"],
            )
        out = _debug_output(caplog)
        assert "card_number=<redacted>" in out
        assert "cvv=<redacted>" in out
        assert "amount=<number>" in out
        assert "values_omitted=true" in out
        assert "4111111111111111" not in out
        assert "123" not in out

    def test_debug_omits_argument_values_without_policy(self, caplog):
        with caplog.at_level(logging.DEBUG, logger=_UTILS_LOGGER):
            log_tool_request("get_weather", {"city": "Tokyo"}, 1)
        out = _debug_output(caplog)
        assert "city=<string chars=5>" in out
        assert "Tokyo" not in out

    def test_debug_does_not_emit_large_argument_body(self, caplog):
        blob = "PAYLOAD_BODY_" + ("x" * 200_000)
        with caplog.at_level(logging.DEBUG, logger=_UTILS_LOGGER):
            log_tool_request("filesystem_write", {"content": blob, "path": "/tmp/out"}, 1)
        out = _debug_output(caplog)
        assert blob not in out
        assert "PAYLOAD_BODY_" not in out
        assert "content=<string chars=200013>" in out
        assert "path=<string chars=8>" in out

    def test_debug_caps_fields_and_field_name_length(self, caplog):
        csi = "\x9b"
        long_key = "field\n" + csi + ("x" * 100)
        arguments = {long_key: "hidden", **{f"key_{i}": i for i in range(25)}}
        with caplog.at_level(logging.DEBUG, logger=_UTILS_LOGGER):
            log_tool_request("wide_tool", arguments, 1)
        out = _debug_output(caplog)
        assert "field??" in out
        assert long_key not in out
        assert csi not in out
        assert "key_18=<number>" in out
        assert "key_19" not in out
        assert "fields_truncated=true" in out

    def test_info_level_skips_argument_summary(self, caplog):
        blob = "SECRET_BODY" * 1000
        with (
            patch(
                "agent_engine_runner_shared.utils._payload_debug_summary",
                side_effect=AssertionError("summary should not run"),
            ),
            caplog.at_level(logging.INFO, logger=_UTILS_LOGGER),
        ):
            log_tool_request("filesystem_write", {"content": blob}, 1)
        assert blob not in caplog.text
        assert _debug_output(caplog) == ""


class TestLogToolResult:
    def test_debug_does_not_stringify_large_result(self, caplog):
        blob = "RESULT_BODY_" + ("y" * 200_000)
        with caplog.at_level(logging.DEBUG, logger=_UTILS_LOGGER):
            log_tool_result("filesystem_write", 1, "success", result=blob, duration_ms=12)
        out = _debug_output(caplog)
        assert blob not in out
        assert "RESULT_BODY_" not in out
        assert "type=string chars=200012" in out
        assert "values_omitted=true" in out

    def test_debug_does_not_emit_result_object_keys(self, caplog):
        sensitive_key = "customer-token-as-key"
        with caplog.at_level(logging.DEBUG, logger=_UTILS_LOGGER):
            log_tool_result(
                "lookup_tool",
                1,
                "success",
                result={sensitive_key: "hidden-value"},
                duration_ms=12,
            )
        out = _debug_output(caplog)
        assert "type=object" in out
        assert sensitive_key not in out
        assert "hidden-value" not in out

    def test_debug_does_not_walk_result_collection(self, caplog):
        class ExplodingValue:
            def __str__(self):
                raise AssertionError("nested result value was stringified")

        result = [ExplodingValue()] * 1_000
        with caplog.at_level(logging.DEBUG, logger=_UTILS_LOGGER):
            log_tool_result("list_tool", 1, "success", result=result)
        assert "type=array items=1000" in _debug_output(caplog)

    def test_info_level_skips_result_summary(self, caplog):
        result = {"customer-token-as-key": "hidden-value"}
        with (
            patch(
                "agent_engine_runner_shared.utils._result_debug_summary",
                side_effect=AssertionError("summary should not run"),
            ),
            caplog.at_level(logging.INFO, logger=_UTILS_LOGGER),
        ):
            log_tool_result("lookup_tool", 1, "success", result=result)
        assert "customer-token-as-key" not in caplog.text
        assert "hidden-value" not in caplog.text
        assert _debug_output(caplog) == ""


class TestToolServerHandleExecuteRedaction:
    @pytest.mark.asyncio
    async def test_applies_tool_metadata_redact_fields(self, caplog):
        from agent_engine_runner_shared.models import ToolPodExecuteRequest
        from agent_engine_runner_shared.server.tool import ToolServer

        mock_runtime = Mock()
        mock_runtime._tools = {"stripe_charge": lambda **kwargs: {"ok": True}}
        mock_runtime._tool_definitions = {"stripe_charge": {"redact_fields": ["card_number"]}}
        server = ToolServer(mock_runtime)

        request = ToolPodExecuteRequest(
            execution_id="exec-1",
            tool_name="stripe_charge",
            arguments={"card_number": "4111111111111111", "amount": 42},
            session_id="session-1",
        )
        with caplog.at_level(logging.DEBUG, logger=_UTILS_LOGGER):
            response = await server._handle_execute(request)

        assert response.status == "success"
        out = _debug_output(caplog)
        assert "card_number=<redacted>" in out
        assert "amount=<number>" in out
        assert "4111111111111111" not in out

    @pytest.mark.asyncio
    async def test_unknown_tool_still_omits_argument_values(self, caplog):
        """No registered policy — values are still omitted."""
        from agent_engine_runner_shared.models import ToolPodExecuteRequest
        from agent_engine_runner_shared.server.tool import ToolServer

        mock_runtime = Mock()
        mock_runtime._tools = {}
        mock_runtime._tool_definitions = {}
        mock_runtime.agent_config.mcp.servers = []
        server = ToolServer(mock_runtime)

        request = ToolPodExecuteRequest(
            execution_id="exec-1",
            tool_name="ghost_tool",
            arguments={"hint": "visible"},
            session_id="session-1",
        )
        with caplog.at_level(logging.DEBUG, logger=_UTILS_LOGGER):
            await server._handle_execute(request)

        out = _debug_output(caplog)
        assert "hint=<string chars=7>" in out
        assert "visible" not in out


class TestExecuteToolRedaction:
    def test_redact_fields_reach_debug_argument_metadata(self, caplog):
        from agent_engine_runner_shared.models import ToolExecuteResponse
        from agent_engine_runner_shared.secure_wrapper import SecureToolWrapper

        wrapper = SecureToolWrapper(oe_url="http://localhost:8080", execution_id="exec-123")

        with patch("agent_engine_runner_shared.secure_wrapper.request_oe_approval") as mock_request:
            mock_request.return_value = ToolExecuteResponse(
                proceed=True,
                status="success",
                result={"ok": True},
                duration_ms=1.0,
            )
            with caplog.at_level(logging.DEBUG, logger=_UTILS_LOGGER):
                wrapper.execute_tool(
                    "stripe_charge",
                    {"card_number": "4111111111111111", "amount": 42},
                    redact_fields=["card_number"],
                )

        out = _debug_output(caplog)
        assert "card_number=<redacted>" in out
        assert "4111111111111111" not in out

    def test_create_secure_tool_function_forwards_redact_fields(self):
        from agent_engine_runner_shared.secure_wrapper import create_secure_tool_function

        wrapper = Mock()
        wrapper.execute_tool = Mock(return_value="ok")

        # create_secure_tool_function imports get_current_wrapper at call time,
        # so both the creation and the invocation must happen under the patch.
        with patch("agent_engine_runner_shared.context.get_current_wrapper", return_value=wrapper):
            wrapped = create_secure_tool_function(
                original_tool=lambda **kwargs: "unused",
                tool_name="stripe_charge",
                redact_fields=["card_number"],
            )
            wrapped(card_number="4111111111111111")

        wrapper.execute_tool.assert_called_once()
        assert wrapper.execute_tool.call_args.kwargs["redact_fields"] == ["card_number"]
