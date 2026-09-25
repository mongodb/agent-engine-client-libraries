"""Tests for the format_llm_error helper."""

from agent_engine_runner_shared.utils import format_llm_error


class _FakeAPIError(Exception):
    """Mimics google.genai.errors.APIError's ``.details`` dict attribute."""

    def __init__(self, details: dict):
        self.details = details
        super().__init__(f"{details}")


class _FakeChatGoogleGenerativeAIError(Exception):
    """Mimics langchain_google_genai's wrapper exception."""


class TestFormatLlmError:
    def test_falls_back_to_str_for_plain_exceptions(self):
        assert format_llm_error(ValueError("boom")) == "boom"

    def test_decodes_nested_json_string_in_cause_details(self):
        inner_json = (
            '{\n  "error": {\n    "code": 400,\n    '
            '"message": "API key not valid. Please pass a valid API key.",\n    '
            '"status": "INVALID_ARGUMENT"\n  }\n}'
        )
        cause = _FakeAPIError({"message": inner_json, "status": "INVALID_ARGUMENT"})

        try:
            try:
                raise cause
            except _FakeAPIError as e:
                raise _FakeChatGoogleGenerativeAIError(
                    f"Error calling model 'gemini-2.5-flash' (Bad Request): {e}"
                ) from e
        except _FakeChatGoogleGenerativeAIError as exc:
            result = format_llm_error(exc)

        # No escaped-newline artifacts from repr()-ing the nested JSON string.
        assert "\\n" not in result
        # The nested JSON string was decoded into a real structure, not left opaque.
        assert '"code": 400' in result
        assert "API key not valid" in result

    def test_reads_details_from_exception_itself_not_only_cause(self):
        exc = _FakeAPIError({"message": "plain message", "status": "INVALID_ARGUMENT"})
        result = format_llm_error(exc)
        assert result == '{\n  "message": "plain message",\n  "status": "INVALID_ARGUMENT"\n}'

    def test_non_json_looking_string_values_are_left_untouched(self):
        exc = _FakeAPIError({"message": "not json", "status": "INVALID_ARGUMENT"})
        result = format_llm_error(exc)
        assert "not json" in result

    def test_circular_reference_in_details_falls_back_to_str_without_raising(self):
        details: dict = {"status": "INVALID_ARGUMENT"}
        details["self"] = details  # circular reference -- json.dumps can't serialize this
        exc = _FakeAPIError(details)

        result = format_llm_error(exc)  # must not raise

        assert result == str(exc)

    def test_non_serializable_value_is_stringified_instead_of_raising(self):
        exc = _FakeAPIError({"status": "INVALID_ARGUMENT", "when": object()})

        result = format_llm_error(exc)  # must not raise

        assert '"status": "INVALID_ARGUMENT"' in result
