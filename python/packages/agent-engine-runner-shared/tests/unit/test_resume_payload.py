"""Tests for OE-to-AER resume payload selection."""

from agent_engine_runner_shared.models import ExecuteRequest
from agent_engine_runner_shared.server.aer import _resolve_resume_payload


def _request(**kwargs) -> ExecuteRequest:
    return ExecuteRequest(
        execution_id="exec-1",
        platform_api_url="http://oe:8000",
        resume=True,
        **kwargs,
    )


class TestResolveResumePayload:
    def test_no_resume_data_is_none(self):
        assert _resolve_resume_payload(_request()) is None

    def test_resume_data_passes_through(self):
        answers = {"int-1": {"approved": True}}
        assert _resolve_resume_payload(_request(resume_data=answers)) == answers

    def test_legacy_resume_message_is_ignored(self):
        assert _resolve_resume_payload(_request(resume_message="yes, go ahead")) is None
