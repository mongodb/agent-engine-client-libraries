"""Tests for token/cost/workspace_id field propagation and execution persistence."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from agent_engine_runner_shared.models import (
    ExecuteRequest,
    Execution,
    ExecutionStatus,
    ExecutionStep,
    InvokeRequest,
    ToolExecuteRequest,
    ToolPodExecuteRequest,
    ToolResultRequest,
)


class TestToolExecuteRequestFields:
    """Tests for ToolExecuteRequest field defaults and round-trips."""

    def _make_request(self, **overrides) -> ToolExecuteRequest:
        defaults = dict(
            execution_id="exec-1",
            tool_name="get_weather",
            arguments={"city": "Tokyo"},
            step_number=1,
        )
        defaults.update(overrides)
        return ToolExecuteRequest(**defaults)

    def test_redact_fields_default_to_empty(self):
        req = self._make_request()
        assert req.redact_fields == []

    def test_is_local_defaults_to_true(self):
        req = self._make_request()
        assert req.is_local is True

    def test_redact_fields_round_trip(self):
        req = self._make_request(redact_fields=["api_key"])
        data = req.model_dump()
        assert data["redact_fields"] == ["api_key"]
        restored = ToolExecuteRequest(**data)
        assert restored.redact_fields == ["api_key"]


class TestExecutionPersistence:
    """Tests for Execution.to_persistence_doc() with workspace_id."""

    def test_workspace_id_in_persistence_doc(self):
        """workspace_id should appear in the persistence document."""
        execution = Execution(
            id="exec-1",
            status=ExecutionStatus.RUNNING,
            message="test",
            workspace_id="my-agent",
        )
        doc = execution.to_persistence_doc()
        assert doc["workspace_id"] == "my-agent"

    def test_workspace_id_none_in_persistence_doc(self):
        """None workspace_id should appear as None in persistence doc."""
        execution = Execution(
            id="exec-1",
            status=ExecutionStatus.RUNNING,
            message="test",
        )
        doc = execution.to_persistence_doc()
        assert doc["workspace_id"] is None

    def test_persistence_doc_has_expected_keys(self):
        """Persistence doc should contain all required keys."""
        execution = Execution(
            id="exec-1",
            status=ExecutionStatus.RUNNING,
            message="test",
            workspace_id="agent-1",
            org_id="org-1",
            session_id="sess-1",
        )
        doc = execution.to_persistence_doc()
        expected_keys = {
            "execution_id",
            "status",
            "message",
            "session_id",
            "user_id",
            "org_id",
            "workspace_id",
            "project_id",
            "result",
            "error",
            "suspend_reason",
            "suspend_context",
            "updated_at",
        }
        assert set(doc.keys()) == expected_keys


class TestPydanticSerialization:
    """Tests for serialization round-trips with new fields."""

    def test_tool_result_request_round_trip(self):
        """ToolResultRequest should survive model_dump → construct round-trip."""
        original = ToolResultRequest(
            execution_id="exec-1",
            step_number=1,
            tool_name="invoke_llm",
            status="success",
            result="Hello",
            duration_ms=100.0,
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            model="gpt-4o",
            workspace_id="my-agent",
        )
        data = original.model_dump()
        restored = ToolResultRequest(**data)
        assert restored.prompt_tokens == 100
        assert restored.completion_tokens == 50
        assert restored.total_tokens == 150
        assert restored.model == "gpt-4o"
        assert restored.workspace_id == "my-agent"

    def test_backward_compat_missing_new_fields(self):
        """Old payloads without token fields should default to None."""
        old_payload = {
            "execution_id": "exec-1",
            "step_number": 1,
            "tool_name": "invoke_llm",
            "status": "success",
            "result": "Hello",
            "duration_ms": 100.0,
            # No token/cost/workspace_id fields
        }
        req = ToolResultRequest(**old_payload)
        assert req.prompt_tokens is None
        assert req.completion_tokens is None
        assert req.total_tokens is None
        assert req.model is None
        assert req.workspace_id is None

    def test_invoke_request_workspace_id(self):
        """InvokeRequest should accept and serialize workspace_id."""
        req = InvokeRequest(message="hello", workspace_id="test-agent")
        data = req.model_dump()
        assert data["workspace_id"] == "test-agent"

    def test_execute_request_workspace_id(self):
        """ExecuteRequest should accept and serialize workspace_id."""
        req = ExecuteRequest(
            execution_id="exec-1",
            message="hello",
            platform_api_url="http://localhost:8080",
            workspace_id="test-agent",
        )
        data = req.model_dump()
        assert data["workspace_id"] == "test-agent"


class TestOwnerCallbackUrlFields:
    """Tests for the optional owner-URL fallback fields.

    These fields drive owner-first callback routing. These tests prove the
    wire-compatible shape.
    """

    def test_execute_request_without_owner_url_still_parses(self):
        """Old callers that don't send platform_api_owner_url still parse."""
        req = ExecuteRequest(
            execution_id="exec-1",
            message="hello",
            platform_api_url="http://localhost:8080",
        )
        assert req.platform_api_owner_url is None

    def test_execute_request_with_owner_url_round_trips(self):
        """New callers can set platform_api_owner_url and read it back."""
        req = ExecuteRequest(
            execution_id="exec-1",
            message="hello",
            platform_api_url="http://localhost:8080",
            platform_api_owner_url="http://oe-replica-1.internal:8080",
        )
        assert req.platform_api_owner_url == "http://oe-replica-1.internal:8080"
        data = req.model_dump()
        assert data["platform_api_owner_url"] == "http://oe-replica-1.internal:8080"
        restored = ExecuteRequest(**data)
        assert restored.platform_api_owner_url == "http://oe-replica-1.internal:8080"

    def test_execute_request_unknown_extra_field_ignored(self):
        """Unknown fields on the wire (e.g. from a newer sender) don't fail parsing."""
        req = ExecuteRequest(
            execution_id="exec-1",
            message="hello",
            platform_api_url="http://localhost:8080",
            some_future_field="unexpected",
        )
        assert not hasattr(req, "some_future_field")

    def test_tool_pod_execute_request_without_owner_url_still_parses(self):
        """Old callers that don't send oe_owner_url still parse."""
        req = ToolPodExecuteRequest(
            execution_id="exec-1",
            tool_name="get_weather",
            arguments={"city": "Tokyo"},
            session_id="sess-1",
        )
        assert req.oe_owner_url is None

    def test_tool_pod_execute_request_with_owner_url_round_trips(self):
        """New callers can set oe_owner_url and read it back."""
        req = ToolPodExecuteRequest(
            execution_id="exec-1",
            tool_name="get_weather",
            arguments={"city": "Tokyo"},
            session_id="sess-1",
            oe_owner_url="http://oe-replica-1.internal:8080",
        )
        assert req.oe_owner_url == "http://oe-replica-1.internal:8080"
        data = req.model_dump()
        assert data["oe_owner_url"] == "http://oe-replica-1.internal:8080"
        restored = ToolPodExecuteRequest(**data)
        assert restored.oe_owner_url == "http://oe-replica-1.internal:8080"

    def test_tool_pod_execute_request_unknown_extra_field_ignored(self):
        """Unknown fields on the wire (e.g. from a newer sender) don't fail parsing."""
        req = ToolPodExecuteRequest(
            execution_id="exec-1",
            tool_name="get_weather",
            arguments={"city": "Tokyo"},
            session_id="sess-1",
            some_future_field="unexpected",
        )
        assert not hasattr(req, "some_future_field")

    def test_tool_pod_execute_request_rejects_oversized_arguments(self, monkeypatch):
        import agent_engine_runner_shared.models as models

        monkeypatch.setattr(models, "MAX_TOOL_ARGUMENT_BYTES", 20)

        with pytest.raises(ValidationError, match="tool arguments exceed 20 bytes"):
            ToolPodExecuteRequest(
                execution_id="exec-1",
                tool_name="write",
                arguments={"content": "more than twenty bytes"},
                session_id="sess-1",
            )

    def test_tool_pod_execute_request_counts_utf8_bytes(self, monkeypatch):
        import agent_engine_runner_shared.models as models

        monkeypatch.setattr(models, "MAX_TOOL_ARGUMENT_BYTES", 19)

        with pytest.raises(ValidationError, match="tool arguments exceed 19 bytes"):
            ToolPodExecuteRequest(
                execution_id="exec-1",
                tool_name="write",
                arguments={"content": "\u20ac\u20ac\u20ac"},
                session_id="sess-1",
            )


class TestExecutionCheckpointPersistence:
    """Resume state is opaque to the platform; no checkpoint fields persist."""

    def test_to_persistence_doc_has_no_checkpoint_fields(self):
        execution = Execution(
            id="exec-1",
            status=ExecutionStatus.SUSPENDED,
            message="test",
        )
        doc = execution.to_persistence_doc()
        assert "checkpoint_id" not in doc
        assert "state_snapshot" not in doc


class TestExecutionFromPersistenceDoc:
    """Tests for Execution.from_persistence_doc() rehydration."""

    def _make_doc(self, **overrides):
        defaults = {
            "execution_id": "exec-1",
            "status": "suspended",
            "message": "I was in an accident",
            "session_id": "sess-1",
            "user_id": "user-1",
            "thread_id": "thread-1",
            "org_id": "org-1",
            "workspace_id": "insurance-agent",
            "result": None,
            "error": None,
            "suspend_reason": "awaiting_human_review",
            "suspend_context": {"claim_id": "C-123", "claim_amount": 15000},
            "checkpoint_id": "ckpt-abc123",
            "state_snapshot": {"key": "value"},
            "created_at": datetime(2025, 5, 20, 12, 0, 0, tzinfo=timezone.utc),
            "updated_at": datetime(2025, 5, 20, 12, 1, 0, tzinfo=timezone.utc),
        }
        defaults.update(overrides)
        return defaults

    def test_round_trip(self):
        """Serialize then deserialize should preserve all fields."""
        original = Execution(
            id="exec-1",
            status=ExecutionStatus.SUSPENDED,
            message="test",
            session_id="sess-1",
            user_id="user-1",
            org_id="org-1",
            workspace_id="my-agent",
            suspend_reason="awaiting_human_review",
            suspend_context={"claim_id": "C-123"},
        )
        doc = original.to_persistence_doc()
        doc["created_at"] = original.created_at
        restored = Execution.from_persistence_doc(doc)

        assert restored.id == original.id
        assert restored.status == original.status
        assert restored.message == original.message
        assert restored.session_id == original.session_id
        assert restored.org_id == original.org_id
        assert restored.suspend_reason == original.suspend_reason
        assert restored.suspend_context == original.suspend_context
        assert restored.workspace_id == original.workspace_id

    def test_missing_optional_fields(self):
        """Old documents missing optional fields should deserialize safely."""
        doc = {
            "execution_id": "exec-old",
            "status": "suspended",
            "message": "old message",
            # No workspace_id, session_id, etc.
        }
        execution = Execution.from_persistence_doc(doc)
        assert execution.id == "exec-old"
        assert execution.status == ExecutionStatus.SUSPENDED
        assert execution.workspace_id is None
        assert execution.session_id is None

    def test_legacy_checkpoint_fields_ignored(self):
        """Documents persisted before checkpoint_id/state_snapshot were removed
        from the Execution model must still deserialize cleanly."""
        doc = self._make_doc()
        execution = Execution.from_persistence_doc(doc)
        assert execution.id == "exec-1"
        assert not hasattr(execution, "checkpoint_id")
        assert not hasattr(execution, "state_snapshot")

    def test_enum_conversion(self):
        """Status string should convert to ExecutionStatus enum."""
        for status_str in [
            "pending",
            "running",
            "suspended",
            "completed",
            "error",
            "resuming",
            "cancelled",
        ]:
            doc = self._make_doc(status=status_str)
            execution = Execution.from_persistence_doc(doc)
            assert execution.status == ExecutionStatus(status_str)

    def test_execution_status_cancelled_is_valid(self):
        assert ExecutionStatus("cancelled") is ExecutionStatus.CANCELLED

    def test_all_fields_populated(self):
        """Document with all fields should deserialize completely."""
        doc = self._make_doc()
        execution = Execution.from_persistence_doc(doc)
        assert execution.id == "exec-1"
        assert execution.status == ExecutionStatus.SUSPENDED
        assert execution.message == "I was in an accident"
        assert execution.session_id == "sess-1"
        assert execution.user_id == "user-1"
        assert execution.org_id == "org-1"
        assert execution.workspace_id == "insurance-agent"
        assert execution.suspend_reason == "awaiting_human_review"
        assert execution.suspend_context == {"claim_id": "C-123", "claim_amount": 15000}
        assert execution.created_at == datetime(2025, 5, 20, 12, 0, 0, tzinfo=timezone.utc)
        assert execution.updated_at == datetime(2025, 5, 20, 12, 1, 0, tzinfo=timezone.utc)


class TestExecutionStepFromLogDoc:
    """Tests for ExecutionStep.from_log_doc() reconstruction from execution_logs."""

    def test_from_log_doc(self):
        """Should reconstruct ExecutionStep from execution_logs document."""
        ts = datetime(2025, 5, 20, 12, 0, 0, tzinfo=timezone.utc)
        doc = {
            "id": "log-1",
            "execution_id": "exec-1",
            "step_number": 3,
            "tool": "file_claim",
            "inputs": {"policy_number": "P-001", "amount": 15000},
            "status": "success",
            "output": {"claim_id": "C-123"},
            "error": None,
            "duration_ms": 250.5,
            "timestamp": ts,
        }
        step = ExecutionStep.from_log_doc(doc)
        assert step.id == "log-1"
        assert step.execution_id == "exec-1"
        assert step.step_number == 3
        assert step.tool_name == "file_claim"
        assert step.arguments == {"policy_number": "P-001", "amount": 15000}
        assert step.status == "success"
        assert step.result == {"claim_id": "C-123"}
        assert step.duration_ms == 250.5
        assert step.timestamp == ts

    def test_from_log_doc_missing_optional_fields(self):
        """Should handle docs with minimal fields."""
        doc = {
            "execution_id": "exec-1",
            "step_number": 1,
            "tool": "invoke_llm",
            "status": "success",
        }
        step = ExecutionStep.from_log_doc(doc)
        assert step.execution_id == "exec-1"
        assert step.tool_name == "invoke_llm"
        assert step.arguments == {}
        assert step.result is None
        assert step.error is None
        assert step.duration_ms is None


class TestEnsureUtc:
    """Tests for Execution._ensure_utc() timezone normalization."""

    def test_naive_datetime_gets_utc(self):
        """PyMongo returns naive datetimes; _ensure_utc should stamp them as UTC."""
        naive = datetime(2025, 5, 20, 12, 0, 0)
        assert naive.tzinfo is None

        result = Execution._ensure_utc(naive)
        assert result.tzinfo == timezone.utc
        assert result.year == 2025
        assert result.hour == 12

    def test_aware_datetime_unchanged(self):
        """Already-aware datetimes should pass through unmodified."""
        aware = datetime(2025, 5, 20, 12, 0, 0, tzinfo=timezone.utc)
        result = Execution._ensure_utc(aware)
        assert result is aware

    def test_from_persistence_doc_normalizes_naive_timestamps(self):
        """Naive timestamps from MongoDB should become UTC-aware after rehydration."""
        naive_created = datetime(2025, 5, 20, 12, 0, 0)
        naive_updated = datetime(2025, 5, 20, 12, 1, 0)
        doc = {
            "execution_id": "exec-1",
            "status": "suspended",
            "created_at": naive_created,
            "updated_at": naive_updated,
        }
        execution = Execution.from_persistence_doc(doc)
        assert execution.created_at.tzinfo == timezone.utc
        assert execution.updated_at.tzinfo == timezone.utc

    def test_from_persistence_doc_created_at_falls_back_to_updated_at(self):
        """When created_at is missing, should fall back to updated_at."""
        updated = datetime(2025, 5, 20, 12, 1, 0, tzinfo=timezone.utc)
        doc = {
            "execution_id": "exec-1",
            "status": "running",
            "updated_at": updated,
        }
        execution = Execution.from_persistence_doc(doc)
        assert execution.created_at == updated

    def test_from_log_doc_normalizes_naive_timestamp(self):
        """Naive timestamp from execution_logs should become UTC-aware."""
        naive_ts = datetime(2025, 5, 20, 12, 0, 0)
        doc = {
            "execution_id": "exec-1",
            "step_number": 1,
            "tool": "test",
            "status": "success",
            "timestamp": naive_ts,
        }
        step = ExecutionStep.from_log_doc(doc)
        assert step.timestamp.tzinfo == timezone.utc
