import warnings
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from agent_engine_runner_shared import TenantRuntime
from agent_engine_runner_shared.context import clear_execution_context, set_execution_context
from agent_engine_runner_shared.generated.workflow.v1.activity_pb2 import (
    ACTIVITY_KIND_MEMORY,
    ActivityContext,
)
from agent_engine_runner_shared.generated.workflow.v1.common_pb2 import (
    WORKFLOW_ERROR_CODE_NONDETERMINISTIC,
    WorkflowIdentity,
)
from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import AttemptContext
from agent_engine_runner_shared.memory import build_context as memory_build_context
from agent_engine_runner_shared.workflow import attempt_context_scope
from agent_engine_runner_shared.workflow.activity import completed_outcome, value_to_json
from agent_engine_runner_shared.workflow.client import (
    ActivityDispatch,
    ActivityReplay,
    WorkflowClientError,
)


def _attempt() -> AttemptContext:
    return AttemptContext(
        attempt_id="attempt-1",
        fencing_token=7,
        owner_id="aer-1",
        workflow_identity=WorkflowIdentity(
            session_id="session-456",
            execution_id="execution-1",
        ),
    )


def _activity_context() -> ActivityContext:
    return ActivityContext(
        workflow_identity=_attempt().workflow_identity,
        activity_id="activity-1",
        attempt_id="attempt-1",
        fencing_token=7,
    )


class _FakeWorkflow:
    def __init__(self, started: object) -> None:
        self.started = started
        self.commands: list[Any] = []
        self.outcomes: list[Any] = []

    def start_activity(self, command: Any) -> object:
        self.commands.append(command)
        if isinstance(self.started, Exception):
            raise self.started
        return self.started

    def report_outcome(self, outcome: Any) -> None:
        self.outcomes.append(outcome)


@contextmanager
def _durable_execution(workflow: _FakeWorkflow):
    tokens = set_execution_context(
        execution_id="execution-1",
        wrapper=SimpleNamespace(workflow=workflow),
        oe_url="http://oe:8000",
        user_id="user-123",
        session_id="session-456",
    )
    try:
        with attempt_context_scope(_attempt()):
            yield
    finally:
        clear_execution_context(tokens)


class TestMemoryBuildContextHelper:
    """Focused tests for agent_engine_runner_shared.memory.build_context max_tokens forwarding."""

    def test_returns_empty_when_memory_is_disabled(self):
        result = memory_build_context(
            memory_engine=None,
            query="q",
            session_id="s1",
            org_id="org1",
            user_id="u1",
            project_id="proj1",
            max_tokens=0,
        )

        assert result == ""

    def test_returns_empty_when_session_is_missing(self):
        engine = Mock()

        result = memory_build_context(
            memory_engine=engine,
            query="q",
            session_id=None,
            org_id="org1",
            user_id="u1",
            project_id="proj1",
            max_tokens=0,
        )

        assert result == ""
        engine.build_context.assert_not_called()

    def test_forwards_explicit_max_tokens_to_engine(self):
        engine = Mock()
        engine.build_context.return_value = SimpleNamespace(formatted_context="ctx")

        result = memory_build_context(
            memory_engine=engine,
            query="q",
            session_id="s1",
            org_id="org1",
            user_id="u1",
            project_id="proj1",
            max_tokens=2048,
        )

        assert result == "ctx"
        kwargs = engine.build_context.call_args[1]
        assert kwargs["max_tokens"] == 2048

    def test_omits_max_tokens_when_absent(self):
        engine = Mock()
        engine.build_context.return_value = SimpleNamespace(formatted_context="ctx")

        result = memory_build_context(
            memory_engine=engine,
            query="q",
            session_id="s1",
            org_id="org1",
            user_id="u1",
            project_id="proj1",
        )

        assert result == "ctx"
        kwargs = engine.build_context.call_args[1]
        assert "max_tokens" not in kwargs

    @pytest.mark.parametrize(
        "max_tokens",
        [0, -1, True, False, 1.5, float("nan"), float("inf"), float("-inf")],
    )
    def test_rejects_invalid_max_tokens(self, max_tokens: object):
        engine = Mock()

        with pytest.raises(ValueError, match="^max_tokens must be a positive integer$"):
            memory_build_context(
                memory_engine=engine,
                query="q",
                session_id="s1",
                org_id="org1",
                user_id="u1",
                project_id="proj1",
                max_tokens=max_tokens,  # type: ignore[arg-type]
            )

        engine.build_context.assert_not_called()


class TestRuntimeMemoryContext:
    """Tests for TenantRuntime memory context delegation."""

    def test_returns_empty_when_memory_is_disabled(self, monkeypatch):
        monkeypatch.setenv("ORG_ID", "org1")
        monkeypatch.setenv("PROJECT_ID", "proj1")
        runtime = TenantRuntime(app_name="Test")
        runtime._memory_engine = None

        assert (
            runtime.build_context(
                query="test",
                user_id="user_123",
                session_id="session_456",
                max_tokens=0,
            )
            == ""
        )

    def test_returns_empty_when_session_is_missing(self, monkeypatch):
        monkeypatch.setenv("ORG_ID", "org1")
        monkeypatch.setenv("PROJECT_ID", "proj1")
        runtime = TenantRuntime(app_name="Test")
        runtime._memory_engine = Mock()

        result = runtime.build_context(
            query="test",
            user_id="user_123",
            session_id=None,
            max_tokens=0,
        )

        assert result == ""
        runtime._memory_engine.build_context.assert_not_called()

    def test_build_memory_context_uses_session_id(self, monkeypatch):
        """TenantRuntime.build_context narrows memory reads by session_id."""
        monkeypatch.setenv("ORG_ID", "org1")
        monkeypatch.setenv("PROJECT_ID", "proj1")
        runtime = TenantRuntime(app_name="Test")
        runtime._memory_engine = Mock()
        context = Mock()
        context.formatted_context = "memory context"
        runtime._memory_engine.build_context.return_value = context

        result = runtime.build_context(
            query="test",
            user_id="user_123",
            session_id="session_456",
        )

        assert result == "memory context"
        runtime._memory_engine.build_context.assert_called_once_with(
            query="test",
            session_id="session_456",
            org_id="org1",
            user_id="user_123",
            project_id="proj1",
            visibility=None,
            enabled_sources=None,
            format_style="jinja2",
        )

    def test_build_context_forwards_explicit_max_tokens(self, monkeypatch):
        """TenantRuntime forwards max_tokens through the helper to the engine."""
        monkeypatch.setenv("ORG_ID", "org1")
        monkeypatch.setenv("PROJECT_ID", "proj1")
        runtime = TenantRuntime(app_name="Test")
        runtime._memory_engine = Mock()
        context = Mock()
        context.formatted_context = "memory context"
        runtime._memory_engine.build_context.return_value = context

        result = runtime.build_context(
            query="test",
            user_id="user_123",
            session_id="session_456",
            max_tokens=2048,
        )

        assert result == "memory context"
        kwargs = runtime._memory_engine.build_context.call_args[1]
        assert kwargs["max_tokens"] == 2048

    def test_build_context_omits_max_tokens_when_absent(self, monkeypatch):
        """Omitted max_tokens stays absent from the engine call kwargs."""
        monkeypatch.setenv("ORG_ID", "org1")
        monkeypatch.setenv("PROJECT_ID", "proj1")
        runtime = TenantRuntime(app_name="Test")
        runtime._memory_engine = Mock()
        context = Mock()
        context.formatted_context = "memory context"
        runtime._memory_engine.build_context.return_value = context

        result = runtime.build_context(
            query="test",
            user_id="user_123",
            session_id="session_456",
        )

        assert result == "memory context"
        kwargs = runtime._memory_engine.build_context.call_args[1]
        assert "max_tokens" not in kwargs

    def test_durable_dispatch_records_memory_context(self, monkeypatch):
        monkeypatch.setenv("ORG_ID", "org1")
        monkeypatch.setenv("PROJECT_ID", "proj1")
        runtime = TenantRuntime(app_name="Test")
        runtime._memory_engine = Mock()
        runtime._memory_engine.build_context.return_value = SimpleNamespace(
            formatted_context="original memory context"
        )
        workflow = _FakeWorkflow(ActivityDispatch(context=_activity_context()))

        with _durable_execution(workflow):
            result = runtime.build_context(
                query="coverage question",
                user_id="user-123",
                session_id="session-456",
                visibility="private",
                metadata_filter={"policy_type": "auto"},
                enabled_sources={"semantic", "episodic"},
                max_tokens=2048,
            )

        assert result == "original memory context"
        runtime._memory_engine.build_context.assert_called_once()
        (command,) = workflow.commands
        assert command.activity_kind == ACTIVITY_KIND_MEMORY
        assert command.activity_name == "memory.build_context"
        assert command.position.activity_ordinal == 1
        assert value_to_json(command.semantic_input) == {
            "query": "coverage question",
            "user_id": "user-123",
            "session_id": "session-456",
            "visibility": "private",
            "metadata_filter": {"policy_type": "auto"},
            "enabled_sources": ["episodic", "semantic"],
            "max_tokens": 2048,
        }
        (outcome,) = workflow.outcomes
        assert value_to_json(outcome.result) == "original memory context"

    def test_durable_dispatch_records_empty_context_without_memory_client(self, monkeypatch):
        monkeypatch.setenv("ORG_ID", "org1")
        monkeypatch.setenv("PROJECT_ID", "proj1")
        runtime = TenantRuntime(app_name="Test")
        runtime._memory_engine = None
        workflow = _FakeWorkflow(ActivityDispatch(context=_activity_context()))

        with _durable_execution(workflow):
            result = runtime.build_context(
                query="coverage question",
                user_id="user-123",
                session_id="session-456",
            )

        assert result == ""
        (command,) = workflow.commands
        assert command.activity_kind == ACTIVITY_KIND_MEMORY
        assert command.activity_name == "memory.build_context"
        assert command.position.activity_ordinal == 1
        (outcome,) = workflow.outcomes
        assert value_to_json(outcome.result) == ""

    def test_durable_replay_does_not_require_memory_client(self, monkeypatch):
        monkeypatch.setenv("ORG_ID", "org1")
        monkeypatch.setenv("PROJECT_ID", "proj1")
        runtime = TenantRuntime(app_name="Test")
        runtime._memory_engine = None
        recorded = completed_outcome(_activity_context(), "recorded memory context")
        workflow = _FakeWorkflow(ActivityReplay(outcome=recorded))

        with _durable_execution(workflow):
            result = runtime.build_context(
                query="coverage question",
                user_id="user-123",
                session_id="session-456",
            )

        assert result == "recorded memory context"
        (command,) = workflow.commands
        assert command.activity_kind == ACTIVITY_KIND_MEMORY
        assert command.activity_name == "memory.build_context"
        assert command.position.activity_ordinal == 1
        assert workflow.outcomes == []

    def test_durable_workflow_failure_is_not_swallowed(self, monkeypatch):
        monkeypatch.setenv("ORG_ID", "org1")
        monkeypatch.setenv("PROJECT_ID", "proj1")
        runtime = TenantRuntime(app_name="Test")
        runtime._memory_engine = Mock()
        workflow = _FakeWorkflow(
            WorkflowClientError(
                WORKFLOW_ERROR_CODE_NONDETERMINISTIC,
                "memory activity changed",
            )
        )

        with _durable_execution(workflow):
            with pytest.raises(WorkflowClientError, match="memory activity changed"):
                runtime.build_context(
                    query="changed question",
                    user_id="user-123",
                    session_id="session-456",
                )

        runtime._memory_engine.build_context.assert_not_called()

    @pytest.mark.parametrize(
        "max_tokens",
        [0, -1, True, False, 1.5, float("nan"), float("inf"), float("-inf")],
    )
    def test_build_context_rejects_invalid_max_tokens(self, monkeypatch, max_tokens: object):
        monkeypatch.setenv("ORG_ID", "org1")
        monkeypatch.setenv("PROJECT_ID", "proj1")
        runtime = TenantRuntime(app_name="Test")
        runtime._memory_engine = Mock()

        with pytest.raises(ValueError, match="^max_tokens must be a positive integer$"):
            runtime.build_context(
                query="test",
                user_id="user_123",
                session_id="session_456",
                max_tokens=max_tokens,  # type: ignore[arg-type]
            )

        runtime._memory_engine.build_context.assert_not_called()

    def test_build_memory_context_uses_runtime_project_id(self, monkeypatch):
        """TenantRuntime supplies its project_id to memory context calls."""
        monkeypatch.setenv("ORG_ID", "org1")
        monkeypatch.setenv("PROJECT_ID", "proj1")
        runtime = TenantRuntime(app_name="Test")
        runtime._memory_engine = Mock()
        context = Mock()
        context.formatted_context = "memory context"
        runtime._memory_engine.build_context.return_value = context

        result = runtime.build_context(
            query="test",
            user_id="user_123",
            session_id="session_456",
        )

        assert result == "memory context"
        runtime._memory_engine.build_context.assert_called_once_with(
            query="test",
            session_id="session_456",
            org_id="org1",
            user_id="user_123",
            project_id="proj1",
            visibility=None,
            enabled_sources=None,
            format_style="jinja2",
        )

    def test_save_semantic_uses_runtime_project_id(self, monkeypatch):
        """TenantRuntime supplies its project_id to semantic writes."""
        monkeypatch.setenv("ORG_ID", "org1")
        monkeypatch.setenv("PROJECT_ID", "proj1")
        runtime = TenantRuntime(app_name="Test")
        runtime._memory_engine = Mock()
        captured_kwargs = {}

        def fake_create_semantic_memory(**kwargs):
            captured_kwargs.update(kwargs)
            return "memory-id"

        monkeypatch.setattr(
            "agent_engine_runner_shared.memory.create_semantic_memory",
            fake_create_semantic_memory,
        )

        result = runtime.save_semantic(
            text="fact",
            label="label",
            user_id="user_123",
        )

        assert result is True
        assert captured_kwargs["project_id"] == "proj1"
        assert captured_kwargs["upsert"] is True

    def test_save_episode_uses_runtime_project_id(self, monkeypatch):
        """TenantRuntime supplies its project_id to episodic writes."""
        monkeypatch.setenv("ORG_ID", "org1")
        monkeypatch.setenv("PROJECT_ID", "proj1")
        runtime = TenantRuntime(app_name="Test")
        runtime._memory_engine = Mock()
        captured_kwargs = {}

        def fake_create_episodic_memory(**kwargs):
            captured_kwargs.update(kwargs)
            return "episode-id"

        monkeypatch.setattr(
            "agent_engine_runner_shared.memory.create_episodic_memory",
            fake_create_episodic_memory,
        )

        result = runtime.save_episode(
            title="Episode",
            content="content",
            user_id="user_123",
            session_id="session_456",
        )

        assert result == "episode-id"
        assert captured_kwargs["project_id"] == "proj1"

    def test_create_taxonomic_passes_project_id(self, monkeypatch):
        """TenantRuntime supplies its project_id to create_taxonomic calls."""
        monkeypatch.setenv("ORG_ID", "org1")
        monkeypatch.setenv("PROJECT_ID", "proj1")
        runtime = TenantRuntime(app_name="Test")
        runtime._memory_engine = Mock()
        runtime._memory_engine.create_taxonomic.return_value = SimpleNamespace(id="tax-1")

        result = runtime.create_taxonomic(
            domain="insurance",
            term="deductible",
            definition="Amount paid out of pocket",
            user_id="user_123",
        )

        assert result == "tax-1"
        call_kwargs = runtime._memory_engine.create_taxonomic.call_args[1]
        assert call_kwargs["project_id"] == "proj1"
        assert call_kwargs["org_id"] == "org1"

    def test_create_taxonomic_forwards_metadata(self, monkeypatch):
        """Developer metadata reaches the engine create_taxonomic call."""
        monkeypatch.setenv("ORG_ID", "org1")
        monkeypatch.setenv("PROJECT_ID", "proj1")
        runtime = TenantRuntime(app_name="Test")
        runtime._memory_engine = Mock()
        runtime._memory_engine.create_taxonomic.return_value = SimpleNamespace(id="tax-1")

        runtime.create_taxonomic(
            domain="insurance",
            term="deductible",
            definition="Amount paid out of pocket",
            user_id="user_123",
            metadata={"k": "v"},
        )

        call_kwargs = runtime._memory_engine.create_taxonomic.call_args[1]
        assert call_kwargs["metadata"] == {"k": "v"}

    def test_discover_procedures_uses_runtime_project_id(self, monkeypatch):
        """TenantRuntime supplies its project_id to procedural discovery."""
        monkeypatch.setenv("ORG_ID", "org1")
        monkeypatch.setenv("PROJECT_ID", "proj1")
        runtime = TenantRuntime(app_name="Test")
        runtime._memory_engine = Mock()
        runtime._memory_engine.discover_procedures.return_value = []

        result = runtime.discover_procedures(query="deploy", user_id="user_123")

        assert result == []
        call_kwargs = runtime._memory_engine.discover_procedures.call_args[1]
        assert call_kwargs["project_id"] == "proj1"

    def test_save_procedure_uses_runtime_project_id(self, monkeypatch):
        """TenantRuntime supplies its project_id to procedural writes."""
        monkeypatch.setenv("ORG_ID", "org1")
        monkeypatch.setenv("PROJECT_ID", "proj1")
        runtime = TenantRuntime(app_name="Test")
        runtime._memory_engine = Mock()
        created = SimpleNamespace(
            id="proc-id",
            procedure="deploy-app",
            has_embedding=True,
            acknowledged=True,
        )
        runtime._memory_engine.create_procedural.return_value = created
        runtime._memory_engine.get_procedural.return_value = None

        result = runtime.save_procedure(
            procedure="deploy-app",
            description="Deploy the app",
            content="Run deploy",
            user_id="user_123",
        )

        assert result == {
            "id": "proc-id",
            "procedure": "deploy-app",
            "has_embedding": True,
            "acknowledged": True,
        }
        call_kwargs = runtime._memory_engine.create_procedural.call_args[1]
        assert call_kwargs["project_id"] == "proj1"

    def test_save_procedure_create_get_procedural_passes_project_id(self, monkeypatch):
        """Post-create get_procedural lookup must include project_id."""
        monkeypatch.setenv("ORG_ID", "org1")
        monkeypatch.setenv("PROJECT_ID", "proj1")
        runtime = TenantRuntime(app_name="Test")
        runtime._memory_engine = Mock()
        created = SimpleNamespace(
            id="proc-id",
            procedure="deploy-app",
            has_embedding=True,
            acknowledged=True,
        )
        fetched = {"id": "proc-id", "procedure": "deploy-app", "content": "Run deploy"}
        runtime._memory_engine.create_procedural.return_value = created
        runtime._memory_engine.get_procedural.return_value = fetched

        result = runtime.save_procedure(
            procedure="deploy-app",
            description="Deploy the app",
            content="Run deploy",
            user_id="user_123",
        )

        assert result == fetched
        get_kwargs = runtime._memory_engine.get_procedural.call_args[1]
        assert get_kwargs["project_id"] == "proj1"
        assert get_kwargs["org_id"] == "org1"
        assert get_kwargs["id"] == "proc-id"

    def test_save_procedure_update_get_procedural_passes_project_id(self, monkeypatch):
        """Update-existing get_procedural lookup must include project_id."""
        monkeypatch.setenv("ORG_ID", "org1")
        monkeypatch.setenv("PROJECT_ID", "proj1")
        runtime = TenantRuntime(app_name="Test")
        runtime._memory_engine = Mock()
        existing = {"id": "proc-id", "procedure": "deploy-app"}
        updated = {"id": "proc-id", "procedure": "deploy-app", "content": "Run deploy v2"}
        runtime._memory_engine.get_procedural.return_value = existing
        runtime._memory_engine.update_procedural.return_value = updated

        result = runtime.save_procedure(
            procedure="deploy-app",
            description="Deploy the app",
            content="Run deploy v2",
            user_id="user_123",
            update_existing=True,
        )

        assert result == updated
        get_kwargs = runtime._memory_engine.get_procedural.call_args[1]
        assert get_kwargs["project_id"] == "proj1"
        assert get_kwargs["org_id"] == "org1"
        assert get_kwargs["procedure"] == "deploy-app"

    def test_save_procedure_create_forwards_metadata(self, monkeypatch):
        """Developer metadata reaches create_procedural on the create path."""
        monkeypatch.setenv("ORG_ID", "org1")
        monkeypatch.setenv("PROJECT_ID", "proj1")
        runtime = TenantRuntime(app_name="Test")
        runtime._memory_engine = Mock()
        created = SimpleNamespace(
            id="proc-id",
            procedure="deploy-app",
            has_embedding=True,
            acknowledged=True,
        )
        runtime._memory_engine.create_procedural.return_value = created
        runtime._memory_engine.get_procedural.return_value = None

        runtime.save_procedure(
            procedure="deploy-app",
            description="Deploy the app",
            content="Run deploy",
            user_id="user_123",
            metadata={"k": "v"},
        )

        call_kwargs = runtime._memory_engine.create_procedural.call_args[1]
        assert call_kwargs["metadata"] == {"k": "v"}

    def test_save_procedure_update_forwards_metadata(self, monkeypatch):
        """Developer metadata reaches update_procedural on the update path."""
        monkeypatch.setenv("ORG_ID", "org1")
        monkeypatch.setenv("PROJECT_ID", "proj1")
        runtime = TenantRuntime(app_name="Test")
        runtime._memory_engine = Mock()
        existing = {"id": "proc-id", "procedure": "deploy-app"}
        updated = {"id": "proc-id", "procedure": "deploy-app", "content": "v2"}
        runtime._memory_engine.get_procedural.return_value = existing
        runtime._memory_engine.update_procedural.return_value = updated

        runtime.save_procedure(
            procedure="deploy-app",
            description="Deploy the app",
            content="v2",
            user_id="user_123",
            update_existing=True,
            metadata={"k": "v"},
        )

        call_kwargs = runtime._memory_engine.update_procedural.call_args[1]
        assert call_kwargs["metadata"] == {"k": "v"}

    def test_save_procedure_post_create_fetch_raises_returns_fallback(self, monkeypatch):
        """When the post-create get_procedural fetch raises, save_procedure returns
        the minimal create result instead of None."""
        monkeypatch.setenv("ORG_ID", "org1")
        monkeypatch.setenv("PROJECT_ID", "proj1")
        runtime = TenantRuntime(app_name="Test")
        runtime._memory_engine = Mock()
        created = SimpleNamespace(
            id="proc-id",
            procedure="deploy-app",
            has_embedding=True,
            acknowledged=True,
        )
        runtime._memory_engine.create_procedural.return_value = created
        runtime._memory_engine.get_procedural.side_effect = Exception(
            "500 vector search unavailable"
        )

        result = runtime.save_procedure(
            procedure="deploy-app",
            description="Deploy the app",
            content="Run deploy",
            user_id="user_123",
        )

        assert result is not None
        assert result["id"] == "proc-id"
        assert result["procedure"] == "deploy-app"
        assert result["has_embedding"] is True
        assert result["acknowledged"] is True

    def test_save_procedure_update_lookup_passes_identity(self, monkeypatch):
        """The update_existing existence check must carry user_id (OE 400s
        identity-less GETs) but must not filter by visibility, so an update
        can change visibility on an existing procedure."""
        monkeypatch.setenv("ORG_ID", "org1")
        monkeypatch.setenv("PROJECT_ID", "proj1")
        runtime = TenantRuntime(app_name="Test")
        runtime._memory_engine = Mock()
        existing = {"id": "proc-id", "procedure": "deploy-app", "visibility": "private"}
        updated = {
            "id": "proc-id",
            "procedure": "deploy-app",
            "content": "v2",
            "visibility": "org",
        }
        runtime._memory_engine.get_procedural.return_value = existing
        runtime._memory_engine.update_procedural.return_value = updated

        result = runtime.save_procedure(
            procedure="deploy-app",
            description="Deploy the app",
            content="v2",
            user_id="user_123",
            visibility="org",
            update_existing=True,
        )

        assert result == updated
        get_kwargs = runtime._memory_engine.get_procedural.call_args[1]
        assert get_kwargs["user_id"] == "user_123"
        assert get_kwargs.get("visibility") is None
        update_kwargs = runtime._memory_engine.update_procedural.call_args[1]
        assert update_kwargs["id"] == "proc-id"
        assert update_kwargs["user_id"] == "user_123"
        assert update_kwargs["visibility"] == "org"
        runtime._memory_engine.create_procedural.assert_not_called()

    def test_save_procedure_update_existing_missing_creates(self, monkeypatch):
        """update_existing=True with no existing procedure falls through to create."""
        monkeypatch.setenv("ORG_ID", "org1")
        monkeypatch.setenv("PROJECT_ID", "proj1")
        runtime = TenantRuntime(app_name="Test")
        runtime._memory_engine = Mock()
        created = SimpleNamespace(
            id="proc-id",
            procedure="deploy-app",
            has_embedding=True,
            acknowledged=True,
        )
        fetched = {"id": "proc-id", "procedure": "deploy-app"}
        runtime._memory_engine.get_procedural.side_effect = [None, fetched]
        runtime._memory_engine.create_procedural.return_value = created

        result = runtime.save_procedure(
            procedure="deploy-app",
            description="Deploy the app",
            content="Run deploy",
            user_id="user_123",
            update_existing=True,
        )

        assert result == fetched
        runtime._memory_engine.create_procedural.assert_called_once()
        runtime._memory_engine.update_procedural.assert_not_called()
        for call in runtime._memory_engine.get_procedural.call_args_list:
            assert call.kwargs["user_id"] == "user_123"

    def test_save_procedure_post_create_fetch_passes_identity(self, monkeypatch):
        """The post-create fetch must carry user_id/visibility too; otherwise
        every create logs a spurious fetch failure and returns the fallback."""
        monkeypatch.setenv("ORG_ID", "org1")
        monkeypatch.setenv("PROJECT_ID", "proj1")
        runtime = TenantRuntime(app_name="Test")
        runtime._memory_engine = Mock()
        created = SimpleNamespace(
            id="proc-id",
            procedure="deploy-app",
            has_embedding=True,
            acknowledged=True,
        )
        fetched = {"id": "proc-id", "procedure": "deploy-app", "content": "Run deploy"}
        runtime._memory_engine.create_procedural.return_value = created
        runtime._memory_engine.get_procedural.return_value = fetched

        result = runtime.save_procedure(
            procedure="deploy-app",
            description="Deploy the app",
            content="Run deploy",
            user_id="user_123",
            visibility="shared",
        )

        assert result == fetched
        get_kwargs = runtime._memory_engine.get_procedural.call_args[1]
        assert get_kwargs["id"] == "proc-id"
        assert get_kwargs["user_id"] == "user_123"
        assert get_kwargs["visibility"] == "shared"

    def test_save_procedure_update_with_model_existing_passes_id(self, monkeypatch):
        """A memory engine returning model objects (not dicts) must still
        resolve the id for the update call."""
        monkeypatch.setenv("ORG_ID", "org1")
        monkeypatch.setenv("PROJECT_ID", "proj1")
        runtime = TenantRuntime(app_name="Test")
        runtime._memory_engine = Mock()
        existing = SimpleNamespace(id="proc-id", procedure="deploy-app")
        updated = {"id": "proc-id", "procedure": "deploy-app", "content": "v2"}
        runtime._memory_engine.get_procedural.return_value = existing
        runtime._memory_engine.update_procedural.return_value = updated

        result = runtime.save_procedure(
            procedure="deploy-app",
            description="Deploy the app",
            content="v2",
            user_id="user_123",
            update_existing=True,
        )

        assert result == updated
        update_kwargs = runtime._memory_engine.update_procedural.call_args[1]
        assert update_kwargs["id"] == "proc-id"

    def test_save_procedure_update_uses_context_user_when_param_omitted(self, monkeypatch):
        """Without an explicit user_id, the execution-context user must flow
        into the existence check and the update."""
        monkeypatch.setenv("ORG_ID", "org1")
        monkeypatch.setenv("PROJECT_ID", "proj1")
        monkeypatch.setattr(
            "agent_engine_runner_shared.runtime.get_current_user_id", lambda: "ctx-user"
        )
        runtime = TenantRuntime(app_name="Test")
        runtime._memory_engine = Mock()
        existing = {"id": "proc-id", "procedure": "deploy-app"}
        updated = {"id": "proc-id", "procedure": "deploy-app", "content": "v2"}
        runtime._memory_engine.get_procedural.return_value = existing
        runtime._memory_engine.update_procedural.return_value = updated

        result = runtime.save_procedure(
            procedure="deploy-app",
            description="Deploy the app",
            content="v2",
            update_existing=True,
        )

        assert result == updated
        get_kwargs = runtime._memory_engine.get_procedural.call_args[1]
        assert get_kwargs["user_id"] == "ctx-user"
        update_kwargs = runtime._memory_engine.update_procedural.call_args[1]
        assert update_kwargs["user_id"] == "ctx-user"

    def test_save_procedure_update_falls_back_to_name_when_existing_has_no_id(self, monkeypatch):
        """If the found procedure carries no id, the update must fall back to
        the scoped by-name path rather than call with a bogus id."""
        monkeypatch.setenv("ORG_ID", "org1")
        monkeypatch.setenv("PROJECT_ID", "proj1")
        runtime = TenantRuntime(app_name="Test")
        runtime._memory_engine = Mock()
        existing = {"procedure": "deploy-app", "content": "old"}
        updated = {"procedure": "deploy-app", "content": "v2"}
        runtime._memory_engine.get_procedural.return_value = existing
        runtime._memory_engine.update_procedural.return_value = updated

        result = runtime.save_procedure(
            procedure="deploy-app",
            description="Deploy the app",
            content="v2",
            user_id="user_123",
            update_existing=True,
        )

        assert result == updated
        update_kwargs = runtime._memory_engine.update_procedural.call_args[1]
        assert update_kwargs["id"] is None
        assert update_kwargs["procedure"] == "deploy-app"
        assert update_kwargs["user_id"] == "user_123"

    def test_deprecation_warning_on_org_id_and_project_id(self, monkeypatch):
        """Passing org_id or project_id to TenantRuntime emits a DeprecationWarning."""
        monkeypatch.setenv("ORG_ID", "org1")
        monkeypatch.setenv("PROJECT_ID", "proj1")

        with pytest.warns(DeprecationWarning, match="org_id") as record:
            TenantRuntime(app_name="Test", org_id="org1")
        assert any("ORG_ID" in str(w.message) for w in record)

        with pytest.warns(DeprecationWarning, match="project_id") as record:
            TenantRuntime(app_name="Test", project_id="proj1")
        assert any("PROJECT_ID" in str(w.message) for w in record)

    def test_constructor_org_id_and_project_id_are_ignored(self, monkeypatch):
        """The deprecated constructor args are ignored; the env vars win.

        This is the footgun fix: a hardcoded org_id/project_id must never
        override the platform-injected env vars, otherwise memory writes get
        tagged with the wrong tenant and the platform UI silently sees nothing.
        """
        monkeypatch.setenv("ORG_ID", "env_org")
        monkeypatch.setenv("PROJECT_ID", "env_proj")

        with pytest.warns(DeprecationWarning):
            runtime = TenantRuntime(
                app_name="Test",
                org_id="hardcoded_org",
                project_id="hardcoded_proj",
            )

        assert runtime.org_id == "env_org"
        assert runtime.project_id == "env_proj"

    def test_no_deprecation_warning_when_args_absent(self, monkeypatch):
        """Constructing without the deprecated args emits no DeprecationWarning."""
        monkeypatch.setenv("ORG_ID", "org1")
        monkeypatch.setenv("PROJECT_ID", "proj1")

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            TenantRuntime(app_name="Test")
