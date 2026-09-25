from agent_engine_runner_shared.generated.workflow.v1 import common_pb2 as _common_pb2
from agent_engine_runner_shared.generated.workflow.v1 import state_pb2 as _state_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class WorkflowDeclaration(_message.Message):
    __slots__ = ("workflow_name", "workflow_version", "adapter_name", "adapter_version", "memory_enabled")
    WORKFLOW_NAME_FIELD_NUMBER: _ClassVar[int]
    WORKFLOW_VERSION_FIELD_NUMBER: _ClassVar[int]
    ADAPTER_NAME_FIELD_NUMBER: _ClassVar[int]
    ADAPTER_VERSION_FIELD_NUMBER: _ClassVar[int]
    MEMORY_ENABLED_FIELD_NUMBER: _ClassVar[int]
    workflow_name: str
    workflow_version: str
    adapter_name: str
    adapter_version: str
    memory_enabled: bool
    def __init__(self, workflow_name: _Optional[str] = ..., workflow_version: _Optional[str] = ..., adapter_name: _Optional[str] = ..., adapter_version: _Optional[str] = ..., memory_enabled: _Optional[bool] = ...) -> None: ...

class BranchLineage(_message.Message):
    __slots__ = ("source_workflow_identity", "source_step_ordinal", "source_state_hash")
    SOURCE_WORKFLOW_IDENTITY_FIELD_NUMBER: _ClassVar[int]
    SOURCE_STEP_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    SOURCE_STATE_HASH_FIELD_NUMBER: _ClassVar[int]
    source_workflow_identity: _common_pb2.WorkflowIdentity
    source_step_ordinal: int
    source_state_hash: str
    def __init__(self, source_workflow_identity: _Optional[_Union[_common_pb2.WorkflowIdentity, _Mapping]] = ..., source_step_ordinal: _Optional[int] = ..., source_state_hash: _Optional[str] = ...) -> None: ...

class AttemptContext(_message.Message):
    __slots__ = ("attempt_id", "fencing_token", "owner_id", "replay_mode", "workflow_identity", "declaration", "heartbeat_interval_ms", "previous_state", "branch_lineage")
    ATTEMPT_ID_FIELD_NUMBER: _ClassVar[int]
    FENCING_TOKEN_FIELD_NUMBER: _ClassVar[int]
    OWNER_ID_FIELD_NUMBER: _ClassVar[int]
    REPLAY_MODE_FIELD_NUMBER: _ClassVar[int]
    WORKFLOW_IDENTITY_FIELD_NUMBER: _ClassVar[int]
    DECLARATION_FIELD_NUMBER: _ClassVar[int]
    HEARTBEAT_INTERVAL_MS_FIELD_NUMBER: _ClassVar[int]
    PREVIOUS_STATE_FIELD_NUMBER: _ClassVar[int]
    BRANCH_LINEAGE_FIELD_NUMBER: _ClassVar[int]
    attempt_id: str
    fencing_token: int
    owner_id: str
    replay_mode: bool
    workflow_identity: _common_pb2.WorkflowIdentity
    declaration: WorkflowDeclaration
    heartbeat_interval_ms: int
    previous_state: _state_pb2.StateSnapshot
    branch_lineage: BranchLineage
    def __init__(self, attempt_id: _Optional[str] = ..., fencing_token: _Optional[int] = ..., owner_id: _Optional[str] = ..., replay_mode: _Optional[bool] = ..., workflow_identity: _Optional[_Union[_common_pb2.WorkflowIdentity, _Mapping]] = ..., declaration: _Optional[_Union[WorkflowDeclaration, _Mapping]] = ..., heartbeat_interval_ms: _Optional[int] = ..., previous_state: _Optional[_Union[_state_pb2.StateSnapshot, _Mapping]] = ..., branch_lineage: _Optional[_Union[BranchLineage, _Mapping]] = ...) -> None: ...

class AttemptStartRequest(_message.Message):
    __slots__ = ("workflow_identity", "owner_id", "declaration")
    WORKFLOW_IDENTITY_FIELD_NUMBER: _ClassVar[int]
    OWNER_ID_FIELD_NUMBER: _ClassVar[int]
    DECLARATION_FIELD_NUMBER: _ClassVar[int]
    workflow_identity: _common_pb2.WorkflowIdentity
    owner_id: str
    declaration: WorkflowDeclaration
    def __init__(self, workflow_identity: _Optional[_Union[_common_pb2.WorkflowIdentity, _Mapping]] = ..., owner_id: _Optional[str] = ..., declaration: _Optional[_Union[WorkflowDeclaration, _Mapping]] = ...) -> None: ...

class AttemptStartResponse(_message.Message):
    __slots__ = ("attempt_context", "error")
    ATTEMPT_CONTEXT_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    attempt_context: AttemptContext
    error: _common_pb2.WorkflowError
    def __init__(self, attempt_context: _Optional[_Union[AttemptContext, _Mapping]] = ..., error: _Optional[_Union[_common_pb2.WorkflowError, _Mapping]] = ...) -> None: ...

class AttemptHeartbeatRequest(_message.Message):
    __slots__ = ("workflow_identity", "attempt_id", "fencing_token", "owner_id")
    WORKFLOW_IDENTITY_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ID_FIELD_NUMBER: _ClassVar[int]
    FENCING_TOKEN_FIELD_NUMBER: _ClassVar[int]
    OWNER_ID_FIELD_NUMBER: _ClassVar[int]
    workflow_identity: _common_pb2.WorkflowIdentity
    attempt_id: str
    fencing_token: int
    owner_id: str
    def __init__(self, workflow_identity: _Optional[_Union[_common_pb2.WorkflowIdentity, _Mapping]] = ..., attempt_id: _Optional[str] = ..., fencing_token: _Optional[int] = ..., owner_id: _Optional[str] = ...) -> None: ...
