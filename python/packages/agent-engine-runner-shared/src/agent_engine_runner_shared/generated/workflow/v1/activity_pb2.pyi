from google.protobuf import struct_pb2 as _struct_pb2
from agent_engine_runner_shared.generated.workflow.v1 import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ActivityKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ACTIVITY_KIND_UNSPECIFIED: _ClassVar[ActivityKind]
    ACTIVITY_KIND_LLM: _ClassVar[ActivityKind]
    ACTIVITY_KIND_TOOL: _ClassVar[ActivityKind]
    ACTIVITY_KIND_MEMORY: _ClassVar[ActivityKind]

class ActivityOutcomeKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ACTIVITY_OUTCOME_KIND_UNSPECIFIED: _ClassVar[ActivityOutcomeKind]
    ACTIVITY_OUTCOME_KIND_COMPLETED: _ClassVar[ActivityOutcomeKind]
    ACTIVITY_OUTCOME_KIND_FAILED: _ClassVar[ActivityOutcomeKind]
    ACTIVITY_OUTCOME_KIND_DENIED: _ClassVar[ActivityOutcomeKind]
    ACTIVITY_OUTCOME_KIND_SUSPENDED: _ClassVar[ActivityOutcomeKind]
ACTIVITY_KIND_UNSPECIFIED: ActivityKind
ACTIVITY_KIND_LLM: ActivityKind
ACTIVITY_KIND_TOOL: ActivityKind
ACTIVITY_KIND_MEMORY: ActivityKind
ACTIVITY_OUTCOME_KIND_UNSPECIFIED: ActivityOutcomeKind
ACTIVITY_OUTCOME_KIND_COMPLETED: ActivityOutcomeKind
ACTIVITY_OUTCOME_KIND_FAILED: ActivityOutcomeKind
ACTIVITY_OUTCOME_KIND_DENIED: ActivityOutcomeKind
ACTIVITY_OUTCOME_KIND_SUSPENDED: ActivityOutcomeKind

class ActivityCommand(_message.Message):
    __slots__ = ("workflow_identity", "attempt_id", "fencing_token", "position", "activity_kind", "activity_name", "semantic_input")
    WORKFLOW_IDENTITY_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ID_FIELD_NUMBER: _ClassVar[int]
    FENCING_TOKEN_FIELD_NUMBER: _ClassVar[int]
    POSITION_FIELD_NUMBER: _ClassVar[int]
    ACTIVITY_KIND_FIELD_NUMBER: _ClassVar[int]
    ACTIVITY_NAME_FIELD_NUMBER: _ClassVar[int]
    SEMANTIC_INPUT_FIELD_NUMBER: _ClassVar[int]
    workflow_identity: _common_pb2.WorkflowIdentity
    attempt_id: str
    fencing_token: int
    position: _common_pb2.ActivityPosition
    activity_kind: ActivityKind
    activity_name: str
    semantic_input: _struct_pb2.Value
    def __init__(self, workflow_identity: _Optional[_Union[_common_pb2.WorkflowIdentity, _Mapping]] = ..., attempt_id: _Optional[str] = ..., fencing_token: _Optional[int] = ..., position: _Optional[_Union[_common_pb2.ActivityPosition, _Mapping]] = ..., activity_kind: _Optional[_Union[ActivityKind, str]] = ..., activity_name: _Optional[str] = ..., semantic_input: _Optional[_Union[_struct_pb2.Value, _Mapping]] = ...) -> None: ...

class ActivityContext(_message.Message):
    __slots__ = ("workflow_identity", "activity_id", "attempt_id", "fencing_token")
    WORKFLOW_IDENTITY_FIELD_NUMBER: _ClassVar[int]
    ACTIVITY_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ID_FIELD_NUMBER: _ClassVar[int]
    FENCING_TOKEN_FIELD_NUMBER: _ClassVar[int]
    workflow_identity: _common_pb2.WorkflowIdentity
    activity_id: str
    attempt_id: str
    fencing_token: int
    def __init__(self, workflow_identity: _Optional[_Union[_common_pb2.WorkflowIdentity, _Mapping]] = ..., activity_id: _Optional[str] = ..., attempt_id: _Optional[str] = ..., fencing_token: _Optional[int] = ...) -> None: ...

class ActivitySuspension(_message.Message):
    __slots__ = ("reason", "context")
    REASON_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    reason: str
    context: _struct_pb2.Struct
    def __init__(self, reason: _Optional[str] = ..., context: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class MemoryWrite(_message.Message):
    __slots__ = ("id", "payload_json")
    ID_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_JSON_FIELD_NUMBER: _ClassVar[int]
    id: str
    payload_json: bytes
    def __init__(self, id: _Optional[str] = ..., payload_json: _Optional[bytes] = ...) -> None: ...

class ActivityOutcome(_message.Message):
    __slots__ = ("workflow_identity", "activity_id", "attempt_id", "fencing_token", "outcome_kind", "result", "error", "suspension")
    WORKFLOW_IDENTITY_FIELD_NUMBER: _ClassVar[int]
    ACTIVITY_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ID_FIELD_NUMBER: _ClassVar[int]
    FENCING_TOKEN_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_KIND_FIELD_NUMBER: _ClassVar[int]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    SUSPENSION_FIELD_NUMBER: _ClassVar[int]
    workflow_identity: _common_pb2.WorkflowIdentity
    activity_id: str
    attempt_id: str
    fencing_token: int
    outcome_kind: ActivityOutcomeKind
    result: _struct_pb2.Value
    error: _common_pb2.WorkflowError
    suspension: ActivitySuspension
    def __init__(self, workflow_identity: _Optional[_Union[_common_pb2.WorkflowIdentity, _Mapping]] = ..., activity_id: _Optional[str] = ..., attempt_id: _Optional[str] = ..., fencing_token: _Optional[int] = ..., outcome_kind: _Optional[_Union[ActivityOutcomeKind, str]] = ..., result: _Optional[_Union[_struct_pb2.Value, _Mapping]] = ..., error: _Optional[_Union[_common_pb2.WorkflowError, _Mapping]] = ..., suspension: _Optional[_Union[ActivitySuspension, _Mapping]] = ...) -> None: ...

class ActivityMemoryCommand(_message.Message):
    __slots__ = ("workflow_identity", "activity_id", "attempt_id", "fencing_token", "memory_writes")
    WORKFLOW_IDENTITY_FIELD_NUMBER: _ClassVar[int]
    ACTIVITY_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ID_FIELD_NUMBER: _ClassVar[int]
    FENCING_TOKEN_FIELD_NUMBER: _ClassVar[int]
    MEMORY_WRITES_FIELD_NUMBER: _ClassVar[int]
    workflow_identity: _common_pb2.WorkflowIdentity
    activity_id: str
    attempt_id: str
    fencing_token: int
    memory_writes: _containers.RepeatedCompositeFieldContainer[MemoryWrite]
    def __init__(self, workflow_identity: _Optional[_Union[_common_pb2.WorkflowIdentity, _Mapping]] = ..., activity_id: _Optional[str] = ..., attempt_id: _Optional[str] = ..., fencing_token: _Optional[int] = ..., memory_writes: _Optional[_Iterable[_Union[MemoryWrite, _Mapping]]] = ...) -> None: ...

class StepSuspensionEntry(_message.Message):
    __slots__ = ("position", "activity_kind", "activity_name", "semantic_input", "suspension")
    POSITION_FIELD_NUMBER: _ClassVar[int]
    ACTIVITY_KIND_FIELD_NUMBER: _ClassVar[int]
    ACTIVITY_NAME_FIELD_NUMBER: _ClassVar[int]
    SEMANTIC_INPUT_FIELD_NUMBER: _ClassVar[int]
    SUSPENSION_FIELD_NUMBER: _ClassVar[int]
    position: _common_pb2.ActivityPosition
    activity_kind: ActivityKind
    activity_name: str
    semantic_input: _struct_pb2.Value
    suspension: ActivitySuspension
    def __init__(self, position: _Optional[_Union[_common_pb2.ActivityPosition, _Mapping]] = ..., activity_kind: _Optional[_Union[ActivityKind, str]] = ..., activity_name: _Optional[str] = ..., semantic_input: _Optional[_Union[_struct_pb2.Value, _Mapping]] = ..., suspension: _Optional[_Union[ActivitySuspension, _Mapping]] = ...) -> None: ...

class StepActivityEntry(_message.Message):
    __slots__ = ("position", "outcome")
    POSITION_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_FIELD_NUMBER: _ClassVar[int]
    position: _common_pb2.ActivityPosition
    outcome: ActivityOutcome
    def __init__(self, position: _Optional[_Union[_common_pb2.ActivityPosition, _Mapping]] = ..., outcome: _Optional[_Union[ActivityOutcome, _Mapping]] = ...) -> None: ...

class ActivityHeartbeatRequest(_message.Message):
    __slots__ = ("workflow_identity", "activity_id", "attempt_id", "fencing_token", "owner_id")
    WORKFLOW_IDENTITY_FIELD_NUMBER: _ClassVar[int]
    ACTIVITY_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ID_FIELD_NUMBER: _ClassVar[int]
    FENCING_TOKEN_FIELD_NUMBER: _ClassVar[int]
    OWNER_ID_FIELD_NUMBER: _ClassVar[int]
    workflow_identity: _common_pb2.WorkflowIdentity
    activity_id: str
    attempt_id: str
    fencing_token: int
    owner_id: str
    def __init__(self, workflow_identity: _Optional[_Union[_common_pb2.WorkflowIdentity, _Mapping]] = ..., activity_id: _Optional[str] = ..., attempt_id: _Optional[str] = ..., fencing_token: _Optional[int] = ..., owner_id: _Optional[str] = ...) -> None: ...
