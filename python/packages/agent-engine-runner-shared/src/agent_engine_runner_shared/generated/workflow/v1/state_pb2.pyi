from google.protobuf import struct_pb2 as _struct_pb2
from agent_engine_runner_shared.generated.workflow.v1 import activity_pb2 as _activity_pb2
from agent_engine_runner_shared.generated.workflow.v1 import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class MessageRole(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    MESSAGE_ROLE_UNSPECIFIED: _ClassVar[MessageRole]
    MESSAGE_ROLE_USER: _ClassVar[MessageRole]
    MESSAGE_ROLE_ASSISTANT: _ClassVar[MessageRole]
    MESSAGE_ROLE_TOOL: _ClassVar[MessageRole]
    MESSAGE_ROLE_SYSTEM: _ClassVar[MessageRole]
MESSAGE_ROLE_UNSPECIFIED: MessageRole
MESSAGE_ROLE_USER: MessageRole
MESSAGE_ROLE_ASSISTANT: MessageRole
MESSAGE_ROLE_TOOL: MessageRole
MESSAGE_ROLE_SYSTEM: MessageRole

class WorkflowMessage(_message.Message):
    __slots__ = ("role", "content", "tool_calls", "tool_call_id", "name", "id", "source_message", "artifacts", "additional_kwargs", "response_metadata", "is_error", "tool_artifact", "platform_artifacts", "invalid_tool_calls")
    ROLE_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    TOOL_CALLS_FIELD_NUMBER: _ClassVar[int]
    TOOL_CALL_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    ARTIFACTS_FIELD_NUMBER: _ClassVar[int]
    ADDITIONAL_KWARGS_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_METADATA_FIELD_NUMBER: _ClassVar[int]
    IS_ERROR_FIELD_NUMBER: _ClassVar[int]
    TOOL_ARTIFACT_FIELD_NUMBER: _ClassVar[int]
    PLATFORM_ARTIFACTS_FIELD_NUMBER: _ClassVar[int]
    INVALID_TOOL_CALLS_FIELD_NUMBER: _ClassVar[int]
    role: MessageRole
    content: _struct_pb2.Value
    tool_calls: _containers.RepeatedCompositeFieldContainer[_struct_pb2.Struct]
    tool_call_id: str
    name: str
    id: str
    source_message: _struct_pb2.Value
    artifacts: _containers.RepeatedCompositeFieldContainer[_struct_pb2.Struct]
    additional_kwargs: _struct_pb2.Struct
    response_metadata: _struct_pb2.Struct
    is_error: bool
    tool_artifact: _struct_pb2.Value
    platform_artifacts: _containers.RepeatedCompositeFieldContainer[_struct_pb2.Struct]
    invalid_tool_calls: _containers.RepeatedCompositeFieldContainer[_struct_pb2.Struct]
    def __init__(self, role: _Optional[_Union[MessageRole, str]] = ..., content: _Optional[_Union[_struct_pb2.Value, _Mapping]] = ..., tool_calls: _Optional[_Iterable[_Union[_struct_pb2.Struct, _Mapping]]] = ..., tool_call_id: _Optional[str] = ..., name: _Optional[str] = ..., id: _Optional[str] = ..., source_message: _Optional[_Union[_struct_pb2.Value, _Mapping]] = ..., artifacts: _Optional[_Iterable[_Union[_struct_pb2.Struct, _Mapping]]] = ..., additional_kwargs: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., response_metadata: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., is_error: _Optional[bool] = ..., tool_artifact: _Optional[_Union[_struct_pb2.Value, _Mapping]] = ..., platform_artifacts: _Optional[_Iterable[_Union[_struct_pb2.Struct, _Mapping]]] = ..., invalid_tool_calls: _Optional[_Iterable[_Union[_struct_pb2.Struct, _Mapping]]] = ...) -> None: ...

class StateSnapshot(_message.Message):
    __slots__ = ("properties", "messages", "message_encoding_version", "replay_properties")
    PROPERTIES_FIELD_NUMBER: _ClassVar[int]
    MESSAGES_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_ENCODING_VERSION_FIELD_NUMBER: _ClassVar[int]
    REPLAY_PROPERTIES_FIELD_NUMBER: _ClassVar[int]
    properties: _struct_pb2.Struct
    messages: _containers.RepeatedCompositeFieldContainer[WorkflowMessage]
    message_encoding_version: int
    replay_properties: _struct_pb2.Struct
    def __init__(self, properties: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., messages: _Optional[_Iterable[_Union[WorkflowMessage, _Mapping]]] = ..., message_encoding_version: _Optional[int] = ..., replay_properties: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class CompleteExecutionCommand(_message.Message):
    __slots__ = ("workflow_identity", "attempt_id", "fencing_token", "state")
    WORKFLOW_IDENTITY_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ID_FIELD_NUMBER: _ClassVar[int]
    FENCING_TOKEN_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    workflow_identity: _common_pb2.WorkflowIdentity
    attempt_id: str
    fencing_token: int
    state: StateSnapshot
    def __init__(self, workflow_identity: _Optional[_Union[_common_pb2.WorkflowIdentity, _Mapping]] = ..., attempt_id: _Optional[str] = ..., fencing_token: _Optional[int] = ..., state: _Optional[_Union[StateSnapshot, _Mapping]] = ...) -> None: ...

class FinalizeStepCommand(_message.Message):
    __slots__ = ("workflow_identity", "attempt_id", "fencing_token", "state", "step_ordinal", "observed_activity_positions", "suspensions")
    WORKFLOW_IDENTITY_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ID_FIELD_NUMBER: _ClassVar[int]
    FENCING_TOKEN_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    STEP_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    OBSERVED_ACTIVITY_POSITIONS_FIELD_NUMBER: _ClassVar[int]
    SUSPENSIONS_FIELD_NUMBER: _ClassVar[int]
    workflow_identity: _common_pb2.WorkflowIdentity
    attempt_id: str
    fencing_token: int
    state: StateSnapshot
    step_ordinal: int
    observed_activity_positions: _containers.RepeatedCompositeFieldContainer[_common_pb2.ActivityPosition]
    suspensions: _containers.RepeatedCompositeFieldContainer[_activity_pb2.StepSuspensionEntry]
    def __init__(self, workflow_identity: _Optional[_Union[_common_pb2.WorkflowIdentity, _Mapping]] = ..., attempt_id: _Optional[str] = ..., fencing_token: _Optional[int] = ..., state: _Optional[_Union[StateSnapshot, _Mapping]] = ..., step_ordinal: _Optional[int] = ..., observed_activity_positions: _Optional[_Iterable[_Union[_common_pb2.ActivityPosition, _Mapping]]] = ..., suspensions: _Optional[_Iterable[_Union[_activity_pb2.StepSuspensionEntry, _Mapping]]] = ...) -> None: ...

class FinalizeStepResponse(_message.Message):
    __slots__ = ("entries", "error")
    ENTRIES_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    entries: _containers.RepeatedCompositeFieldContainer[_activity_pb2.StepActivityEntry]
    error: _common_pb2.WorkflowError
    def __init__(self, entries: _Optional[_Iterable[_Union[_activity_pb2.StepActivityEntry, _Mapping]]] = ..., error: _Optional[_Union[_common_pb2.WorkflowError, _Mapping]] = ...) -> None: ...
