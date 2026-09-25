from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class WorkflowErrorCode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    WORKFLOW_ERROR_CODE_UNSPECIFIED: _ClassVar[WorkflowErrorCode]
    WORKFLOW_ERROR_CODE_INVALID_ARGUMENT: _ClassVar[WorkflowErrorCode]
    WORKFLOW_ERROR_CODE_NOT_FOUND: _ClassVar[WorkflowErrorCode]
    WORKFLOW_ERROR_CODE_CONFLICT: _ClassVar[WorkflowErrorCode]
    WORKFLOW_ERROR_CODE_STALE_FENCE: _ClassVar[WorkflowErrorCode]
    WORKFLOW_ERROR_CODE_NONDETERMINISTIC: _ClassVar[WorkflowErrorCode]
    WORKFLOW_ERROR_CODE_VERSION_UNAVAILABLE: _ClassVar[WorkflowErrorCode]
    WORKFLOW_ERROR_CODE_OUTCOME_UNKNOWN: _ClassVar[WorkflowErrorCode]
    WORKFLOW_ERROR_CODE_UNAUTHORIZED: _ClassVar[WorkflowErrorCode]
WORKFLOW_ERROR_CODE_UNSPECIFIED: WorkflowErrorCode
WORKFLOW_ERROR_CODE_INVALID_ARGUMENT: WorkflowErrorCode
WORKFLOW_ERROR_CODE_NOT_FOUND: WorkflowErrorCode
WORKFLOW_ERROR_CODE_CONFLICT: WorkflowErrorCode
WORKFLOW_ERROR_CODE_STALE_FENCE: WorkflowErrorCode
WORKFLOW_ERROR_CODE_NONDETERMINISTIC: WorkflowErrorCode
WORKFLOW_ERROR_CODE_VERSION_UNAVAILABLE: WorkflowErrorCode
WORKFLOW_ERROR_CODE_OUTCOME_UNKNOWN: WorkflowErrorCode
WORKFLOW_ERROR_CODE_UNAUTHORIZED: WorkflowErrorCode

class TenantScope(_message.Message):
    __slots__ = ("org_id", "project_id", "workspace_id")
    ORG_ID_FIELD_NUMBER: _ClassVar[int]
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    WORKSPACE_ID_FIELD_NUMBER: _ClassVar[int]
    org_id: str
    project_id: str
    workspace_id: str
    def __init__(self, org_id: _Optional[str] = ..., project_id: _Optional[str] = ..., workspace_id: _Optional[str] = ...) -> None: ...

class WorkflowIdentity(_message.Message):
    __slots__ = ("tenant_scope", "session_id", "execution_id")
    TENANT_SCOPE_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_ID_FIELD_NUMBER: _ClassVar[int]
    tenant_scope: TenantScope
    session_id: str
    execution_id: str
    def __init__(self, tenant_scope: _Optional[_Union[TenantScope, _Mapping]] = ..., session_id: _Optional[str] = ..., execution_id: _Optional[str] = ...) -> None: ...

class OperationPathSegment(_message.Message):
    __slots__ = ("name", "ordinal")
    NAME_FIELD_NUMBER: _ClassVar[int]
    ORDINAL_FIELD_NUMBER: _ClassVar[int]
    name: str
    ordinal: int
    def __init__(self, name: _Optional[str] = ..., ordinal: _Optional[int] = ...) -> None: ...

class OperationPath(_message.Message):
    __slots__ = ("segments",)
    SEGMENTS_FIELD_NUMBER: _ClassVar[int]
    segments: _containers.RepeatedCompositeFieldContainer[OperationPathSegment]
    def __init__(self, segments: _Optional[_Iterable[_Union[OperationPathSegment, _Mapping]]] = ...) -> None: ...

class ActivityPosition(_message.Message):
    __slots__ = ("operation_path", "activity_ordinal", "step_ordinal")
    OPERATION_PATH_FIELD_NUMBER: _ClassVar[int]
    ACTIVITY_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    STEP_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    operation_path: OperationPath
    activity_ordinal: int
    step_ordinal: int
    def __init__(self, operation_path: _Optional[_Union[OperationPath, _Mapping]] = ..., activity_ordinal: _Optional[int] = ..., step_ordinal: _Optional[int] = ...) -> None: ...

class WorkflowError(_message.Message):
    __slots__ = ("code", "message")
    CODE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    code: WorkflowErrorCode
    message: str
    def __init__(self, code: _Optional[_Union[WorkflowErrorCode, str]] = ..., message: _Optional[str] = ...) -> None: ...
