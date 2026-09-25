"""Function mode for the Runner SDK.

Function mode runs a tool CLI-style: the process boots, runs a single named
tool from the registry, reports the result to the OE, and exits. It is the
per-call counterpart to the long-running tool server (mode ``tool``).
``ToolFunctionRunner`` reuses the ``ToolExecution`` mixin for tool/built-in/
named-LLM registration (``prepare``) and the invocation core (``invoke_tool``),
but it is not a server: it binds no listener, reads one invocation, runs the
tool, reports the result, and returns.

Because fctr's ``Run`` returns at guest boot rather than at completion, the
result cannot ride back through the call that launched the VM. Instead the OE
delivers the invocation in ``RunRequest.metadata`` (the initial keyset on the
``Run`` RPC), which fctr seeds into the guest metadata directory before the
workload starts, so it is present locally at startup with no fetch.

The result is reported as a ``ToolResultRequest`` POSTed to
``{request.oe_url}/tool/result``, the same contract and endpoint the AER
already uses to report tool results (see ``secure_wrapper``). Function mode
needs no new OE endpoint or result shape; it just takes on the reporting role
the synchronous caller would have played. (The OE must allow the function-mode
pod to reach ``/tool/result``, the same as for the AER.)

Invocation contract: the OE sets one key, ``request``, in ``RunRequest.metadata``.
It is a JSON ``ToolFunctionRequest`` envelope carrying:

- ``request``: a ``ToolPodExecuteRequest`` (the object the OE already builds for
  server mode, so every field rides at full fidelity: ``oe_url``, delegated
  ``authorization``, MCP ``metadata``, ``payload``, the tool ``arguments``).
- ``step``: the tool-call step number, required by ``ToolResultRequest`` and
  owned by the OE dispatch (not part of the tool invocation itself).

It is read once at startup, before the tool runs. Delivering via metadata
rather than env avoids env's per-variable size limit and keeps delegated
credentials out of ``os.environ`` (where child processes the tool spawns would
inherit them).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from agent_engine_runner_shared.context import (
    OwnerUrlFailureState,
    clear_execution_context,
    set_execution_context,
)
from agent_engine_runner_shared.models import (
    ToolPodExecuteRequest,
    ToolPodExecuteResponse,
    ToolResultRequest,
)
from agent_engine_runner_shared.server.http_retry import post_json_with_retries
from agent_engine_runner_shared.server.oe_url import resolve_oe_url
from agent_engine_runner_shared.server.owner_url import resolve_owner_url
from agent_engine_runner_shared.server.tool import ToolExecution
from agent_engine_runner_shared.tls_client import create_async_httpx_client_with_tls
from agent_engine_runner_shared.utils import get_request_timeout

logger = logging.getLogger(__name__)

# Guest metadata directory, mirroring fctr's ``util.MetadataDir``. fctr
# materializes ``RunRequest.metadata`` into ``<dir>/<key>`` files before the
# workload starts. A fixed convention, like the EC2 IMDS address: the
# platform does not announce it, so it is not read from the environment.
_METADATA_DIR = "/run/meta"
_REQUEST_KEY = "request"
_TOOL_RESULT_PATH = "/tool/result"


def _current_trace_context() -> tuple[Optional[str], Optional[str]]:
    """Best-effort (trace_id, span_id) of the active OTel span.

    Lazily imported: this module runs unconditionally in function mode, while
    OTel itself lives behind the optional ``tracing`` extra. A top-level
    import here would make tool execution depend on an extra that has
    nothing to do with it.
    """
    try:
        from agent_engine_runner_shared.tracing.setup import get_current_trace_context
    except ImportError:
        return None, None
    return get_current_trace_context()


class ToolFunctionRequest(BaseModel):
    """The function-mode invocation envelope delivered at ``/run/meta/request``.

    The OE sets a single metadata key carrying both the tool invocation and the
    tool-call ``step`` number. ``step`` is not a field of ``ToolPodExecuteRequest``
    (it is an OE-owned reporting field required by ``ToolResultRequest``), so it
    rides alongside the request here rather than polluting the invocation type
    that server mode also uses.
    """

    model_config = ConfigDict(hide_input_in_errors=True)

    request: ToolPodExecuteRequest
    step: int


def _read_metadata(metadata_dir: str, key: str) -> str:
    """Read a metadata key from ``<metadata_dir>/<key>``.

    fctr seeds the key before the workload starts, so it is present at startup.
    Raises ``ValueError`` when the key is missing or empty. A malformed
    contract is an OE bug, not tool data, so the process should exit non-zero.
    """
    path = Path(metadata_dir) / key
    try:
        value = path.read_text()
    except FileNotFoundError as exc:
        raise ValueError(f"function mode requires metadata key {key!r} at {path}") from exc
    if not value.strip():
        raise ValueError(f"function mode metadata key {key!r} at {path} is empty")
    return value


def _read_function_request(metadata_dir: str) -> ToolFunctionRequest:
    """Parse the ``request`` metadata key into a ``ToolFunctionRequest`` envelope.

    Raises ``ValueError`` on bad JSON or a schema violation (including a missing
    ``step`` or an invalid inner request) so the failure is visible at the
    process boundary.
    """
    raw = _read_metadata(metadata_dir, _REQUEST_KEY)
    try:
        return ToolFunctionRequest.model_validate_json(raw)
    except ValidationError as exc:
        raise ValueError(
            f"metadata key {_REQUEST_KEY!r} is not a valid ToolFunctionRequest: {exc}"
        ) from exc


def _result_request(
    request: ToolPodExecuteRequest,
    response: ToolPodExecuteResponse,
    step: int,
    duration_ms: float,
) -> ToolResultRequest:
    trace_id, span_id = _current_trace_context()
    return ToolResultRequest(
        execution_id=request.execution_id,
        step_number=step,
        tool_name=request.tool_name,
        status=response.status,
        result=response.result,
        error=response.error,
        duration_ms=duration_ms,
        pod_name=response.pod_name,
        metadata=response.metadata,
        trace_id=trace_id,
        span_id=span_id,
        tool_api_error=response.tool_api_error,
    )


def _result_payload(result: ToolResultRequest, execution_id: str) -> Dict[str, Any]:
    """Encode the result to a JSON-safe dict for the ``/tool/result`` POST.

    A tool may return a value the JSON encoder cannot handle, in either the
    ``result`` or the observability ``metadata``. The HTTP ``/execute`` path
    surfaces that as a 500 the OE observes; function mode has no synchronous
    channel, so a raw encode failure would drop the result entirely. Downgrade
    any encode failure to an error report so the OE still receives a terminal
    signal for the step. The downgrade clears both ``result`` and ``metadata``
    so the error report itself carries only primitive fields and cannot fail to
    encode in turn.
    """
    try:
        payload = result.model_dump(mode="json")
        json.dumps(payload)
        return payload
    except Exception as exc:  # noqa: BLE001 - any encode failure becomes a terminal error report
        logger.error(
            "function mode: result for execution_id=%s is not JSON-serializable: %s",
            execution_id,
            exc,
        )
        downgraded = result.model_copy(
            update={
                "status": "error",
                "result": None,
                "metadata": {},
                "error": f"tool result is not JSON-serializable: {exc}",
            }
        )
        return downgraded.model_dump(mode="json")


class ToolFunctionRunner(ToolExecution):
    """Runs one metadata-delivered tool call and exits. Not a server.

    Reuses the ``ToolExecution`` mixin for tool/built-in/named-LLM registration
    (``prepare``) and the invocation core (``invoke_tool``); it binds no HTTP
    listener. ``metadata_dir`` and ``client`` are injectable for tests.
    """

    def __init__(
        self,
        runtime: Any,
        *,
        metadata_dir: str = _METADATA_DIR,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.runtime = runtime
        self._metadata_dir = metadata_dir
        self._client = client
        # Shared preparation attempts registry construction once before the tool call.
        self._llm_registry_lock = asyncio.Lock()
        self._llm_registry_loaded = False
        self._llm_registry_load_error = None
        self._connector_runtime = None

    async def run(self) -> None:
        """Run one tool from the metadata-delivered request and report the result.

        Re-raises if the result POST cannot be delivered after retries so fctr
        records the execution as failed.
        """
        function_request = _read_function_request(self._metadata_dir)
        request = function_request.request
        step = function_request.step
        # Pin the callback base to the runner's own configured OE (deploy-time
        # OE_URL) before it is used as the callback base or the owner-URL trust
        # anchor. The request field is only honoured when no OE_URL is stamped
        # (local dev / tests). Resolving once here means the owner URL is
        # validated against a trusted anchor, not the raw request value.
        oe_url = resolve_oe_url(request.oe_url or "").rstrip("/")
        if not oe_url:
            raise ValueError("function mode requires oe_url on the request to report the result")
        owner_url = resolve_owner_url(request.oe_owner_url, oe_url)
        owner_failure = OwnerUrlFailureState()
        # Install before prepare() and keep through result delivery / connector
        # shutdown so startup and callback failures still emit platform_trace_id.
        tokens = set_execution_context(
            execution_id=request.execution_id,
            wrapper=None,
            oe_url=oe_url,
            oe_owner_url=owner_url,
            owner_url_failure=owner_failure,
            user_id=request.user_id,
            session_id=request.session_id,
            authorization=request.authorization,
            custom_headers=request.custom_headers,
            payload=request.payload,
            trace_id=request.platform_trace_id,
        )

        try:
            try:
                await self.prepare()

                logger.info(
                    "function mode: invoking tool=%r execution_id=%s step=%d",
                    request.tool_name,
                    request.execution_id,
                    step,
                )
                started = time.monotonic()
                response = await self.invoke_tool(request, owner_url_failure=owner_failure)
                duration_ms = (time.monotonic() - started) * 1000.0

                payload = _result_payload(
                    _result_request(request, response, step, duration_ms), request.execution_id
                )

                owns_client = self._client is None
                client = self._client or await create_async_httpx_client_with_tls(
                    oe_url, get_request_timeout()
                )
                try:
                    await post_json_with_retries(
                        client,
                        f"{oe_url}{_TOOL_RESULT_PATH}",
                        payload,
                        owner_url=(
                            f"{owner_url}{_TOOL_RESULT_PATH}"
                            if owner_url and not owner_failure.failed
                            else None
                        ),
                        on_owner_failure=lambda: setattr(owner_failure, "failed", True),
                    )
                    logger.info(
                        "function mode: reported %s result for execution_id=%s step=%d",
                        payload["status"],
                        request.execution_id,
                        step,
                    )
                finally:
                    if owns_client:
                        await client.aclose()
            finally:
                await self.close_connector_runtime()
        finally:
            clear_execution_context(tokens)
