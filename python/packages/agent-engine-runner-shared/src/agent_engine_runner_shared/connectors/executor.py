"""Synchronous HTTP execution core, invoked only after normal tool authorization.

Own one executor per loaded connector per Tool Pod process and close it at shutdown.
The transport reuses connections, never credentials or cookies. No runtime registration
occurs here.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
import re
import time
from collections.abc import Mapping
from email.utils import parsedate_to_datetime
from typing import Any, Self, cast
from urllib.parse import quote, urlencode, urljoin

import httpx
from jsonschema import Draft202012Validator

from agent_engine_runner_shared.connectors.definitions import (
    Operation,
    ToolDefinitions,
    validate_segment,
    validate_url,
)
from agent_engine_runner_shared.utils import tenant_env_vars

__all__ = ["DEFAULT_EXECUTION_TIMEOUT_SECONDS", "ConnectorExecutor"]

DEFAULT_EXECUTION_TIMEOUT_SECONDS = 30
MAX_RESPONSE_BYTES = 200_000
MAX_REDIRECTS = 5
MAX_RETRIES = 3
# Mirrors the classifier's envelope cap: only a body small enough to be parsed
# as a provider error envelope is retained on a raised HTTP error.
MAX_ERROR_ENVELOPE_BYTES = 4096


def _origin(url: httpx.URL) -> tuple[str, str, int | None]:
    """Reduce a URL to the fields that must remain fixed across redirects."""
    return url.scheme, url.host, url.port


def _retry_delay(retry_after: str, attempt: int) -> float:
    """Apply bounded provider retry advice without exceeding the invocation budget."""
    delay = min(2 ** (attempt + 1), 8)
    if retry_after.isascii() and retry_after.isdigit():
        # Three significant digits suffice for the 30-second cap, without parsing
        # an unbounded integer or discarding a valid zero-prefixed value.
        seconds = int(retry_after.lstrip("0")[:3] or "0")
        return max(delay, min(seconds, 30))
    if retry_after:
        try:
            seconds = parsedate_to_datetime(retry_after).timestamp() - time.time()
            return max(delay, min(seconds, 30))
        except (ValueError, TypeError, OverflowError):
            pass  # Invalid provider advice does not replace bounded backoff.
    return delay


class ConnectorExecutor:
    def __init__(
        self,
        definitions: ToolDefinitions,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout_seconds: float = DEFAULT_EXECUTION_TIMEOUT_SECONDS,
    ) -> None:
        """Load one validated catalog and create its reusable stateless transport."""
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("connector timeout must be a positive finite number")
        self._definitions = ToolDefinitions.model_validate(definitions.model_dump())
        if re.search(r"\{[^{}]+\}", self._definitions.source.base_url) and not any(
            credential.location == "path"
            for credential in (
                self._definitions.auth.credential_bindings() if self._definitions.auth else []
            )
        ):
            raise ValueError("connector path credential must be materialized before execution")
        self._operations = {tool.name: tool for tool in self._definitions.tools}
        # The transport provides pooling without Client's cookie or redirect state.
        self._transport = (
            transport if transport is not None else httpx.HTTPTransport(trust_env=False)
        )
        self._timeout_seconds = timeout_seconds

    def __enter__(self) -> Self:
        """Allow deterministic transport cleanup with a context manager."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Close the pooled transport when the executor leaves its context."""
        self.close()

    def close(self) -> None:
        """Release pooled connections after in-flight calls have finished."""
        self._transport.close()

    def _prepare_request(
        self, tool: Operation, arguments: dict[str, Any], env: Mapping[str, str] | None
    ) -> tuple[
        httpx.URL,
        dict[str, str],
        bytes | None,
        list[tuple[str, str]],
        bool,
    ]:
        """Validate and serialize request-local arguments and credentials before I/O."""
        properties = tool.inputSchema.get("properties", {})
        if not isinstance(arguments, dict):
            raise ValueError("arguments do not match the compiled input schema")
        arguments = {name: arguments[name] for name in properties if name in arguments}
        if not Draft202012Validator(tool.inputSchema).is_valid(arguments):
            raise ValueError("arguments do not match the compiled input schema")
        try:
            json.dumps(arguments, allow_nan=False)
        except (ValueError, TypeError):
            raise ValueError("arguments must be JSON values") from None
        path = tool.path
        for key in tool.params.path:
            value = self._param_value(arguments, key)
            # Path arrays serialize simple-style: each element encoded, then
            # joined with the literal structural comma.
            if isinstance(value, list):
                rendered = ",".join(quote(validate_segment(item), safe="") for item in value)
            else:
                rendered = quote(validate_segment(value), safe="")
            path = path.replace("{" + key + "}", rendered)
        # Query values serialize per the encoding the generator recorded:
        # repeated keys by default (form/explode), comma-joined for
        # explode: false, bracketed names for deepObject lists. Union-typed
        # arguments take the branch their value matches: mappings expand to
        # bracketed fields, which is the only object spelling query strings
        # support.
        query: list[tuple[str, Any]] = []
        for key in tool.params.query:
            if key not in arguments:
                continue
            raw = arguments[key]
            encoding = tool.params.query_style.get(key, "repeat")
            if encoding == "json":
                query.append((key, json.dumps(raw, allow_nan=False, separators=(",", ":"))))
                continue
            if raw is None:
                # Query strings carry no null spelling; nullable arguments
                # omit the parameter. JSON-valued queries are handled above
                # because application/json has an explicit null spelling.
                continue
            if isinstance(raw, dict):
                query.extend(self._bracket_pairs(key, raw))
                continue
            value = self._param_value(arguments, key)
            if isinstance(value, list) and encoding == "comma":
                query.append((key, ",".join(value)))
            elif isinstance(value, list) and encoding == "brackets":
                query.extend((f"{key}[]", item) for item in value)
            elif isinstance(value, list):
                query.extend((key, item) for item in value)
            else:
                query.append((key, value))
        headers = {}
        for key in tool.params.header:
            if key not in arguments or arguments[key] is None:
                # Headers carry no null spelling either.
                continue
            value = self._param_value(arguments, key)
            # Header lists are comma-joined per HTTP list-header semantics.
            headers[key] = ",".join(value) if isinstance(value, list) else value
        if tool.params.cookie:
            pairs = [
                f"{key}={quote(self._scalar(arguments[key]), safe='')}"
                for key in tool.params.cookie
                if key in arguments and arguments[key] is not None
            ]
            if pairs:
                headers["Cookie"] = "; ".join(pairs)
        base_url, auth_query, has_path_auth = self._apply_auth(headers, query, env)
        content, content_type = self._request_body(tool, arguments)
        if content_type:
            headers["Content-Type"] = content_type
        url = httpx.URL(base_url + path)
        if query:
            url = url.copy_merge_params(query)
        return url, headers, content, auth_query, has_path_auth

    @staticmethod
    def _secret(env: Mapping[str, str], name: str) -> str:
        """Read one printable secret without placing its value in an error."""
        value = env.get(name)
        if not value or any(ord(character) < 32 or ord(character) >= 127 for character in value):
            raise ValueError("configured connector secret is missing or invalid")
        return value

    def _apply_auth(
        self,
        headers: dict[str, str],
        query: list[tuple[str, Any]],
        env: Mapping[str, str] | None,
    ) -> tuple[str, list[tuple[str, str]], bool]:
        """Place materialized credentials exactly where the definition declares them.

        Query credentials are returned so same-origin redirects can preserve them.
        Path-authenticated requests reject redirects because moving the credential
        to another path cannot be proven safe from a redirect target alone.
        """
        auth = self._definitions.auth
        base_url = self._definitions.source.base_url.rstrip("/")
        if not auth or auth.type == "none":
            return base_url, [], False

        secrets = env if env is not None else tenant_env_vars()
        auth_query: list[tuple[str, str]] = []
        has_path_auth = False
        if auth.type == "bearer":
            assert auth.env
            headers["Authorization"] = f"Bearer {self._secret(secrets, auth.env)}"
        elif auth.type == "basic":
            assert auth.username_env and auth.password_env
            username = self._secret(secrets, auth.username_env)
            if ":" in username:
                raise ValueError("configured connector basic username is invalid")
            value = f"{username}:{self._secret(secrets, auth.password_env)}"
            headers["Authorization"] = "Basic " + base64.b64encode(value.encode("ascii")).decode(
                "ascii"
            )
        else:
            for credential in auth.credential_bindings():
                value = credential.prefix + self._secret(secrets, credential.env)
                if credential.location == "header":
                    headers[credential.name] = value
                elif credential.location == "query":
                    pair = (credential.name, value)
                    auth_query.append(pair)
                    query.append(pair)
                elif credential.location == "cookie":
                    pair = f"{credential.name}={quote(value, safe='')}"
                    headers["Cookie"] = "; ".join(filter(None, (headers.get("Cookie"), pair)))
                else:
                    encoded = quote(validate_segment(value), safe="")
                    has_path_auth = True
                    base_url = base_url.replace("{" + credential.name + "}", encoded)
        headers.update(auth.headers or {})
        return base_url, auth_query, has_path_auth

    def _request_body(
        self, tool: Operation, arguments: dict[str, Any]
    ) -> tuple[bytes | None, str | None]:
        """Encode the validated body using only the catalog's compiled wire metadata."""
        if tool.params.wrapped and "body" in arguments:
            body = arguments["body"]
            if tool.body_encoding == "form":
                if not isinstance(body, dict):
                    raise ValueError("form request body must be an object")
                return (
                    urlencode(self._form_fields(body, tool.params.form_style), doseq=True).encode(),
                    "application/x-www-form-urlencoded",
                )
            if tool.body_encoding == "multipart":
                if not isinstance(body, dict):
                    raise ValueError("multipart request body must be an object")
                parts = [
                    (
                        key,
                        (
                            None,
                            json.dumps(value, allow_nan=False)
                            if isinstance(value, (dict, list))
                            else self._scalar(value),
                            "application/json" if isinstance(value, (dict, list)) else "text/plain",
                        ),
                    )
                    for key, value in body.items()
                    if value is not None
                ]
                request = httpx.Request("POST", "https://multipart.invalid", files=parts)
                return request.read(), request.headers["Content-Type"]
            if tool.body_encoding == "text":
                if not isinstance(body, str):
                    raise ValueError("text request body must be a string")
                return body.encode(), "text/plain"
            if tool.body_encoding == "binary":
                try:
                    return (
                        base64.b64decode(self._scalar(body), validate=True),
                        "application/octet-stream",
                    )
                except binascii.Error:
                    raise ValueError("binary request body must be valid base64") from None
            return json.dumps(
                body, allow_nan=False
            ).encode(), tool.content_type or "application/json"

        if tool.params.body:
            body = {key: arguments[key] for key in tool.params.body if key in arguments}
            if body:
                return (
                    json.dumps(body, allow_nan=False).encode(),
                    tool.content_type or "application/json",
                )
        return None, None

    def execute(
        self, name: str, arguments: dict[str, Any], *, env: Mapping[str, str] | None = None
    ) -> str:
        """Execute a named compiled operation and return bounded UTF-8 response text.

        Invalid definitions/arguments/secrets raise ValueError before dispatch.
        HTTP and transport failures use httpx exceptions understood by ToolAPIError.
        Only GET/HEAD/OPTIONS retry 429/503; writes are never automatically replayed.
        """
        deadline = time.monotonic() + self._timeout_seconds
        tool = self._operations.get(name)
        if tool is None:
            raise ValueError("unknown connector tool")
        try:
            url, headers, content, auth_query, has_path_auth = self._prepare_request(
                tool, arguments, env
            )
        except RecursionError:
            # Pathological nesting exhausts the interpreter stack in the
            # validator, the JSON round-trip, or bracket expansion; the
            # contract promises ValueError for invalid arguments.
            raise ValueError("arguments are too deeply nested") from None
        for attempt in range(MAX_RETRIES + 1):
            status, response_headers, text = self._request(
                tool.method,
                url,
                headers,
                content,
                deadline,
                auth_query=auth_query,
                has_path_auth=has_path_auth,
            )
            if 200 <= status < 300:
                return text
            if (
                status in (429, 503)
                and tool.method in ("GET", "HEAD", "OPTIONS")
                and attempt < MAX_RETRIES
            ):
                delay = _retry_delay(response_headers.get("retry-after", ""), attempt)
                if delay >= self._remaining_seconds(deadline):
                    raise self._timeout_error()
                time.sleep(delay)
                continue
            raise self._http_error(tool.method, status, text)
        raise AssertionError("retry loop exhausted without a result")

    @staticmethod
    def _scalar(value: Any) -> str:
        """Render a JSON scalar safely for an HTTP path, header, cookie, or query."""
        if not isinstance(value, (str, int, float, bool)):
            raise ValueError("non-body arguments must be scalar values")
        result = str(value).lower() if isinstance(value, bool) else str(value)
        if any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in result):
            # C0, DEL, and C1: CR/LF ride here for response splitting, and
            # C1 decodes ambiguously across the latin-1/UTF-8 boundary.
            raise ValueError("parameter contains control characters")
        return result

    def _param_value(self, arguments: dict[str, Any], key: str) -> Any:
        """Return a validated scalar or list of scalars."""
        value = arguments[key]
        if isinstance(value, list):
            return [self._scalar(item) for item in value]
        return self._scalar(value)

    def _bracket_pairs(self, prefix: str, value: Any) -> list[tuple[str, str]]:
        """Expand a JSON value using bracket notation."""
        if isinstance(value, dict):
            pairs: list[tuple[str, str]] = []
            for sub, subvalue in value.items():
                pairs.extend(self._bracket_pairs(f"{prefix}[{self._scalar(sub)}]", subvalue))
            return pairs
        if isinstance(value, list):
            pairs = []
            for item in value:
                pairs.extend(self._bracket_pairs(f"{prefix}[]", item))
            return pairs
        if value is None:
            return []
        return [(prefix, self._scalar(value))]

    def _form_fields(self, body: dict[str, Any], styles: dict[str, str]) -> list[tuple[str, str]]:
        """Serialize a validated object as form fields."""
        fields: list[tuple[str, str]] = []
        for key, value in body.items():
            if value is None:
                continue
            style = styles.get(key)
            if style == "brackets" or isinstance(value, dict):
                fields.extend(self._bracket_pairs(key, value))
            elif isinstance(value, list):
                rendered = [self._scalar(item) for item in value if item is not None]
                if style == "comma":
                    fields.append((key, ",".join(rendered)))
                else:
                    fields.extend((key, item) for item in rendered)
            else:
                fields.append((key, self._scalar(value)))
        return fields

    def _request(
        self,
        method: str,
        url: httpx.URL,
        headers: dict[str, str],
        content: bytes | None,
        deadline: float,
        *,
        auth_query: list[tuple[str, str]] | None = None,
        has_path_auth: bool = False,
    ) -> tuple[int, httpx.Headers, str]:
        """Dispatch one bounded request while containing redirects and response size.

        Redirects stay on the original origin, credentials remain present only in
        their declared locations, and every response is closed before returning.
        """
        origin = _origin(url)
        for redirect in range(MAX_REDIRECTS + 1):
            remaining_seconds = self._remaining_seconds(deadline)
            # HTTPX applies these values to each blocking I/O operation. The
            # monotonic checks carry the invocation budget between operations.
            request = httpx.Request(
                method,
                url,
                headers=headers,
                content=content,
                extensions={"timeout": httpx.Timeout(remaining_seconds).as_dict()},
            )
            try:
                response = self._transport.handle_request(request)
                response.request = request
                try:
                    if response.status_code in (301, 302, 303, 307, 308):
                        location = response.headers.get("location")
                        if not location or redirect == MAX_REDIRECTS:
                            raise self._http_error(method, response.status_code)
                        try:
                            next_url = httpx.URL(validate_url(urljoin(str(url), location)))
                        except (ValueError, httpx.InvalidURL):
                            raise self._http_error(method, response.status_code) from None
                        if _origin(next_url) != origin:
                            raise self._http_error(method, response.status_code)
                        if has_path_auth:
                            raise self._http_error(method, response.status_code)
                        if auth_query:
                            auth_names = {name for name, _ in auth_query}
                            query = [
                                pair
                                for pair in next_url.params.multi_items()
                                if pair[0] not in auth_names
                            ]
                            next_url = next_url.copy_with(
                                query=str(httpx.QueryParams(cast(Any, query + auth_query))).encode(
                                    "ascii"
                                )
                            )
                        url = next_url
                        if (
                            response.status_code == 303
                            and method != "HEAD"
                            or response.status_code in (301, 302)
                            and method == "POST"
                        ):
                            method, content = "GET", None
                            headers = {
                                k: v for k, v in headers.items() if k.lower() != "content-type"
                            }
                        continue
                    body = bytearray()
                    truncated = False
                    for chunk in response.iter_bytes():
                        self._remaining_seconds(deadline)
                        room = MAX_RESPONSE_BYTES - len(body)
                        body.extend(chunk[:room])
                        if len(chunk) > room:
                            truncated = True
                            break
                    text = body.decode("utf-8", errors="replace")
                    if truncated:
                        text += "\n...[truncated]"
                    self._remaining_seconds(deadline)
                    return response.status_code, response.headers, text
                finally:
                    response.close()
            except httpx.DecodingError:
                raise httpx.ConnectError("connector response decoding failed") from None
            except httpx.TimeoutException:
                raise httpx.ReadTimeout("connector request timed out") from None
            except httpx.TransportError:
                raise httpx.ConnectError("connector transport failed") from None
        raise AssertionError("redirect loop exhausted without a result")

    @staticmethod
    def _timeout_error() -> httpx.ReadTimeout:
        """Use one sanitized timeout shape for all exhausted budget paths."""
        return httpx.ReadTimeout("connector request timed out")

    @classmethod
    def _remaining_seconds(cls, deadline: float) -> float:
        """Carry one invocation deadline across retries, redirects, and body reads."""
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise cls._timeout_error()
        return remaining

    def _http_error(
        self, method: str, status: int, body: str | None = None
    ) -> httpx.HTTPStatusError:
        """Return a classifiable HTTP error that retains only a bounded error envelope.

        Providers explain 4xx/5xx responses in their body; the shared classifier
        reads a small recognized envelope so the failure can name the cause.
        The retained body is raw (not pre-redacted) and must never be logged or
        serialized; only the classifier's extracted, bounded, credential-redacted
        explanation is returned. Larger bodies, redirect locations, and
        credential-bearing requests are never retained.
        """
        request = httpx.Request(method, self._definitions.source.base_url)
        content: bytes | None = None
        if body:
            encoded = body.encode("utf-8", errors="replace")
            if len(encoded) <= MAX_ERROR_ENVELOPE_BYTES:
                content = encoded
        return httpx.HTTPStatusError(
            f"connector API returned HTTP {status}",
            request=request,
            response=httpx.Response(status, request=request, content=content),
        )
