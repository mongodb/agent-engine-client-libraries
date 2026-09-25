import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from threading import Barrier

import httpx
import pytest

from agent_engine_runner_shared.connectors import ConnectorExecutor, ToolDefinitions, load_tool_defs
from agent_engine_runner_shared.connectors.definitions import Auth
from agent_engine_runner_shared.tool_api_error import classify_tool_api_error


@pytest.fixture
def fake_time(monkeypatch):
    from agent_engine_runner_shared.connectors import executor

    class Clock:
        def __init__(self):
            self.now = 100.0
            self.sleeps = []

        def sleep(self, seconds):
            self.sleeps.append(seconds)
            self.now += seconds

    clock = Clock()
    monkeypatch.setattr(executor.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(executor.time, "sleep", clock.sleep)
    return clock


@pytest.fixture
def document():
    return load_tool_defs(Path(__file__).parent / "fixtures/jira.tool_defs.yaml").model_dump()


def test_jira_read_write_and_roundtrip(document, tmp_path):
    definitions = ToolDefinitions.model_validate(document)
    path = tmp_path / "tool_defs.yaml"
    definitions.write(path)
    loaded = load_tool_defs(path)
    assert loaded == definitions
    schemas = loaded.tool_schemas()
    json.dumps(schemas, allow_nan=False)
    assert schemas[0] == {
        "name": "jiradc_getComments",
        "description": "Returns comments for an issue.",
        "inputSchema": {
            "type": "object",
            "properties": {"issueIdOrKey": {"type": "string"}},
            "required": ["issueIdOrKey"],
        },
    }
    assert schemas[1]["name"] == "jiradc_addComment"
    schemas[0]["inputSchema"]["properties"].clear()
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"id": "42"})

    with ConnectorExecutor(loaded, transport=httpx.MockTransport(handler)) as executor:
        assert json.loads(
            executor.execute(
                "jiradc_getComments", {"issueIdOrKey": "DEMO-1"}, env={"JIRA_TOKEN": "token"}
            )
        ) == {"id": "42"}
        executor.execute(
            "jiradc_addComment",
            {"issueIdOrKey": "DEMO-1", "body": "Hello"},
            env={"JIRA_TOKEN": "token"},
        )
    assert str(requests[0].url) == "https://jira.example/rest/api/2/issue/DEMO-1/comment"
    assert requests[1].method == "POST"
    assert json.loads(requests[1].content) == {"body": "Hello"}
    assert requests[1].headers["authorization"] == "Bearer token"


@pytest.mark.parametrize(
    "location,source,expected_url,expected_header",
    [
        (
            "header",
            "https://jira.example",
            "https://jira.example/api/2/issue/DEMO-1/comment",
            ("x-service-key", "secret"),
        ),
        (
            "query",
            "https://jira.example",
            "https://jira.example/api/2/issue/DEMO-1/comment?api_key=secret",
            None,
        ),
        (
            "cookie",
            "https://jira.example",
            "https://jira.example/api/2/issue/DEMO-1/comment",
            ("cookie", "api_key=secret"),
        ),
        (
            "path",
            "https://jira.example/bot{api_key}",
            "https://jira.example/botsecret/api/2/issue/DEMO-1/comment",
            None,
        ),
    ],
)
def test_api_key_locations(document, location, source, expected_url, expected_header):
    document["source"]["base_url"] = source
    document["auth"] = Auth(
        type="api_key",
        env="SERVICE_KEY",
        location=location,
        name="X-Service-Key" if location == "header" else "api_key",
    ).model_dump(exclude_none=True)
    requests = []
    with ConnectorExecutor(
        ToolDefinitions.model_validate(document),
        transport=httpx.MockTransport(
            lambda request: requests.append(request) or httpx.Response(200, text="ok")
        ),
    ) as executor:
        assert (
            executor.execute(
                "jiradc_getComments", {"issueIdOrKey": "DEMO-1"}, env={"SERVICE_KEY": "secret"}
            )
            == "ok"
        )
    assert str(requests[0].url) == expected_url
    if expected_header:
        assert requests[0].headers[expected_header[0]] == expected_header[1]


def test_basic_and_compound_credentials(document):
    requests = []
    document["auth"] = {
        "type": "basic",
        "username_env": "SERVICE_USER",
        "password_env": "SERVICE_PASS",
        "headers": {"API-Version": "2026-01-01"},
    }
    with ConnectorExecutor(
        ToolDefinitions.model_validate(document),
        transport=httpx.MockTransport(
            lambda request: requests.append(request) or httpx.Response(200, text="ok")
        ),
    ) as executor:
        executor.execute(
            "jiradc_getComments",
            {"issueIdOrKey": "DEMO-1"},
            env={"SERVICE_USER": "user", "SERVICE_PASS": "pass"},
        )
    assert requests[-1].headers["authorization"] == "Basic dXNlcjpwYXNz"
    assert requests[-1].headers["api-version"] == "2026-01-01"

    document["auth"] = {
        "type": "api_keys",
        "credentials": [
            {"env": "CLIENT_ID", "location": "header", "name": "X-Client-ID"},
            {"env": "CLIENT_SECRET", "location": "header", "name": "X-Client-Secret"},
        ],
        "headers": {"API-Version": "2026-01-01"},
    }
    with ConnectorExecutor(
        ToolDefinitions.model_validate(document),
        transport=httpx.MockTransport(
            lambda request: requests.append(request) or httpx.Response(200, text="ok")
        ),
    ) as executor:
        executor.execute(
            "jiradc_getComments",
            {"issueIdOrKey": "DEMO-1"},
            env={"CLIENT_ID": "id", "CLIENT_SECRET": "secret"},
        )
    assert requests[-1].headers["x-client-id"] == "id"
    assert requests[-1].headers["x-client-secret"] == "secret"
    assert requests[-1].headers["api-version"] == "2026-01-01"


@pytest.mark.parametrize(
    "encoding,value,expected,content_type",
    [
        ("text", "hello\nworld\t!", b"hello\nworld\t!", "text/plain"),
        ("binary", "AAEC", b"\x00\x01\x02", "application/octet-stream"),
    ],
)
def test_raw_request_body_encodings(document, encoding, value, expected, content_type):
    tool = document["tools"][1]
    tool["params"] = {"path": ["issueIdOrKey"], "wrapped": True}
    tool["inputSchema"] = {
        "type": "object",
        "properties": {
            "issueIdOrKey": {"type": "string"},
            "body": {"type": "string"},
        },
        "required": ["issueIdOrKey", "body"],
    }
    tool["body_encoding"] = encoding
    requests = []
    with ConnectorExecutor(
        ToolDefinitions.model_validate(document),
        transport=httpx.MockTransport(
            lambda request: requests.append(request) or httpx.Response(200, text="ok")
        ),
    ) as executor:
        executor.execute(
            "jiradc_addComment",
            {"issueIdOrKey": "DEMO-1", "body": value},
            env={"JIRA_TOKEN": "token"},
        )
    assert requests[0].content == expected
    assert requests[0].headers["content-type"] == content_type


def test_json_query_and_multipart_body_encodings(document):
    tool = document["tools"][1]
    tool["params"] = {
        "path": ["issueIdOrKey"],
        "query": ["filter"],
        "query_style": {"filter": "json"},
        "wrapped": True,
    }
    tool["inputSchema"] = {
        "type": "object",
        "properties": {
            "issueIdOrKey": {"type": "string"},
            "filter": {"type": "object"},
            "body": {
                "type": "object",
                "properties": {
                    "message": {"type": "string"},
                    "metadata": {"type": "object"},
                },
                "required": ["message"],
            },
        },
        "required": ["issueIdOrKey", "body"],
    }
    tool["body_encoding"] = "multipart"
    requests = []
    with ConnectorExecutor(
        ToolDefinitions.model_validate(document),
        transport=httpx.MockTransport(
            lambda request: requests.append(request) or httpx.Response(200, text="ok")
        ),
    ) as executor:
        executor.execute(
            "jiradc_addComment",
            {
                "issueIdOrKey": "DEMO-1",
                "filter": {"open": True},
                "body": {"message": "hello", "metadata": {"priority": 1}},
            },
            env={"JIRA_TOKEN": "token"},
        )
    request = requests[0]
    assert request.url.params["filter"] == '{"open":true}'
    assert request.headers["content-type"].startswith("multipart/form-data; boundary=")
    assert b"\r\n\r\nhello\r\n" in request.content
    assert b'{"priority": 1}' in request.content


def test_required_nullable_json_query_serializes_null(document):
    tool = document["tools"][0]
    tool["params"]["query"] = ["filter"]
    tool["params"]["query_style"] = {"filter": "json"}
    tool["inputSchema"]["properties"]["filter"] = {"type": ["object", "null"]}
    tool["inputSchema"]["required"].append("filter")
    requests = []
    with ConnectorExecutor(
        ToolDefinitions.model_validate(document),
        transport=httpx.MockTransport(
            lambda request: requests.append(request) or httpx.Response(200, text="ok")
        ),
    ) as executor:
        executor.execute(
            "jiradc_getComments",
            {"issueIdOrKey": "DEMO-1", "filter": None},
            env={"JIRA_TOKEN": "token"},
        )

    assert requests[0].url.params["filter"] == "null"


def test_static_auth_contract_fails_closed(document):
    with pytest.raises(ValueError, match="different secrets"):
        Auth(type="basic", username_env="SAME", password_env="SAME")
    with pytest.raises(ValueError, match="duplicate credential destinations"):
        Auth(
            type="api_keys",
            credentials=[
                {"env": "ONE", "location": "header", "name": "X-Key"},
                {"env": "TWO", "location": "header", "name": "x-key"},
            ],
        )
    document["auth"] = {
        "type": "api_key",
        "env": "TOKEN",
        "location": "path",
        "name": "token",
    }
    with pytest.raises(ValueError, match="base URL placeholders"):
        ToolDefinitions.model_validate(document)


def test_path_credential_cannot_control_the_base_url_host(document):
    document["source"]["base_url"] = "https://{host}/api"
    document["auth"] = {
        "type": "api_key",
        "env": "HOST",
        "location": "path",
        "name": "host",
    }

    with pytest.raises(ValueError, match="only in the URL path"):
        ToolDefinitions.model_validate(document)


def test_externalized_path_credential_requires_materialization(document):
    document["source"]["base_url"] = "https://jira.example/bot{token}"
    document["auth"] = None
    document["externalize"] = {"auth": {"type": "api_key", "location": "path", "name": "token"}}
    definitions = ToolDefinitions.model_validate(document)

    with pytest.raises(ValueError, match="materialized before execution"):
        ConnectorExecutor(definitions)


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(auth={"type": "oauth"}),
        lambda d: d["auth"].update(env="AGENTIC_API_KEY"),
        lambda d: d["source"].update(base_url="https://user:pass@jira.example"),
        lambda d: d["tools"].append(d["tools"][0]),
        lambda d: d["tools"][0].update(path="//other.example/path"),
        lambda d: d["tools"][0]["params"].update(query=["missing"]),
        lambda d: d["tools"][0]["inputSchema"]["properties"].update(extra={"type": "string"}),
    ],
    ids=[
        "oauth",
        "platform-secret",
        "userinfo",
        "duplicate-name",
        "network-path",
        "unmapped-parameter",
        "unplaced-property",
    ],
)
def test_reject_invalid_definitions(document, change):
    change(document)
    with pytest.raises(ValueError):
        ToolDefinitions.model_validate(document)


@pytest.mark.parametrize("value", [date(2026, 9, 9), float("nan")], ids=["date", "nan"])
def test_reject_non_json_schema_values(document, value):
    document["tools"][0]["inputSchema"]["properties"]["issueIdOrKey"]["default"] = value
    with pytest.raises(ValueError, match="schema must contain JSON values"):
        ToolDefinitions.model_validate(document)


@pytest.mark.parametrize(
    "keyword,value",
    [
        ("dependentRequired", {"issueIdOrKey": ["ghost"]}),
        ("allOf", [{"required": ["ghost"]}]),
        ("patternProperties", {"^ghost$": {"type": "string"}}),
        ("additionalProperties", False),
    ],
)
def test_reject_schema_composition_around_tool_argument_envelope(document, keyword, value):
    document["tools"][0]["inputSchema"][keyword] = value
    with pytest.raises(ValueError, match="tool argument envelope"):
        ToolDefinitions.model_validate(document)


def test_schema_composition_inside_wrapped_body_is_preserved_and_enforced(document):
    tool = document["tools"][1]
    tool["path"] = "/comment"
    tool["params"] = {"wrapped": True}
    tool["inputSchema"] = {
        "type": "object",
        "properties": {
            "body": {
                "type": "object",
                "properties": {"known": {"type": "string"}},
                "dependentRequired": {"known": ["ghost"]},
                "allOf": [{"required": ["known"]}],
                "patternProperties": {"^ghost$": {"type": "string"}},
                "additionalProperties": True,
            }
        },
        "required": ["body"],
    }
    requests = []

    with ConnectorExecutor(
        ToolDefinitions.model_validate(document),
        transport=httpx.MockTransport(
            lambda request: requests.append(request) or httpx.Response(200, text="ok")
        ),
    ) as executor:
        with pytest.raises(ValueError, match="compiled input schema"):
            executor.execute(
                "jiradc_addComment",
                {"body": {"known": "value"}},
                env={"JIRA_TOKEN": "token"},
            )
        assert requests == []
        assert (
            executor.execute(
                "jiradc_addComment",
                {"body": {"known": "value", "ghost": "value"}},
                env={"JIRA_TOKEN": "token"},
            )
            == "ok"
        )
    assert json.loads(requests[0].content) == {"known": "value", "ghost": "value"}


def test_recursive_yaml_schema_alias_is_rejected(tmp_path):
    path = tmp_path / "tool_defs.yaml"
    path.write_text(
        """source:
  name: recursive
  base_url: https://example.com
tools:
  - name: get_item
    method: POST
    path: /item
    params:
      wrapped: true
    inputSchema:
      type: object
      properties:
        body: &schema
          type: object
          properties:
            self: *schema
      required: [body]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema must contain JSON values"):
        load_tool_defs(path)


def test_deeply_nested_schema_is_rejected_with_value_error(document):
    schema = {"type": "string"}
    for _ in range(200):
        schema = {"type": "object", "properties": {"child": schema}}
    tool = document["tools"][1]
    tool["path"] = "/comment"
    tool["params"] = {"wrapped": True}
    tool["inputSchema"] = {
        "type": "object",
        "properties": {"body": schema},
        "required": ["body"],
    }

    with pytest.raises(ValueError, match="schema is too deeply nested"):
        ToolDefinitions.model_validate(document)


@pytest.mark.parametrize(
    "auth,header",
    [
        ({"type": "bearer", "env": "JIRA_TOKEN"}, "aUtHoRiZaTiOn"),
        ({"type": "api_key", "env": "JIRA_TOKEN", "header": "X-API-Key"}, "x-api-KEY"),
    ],
)
def test_auth_header_cannot_be_a_model_parameter(document, auth, header):
    document["auth"] = auth
    tool = document["tools"][0]
    tool["params"]["header"] = [header]
    tool["inputSchema"]["properties"][header] = {"type": "string"}
    with pytest.raises(ValueError, match="auth header|auth or transport headers"):
        ToolDefinitions.model_validate(document)


@pytest.mark.parametrize(
    "arguments,env",
    [
        ({"issueIdOrKey": "DEMO-1"}, {}),
        ({"issueIdOrKey": ".."}, {"JIRA_TOKEN": "token"}),
        ({"issueIdOrKey": "%2e%2e"}, {"JIRA_TOKEN": "token"}),
        ({"issueIdOrKey": "a/b"}, {"JIRA_TOKEN": "token"}),
        ({}, {"JIRA_TOKEN": "token"}),
    ],
)
def test_fail_before_dispatch(document, arguments, env):
    def handler(request):
        pytest.fail("invalid invocation reached network")

    with ConnectorExecutor(
        ToolDefinitions.model_validate(document), transport=httpx.MockTransport(handler)
    ) as executor:
        with pytest.raises(ValueError):
            executor.execute("jiradc_getComments", arguments, env=env)


def test_unknown_top_level_arguments_are_ignored(document):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, text="ok")

    with ConnectorExecutor(
        ToolDefinitions.model_validate(document), transport=httpx.MockTransport(handler)
    ) as executor:
        assert (
            executor.execute(
                "jiradc_getComments",
                {
                    "issueIdOrKey": "DEMO-1",
                    "unmapped": {"arbitrary": "value"},
                    "Authorization": "evil",
                },
                env={"JIRA_TOKEN": "token"},
            )
            == "ok"
        )
    assert len(requests) == 1
    assert str(requests[0].url) == "https://jira.example/rest/api/2/issue/DEMO-1/comment"
    assert requests[0].headers["authorization"] == "Bearer token"


def test_path_segment_validation_rejects_nested_encoding_without_recursive_work(document):
    nested_separator = "%" + "25" * 10_000 + "2F"

    def handler(request):
        pytest.fail("nested path encoding reached network")

    with ConnectorExecutor(
        ToolDefinitions.model_validate(document), transport=httpx.MockTransport(handler)
    ) as executor:
        with pytest.raises(ValueError, match="single non-traversing segment"):
            executor.execute(
                "jiradc_getComments",
                {"issueIdOrKey": nested_separator},
                env={"JIRA_TOKEN": "token"},
            )


def test_encoded_percent_marker_is_rejected_before_dispatch(document):
    def handler(request):
        pytest.fail("encoded percent marker reached network")

    with ConnectorExecutor(
        ToolDefinitions.model_validate(document), transport=httpx.MockTransport(handler)
    ) as executor:
        with pytest.raises(ValueError, match="single non-traversing segment"):
            executor.execute(
                "jiradc_getComments",
                {"issueIdOrKey": "a%25b"},
                env={"JIRA_TOKEN": "token"},
            )


def test_raw_percent_in_path_segment_is_encoded_by_executor(document):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, text="ok")

    with ConnectorExecutor(
        ToolDefinitions.model_validate(document), transport=httpx.MockTransport(handler)
    ) as executor:
        executor.execute(
            "jiradc_getComments", {"issueIdOrKey": "100%"}, env={"JIRA_TOKEN": "token"}
        )
    assert requests[0].url.path.endswith("/issue/100%/comment")
    assert str(requests[0].url).endswith("/issue/100%25/comment")


@pytest.mark.parametrize(
    "location",
    [
        "https://evil.example/",
        "http://jira.example/",
        "https://user@jira.example/",
        "//evil.example/",
    ],
)
def test_redirect_cannot_leak_credentials(document, location):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(302, headers={"location": location})

    with ConnectorExecutor(
        ToolDefinitions.model_validate(document), transport=httpx.MockTransport(handler)
    ) as executor:
        with pytest.raises(httpx.HTTPStatusError) as caught:
            executor.execute(
                "jiradc_getComments", {"issueIdOrKey": "DEMO-1"}, env={"JIRA_TOKEN": "token"}
            )
    assert len(requests) == 1
    error, _ = classify_tool_api_error(caught.value, "jira")
    assert error.http_status == 302
    assert error.classification == "UNKNOWN"
    assert "token" not in str(caught.value)
    assert "location" not in caught.value.response.headers
    assert "authorization" not in caught.value.request.headers


def test_http_error_carries_a_bounded_provider_error_envelope(document):
    def handler(request):
        return httpx.Response(
            400,
            json={
                "errorMessages": ["The value 'OpenJira' does not exist for the field 'project'."],
                "errors": {},
            },
        )

    with ConnectorExecutor(
        ToolDefinitions.model_validate(document), transport=httpx.MockTransport(handler)
    ) as executor:
        with pytest.raises(httpx.HTTPStatusError) as caught:
            executor.execute(
                "jiradc_getComments", {"issueIdOrKey": "DEMO-1"}, env={"JIRA_TOKEN": "token"}
            )

    error, message = classify_tool_api_error(caught.value, "jira")
    assert error.reason == "The value 'OpenJira' does not exist for the field 'project'."
    assert "does not exist for the field" in message
    # The explanation is extracted from the retained envelope, not synthesized.
    retained = caught.value.response._content or b""
    assert b'"errorMessages"' in retained


def test_near_limit_error_envelope_keeps_its_explanation(document):
    # A compact envelope at the retention limit must survive the size check.
    base = {"message": "Invalid credential", "pad": ""}
    base_size = len(json.dumps(base, separators=(",", ":")).encode())
    body = {"message": "Invalid credential", "pad": "x" * (4096 - base_size)}
    assert len(json.dumps(body, separators=(",", ":")).encode()) == 4096

    def handler(request):
        return httpx.Response(400, json=body)

    with ConnectorExecutor(
        ToolDefinitions.model_validate(document), transport=httpx.MockTransport(handler)
    ) as executor:
        with pytest.raises(httpx.HTTPStatusError) as caught:
            executor.execute(
                "jiradc_getComments",
                {"issueIdOrKey": "DEMO-1"},
                env={"JIRA_TOKEN": "token"},
            )

    error, message = classify_tool_api_error(caught.value, "jira")
    assert error.reason is not None and error.reason.startswith("Invalid credential")
    assert "Invalid credential" in message


def test_http_error_drops_an_oversized_provider_body(document):
    def handler(request):
        return httpx.Response(400, json={"message": "x" * 5000})

    with ConnectorExecutor(
        ToolDefinitions.model_validate(document), transport=httpx.MockTransport(handler)
    ) as executor:
        with pytest.raises(httpx.HTTPStatusError) as caught:
            executor.execute(
                "jiradc_getComments", {"issueIdOrKey": "DEMO-1"}, env={"JIRA_TOKEN": "token"}
            )

    error, message = classify_tool_api_error(caught.value, "jira")
    assert error.reason is None
    assert message == "jira API call failed: HTTP 400 UNKNOWN"


def test_path_authenticated_request_does_not_follow_redirects(document):
    document["source"]["base_url"] = "https://jira.example/bot/{token}"
    document["auth"] = {
        "type": "api_key",
        "env": "TOKEN",
        "location": "path",
        "name": "token",
    }
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(302, headers={"location": "/moved/secret"})

    with ConnectorExecutor(
        ToolDefinitions.model_validate(document), transport=httpx.MockTransport(handler)
    ) as executor:
        with pytest.raises(httpx.HTTPStatusError):
            executor.execute(
                "jiradc_getComments", {"issueIdOrKey": "DEMO-1"}, env={"TOKEN": "secret"}
            )

    assert len(requests) == 1


def test_concurrent_credentials_and_cookies_are_isolated(document):
    concurrent_requests = Barrier(8, timeout=5)

    def handler(request):
        assert "cookie" not in request.headers
        if request.url.path.endswith("comment"):
            concurrent_requests.wait()
            return httpx.Response(
                307, headers={"location": "/done", "set-cookie": "session=secret; Path=/"}
            )
        return httpx.Response(200, text=request.headers["authorization"])

    with ConnectorExecutor(
        ToolDefinitions.model_validate(document), transport=httpx.MockTransport(handler)
    ) as executor:

        def call(i):
            return executor.execute(
                "jiradc_getComments", {"issueIdOrKey": "DEMO-1"}, env={"JIRA_TOKEN": str(i)}
            )

        with ThreadPoolExecutor(max_workers=8) as pool:
            assert list(pool.map(call, range(16))) == [f"Bearer {i}" for i in range(16)]


@pytest.mark.parametrize(
    "status,classification",
    [
        (401, "AUTH_FAILED"),
        (403, "AUTH_FAILED"),
        (429, "RATE_LIMITED"),
        (503, "PROVIDER_UNAVAILABLE"),
        (400, "UNKNOWN"),
    ],
)
def test_structured_errors_expose_provider_explanation_without_platform_secrets(
    document, fake_time, status, classification
):
    def handler(request):
        return httpx.Response(status, json={"reason": "provider detail"})

    with ConnectorExecutor(
        ToolDefinitions.model_validate(document),
        transport=httpx.MockTransport(handler),
    ) as executor:
        with pytest.raises(httpx.HTTPStatusError) as caught:
            executor.execute(
                "jiradc_getComments",
                {"issueIdOrKey": "DEMO-1"},
                env={"JIRA_TOKEN": "platform-secret-value"},
            )
    error, message = classify_tool_api_error(caught.value, "jira")
    assert error.classification == classification
    assert error.reason == "provider detail"
    assert "provider detail" in message
    # The platform's own credential never rides into the surfaced error.
    combined = str(caught.value) + message + error.model_dump_json()
    assert "platform-secret-value" not in combined
    assert "authorization" not in caught.value.request.headers


def test_provider_envelope_redacts_secret_patterns(document):
    def handler(request):
        return httpx.Response(400, json={"reason": "token=sk-live-provider-secret"})

    with ConnectorExecutor(
        ToolDefinitions.model_validate(document), transport=httpx.MockTransport(handler)
    ) as executor:
        with pytest.raises(httpx.HTTPStatusError) as caught:
            executor.execute(
                "jiradc_getComments",
                {"issueIdOrKey": "DEMO-1"},
                env={"JIRA_TOKEN": "provider-token-value"},
            )

    error, message = classify_tool_api_error(caught.value, "jira")
    assert "sk-live-provider-secret" not in (error.reason or "") + message
    assert "<redacted>" in (error.reason or "")


def test_provider_echo_of_the_active_credential_is_redacted(document):
    def handler(request):
        return httpx.Response(401, json={"reason": "Rejected credential platform-secret-value"})

    with ConnectorExecutor(
        ToolDefinitions.model_validate(document), transport=httpx.MockTransport(handler)
    ) as executor:
        with pytest.raises(httpx.HTTPStatusError) as caught:
            executor.execute(
                "jiradc_getComments",
                {"issueIdOrKey": "DEMO-1"},
                env={"JIRA_TOKEN": "platform-secret-value"},
            )

    error, message = classify_tool_api_error(
        caught.value, "jira", credentials=["platform-secret-value"]
    )
    combined = message + error.model_dump_json()
    assert "platform-secret-value" not in combined
    assert error.reason == "Rejected credential <redacted>"


def test_timeout_is_classifiable(document):
    def handler(request):
        raise httpx.ReadTimeout("unsafe provider detail")

    with ConnectorExecutor(
        ToolDefinitions.model_validate(document), transport=httpx.MockTransport(handler)
    ) as executor:
        with pytest.raises(httpx.TimeoutException) as caught:
            executor.execute(
                "jiradc_getComments", {"issueIdOrKey": "DEMO-1"}, env={"JIRA_TOKEN": "t"}
            )
    assert classify_tool_api_error(caught.value, "jira")[0].classification == "TIMEOUT"
    assert "unsafe" not in str(caught.value)


def test_execution_budget_is_applied_across_retries(document, fake_time):
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(503, headers={"Retry-After": "10"})
        return httpx.Response(200, text="ok")

    with ConnectorExecutor(
        ToolDefinitions.model_validate(document),
        transport=httpx.MockTransport(handler),
    ) as executor:
        assert (
            executor.execute(
                "jiradc_getComments", {"issueIdOrKey": "DEMO-1"}, env={"JIRA_TOKEN": "t"}
            )
            == "ok"
        )

    assert fake_time.sleeps == [10]
    assert requests[0].extensions["timeout"] == {
        "connect": 30,
        "read": 30,
        "write": 30,
        "pool": 30,
    }
    assert requests[1].extensions["timeout"] == {
        "connect": 20,
        "read": 20,
        "write": 20,
        "pool": 20,
    }


def test_retry_after_cannot_exceed_execution_budget(document, fake_time):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(503, headers={"Retry-After": "30"})

    with ConnectorExecutor(
        ToolDefinitions.model_validate(document), transport=httpx.MockTransport(handler)
    ) as executor:
        with pytest.raises(httpx.ReadTimeout, match="connector request timed out"):
            executor.execute(
                "jiradc_getComments", {"issueIdOrKey": "DEMO-1"}, env={"JIRA_TOKEN": "t"}
            )

    assert len(requests) == 1
    assert fake_time.sleeps == []


def test_malformed_compressed_response_is_sanitized_and_closed(document):
    class Body(httpx.SyncByteStream):
        closed = False

        def __iter__(self):
            yield b"unsafe malformed gzip content"

        def close(self):
            self.closed = True

    body = Body()
    with ConnectorExecutor(
        ToolDefinitions.model_validate(document),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={"Content-Encoding": "gzip"},
                stream=body,
            )
        ),
    ) as executor:
        with pytest.raises(httpx.ConnectError) as caught:
            executor.execute(
                "jiradc_getComments", {"issueIdOrKey": "DEMO-1"}, env={"JIRA_TOKEN": "t"}
            )
    error, message = classify_tool_api_error(caught.value, "jira")
    assert error.classification == "CONNECTION_ERROR"
    assert "unsafe" not in str(caught.value) + message + error.model_dump_json()
    assert body.closed


def test_streaming_connection_error_is_sanitized_and_closed(document):
    class Body(httpx.SyncByteStream):
        closed = False

        def __iter__(self):
            raise httpx.ReadError("unsafe provider detail")
            yield b""  # pragma: no cover

        def close(self):
            self.closed = True

    body = Body()
    with ConnectorExecutor(
        ToolDefinitions.model_validate(document),
        transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=body)),
    ) as executor:
        with pytest.raises(httpx.ConnectError) as caught:
            executor.execute(
                "jiradc_getComments", {"issueIdOrKey": "DEMO-1"}, env={"JIRA_TOKEN": "t"}
            )
    error, message = classify_tool_api_error(caught.value, "jira")
    assert error.classification == "CONNECTION_ERROR"
    assert "unsafe" not in str(caught.value) + message + error.model_dump_json()
    assert body.closed


@pytest.mark.parametrize("status", [429, 503])
def test_retries_are_bounded_and_writes_never_retry(document, fake_time, status):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(status, headers={"Retry-After": "0"})

    with ConnectorExecutor(
        ToolDefinitions.model_validate(document), transport=httpx.MockTransport(handler)
    ) as executor:
        with pytest.raises(httpx.HTTPStatusError):
            executor.execute(
                "jiradc_getComments", {"issueIdOrKey": "DEMO-1"}, env={"JIRA_TOKEN": "t"}
            )
        assert len(requests) == 4
        assert fake_time.sleeps == [2, 4, 8]
        requests.clear()
        with pytest.raises(httpx.HTTPStatusError):
            executor.execute(
                "jiradc_addComment",
                {"issueIdOrKey": "DEMO-1", "body": "hello"},
                env={"JIRA_TOKEN": "t"},
            )
        assert len(requests) == 1


def test_retry_after_is_capped(document, fake_time):
    starts = []

    def handler(request):
        starts.append(fake_time.now)
        return httpx.Response(429, headers={"Retry-After": "9" * 5000})

    with ConnectorExecutor(
        ToolDefinitions.model_validate(document),
        transport=httpx.MockTransport(handler),
        timeout_seconds=120,
    ) as executor:
        with pytest.raises(httpx.HTTPStatusError):
            executor.execute(
                "jiradc_getComments", {"issueIdOrKey": "DEMO-1"}, env={"JIRA_TOKEN": "t"}
            )
    assert starts == [100, 130, 160, 190]
    assert fake_time.sleeps == [30, 30, 30]


def test_response_is_bounded_and_stream_is_closed(document):
    class Body(httpx.SyncByteStream):
        closed = False
        yielded = 0

        def __iter__(self):
            for _ in range(1000):
                self.yielded += 1
                yield b"x" * 8192

        def close(self):
            self.closed = True

    body = Body()
    with ConnectorExecutor(
        ToolDefinitions.model_validate(document),
        transport=httpx.MockTransport(lambda r: httpx.Response(200, stream=body)),
    ) as executor:
        result = executor.execute(
            "jiradc_getComments", {"issueIdOrKey": "DEMO-1"}, env={"JIRA_TOKEN": "t"}
        )
    assert result == "x" * 200_000 + "\n...[truncated]"
    assert body.closed
    assert body.yielded == 25


def test_streaming_response_checks_budget_between_chunks(document, fake_time):
    class Body(httpx.SyncByteStream):
        closed = False

        def __iter__(self):
            for _ in range(100):
                fake_time.now += 1
                yield b"x"

        def close(self):
            self.closed = True

    body = Body()
    with ConnectorExecutor(
        ToolDefinitions.model_validate(document),
        transport=httpx.MockTransport(lambda r: httpx.Response(200, stream=body)),
    ) as executor:
        with pytest.raises(httpx.ReadTimeout, match="connector request timed out"):
            executor.execute(
                "jiradc_getComments", {"issueIdOrKey": "DEMO-1"}, env={"JIRA_TOKEN": "t"}
            )
    assert fake_time.now == 130
    assert body.closed


def test_redirect_limit_closes_every_response(document):
    responses = []

    def handler(request):
        response = httpx.Response(307, headers={"location": "/loop"}, stream=httpx.ByteStream(b""))
        responses.append(response)
        return response

    with ConnectorExecutor(
        ToolDefinitions.model_validate(document), transport=httpx.MockTransport(handler)
    ) as executor:
        with pytest.raises(httpx.HTTPStatusError) as caught:
            executor.execute(
                "jiradc_getComments", {"issueIdOrKey": "DEMO-1"}, env={"JIRA_TOKEN": "t"}
            )
    assert len(responses) == 6
    assert all(response.is_closed for response in responses)
    assert classify_tool_api_error(caught.value, "jira")[0].http_status == 307


@pytest.mark.parametrize("location", [None, "https://[broken", "https://jira.example:bad/"])
def test_malformed_redirect_remains_a_sanitized_provider_error(document, location):
    responses = []

    def handler(request):
        response = httpx.Response(
            302,
            headers={"location": location} if location else {},
            stream=httpx.ByteStream(b"secret"),
        )
        responses.append(response)
        return response

    with ConnectorExecutor(
        ToolDefinitions.model_validate(document), transport=httpx.MockTransport(handler)
    ) as executor:
        with pytest.raises(httpx.HTTPStatusError) as caught:
            executor.execute(
                "jiradc_getComments", {"issueIdOrKey": "DEMO-1"}, env={"JIRA_TOKEN": "secret"}
            )
    assert len(responses) == 1
    assert responses[0].is_closed
    assert classify_tool_api_error(caught.value, "jira")[0].http_status == 302
    assert not caught.value.response.content
    assert "secret" not in str(caught.value)


@pytest.mark.parametrize("body", [None, [], [1, 2], {}, False, 0])
def test_wrapped_json_body_preserves_value(document, body):
    tool = document["tools"][1]
    tool["params"].update(
        body=[], wrapped=True, query=["expand"], header=["X-Trace"], cookie=["session"]
    )
    tool["inputSchema"]["properties"].update(
        body={},
        expand={"type": "boolean"},
        **{"X-Trace": {"type": "string"}, "session": {"type": "string"}},
    )
    document["auth"] = {"type": "api_key", "env": "JIRA_TOKEN", "header": "X-API-Key"}
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, text="ok")

    with ConnectorExecutor(
        ToolDefinitions.model_validate(document), transport=httpx.MockTransport(handler)
    ) as executor:
        executor.execute(
            "jiradc_addComment",
            {
                "issueIdOrKey": "DEMO-1",
                "body": body,
                "expand": False,
                "X-Trace": "trace",
                "session": "a;b=evil",
            },
            env={"JIRA_TOKEN": "t"},
        )
    request = requests[0]
    assert json.loads(request.content) == body
    assert type(json.loads(request.content)) is type(body)
    assert request.headers["content-type"] == "application/json"
    assert request.url.params["expand"] == "false"
    assert request.headers["x-trace"] == "trace"
    assert request.headers["cookie"] == "session=a%3Bb%3Devil"
    assert request.headers["x-api-key"] == "t"


def test_no_auth_and_redirect_query_is_not_reapplied(document):
    document["auth"] = {"type": "none"}
    tool = document["tools"][0]
    tool["params"]["query"] = ["startAt"]
    tool["inputSchema"]["properties"]["startAt"] = {"type": "integer"}
    requests = []

    def handler(request):
        requests.append(request)
        assert "authorization" not in request.headers
        if len(requests) == 1:
            return httpx.Response(302, headers={"location": "/page?startAt=10"})
        return httpx.Response(200, text="ok")

    with ConnectorExecutor(
        ToolDefinitions.model_validate(document), transport=httpx.MockTransport(handler)
    ) as executor:
        executor.execute("jiradc_getComments", {"issueIdOrKey": "DEMO-1", "startAt": 0}, env={})
    assert requests[0].url.params["startAt"] == "0"
    assert requests[1].url.params["startAt"] == "10"


def test_duplicate_yaml_keys_are_rejected(document, tmp_path):
    path = tmp_path / "tool_defs.yaml"
    ToolDefinitions.model_validate(document).write(path)
    with path.open("a") as output:
        output.write("auth: {type: none}\n")
    with pytest.raises(ValueError, match="unique"):
        load_tool_defs(path)


@pytest.mark.parametrize(
    "schema",
    [True, {"$ref": "https://evil.example/schema"}],
)
def test_unsupported_parameter_schemas_fail_explicitly(document, schema):
    document["tools"][0]["inputSchema"]["properties"]["issueIdOrKey"] = schema
    with pytest.raises(ValueError):
        ToolDefinitions.model_validate(document)


@pytest.mark.parametrize("retry_after", ["Thu, 01 Jan 1970 00:00:10 GMT", "000000000010"])
def test_retry_after_delay(document, fake_time, monkeypatch, retry_after):
    from agent_engine_runner_shared.connectors import executor

    monkeypatch.setattr(executor.time, "time", lambda: 0)
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(503, headers={"Retry-After": retry_after})
        return httpx.Response(200, text="ok")

    with ConnectorExecutor(
        ToolDefinitions.model_validate(document), transport=httpx.MockTransport(handler)
    ) as executor:
        assert (
            executor.execute(
                "jiradc_getComments", {"issueIdOrKey": "DEMO-1"}, env={"JIRA_TOKEN": "t"}
            )
            == "ok"
        )
    assert fake_time.sleeps == [10]


def test_operation_without_arguments(document):
    document["tools"] = [
        {
            "name": "status",
            "method": "GET",
            "path": "/status",
            "params": {},
            "inputSchema": {"type": "object"},
        }
    ]
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, text="healthy")

    with ConnectorExecutor(
        ToolDefinitions.model_validate(document), transport=httpx.MockTransport(handler)
    ) as executor:
        assert executor.execute("status", {}, env={"JIRA_TOKEN": "t"}) == "healthy"
    assert str(requests[0].url) == "https://jira.example/rest/status"
