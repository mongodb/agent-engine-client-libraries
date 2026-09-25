"""Generator battery: compile, scaffold, and regenerate authoring artifacts.

The generator library reads local files only; downloading specs and catalogs
is the CLI's responsibility, so network behavior is not exercised here.
"""

from __future__ import annotations

import hashlib
import shlex
from pathlib import Path

import httpx
import pytest
import yaml
from pydantic import ValidationError

from agent_engine_runner_shared.connectors import (
    ConnectorExecutor,
    GenerateError,
    compile_catalog,
    load_tool_defs,
    regenerate_catalog,
    scaffold_tool_yaml,
)
from agent_engine_runner_shared.connectors.definitions import (
    Auth,
    Credential,
    ToolDefinitions,
)

SPEC = """\
openapi: 3.0.1
info:
  title: Weather
  version: "1.0"
servers:
  - url: https://weather.example/api
paths:
  /forecast/{city}:
    get:
      operationId: getForecast
      summary: Get the forecast for a city
      parameters:
        - name: city
          in: path
          required: true
          schema: {type: string}
        - name: days
          in: query
          schema: {type: integer}
      responses:
        "200": {description: ok}
  /alerts:
    post:
      operationId: createAlert
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                region: {type: string}
                threshold: {type: number}
              required: [region]
      responses:
        "200": {description: ok}
"""


def _connector_dir(tmp_path: Path, spec: str = SPEC) -> Path:
    out = tmp_path / "connectors" / "weather"
    out.mkdir(parents=True)
    (out / "spec.yaml").write_text(spec, encoding="utf-8")
    return out


class TestCompile:
    def test_compiles_catalog_and_scaffolds_tool_yaml(self, tmp_path, monkeypatch):
        out = _connector_dir(tmp_path)
        monkeypatch.chdir(out)

        result = compile_catalog("spec.yaml", name="weather", out_dir=".")

        definitions = load_tool_defs(result.catalog)
        assert result.tool_yaml is not None
        assert [tool.name for tool in definitions.tools] == [
            "weather_getForecast",
            "weather_createAlert",
        ]
        assert definitions.source.base_url == "https://weather.example/api"
        forecast = definitions.tools[0]
        assert forecast.params.path == ["city"]
        assert forecast.params.query == ["days"]
        alert = definitions.tools[1]
        # Optional body with nested required properties travels as one
        # optional argument whose nested schema stays enforced.
        assert alert.params.wrapped is True
        assert alert.params.body == []
        assert "required" not in alert.inputSchema
        assert alert.inputSchema["properties"]["body"]["required"] == ["region"]

        authoring = yaml.safe_load(result.tool_yaml.read_text(encoding="utf-8"))
        assert authoring["name"] == "weather"
        assert authoring["tool_defs"] == "tool_defs.yaml"
        assert authoring["source"] == {
            "type": "openapi",
            "base_url": "https://weather.example/api",
            "spec": "spec.yaml",
        }
        assert authoring["expose"] == {"allow_all": True}
        assert "auth" not in authoring

    def test_compiled_catalog_matches_the_runtime_model_exactly(self, tmp_path):
        out = _connector_dir(tmp_path)

        result = compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out)

        definitions = load_tool_defs(result.catalog)
        dumped = yaml.safe_load(result.catalog.read_text(encoding="utf-8"))
        assert dumped == definitions.model_dump(exclude_none=True)

    def test_form_body_keeps_its_schema_and_wire_style(self, tmp_path):
        spec = SPEC.replace(
            "          application/json:",
            "          application/x-www-form-urlencoded:\n"
            "            encoding:\n"
            "              tags: {style: deepObject}",
        ).replace(
            "                threshold: {type: number}",
            "                tags: {type: array, items: {type: string}}",
        )
        out = _connector_dir(tmp_path, spec)
        definitions = load_tool_defs(
            compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out).catalog
        )
        operation = definitions.tools[1]

        assert operation.params.wrapped is True
        assert operation.params.form_style == {"tags": "brackets"}
        assert operation.inputSchema["properties"]["body"]["required"] == ["region"]

        requests = []
        with ConnectorExecutor(
            definitions,
            transport=httpx.MockTransport(
                lambda request: requests.append(request) or httpx.Response(200, text="ok")
            ),
        ) as executor:
            executor.execute(
                operation.name,
                {"body": {"region": "us", "tags": ["a", "b"]}},
                env={},
            )

        assert requests[0].content == b"region=us&tags%5B%5D=a&tags%5B%5D=b"

    def test_compile_is_byte_stable_across_runs(self, tmp_path, monkeypatch):
        first = _connector_dir(tmp_path / "a")
        second = _connector_dir(tmp_path / "b")

        monkeypatch.chdir(first)
        one = compile_catalog("spec.yaml", name="weather", out_dir=".")
        monkeypatch.chdir(second)
        two = compile_catalog("spec.yaml", name="weather", out_dir=".")

        assert one.catalog.read_bytes() == two.catalog.read_bytes()
        assert one.tool_yaml is not None and two.tool_yaml is not None
        assert one.tool_yaml.read_bytes() == two.tool_yaml.read_bytes()

    def test_provenance_records_source_digest_and_generator(self, tmp_path):
        out = _connector_dir(tmp_path)
        spec = str(out / "spec.yaml")

        result = compile_catalog(spec, name="weather", out_dir=out)

        definitions = load_tool_defs(result.catalog)
        raw = (out / "spec.yaml").read_bytes()
        assert definitions.source.openapi == spec
        assert definitions.source.spec_sha256 == hashlib.sha256(raw).hexdigest()
        assert definitions.source.generator is not None
        assert definitions.source.generator.startswith("agent-engine-runner-shared/")

    def test_colliding_operation_names_get_deterministic_digest_suffix(self, tmp_path):
        spec = SPEC.replace(
            "  /alerts:",
            """  /collide:
    get:
      operationId: get.issue
      responses:
        "200": {description: ok}
  /collide2:
    get:
      operationId: get_issue
      responses:
        "200": {description: ok}
  /alerts:""",
        )
        out = _connector_dir(tmp_path, spec)

        first = compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out)
        second = compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out / "again")

        names = [tool.name for tool in load_tool_defs(first.catalog).tools]
        assert len(names) == len(set(names))
        collided = [name for name in names if name.startswith("weather_get_issue")]
        assert len(collided) == 2
        assert all(len(name) <= 64 for name in collided)
        assert [tool.name for tool in load_tool_defs(second.catalog).tools] == names

    def test_collision_names_are_independent_of_document_order(self, tmp_path):
        block = '  {key}:\n    get:\n      operationId: {op}\n      responses:\n        "200": {{description: ok}}\n'
        colliding_forward = block.format(key="/collide", op="get.issue") + block.format(
            key="/collide2", op="get_issue"
        )
        colliding_backward = block.format(key="/collide2", op="get_issue") + block.format(
            key="/collide", op="get.issue"
        )
        first = _connector_dir(
            tmp_path / "fwd", SPEC.replace("  /alerts:", colliding_forward + "  /alerts:")
        )
        second = _connector_dir(
            tmp_path / "bwd", SPEC.replace("  /alerts:", colliding_backward + "  /alerts:")
        )

        d1 = load_tool_defs(
            compile_catalog(str(first / "spec.yaml"), name="weather", out_dir=first).catalog
        )
        d2 = load_tool_defs(
            compile_catalog(str(second / "spec.yaml"), name="weather", out_dir=second).catalog
        )

        m1 = {tool.path: tool.name for tool in d1.tools}
        m2 = {tool.path: tool.name for tool in d2.tools}
        assert m1["/collide"] == m2["/collide"]
        assert m1["/collide2"] == m2["/collide2"]
        assert m1["/collide"] != m1["/collide2"]

    def test_missing_operation_id_gets_deterministic_fallback(self, tmp_path):
        spec = SPEC.replace("      operationId: getForecast\n", "")
        out = _connector_dir(tmp_path, spec)

        result = compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out)

        definitions = load_tool_defs(result.catalog)
        assert definitions.tools[0].name == "weather_get_forecast_city"

    def test_compile_refuses_existing_tool_yaml(self, tmp_path):
        out = _connector_dir(tmp_path)
        (out / "tool.yaml").write_text("name: weather\n", encoding="utf-8")

        with pytest.raises(GenerateError, match="refusing to overwrite"):
            compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out)
        assert (out / "tool.yaml").read_text(encoding="utf-8") == "name: weather\n"

    def test_wrapped_body_for_non_object_schema(self, tmp_path):
        spec = SPEC.replace(
            """            schema:
              type: object
              properties:
                region: {type: string}
                threshold: {type: number}
              required: [region]""",
            "            schema: {type: array, items: {type: string}}",
        )
        out = _connector_dir(tmp_path, spec)

        definitions = load_tool_defs(
            compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out).catalog
        )

        alert = definitions.tools[1]
        assert alert.params.wrapped is True
        assert alert.params.body == []
        assert alert.inputSchema["properties"]["body"] == {
            "type": "array",
            "items": {"type": "string"},
        }

    def test_required_wrapped_body_is_required(self, tmp_path):
        spec = SPEC.replace(
            "      requestBody:\n",
            "      requestBody:\n        required: true\n",
        ).replace(
            """            schema:
              type: object
              properties:
                region: {type: string}
                threshold: {type: number}
              required: [region]""",
            "            schema: {type: array, items: {type: string}}",
        )
        out = _connector_dir(tmp_path, spec)

        definitions = load_tool_defs(
            compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out).catalog
        )

        alert = definitions.tools[1]
        assert alert.params.wrapped is True
        assert alert.inputSchema["required"] == ["body"]

    def test_required_object_body_without_required_properties_becomes_wrapped(self, tmp_path):
        """A required object body with no required properties must stay
        expressible: flattening would let the agent omit every field and send
        no body at all to an API that requires one."""
        spec = SPEC.replace(
            "      requestBody:\n",
            "      requestBody:\n        required: true\n",
        ).replace("              required: [region]\n", "")
        out = _connector_dir(tmp_path, spec)

        definitions = load_tool_defs(
            compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out).catalog
        )

        alert = definitions.tools[1]
        assert alert.params.wrapped is True
        assert alert.params.body == []
        assert alert.inputSchema["required"] == ["body"]
        assert set(alert.inputSchema["properties"]) == {"body"}

    def test_required_object_body_with_required_properties_stays_flattened(self, tmp_path):
        spec = SPEC.replace(
            "      requestBody:\n",
            "      requestBody:\n        required: true\n",
        )
        out = _connector_dir(tmp_path, spec)

        definitions = load_tool_defs(
            compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out).catalog
        )

        alert = definitions.tools[1]
        assert alert.params.wrapped is False
        assert alert.params.body == ["region", "threshold"]
        assert alert.inputSchema["required"] == ["region"]

    def test_optional_body_with_nested_required_stays_omittable(self, tmp_path):
        """Execution-boundary semantics: an optional body can be omitted, but
        supplying it must still satisfy the body's own required properties.
        Unknown top-level arguments are ignored by preview contract."""
        import jsonschema

        out = _connector_dir(tmp_path)
        definitions = load_tool_defs(
            compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out).catalog
        )
        schema = jsonschema.Draft202012Validator(definitions.tools[1].inputSchema)

        assert schema.is_valid({}) is True
        assert schema.is_valid({"region": "us"}) is True
        assert schema.is_valid({"body": {"region": "us"}}) is True
        assert schema.is_valid({"body": {"threshold": 3}}) is False
        assert schema.is_valid({"body": {"region": "us", "threshold": 3}}) is True

    def test_parameter_reference_objects_are_resolved(self, tmp_path):
        spec = SPEC.replace(
            "servers:",
            """components:
  parameters:
    Days: {name: days, in: query, schema: {type: integer}}
servers:""",
        ).replace(
            """        - name: days
          in: query
          schema: {type: integer}""",
            "        - $ref: '#/components/parameters/Days'",
        )
        out = _connector_dir(tmp_path, spec)

        definitions = load_tool_defs(
            compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out).catalog
        )

        assert definitions.tools[0].params.query == ["days"]

    def test_request_body_reference_object_is_resolved(self, tmp_path):
        spec = SPEC.replace(
            "servers:",
            """components:
  requestBodies:
    AlertBody:
      content:
        application/json:
          schema:
            type: object
            properties:
              region: {type: string}
            required: [region]
servers:""",
        ).replace(
            """      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                region: {type: string}
                threshold: {type: number}
              required: [region]""",
            "      requestBody:\n        $ref: '#/components/requestBodies/AlertBody'",
        )
        out = _connector_dir(tmp_path, spec)

        definitions = load_tool_defs(
            compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out).catalog
        )

        alert = definitions.tools[1]
        assert alert.params.wrapped is True
        assert alert.inputSchema["properties"]["body"]["required"] == ["region"]

    def test_path_item_reference_operations_are_compiled(self, tmp_path):
        """An internal path-item reference must not silently drop the aliased
        operations from the catalog."""
        spec = SPEC + "  /alias:\n    $ref: '#/paths/~1alerts'\n"
        out = _connector_dir(tmp_path, spec)

        definitions = load_tool_defs(
            compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out).catalog
        )

        paths = {tool.path for tool in definitions.tools}
        assert paths == {"/forecast/{city}", "/alerts", "/alias"}
        alias = next(tool for tool in definitions.tools if tool.path == "/alias")
        assert alias.params.wrapped is True

    def test_anonymous_security_override_is_accepted(self, tmp_path):
        spec = SPEC.replace(
            "      operationId: getForecast",
            "      operationId: getForecast\n      security:\n        - {}",
        ).replace(
            "servers:",
            """security:
  - bearer: []
components:
  securitySchemes:
    bearer: {type: http, scheme: bearer}
servers:""",
        )
        out = _connector_dir(tmp_path, spec)

        definitions = load_tool_defs(
            compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out).catalog
        )

        dumped = definitions.model_dump(exclude_none=True)
        assert dumped["externalize"] == {"auth": {"type": "bearer"}}

    @pytest.mark.parametrize("scope", ["document", "operation"])
    def test_anonymous_security_alternative_does_not_require_auth(self, tmp_path, scope):
        optional_security = "security:\n  - {}\n  - bearer: []"
        spec = SPEC.replace(
            "servers:",
            """components:
  securitySchemes:
    bearer: {type: http, scheme: bearer}
servers:""",
        )
        if scope == "document":
            spec = spec.replace("servers:", f"{optional_security}\nservers:")
        else:
            spec = spec.replace(
                "      operationId: getForecast",
                "      operationId: getForecast\n"
                + "\n".join(f"      {line}" for line in optional_security.splitlines()),
            )
        out = _connector_dir(tmp_path, spec)

        definitions = load_tool_defs(
            compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out).catalog
        )

        assert definitions.externalize is None

    def test_undeclared_path_parameter_fails_closed(self, tmp_path):
        spec = SPEC.replace(
            """        - name: city
          in: path
          required: true
          schema: {type: string}
""",
            "",
        )
        out = _connector_dir(tmp_path, spec)

        with pytest.raises(GenerateError, match="no parameter declaration"):
            compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out)

    def test_invalid_connector_name_is_rejected(self, tmp_path):
        out = _connector_dir(tmp_path)

        with pytest.raises(GenerateError, match="connector name must"):
            compile_catalog(str(out / "spec.yaml"), name="../evil", out_dir=out)

    def test_url_specs_are_rejected_downloading_is_the_cli_responsibility(self, tmp_path):
        out = _connector_dir(tmp_path)

        with pytest.raises(GenerateError, match="CLI"):
            compile_catalog("https://specs.example/weather.yaml", name="weather", out_dir=out)

    def test_spec_outside_connector_folder_is_not_declared_for_regeneration(self, tmp_path):
        out = tmp_path / "connectors" / "weather"
        out.mkdir(parents=True)
        (tmp_path / "spec.yaml").write_text(SPEC, encoding="utf-8")

        result = compile_catalog(str(tmp_path / "spec.yaml"), name="weather", out_dir=out)

        assert result.tool_yaml is not None
        authoring = yaml.safe_load(result.tool_yaml.read_text(encoding="utf-8"))
        assert "spec" not in authoring["source"]


class TestCompileRejections:
    def test_non_scalar_query_parameter_names_the_operation(self, tmp_path):
        spec = SPEC.replace("schema: {type: integer}", "schema: {type: object}")
        out = _connector_dir(tmp_path, spec)

        with pytest.raises(GenerateError, match=r"getForecast.*object"):
            compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out)

    @pytest.mark.parametrize(
        ("scheme", "expected"),
        [
            ("{type: http, scheme: bearer}", {"type": "bearer"}),
            ("{type: http, scheme: basic}", {"type": "basic"}),
            ("{type: oauth2, flows: {}}", {"type": "bearer"}),
            (
                "{type: openIdConnect, "
                "openIdConnectUrl: https://auth.example/.well-known/openid-configuration}",
                {"type": "bearer"},
            ),
            (
                "{type: apiKey, in: header, name: X-Key}",
                {"type": "api_key", "location": "header", "name": "X-Key"},
            ),
            (
                "{type: apiKey, in: query, name: key}",
                {"type": "api_key", "location": "query", "name": "key"},
            ),
            (
                "{type: apiKey, in: cookie, name: session}",
                {"type": "api_key", "location": "cookie", "name": "session"},
            ),
        ],
        ids=["bearer", "basic", "oauth2", "oidc", "header-key", "query-key", "cookie-key"],
    )
    def test_source_auth_maps_to_supported_presentation(self, tmp_path, scheme, expected):
        spec = SPEC.replace(
            "servers:",
            """security:
  - auth: []
components:
  securitySchemes:
    auth: SCHEME
servers:""".replace("SCHEME", scheme),
        )
        out = _connector_dir(tmp_path, spec)

        result = compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out)
        definitions = load_tool_defs(result.catalog)
        assert definitions.externalize == {"auth": expected}

    def test_required_api_keys_compile_to_the_compound_auth_primitive(self, tmp_path):
        spec = SPEC.replace(
            "servers:",
            """security:
  - client_id: []
    client_secret: []
components:
  securitySchemes:
    client_id: {type: apiKey, in: header, name: X-Client-ID}
    client_secret: {type: apiKey, in: query, name: client_secret}
servers:""",
        ).replace(
            "        - name: days\n          in: query",
            "        - name: client_secret\n          in: query\n"
            "          schema: {type: string}\n"
            "        - name: days\n          in: query",
        )
        out = _connector_dir(tmp_path, spec)

        definitions = load_tool_defs(
            compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out).catalog
        )

        assert definitions.externalize == {
            "auth": {
                "type": "api_keys",
                "credentials": [
                    {"location": "header", "name": "X-Client-ID"},
                    {"location": "query", "name": "client_secret"},
                ],
            }
        }
        assert "client_secret" not in definitions.tools[0].inputSchema["properties"]

        materialized = definitions.model_dump(exclude_none=True)
        materialized["auth"] = {
            "type": "api_keys",
            "credentials": [
                {"env": "CLIENT_ID", "location": "header", "name": "X-Client-ID"},
                {
                    "env": "CLIENT_SECRET",
                    "location": "query",
                    "name": "client_secret",
                },
            ],
        }
        requests = []
        with ConnectorExecutor(
            ToolDefinitions.model_validate(materialized),
            transport=httpx.MockTransport(
                lambda request: requests.append(request) or httpx.Response(200, text="ok")
            ),
        ) as executor:
            executor.execute(
                "weather_getForecast",
                {"city": "Paris"},
                env={"CLIENT_ID": "id", "CLIENT_SECRET": "secret"},
            )
        assert requests[0].headers["x-client-id"] == "id"
        assert requests[0].url.params["client_secret"] == "secret"

    def test_mixed_compound_auth_is_rejected(self, tmp_path):
        spec = SPEC.replace(
            "servers:",
            """security:
  - key: []
    bearer: []
components:
  securitySchemes:
    key: {type: apiKey, in: header, name: X-Key}
    bearer: {type: http, scheme: bearer}
servers:""",
        )
        out = _connector_dir(tmp_path, spec)

        with pytest.raises(GenerateError, match="must contain only API keys"):
            compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out)

    def test_unsupported_media_type_names_the_operation(self, tmp_path):
        spec = SPEC.replace("application/json:", "text/xml:")
        out = _connector_dir(tmp_path, spec)

        with pytest.raises(GenerateError, match=r"createAlert.*text/xml"):
            compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out)

    def test_external_schema_reference_is_rejected(self, tmp_path):
        spec = SPEC.replace("schema: {type: integer}", "schema: {$ref: './other.yaml#/x'}")
        out = _connector_dir(tmp_path, spec)

        with pytest.raises(GenerateError, match="external schema references"):
            compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out)

    def test_reference_cycle_fails_at_the_depth_limit(self, tmp_path):
        spec = SPEC.replace(
            "schema: {type: integer}",
            "schema: {$ref: '#/components/schemas/loop'}",
        ).replace(
            "servers:",
            """components:
  schemas:
    loop:
      type: object
      properties:
        again: {$ref: '#/components/schemas/loop'}
servers:""",
        )
        out = _connector_dir(tmp_path, spec)

        with pytest.raises(GenerateError, match="recursive|deep|limit"):
            compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out)

    def test_authorization_header_parameter_is_rejected(self, tmp_path):
        spec = SPEC.replace(
            "        - name: days\n          in: query\n",
            "        - name: Authorization\n          in: header\n",
        ).replace("schema: {type: integer}", "schema: {type: string}")
        out = _connector_dir(tmp_path, spec)

        with pytest.raises(GenerateError, match="getForecast"):
            compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out)

    def test_error_messages_are_control_byte_safe(self, tmp_path):
        spec = SPEC.replace("application/json:", "text\x07/xml:")
        out = _connector_dir(tmp_path, spec)

        with pytest.raises(GenerateError) as exc_info:
            compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out)

        assert "\x07" not in str(exc_info.value)

    def test_control_bytes_in_paths_and_auth_schemes_are_sanitized(self, tmp_path):
        spec = SPEC.replace(
            "      operationId: getForecast",
            '      operationId: "get\\u0007Forecast"\n      security:\n        - bad: []',
        ).replace(
            "servers:",
            """components:
  securitySchemes:
    bad: {type: http, scheme: "bad\\u0007scheme"}
servers:""",
        )
        out = _connector_dir(tmp_path, spec)

        with pytest.raises(GenerateError) as exc_info:
            compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out)

        assert "\x07" not in str(exc_info.value)
        assert "operation" in str(exc_info.value)

    def test_conflicting_security_schemes_are_rejected(self, tmp_path):
        spec = SPEC.replace(
            "      operationId: getForecast",
            """      operationId: getForecast
      security:
        - key: []""",
        ).replace(
            "servers:",
            """components:
  securitySchemes:
    key: {type: apiKey, in: header, name: X-Key}
    bearer: {type: http, scheme: bearer}
security:
  - bearer: []
servers:""",
        )
        out = _connector_dir(tmp_path, spec)

        with pytest.raises(GenerateError, match="different credential"):
            compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out)

    def test_compatibility_mode_skips_operations_using_another_credential(self, tmp_path):
        spec = SPEC.replace(
            "      operationId: getForecast",
            """      operationId: getForecast
      security:
        - key: []""",
        ).replace(
            "servers:",
            """components:
  securitySchemes:
    key: {type: apiKey, in: header, name: X-Key}
    bearer: {type: http, scheme: bearer}
security:
  - bearer: []
servers:""",
        )
        out = _connector_dir(tmp_path, spec)

        result = compile_catalog(
            str(out / "spec.yaml"), name="weather", skip_unsupported=True, out_dir=out
        )
        definitions = load_tool_defs(result.catalog)

        assert [tool.name for tool in definitions.tools] == ["weather_getForecast"]
        assert definitions.externalize == {
            "auth": {"type": "api_key", "location": "header", "name": "X-Key"},
            "skipped": list(result.skipped),
        }
        assert len(result.skipped) == 1
        assert "different credential" in result.skipped[0]

    def test_catalog_auth_comes_from_a_compilable_operation(self, tmp_path):
        spec = (
            SPEC.replace(
                "      operationId: getForecast",
                """      operationId: getForecast
      security:
        - key: []""",
            )
            .replace(
                "          in: query\n          schema: {type: integer}",
                "          in: query\n          allowReserved: true\n"
                "          schema: {type: integer}",
            )
            .replace(
                "servers:",
                """components:
  securitySchemes:
    key: {type: apiKey, in: header, name: X-Key}
    bearer: {type: http, scheme: bearer}
security:
  - bearer: []
servers:""",
            )
        )
        out = _connector_dir(tmp_path, spec)

        result = compile_catalog(
            str(out / "spec.yaml"), name="weather", skip_unsupported=True, out_dir=out
        )
        definitions = load_tool_defs(result.catalog)

        assert [tool.name for tool in definitions.tools] == ["weather_createAlert"]
        assert definitions.externalize is not None
        assert definitions.externalize["auth"] == {"type": "bearer"}
        assert len(result.skipped) == 1
        assert "allowReserved" in result.skipped[0]

    def test_anonymous_operation_cannot_reuse_catalog_credential_parameter(self, tmp_path):
        spec = SPEC.replace(
            "      operationId: getForecast",
            "      operationId: getForecast\n      security: []",
        ).replace(
            "servers:",
            """components:
  securitySchemes:
    key: {type: apiKey, in: query, name: days}
security:
  - key: []
servers:""",
        )
        out = _connector_dir(tmp_path, spec)

        result = compile_catalog(
            str(out / "spec.yaml"), name="weather", skip_unsupported=True, out_dir=out
        )
        definitions = load_tool_defs(result.catalog)

        assert [tool.name for tool in definitions.tools] == ["weather_createAlert"]
        assert len(result.skipped) == 1
        assert "parameter reserved by catalog auth" in result.skipped[0]

    def test_catalog_records_unambiguous_scheme_metadata_without_credentials(self, tmp_path):
        spec = SPEC.replace(
            "servers:",
            """security:
  - bearer: []
components:
  securitySchemes:
    bearer: {type: http, scheme: bearer}
servers:""",
        )
        out = _connector_dir(tmp_path, spec)

        definitions = load_tool_defs(
            compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out).catalog
        )

        dumped = definitions.model_dump(exclude_none=True)
        assert dumped["externalize"] == {"auth": {"type": "bearer"}}
        assert "auth" not in dumped

    def test_declared_security_without_schemes_fails_closed(self, tmp_path):
        """A named requirement with no securitySchemes registry must fail with
        the operation named, never compile into anonymous tools."""
        spec = SPEC.replace(
            "servers:",
            """security:
  - bearer: []
servers:""",
        )
        out = _connector_dir(tmp_path, spec)

        with pytest.raises(GenerateError, match=r"GET /forecast/\{city\}.*bearer"):
            compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out)

    def test_unsupported_trace_operation_fails_loudly(self, tmp_path):
        spec = SPEC.replace(
            "  /alerts:",
            """  /debug:
    trace:
      operationId: tracePipe
      responses:
        "200": {description: ok}
  /alerts:""",
        )
        out = _connector_dir(tmp_path, spec)

        with pytest.raises(GenerateError, match=r"TRACE /debug"):
            compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out)

    def test_media_type_selection_is_deterministic_and_json_preferred(self, tmp_path):
        vendor_line = "          application/vnd.example+json:\n            schema: {type: object, properties: {b: {type: string}}}\n"
        # Same content map with the vendor type before (forward) and after
        # (backward) the application/json entry.
        forward = SPEC.replace(
            "          application/json:\n", vendor_line + "          application/json:\n"
        )
        backward = SPEC.replace(
            "              required: [region]\n", "              required: [region]\n" + vendor_line
        )
        first = _connector_dir(tmp_path / "fwd", forward)
        second = _connector_dir(tmp_path / "bwd", backward)

        d1 = load_tool_defs(
            compile_catalog(str(first / "spec.yaml"), name="weather", out_dir=first).catalog
        )
        d2 = load_tool_defs(
            compile_catalog(str(second / "spec.yaml"), name="weather", out_dir=second).catalog
        )

        body1 = d1.tools[1].inputSchema["properties"]["body"]
        body2 = d2.tools[1].inputSchema["properties"]["body"]
        assert body1 == body2
        assert set(body1["properties"]) == {"region", "threshold"}

    def test_malformed_reference_values_fail_with_generate_error(self, tmp_path):
        spec = SPEC.replace("schema: {type: integer}", "schema: {$ref: 7}")
        out = _connector_dir(tmp_path, spec)

        with pytest.raises(GenerateError, match="getForecast"):
            compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out)

    def test_scalar_parameter_constraints_survive_compilation(self, tmp_path):
        spec = SPEC.replace(
            "schema: {type: integer}",
            "schema: {type: integer, minimum: 1, maximum: 5, enum: [1, 3, 5]}",
        )
        out = _connector_dir(tmp_path, spec)

        definitions = load_tool_defs(
            compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out).catalog
        )

        days = definitions.tools[0].inputSchema["properties"]["days"]
        assert days["minimum"] == 1
        assert days["maximum"] == 5
        assert days["enum"] == [1, 3, 5]

    def test_openapi_boolean_exclusive_bounds_translate_to_draft_2020_12(self, tmp_path):
        spec = SPEC.replace(
            "schema: {type: integer}",
            "schema: {type: integer, minimum: 0, exclusiveMinimum: true, maximum: 10, exclusiveMaximum: false}",
        )
        out = _connector_dir(tmp_path, spec)

        definitions = load_tool_defs(
            compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out).catalog
        )

        days = definitions.tools[0].inputSchema["properties"]["days"]
        assert days["exclusiveMinimum"] == 0
        assert days["maximum"] == 10
        assert "exclusiveMaximum" not in days
        import jsonschema

        validator = jsonschema.Draft202012Validator(
            {"type": "object", "properties": {"days": days}, "required": ["days"]}
        )
        assert validator.is_valid({"days": 1}) is True
        assert validator.is_valid({"days": 0}) is False

    def test_exclusive_bound_without_paired_bound_fails(self, tmp_path):
        spec = SPEC.replace(
            "schema: {type: integer}", "schema: {type: integer, exclusiveMinimum: true}"
        )
        out = _connector_dir(tmp_path, spec)

        with pytest.raises(GenerateError, match="exclusiveMinimum"):
            compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out)

    def test_exclusive_maximum_translates_and_enforces_boundary(self, tmp_path):
        spec = SPEC.replace(
            "schema: {type: integer}",
            "schema: {type: integer, maximum: 10, exclusiveMaximum: true}",
        )
        out = _connector_dir(tmp_path, spec)

        definitions = load_tool_defs(
            compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out).catalog
        )

        days = definitions.tools[0].inputSchema["properties"]["days"]
        assert days["exclusiveMaximum"] == 10
        assert "maximum" not in days
        import jsonschema

        validator = jsonschema.Draft202012Validator(
            {"type": "object", "properties": {"days": days}, "required": ["days"]}
        )
        assert validator.is_valid({"days": 9}) is True
        assert validator.is_valid({"days": 10}) is False

    def test_vendor_json_body_preserves_its_media_type(self, tmp_path):
        spec = SPEC.replace(
            "          application/json:\n            schema:\n              type: object\n              properties:\n                region: {type: string}\n                threshold: {type: number}\n              required: [region]",
            "          application/vnd.example+json:\n            schema: {type: object, properties: {b: {type: string}}}",
        )
        out = _connector_dir(tmp_path, spec)

        result = compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out)

        definitions = load_tool_defs(result.catalog)
        assert definitions.tools[1].content_type == "application/vnd.example+json"

    def test_recursive_json_body_uses_self_contained_local_definitions(self, tmp_path):
        spec = SPEC.replace(
            "servers:",
            """components:
  schemas:
    Node:
      type: object
      properties:
        value: {type: string}
        child: {$ref: '#/components/schemas/Node'}
servers:""",
        ).replace(
            "                region: {type: string}",
            "                region: {$ref: '#/components/schemas/Node'}",
        )
        out = _connector_dir(tmp_path, spec)

        result = compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out)
        schema = load_tool_defs(result.catalog).tools[1].inputSchema

        body = schema["properties"]["body"]
        assert body["properties"]["region"]["$ref"].startswith("#/$defs/")
        assert schema["$defs"]
        import jsonschema

        jsonschema.Draft202012Validator(schema).validate(
            {"body": {"region": {"value": "root", "child": {"value": "leaf"}}}}
        )

    def test_wrong_kind_path_item_reference_fails_with_path_context(self, tmp_path):
        spec = SPEC.replace(
            "servers:",
            """components:
  schemas:
    NotAPathItem: {type: object}
servers:""",
        ).replace(
            "  /alerts:",
            "  /broken:\n    $ref: '#/components/schemas/NotAPathItem'\n  /alerts:",
        )
        out = _connector_dir(tmp_path, spec)

        with pytest.raises(GenerateError, match=r"/broken.*Path Item"):
            compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out)


class TestScaffold:
    def _compiled(self, tmp_path: Path) -> Path:
        out = _connector_dir(tmp_path)
        return compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out).catalog

    def test_scaffolds_from_local_catalog_with_allow_list(self, tmp_path):
        self._compiled(tmp_path)
        out = tmp_path / "connectors" / "consumer"

        tool_yaml = scaffold_tool_yaml(
            "../weather/tool_defs.yaml",
            base_url="https://tenant.example/api",
            allow=["weather_getForecast"],
            out_dir=out,
        )

        authoring = yaml.safe_load(tool_yaml.read_text(encoding="utf-8"))
        assert authoring == {
            "name": "weather",
            "tool_defs": "../weather/tool_defs.yaml",
            "source": {"type": "openapi", "base_url": "https://tenant.example/api"},
            "expose": {"allow": ["weather_getForecast"]},
        }

    def test_scaffold_defaults_to_allow_all(self, tmp_path):
        self._compiled(tmp_path)

        tool_yaml = scaffold_tool_yaml(
            "../weather/tool_defs.yaml",
            base_url="https://tenant.example/api",
            out_dir=tmp_path / "connectors" / "consumer",
        )

        authoring = yaml.safe_load(tool_yaml.read_text(encoding="utf-8"))
        assert authoring["expose"] == {"allow_all": True}
        assert "auth" not in authoring

    def test_scaffold_binds_auth_only_when_supplied(self, tmp_path):
        self._compiled(tmp_path)

        tool_yaml = scaffold_tool_yaml(
            "../weather/tool_defs.yaml",
            base_url="https://tenant.example/api",
            auth=Auth(type="bearer", env="WEATHER_TOKEN"),
            out_dir=tmp_path / "connectors" / "consumer",
        )

        authoring = yaml.safe_load(tool_yaml.read_text(encoding="utf-8"))
        assert authoring["auth"] == {"type": "bearer", "env": "WEATHER_TOKEN"}

    def test_scaffold_refuses_overwrite(self, tmp_path):
        self._compiled(tmp_path)
        out = tmp_path / "connectors" / "consumer"
        scaffold_tool_yaml("../weather/tool_defs.yaml", base_url="https://t.example", out_dir=out)

        with pytest.raises(GenerateError, match="refusing to overwrite"):
            scaffold_tool_yaml(
                "../weather/tool_defs.yaml", base_url="https://t.example", out_dir=out
            )

    def test_unmatched_allow_rule_lists_available_operations(self, tmp_path):
        self._compiled(tmp_path)

        with pytest.raises(GenerateError, match="weather_getForecast"):
            scaffold_tool_yaml(
                "../weather/tool_defs.yaml",
                base_url="https://t.example",
                allow=["nope"],
                out_dir=tmp_path / "connectors" / "consumer",
            )

    def test_nonexistent_catalog_reference_is_rejected(self, tmp_path):
        with pytest.raises(GenerateError, match="does not exist"):
            scaffold_tool_yaml(
                "../missing/tool_defs.yaml",
                base_url="https://t.example",
                out_dir=tmp_path / "connectors" / "consumer",
            )

    def test_absolute_catalog_reference_is_rejected(self, tmp_path):
        with pytest.raises(GenerateError, match="relative path"):
            scaffold_tool_yaml(
                "/etc/tool_defs.yaml",
                base_url="https://t.example",
                out_dir=tmp_path / "connectors" / "consumer",
            )

    def test_invalid_base_url_binding_is_rejected_through_the_runtime_model(self, tmp_path):
        self._compiled(tmp_path)

        with pytest.raises(GenerateError, match="query"):
            scaffold_tool_yaml(
                "../weather/tool_defs.yaml",
                base_url="https://t.example/?x=1",
                out_dir=tmp_path / "connectors" / "consumer",
            )

    def test_scaffold_empty_allow_list_raises_generate_error_without_writing(self, tmp_path):
        self._compiled(tmp_path)
        out = tmp_path / "connectors" / "consumer"

        with pytest.raises(GenerateError, match="non-empty"):
            scaffold_tool_yaml(
                "../weather/tool_defs.yaml",
                base_url="https://t.example",
                allow=[],
                out_dir=out,
            )

        assert not (out / "tool.yaml").exists()


class TestRegenerate:
    def test_regeneration_preserves_authoring_and_rewrites_catalog(self, tmp_path):
        out = _connector_dir(tmp_path)
        result = compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out)
        authoring = result.tool_yaml
        assert authoring is not None
        authoring.write_text(
            "name: weather\n"
            "tool_defs: tool_defs.yaml\n"
            "source:\n"
            "  type: openapi\n"
            "  base_url: https://tenant.example/api\n"
            "  spec: spec.yaml\n"
            "expose:\n"
            "  allow:\n"
            "    - weather_getForecast\n"
            "auth:\n"
            "  type: bearer\n"
            "  env: WEATHER_TOKEN\n",
            encoding="utf-8",
        )
        before = authoring.read_text(encoding="utf-8")

        regeneration = regenerate_catalog(authoring)

        assert regeneration.catalog == out / "tool_defs.yaml"
        assert authoring.read_text(encoding="utf-8") == before
        definitions = load_tool_defs(regeneration.catalog)
        assert definitions.source.base_url == "https://tenant.example/api"
        assert [tool.name for tool in definitions.tools] == [
            "weather_getForecast",
            "weather_createAlert",
        ]

    def test_regeneration_of_edited_spec_picks_up_changes(self, tmp_path):
        out = _connector_dir(tmp_path)
        compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out)
        spec_path = out / "spec.yaml"
        spec_path.write_text(
            SPEC.replace("operationId: getForecast\n", "operationId: getForecastByCity\n"),
            encoding="utf-8",
        )

        regeneration = regenerate_catalog(out / "tool.yaml")

        names = [tool.name for tool in load_tool_defs(regeneration.catalog).tools]
        assert "weather_getForecastByCity" in names

    def test_regeneration_requires_declared_spec(self, tmp_path):
        out = _connector_dir(tmp_path)
        compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out)
        authoring = out / "tool.yaml"
        document = yaml.safe_load(authoring.read_text(encoding="utf-8"))
        del document["source"]["spec"]
        authoring.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

        with pytest.raises(GenerateError, match="does not declare a local OpenAPI source"):
            regenerate_catalog(authoring)

    def test_regeneration_of_public_catalog_reference_is_refused(self, tmp_path):
        out = tmp_path / "connectors" / "consumer"
        out.mkdir(parents=True)
        (out / "tool.yaml").write_text(
            "name: jira\n"
            "tool_defs: https://connectors.example.com/jira/tool_defs.yaml\n"
            "source:\n"
            "  type: openapi\n"
            "  base_url: https://jira.corp/rest\n",
            encoding="utf-8",
        )

        with pytest.raises(GenerateError, match="need no regeneration"):
            regenerate_catalog(out / "tool.yaml")

    def test_regeneration_accepts_spec_outside_the_connector_folder(self, tmp_path):
        (tmp_path / "spec.yaml").write_text(SPEC, encoding="utf-8")
        out = tmp_path / "connectors" / "weather"
        out.mkdir(parents=True)
        (out / "tool.yaml").write_text(
            "name: weather\n"
            "tool_defs: tool_defs.yaml\n"
            "source:\n"
            "  type: openapi\n"
            "  base_url: https://weather.example/api\n"
            "  spec: ../../spec.yaml\n",
            encoding="utf-8",
        )

        regeneration = regenerate_catalog(out / "tool.yaml")

        definitions = load_tool_defs(regeneration.catalog)
        assert definitions.source.openapi == "../../spec.yaml"
        assert [tool.name for tool in definitions.tools] == [
            "weather_getForecast",
            "weather_createAlert",
        ]

    def test_regeneration_refuses_to_write_through_a_symlink(self, tmp_path):
        out = _connector_dir(tmp_path)
        compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out)
        outside = tmp_path / "outside"
        outside.mkdir()
        (out / "tool_defs.yaml").unlink()
        (out / "tool_defs.yaml").symlink_to(outside / "tool_defs.yaml")

        with pytest.raises(GenerateError, match="outside the connector folder"):
            regenerate_catalog(out / "tool.yaml")

    def test_regeneration_traversal_write_is_contained(self, tmp_path):
        out = _connector_dir(tmp_path)
        compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out)
        sentinel = tmp_path / "README.md"
        sentinel.write_text("do not touch", encoding="utf-8")
        authoring = out / "tool.yaml"
        document = yaml.safe_load(authoring.read_text(encoding="utf-8"))
        document["tool_defs"] = "../../README.md"
        authoring.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

        with pytest.raises(GenerateError, match="outside the connector folder"):
            regenerate_catalog(authoring)

        assert sentinel.read_text(encoding="utf-8") == "do not touch"

    def test_regeneration_ancestor_symlink_escape_is_contained(self, tmp_path):
        out = _connector_dir(tmp_path)
        compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out)
        outside = tmp_path / "outside"
        outside.mkdir()
        (out / "sub").mkdir()
        (out / "sub" / "link").symlink_to(outside, target_is_directory=True)
        authoring = out / "tool.yaml"
        document = yaml.safe_load(authoring.read_text(encoding="utf-8"))
        document["tool_defs"] = "sub/link/tool_defs.yaml"
        authoring.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

        with pytest.raises(GenerateError, match="outside the connector folder"):
            regenerate_catalog(authoring)

        assert not (outside / "tool_defs.yaml").exists()


class TestSharedValidation:
    def test_runtime_models_reject_malformed_generated_schemas(self):
        """The generator validates through the same models the runtime parses:
        a catalog whose tool names collide must fail identically at both ends."""
        from agent_engine_runner_shared.connectors.definitions import ToolDefinitions

        document = {
            "source": {"name": "x", "base_url": "https://x.example"},
            "tools": [
                {
                    "name": "x_op",
                    "method": "GET",
                    "path": "/a",
                    "params": {},
                    "inputSchema": {"type": "object", "properties": {}},
                },
                {
                    "name": "x_op",
                    "method": "GET",
                    "path": "/b",
                    "params": {},
                    "inputSchema": {"type": "object", "properties": {}},
                },
            ],
        }
        with pytest.raises(ValidationError, match="duplicate tool names"):
            ToolDefinitions.model_validate(document)


class TestRoundTwoFeedback:
    """Findings from the 2026-09-14 human + augment review passes."""

    def test_nested_servers_do_not_override_the_catalog_base_url(self, tmp_path):
        spec = SPEC.replace(
            "      operationId: createAlert",
            "      operationId: createAlert\n      servers:\n        - url: https://other.example/api",
        ).replace(
            "  /forecast/{city}:",
            "  /forecast/{city}:\n    servers:\n      - url: https://other.example/api",
        )
        out = _connector_dir(tmp_path, spec)

        result = compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out)

        definitions = load_tool_defs(result.catalog)
        assert definitions.source.base_url == "https://weather.example/api"
        assert len(definitions.tools) == 2

    def test_explicit_base_url_overrides_operation_servers(self, tmp_path):
        spec = SPEC.replace(
            "      operationId: createAlert",
            "      operationId: createAlert\n      servers:\n        - url: https://other.example/api",
        )
        out = _connector_dir(tmp_path, spec)

        result = compile_catalog(
            str(out / "spec.yaml"),
            name="weather",
            base_url="https://tenant.example/api",
            out_dir=out,
        )

        definitions = load_tool_defs(result.catalog)
        assert definitions.source.base_url == "https://tenant.example/api"
        assert len(definitions.tools) == 2

    def test_nullable_scalar_parameter_preserves_nullability(self, tmp_path):
        spec = SPEC.replace("schema: {type: integer}", "schema: {type: integer, nullable: true}")
        out = _connector_dir(tmp_path, spec)

        result = compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out)
        definitions = load_tool_defs(result.catalog)
        assert definitions.tools[0].inputSchema["properties"]["days"]["type"] == [
            "integer",
            "null",
        ]

    def test_nullable_body_schema_preserves_nullability(self, tmp_path):
        spec = SPEC.replace(
            "                region: {type: string}",
            "                region: {type: string, nullable: true}",
        )
        out = _connector_dir(tmp_path, spec)

        result = compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out)
        definitions = load_tool_defs(result.catalog)
        body = definitions.tools[1].inputSchema["properties"]["body"]
        assert body["properties"]["region"]["type"] == ["string", "null"]

    @pytest.mark.parametrize(
        ("location", "name"),
        [("path", "city"), ("query", "days"), ("header", "days"), ("cookie", "days")],
    )
    def test_required_nullable_parameters_are_not_compiled(self, tmp_path, location, name):
        if location == "path":
            spec = SPEC.replace(
                "schema: {type: string}",
                "schema: {type: string, nullable: true}",
                1,
            )
        else:
            spec = SPEC.replace(
                "in: query\n          schema: {type: integer}",
                f"in: {location}\n          required: true\n"
                "          schema: {type: integer, nullable: true}",
            )
        out = _connector_dir(tmp_path, spec)

        with pytest.raises(GenerateError, match=rf"required nullable.*{name}"):
            compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out)

    @pytest.mark.parametrize(
        "media_type",
        ["application/x-www-form-urlencoded", "multipart/form-data"],
    )
    def test_required_nullable_form_parts_are_not_compiled(self, tmp_path, media_type):
        spec = SPEC.replace("application/json", media_type).replace(
            "region: {type: string}",
            "region: {type: string, nullable: true}",
        )
        out = _connector_dir(tmp_path, spec)

        with pytest.raises(GenerateError, match=r"required nullable.*region"):
            compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out)

    def test_allow_reserved_query_is_skipped_with_provenance(self, tmp_path):
        spec = SPEC.replace(
            "in: query\n          schema: {type: integer}",
            "in: query\n          allowReserved: true\n          schema: {type: integer}",
        )
        out = _connector_dir(tmp_path, spec)

        result = compile_catalog(
            str(out / "spec.yaml"),
            name="weather",
            skip_unsupported=True,
            out_dir=out,
        )

        definitions = load_tool_defs(result.catalog)
        assert [tool.name for tool in definitions.tools] == ["weather_createAlert"]
        assert len(result.skipped) == 1
        assert "allowReserved" in result.skipped[0]

    def test_allow_reserved_form_field_is_not_compiled(self, tmp_path):
        spec = SPEC.replace(
            "          application/json:",
            "          application/x-www-form-urlencoded:\n"
            "            encoding:\n"
            "              region: {allowReserved: true}",
        )
        out = _connector_dir(tmp_path, spec)

        with pytest.raises(GenerateError, match="allowReserved"):
            compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out)

    @pytest.mark.parametrize(
        ("reference", "name"),
        [("a%7E1b", "a/b"), ("a%7E0b", "a~b")],
    )
    def test_uri_decoding_precedes_json_pointer_unescaping(self, tmp_path, reference, name):
        spec = (
            SPEC.replace(
                "schema: {type: integer}",
                f"schema: {{$ref: '#/components/schemas/{reference}'}}",
            )
            + f'\ncomponents:\n  schemas:\n    "{name}": {{type: integer}}\n'
        )
        out = _connector_dir(tmp_path, spec)

        result = compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out)

        definitions = load_tool_defs(result.catalog)
        assert definitions.tools[0].inputSchema["properties"]["days"] == {"type": "integer"}

    def test_uri_decoding_precedes_json_pointer_tokenization(self, tmp_path):
        spec = (
            SPEC.replace(
                "schema: {type: integer}",
                "schema: {$ref: '#/components%2Fschemas%2FThing'}",
            )
            + "\ncomponents:\n  schemas:\n    Thing: {type: integer}\n"
        )
        out = _connector_dir(tmp_path, spec)

        definitions = load_tool_defs(
            compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out).catalog
        )

        assert definitions.tools[0].inputSchema["properties"]["days"] == {"type": "integer"}

    @pytest.mark.parametrize("required", [False, True], ids=["optional", "required"])
    @pytest.mark.parametrize(
        "media_type",
        ["application/x-www-form-urlencoded", "multipart/form-data", "text/plain"],
    )
    def test_nullable_non_json_body_is_not_compiled(self, tmp_path, media_type, required):
        if media_type == "text/plain":
            body_schema = "schema: {type: string, nullable: true}"
        else:
            body_schema = (
                "schema: {type: object, nullable: true, properties: {value: {type: string}}}"
            )
        spec = SPEC.replace("application/json", media_type).replace(
            "            schema:\n"
            "              type: object\n"
            "              properties:\n"
            "                region: {type: string}\n"
            "                threshold: {type: number}\n"
            "              required: [region]",
            f"            {body_schema}",
        )
        if required:
            spec = spec.replace(
                "      requestBody:\n", "      requestBody:\n        required: true\n"
            )
        out = _connector_dir(tmp_path, spec)

        with pytest.raises(GenerateError, match="nullable.*body"):
            compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out)

    def test_optional_nullable_non_json_body_is_skipped_with_provenance(self, tmp_path):
        spec = SPEC.replace("application/json", "text/plain").replace(
            "            schema:\n"
            "              type: object\n"
            "              properties:\n"
            "                region: {type: string}\n"
            "                threshold: {type: number}\n"
            "              required: [region]",
            "            schema: {type: string, nullable: true}",
        )
        out = _connector_dir(tmp_path, spec)

        result = compile_catalog(
            str(out / "spec.yaml"), name="weather", skip_unsupported=True, out_dir=out
        )
        definitions = load_tool_defs(result.catalog)

        assert [tool.name for tool in definitions.tools] == ["weather_getForecast"]
        assert result.skipped == (
            "operation createAlert (POST /alerts): nullable text body is not supported",
        )

    def test_required_nullable_json_body_is_compiled(self, tmp_path):
        spec = SPEC.replace(
            "      requestBody:\n", "      requestBody:\n        required: true\n"
        ).replace(
            "              type: object", "              type: object\n              nullable: true"
        )
        out = _connector_dir(tmp_path, spec)

        definitions = load_tool_defs(
            compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out).catalog
        )

        body = definitions.tools[1].inputSchema["properties"]["body"]
        assert body["type"] == ["object", "null"]
        assert "body" in definitions.tools[1].inputSchema["required"]

    def test_non_default_parameter_style_is_rejected(self, tmp_path):
        spec = SPEC.replace(
            "        - name: city\n          in: path\n          required: true\n          schema: {type: string}",
            "        - name: city\n          in: path\n          required: true\n          style: label\n          schema: {type: string}",
        )
        out = _connector_dir(tmp_path, spec)

        with pytest.raises(GenerateError, match=r"style label"):
            compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out)

    def test_default_parameter_styles_are_accepted(self, tmp_path):
        spec = SPEC.replace(
            "        - name: city\n          in: path\n          required: true\n          schema: {type: string}",
            "        - name: city\n          in: path\n          required: true\n          style: simple\n          schema: {type: string}",
        ).replace(
            "        - name: days\n          in: query\n          schema: {type: integer}",
            "        - name: days\n          in: query\n          style: form\n          schema: {type: integer}",
        )
        out = _connector_dir(tmp_path, spec)

        definitions = load_tool_defs(
            compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out).catalog
        )

        assert definitions.tools[0].params.query == ["days"]

    def test_object_level_body_constraints_disable_flattening(self, tmp_path):
        spec = SPEC.replace(
            "              required: [region]",
            "              required: [region]\n              maxProperties: 2",
        )
        out = _connector_dir(tmp_path, spec)

        definitions = load_tool_defs(
            compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out).catalog
        )

        alert = definitions.tools[1]
        assert alert.params.wrapped is True
        nested = alert.inputSchema["properties"]["body"]
        assert nested["maxProperties"] == 2
        assert nested["required"] == ["region"]

    def test_security_scheme_reference_objects_resolve(self, tmp_path):
        spec = SPEC.replace(
            "servers:",
            """security:
  - bearer: []
components:
  securitySchemes:
    bearer: {$ref: '#/components/securitySchemes/Bearer'}
    Bearer: {type: http, scheme: bearer}
servers:""",
        )
        out = _connector_dir(tmp_path, spec)

        definitions = load_tool_defs(
            compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out).catalog
        )

        dumped = definitions.model_dump(exclude_none=True)
        assert dumped["externalize"] == {"auth": {"type": "bearer"}}

    def _compiled(self, tmp_path: Path) -> Path:
        out = _connector_dir(tmp_path)
        return compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out).catalog

    def test_scaffold_rejects_catalogs_with_invalid_connector_names(self, tmp_path):
        self._compiled(tmp_path)
        catalog = tmp_path / "connectors" / "weather" / "tool_defs.yaml"
        text = catalog.read_text(encoding="utf-8")
        # Only the source identity is corrupted; tool names stay valid so the
        # failure surfaces from the binding validation, not the loader.
        catalog.write_text(text.replace("  name: weather", "  name: My API!"), encoding="utf-8")

        with pytest.raises(GenerateError, match="connector name must"):
            scaffold_tool_yaml(
                "../weather/tool_defs.yaml",
                base_url="https://t.example",
                out_dir=tmp_path / "connectors" / "consumer",
            )

    def test_regenerate_diagnostics_are_control_byte_safe(self, tmp_path):
        out = _connector_dir(tmp_path)
        compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out)
        broken = tmp_path / "to\x07ol.yaml"
        broken.write_text("name: [unclosed\n", encoding="utf-8")

        with pytest.raises(GenerateError) as exc_info:
            regenerate_catalog(broken)

        assert "\x07" not in str(exc_info.value)


class TestBatchThreeFeedback:
    """Findings from the 2026-09-14 third review pass."""

    def test_duplicate_yaml_keys_are_rejected(self, tmp_path):
        out = _connector_dir(tmp_path)
        duplicate = SPEC + "\nservers:\n  - url: https://other.example/api\n"
        (out / "spec.yaml").write_text(duplicate, encoding="utf-8")

        with pytest.raises(GenerateError, match="unique|string"):
            compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out)

    def test_duplicate_json_keys_rejected_via_object_hook(self, tmp_path):
        duplicate = (
            '{"openapi": "3.0.1", "info": {"title": "t", "version": "1"}, "paths": {}, "paths": {}}'
        )
        out = _connector_dir(tmp_path)
        (out / "spec.json").write_text(duplicate, encoding="utf-8")

        with pytest.raises(GenerateError, match="duplicate key"):
            compile_catalog(str(out / "spec.json"), name="weather", out_dir=out)

    def test_duplicate_authoring_keys_are_rejected(self, tmp_path):
        out = _connector_dir(tmp_path)
        compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out)
        authoring = out / "tool.yaml"
        # A duplicated tool_defs key must not hide which value drives the
        # destructive regeneration write.
        authoring.write_text(
            authoring.read_text(encoding="utf-8") + "tool_defs: other.tool_defs.yaml\n",
            encoding="utf-8",
        )

        with pytest.raises(GenerateError, match="unique|string"):
            regenerate_catalog(authoring)

    def test_non_mapping_security_schemes_fail_with_generate_error(self, tmp_path):
        spec = SPEC.replace(
            "servers:",
            "components:\n  securitySchemes: []\nservers:",
        )
        out = _connector_dir(tmp_path, spec)

        with pytest.raises(GenerateError, match="securitySchemes is malformed"):
            compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out)


class TestBatchFourFeedback:
    """Findings from the 2026-09-14 fourth review pass."""

    def test_malformed_media_entry_is_operation_scoped(self, tmp_path):
        spec = SPEC.replace(
            "          application/json:\n            schema:\n              type: object\n              properties:\n                region: {type: string}\n                threshold: {type: number}\n              required: [region]",
            "          application/json: null",
        )
        out = _connector_dir(tmp_path, spec)

        with pytest.raises(GenerateError, match=r"createAlert.*malformed"):
            compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out)

    def test_non_mapping_properties_fail_scoped(self, tmp_path):
        """A non-mapping properties value is invalid JSON Schema: the failure
        must stay operation-scoped, not leak a raw exception."""
        spec = SPEC.replace(
            "              properties:\n                region: {type: string}\n                threshold: {type: number}\n              required: [region]",
            "              properties: [1]",
        )
        out = _connector_dir(tmp_path, spec)

        with pytest.raises(GenerateError, match="createAlert"):
            compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out)

    def test_c1_and_bidi_controls_are_safe(self, tmp_path):
        from agent_engine_runner_shared.connectors.generate.spec import _quote

        # Single-character CSI (C1) and right-to-left override (bidi) must
        # not survive the sanitizer, matching the CLI's termsafe rule.
        assert _quote("a\u009b[2Jb") == "a?[2Jb"
        assert _quote("bad\u202efake") == "bad?fake"
        assert _quote("keep spaces and ünïcode") == "keep spaces and ünïcode"

        spec = SPEC.replace("operationId: getForecast", "operationId: get\u009bForecast")
        out = _connector_dir(tmp_path, spec)

        with pytest.raises(GenerateError, match="not valid JSON or YAML") as exc_info:
            compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out)

        assert "\u009b" not in str(exc_info.value)


class TestAuthoringGuidance:
    """Generated tool.yaml names the agent.yaml policy the connector needs."""

    def test_compile_states_egress_and_secret(self, tmp_path, monkeypatch):
        out = _connector_dir(tmp_path)
        monkeypatch.chdir(out)

        result = compile_catalog(
            "spec.yaml",
            name="weather",
            auth=Auth(type="bearer", env="WEATHER_TOKEN"),
            out_dir=".",
        )

        assert result.tool_yaml is not None
        text = result.tool_yaml.read_text(encoding="utf-8")
        assert "#   egress   allow weather.example:443 from the tool sandbox" in text
        assert (
            "#            agentengine agent egress add --component tool weather.example:443" in text
        )
        assert "#   secret   grant WEATHER_TOKEN to the tool sandbox under" in text
        assert "#            sandboxes.tool.secrets (sandboxes requires an agent block)" in text
        # Guidance is a comment only: the parsed authoring file is unchanged.
        assert yaml.safe_load(text) == {
            "name": "weather",
            "tool_defs": "tool_defs.yaml",
            "source": {
                "type": "openapi",
                "base_url": "https://weather.example/api",
                "spec": "spec.yaml",
            },
            "expose": {"allow_all": True},
            "auth": {"type": "bearer", "env": "WEATHER_TOKEN"},
        }

    def test_guidance_reports_the_bound_port(self, tmp_path, monkeypatch):
        out = _connector_dir(tmp_path)
        monkeypatch.chdir(out)

        result = compile_catalog(
            "spec.yaml",
            name="weather",
            base_url="https://weather.example:8443/api",
            out_dir=".",
        )

        assert result.tool_yaml is not None
        assert "allow weather.example:8443 from the tool sandbox" in result.tool_yaml.read_text(
            encoding="utf-8"
        )

    def test_anonymous_connector_states_no_secret_is_required(self, tmp_path, monkeypatch):
        out = _connector_dir(tmp_path)
        monkeypatch.chdir(out)

        result = compile_catalog("spec.yaml", name="weather", out_dir=".")

        assert result.tool_yaml is not None
        text = result.tool_yaml.read_text(encoding="utf-8")
        assert "#   secret   none required; this connector authenticates anonymously" in text
        assert "sandboxes.tool.secrets" not in text

    def test_scaffold_states_every_credential_secret(self, tmp_path):
        out = _connector_dir(tmp_path)
        compile_catalog(str(out / "spec.yaml"), name="weather", out_dir=out)

        tool_yaml = scaffold_tool_yaml(
            "../weather/tool_defs.yaml",
            base_url="https://tenant.example/api",
            auth=Auth(
                type="api_keys",
                credentials=[
                    Credential(env="FIRST_KEY", location="header", name="X-First"),
                    Credential(env="SECOND_KEY", location="query", name="second"),
                ],
            ),
            out_dir=tmp_path / "connectors" / "consumer",
        )

        text = tool_yaml.read_text(encoding="utf-8")
        assert "allow tenant.example:443 from the tool sandbox" in text
        assert "grant FIRST_KEY, SECOND_KEY to the tool sandbox under" in text

    def test_basic_auth_states_both_credential_secrets(self, tmp_path, monkeypatch):
        out = _connector_dir(tmp_path)
        monkeypatch.chdir(out)

        result = compile_catalog(
            "spec.yaml",
            name="weather",
            auth=Auth(type="basic", username_env="API_USER", password_env="API_PASS"),
            out_dir=".",
        )

        assert result.tool_yaml is not None
        text = result.tool_yaml.read_text(encoding="utf-8")
        assert "#   secret   grant API_USER, API_PASS to the tool sandbox under" in text

    def test_metacharacter_destination_is_shell_quoted(self, tmp_path, monkeypatch):
        out = _connector_dir(tmp_path)
        monkeypatch.chdir(out)

        result = compile_catalog(
            "spec.yaml",
            name="weather",
            base_url="https://a;b$(id).example/api",
            out_dir=".",
        )

        assert result.tool_yaml is not None
        text = result.tool_yaml.read_text(encoding="utf-8")
        command = next(
            line.removeprefix("#            ")
            for line in text.splitlines()
            if line.startswith("#            agentengine agent egress add")
        )
        # The paste-ready command carries the destination as one quoted token,
        # so a shell cannot interpret its metacharacters as syntax.
        assert shlex.split(command)[-1] == "a;b$(id).example:443"
        assert "'a;b$(id).example:443'" in command
        assert "--component tool a;b$(id).example:443" not in text

    def test_comment_values_cannot_escape_the_comment(self):
        from agent_engine_runner_shared.connectors.generate.authoring import _guidance_comment

        comment = _guidance_comment("https://ho\u009bst\u202e.example/api", None)

        assert "\u009b" not in comment and "\u202e" not in comment
        assert "allow ho?st?.example:443" in comment
