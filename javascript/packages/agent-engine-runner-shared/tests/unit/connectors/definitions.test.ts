import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import {
  AuthSchema,
  CredentialSchema,
  SourceSchema,
  ToolDefinitionsSchema,
  authHeaderNames,
  loadToolDefs,
  toolSchemas,
  validateSegment,
  validateUrl,
} from "../../../src/connectors/index.js";
import {
  loadAuthoringYaml,
  loadConnectorYaml,
} from "../../../src/connectors/definitions.js";

const FIXTURE = fileURLToPath(
  new URL("./fixtures/jira.tool_defs.yaml", import.meta.url),
);

function fixtureDocument(): Record<string, unknown> {
  const document = loadToolDefs(FIXTURE);
  return structuredClone(document) as unknown as Record<string, unknown>;
}

function messages(result: {
  success: boolean;
  error?: { issues: Array<{ path: Array<string | number>; message: string }> };
}): string {
  return result.success
    ? ""
    : (result.error?.issues ?? [])
        .map((issue) => `${issue.path.join(".")}: ${issue.message}`)
        .join("; ");
}

describe("connector definitions", () => {
  it("loads the shared catalog and exposes detached schemas", () => {
    const definitions = loadToolDefs(FIXTURE);
    expect(definitions.source.name).toBe("jiradc");
    expect(definitions.source.base_url).toBe("https://jira.example/rest");
    expect(definitions.tools.map((tool) => tool.name)).toEqual([
      "jiradc_getComments",
      "jiradc_addComment",
    ]);

    const schemas = toolSchemas(definitions);
    expect(schemas[0]).toEqual({
      name: "jiradc_getComments",
      description: "Returns comments for an issue.",
      inputSchema: {
        type: "object",
        properties: { issueIdOrKey: { type: "string" } },
        required: ["issueIdOrKey"],
      },
    });
    const schema = schemas[0];
    if (schema === undefined) throw new Error("missing schema fixture");
    schema.inputSchema["name"] = "mutated";
    expect(definitions.tools[0]?.inputSchema).not.toHaveProperty("name");
  });

  it("derives auth headers", () => {
    const definitions = loadToolDefs(FIXTURE);
    expect(authHeaderNames(definitions.auth)).toEqual(
      new Set(["authorization"]),
    );

    const delegated = ToolDefinitionsSchema.parse({
      ...fixtureDocument(),
      auth: { provider: "custom-github", scopes: ["repo"] },
    });
    expect(authHeaderNames(delegated.auth)).toEqual(new Set(["authorization"]));
  });

  it("rejects auth headers used as model arguments", () => {
    const document = fixtureDocument();
    const first = (
      document["tools"] as Array<Record<string, unknown>>
    )[0] as Record<string, unknown>;
    first["params"] = { header: ["Authorization"] };
    first["inputSchema"] = {
      type: "object",
      properties: { Authorization: { type: "string" } },
      required: ["Authorization"],
    };
    const result = ToolDefinitionsSchema.safeParse(document);
    expect(result.success).toBe(false);
    expect(messages(result)).toContain(
      "header parameters cannot override auth or transport headers",
    );
  });

  it("rejects external schema references", () => {
    const document = fixtureDocument();
    const first = (
      document["tools"] as Array<Record<string, unknown>>
    )[0] as Record<string, unknown>;
    first["inputSchema"] = {
      type: "object",
      properties: {
        issueIdOrKey: { $ref: "https://example.test/schema.json" },
      },
      required: ["issueIdOrKey"],
    };
    const result = ToolDefinitionsSchema.safeParse(document);
    expect(result.success).toBe(false);
    expect(messages(result)).toContain(
      "compiled schemas may reference only local definitions",
    );
  });

  it("rejects duplicate tool names", () => {
    const document = fixtureDocument();
    const tools = document["tools"] as Array<Record<string, unknown>>;
    tools.push(structuredClone(tools[0] as Record<string, unknown>));
    const result = ToolDefinitionsSchema.safeParse(document);
    expect(result.success).toBe(false);
    expect(messages(result)).toContain("duplicate tool names");
  });

  it("rejects a non-loopback HTTP base URL and accepts loopback", () => {
    expect(() => validateUrl("http://jira.example/rest")).toThrow("HTTPS");
    expect(validateUrl("http://127.0.0.1:8080/rest")).toBe(
      "http://127.0.0.1:8080/rest",
    );
    expect(validateUrl("http://localhost:8080/rest")).toBe(
      "http://localhost:8080/rest",
    );
  });

  it("rejects base URL placeholders outside the path", () => {
    const base = { name: "example" };
    expect(
      SourceSchema.safeParse({
        ...base,
        base_url: "https://{token}.example/{token}",
      }).success,
    ).toBe(false);
    expect(
      SourceSchema.safeParse({
        ...base,
        base_url: "https://api.example/{token}/rest",
      }).success,
    ).toBe(true);
  });

  it("rejects unsafe path segments", () => {
    expect(() => validateSegment("..")).toThrow("non-traversing");
    expect(() => validateSegment("%2e%2e")).toThrow("non-traversing");
    expect(() => validateSegment("a%2fb")).toThrow("non-traversing");
    expect(validateSegment("DEMO-1")).toBe("DEMO-1");
  });

  it("validates credential bindings independently", () => {
    expect(
      CredentialSchema.safeParse({
        env: "JIRA_TOKEN",
        location: "header",
        name: "X-API-Key",
      }).success,
    ).toBe(true);
    expect(
      CredentialSchema.safeParse({
        env: "OPENAI_API_KEY",
        location: "header",
        name: "X-API-Key",
      }).success,
    ).toBe(false);
  });

  it("normalizes static auth bindings", () => {
    const apiKey = AuthSchema.parse({
      api_keys: [{ env: "API_KEY", location: "query", name: "api_key" }],
    });
    expect(authHeaderNames(apiKey)).toEqual(new Set());
    expect(apiKey.api_keys).toEqual([
      { env: "API_KEY", location: "query", name: "api_key", prefix: "" },
    ]);

    const basic = AuthSchema.parse({
      basic: { username: "JIRA_USER", password: "JIRA_PASSWORD" },
    });
    expect(authHeaderNames(basic)).toEqual(new Set(["authorization"]));

    const multi = AuthSchema.parse({
      api_keys: [
        {
          env: "FIRST_KEY",
          location: "header",
          name: "X-First-Key",
          prefix: "",
        },
        {
          env: "SECOND_KEY",
          location: "header",
          name: "X-Second-Key",
          prefix: "",
        },
      ],
    });
    expect(authHeaderNames(multi)).toEqual(
      new Set(["x-first-key", "x-second-key"]),
    );
  });

  it("rejects a form body that is not wrapped", () => {
    const document = fixtureDocument();
    const first = (
      document["tools"] as Array<Record<string, unknown>>
    )[0] as Record<string, unknown>;
    first["body_encoding"] = "form";
    const result = ToolDefinitionsSchema.safeParse(document);
    expect(result.success).toBe(false);
    expect(messages(result)).toContain("form bodies must be wrapped objects");
  });

  it("rejects unknown query serialization styles", () => {
    const document = fixtureDocument();
    const first = (
      document["tools"] as Array<Record<string, unknown>>
    )[0] as Record<string, unknown>;
    first["params"] = {
      path: ["issueIdOrKey"],
      query: ["expand"],
      query_style: { expand: "explode" },
    };
    first["inputSchema"] = {
      type: "object",
      properties: {
        issueIdOrKey: { type: "string" },
        expand: { type: "string" },
      },
      required: ["issueIdOrKey"],
    };
    const result = ToolDefinitionsSchema.safeParse(document);
    expect(result.success).toBe(false);
    expect(messages(result)).toContain("unknown query serialization style");
  });

  it("accepts local $defs references inside a wrapped body", () => {
    const document = fixtureDocument();
    const first = (
      document["tools"] as Array<Record<string, unknown>>
    )[0] as Record<string, unknown>;
    first["path"] = "/api/2/issue";
    first["params"] = { wrapped: true };
    first["inputSchema"] = {
      type: "object",
      properties: { body: { $ref: "#/$defs/Body" } },
      required: ["body"],
      $defs: {
        Body: { type: "object", properties: { text: { type: "string" } } },
      },
    };
    const result = ToolDefinitionsSchema.safeParse(document);
    expect(result.success).toBe(true);
  });

  it("requires the compiled operation params field", () => {
    const document = fixtureDocument();
    const first = (
      document["tools"] as Array<Record<string, unknown>>
    )[0] as Record<string, unknown>;
    delete first["params"];
    const missing = ToolDefinitionsSchema.safeParse(document);
    expect(missing.success).toBe(false);
    expect(messages(missing)).toContain("params");

    first["path"] = "/api/2/issue";
    first["params"] = {};
    first["inputSchema"] = { type: "object", properties: {} };
    expect(ToolDefinitionsSchema.safeParse(document).success).toBe(true);
  });

  it("keeps date-like YAML values as strings", () => {
    const temporary = mkdtempSync(join(tmpdir(), "connector-dates-"));
    try {
      const path = join(temporary, "dated.tool_defs.yaml");
      writeFileSync(
        path,
        [
          "source:",
          "  name: dated",
          "  base_url: https://jira.example/rest",
          "tools:",
          "  - name: dated_tool",
          "    description: Dated tool",
          "    method: GET",
          "    path: /api/2/issue",
          "    params:",
          "      query: [date]",
          "    inputSchema:",
          "      type: object",
          "      properties:",
          "        date:",
          "          type: string",
          "          default: 2024-01-01",
        ].join("\n"),
        "utf8",
      );
      const definitions = loadToolDefs(path);
      const dateSchema = definitions.tools[0]?.inputSchema[
        "properties"
      ] as Record<string, unknown>;
      const date = (dateSchema["date"] as Record<string, unknown>)["default"];
      expect(typeof date).toBe("string");
      expect(date).toBe("2024-01-01");
    } finally {
      rmSync(temporary, { recursive: true, force: true });
    }
  });

  it("matches PyYAML catalog scalar resolution", () => {
    const parsed = loadConnectorYaml(
      [
        "plain:",
        "  exponent: 1e3",
        "  signed_exponent: 1e+3",
        "  decimal_exponent: 1.0e+3",
        "  leading_zero: 08",
        "  octal: 010",
        "  underscored: 1_000",
        "  date: 2024-01-01",
        "  strict_true: true",
        "  yes_is_string: yes",
        "  nullish: ~",
        "  equals: =",
      ].join("\n"),
    ) as Record<string, unknown>;
    expect(parsed).toEqual({
      plain: {
        exponent: "1e3",
        signed_exponent: "1e+3",
        decimal_exponent: 1000,
        leading_zero: "08",
        octal: 8,
        underscored: 1000,
        date: "2024-01-01",
        strict_true: true,
        yes_is_string: "yes",
        nullish: null,
        equals: "=",
      },
    });
  });

  it("matches yaml.safe_load for authoring merge keys and booleans", () => {
    const parsed = loadAuthoringYaml(
      [
        "shared_auth: &shared_auth",
        "  bearer: JIRA_TOKEN",
        "auth:",
        "  <<: *shared_auth",
        "allow_all: yes",
      ].join("\n"),
    ) as Record<string, unknown>;
    expect(parsed).toEqual({
      shared_auth: { bearer: "JIRA_TOKEN" },
      auth: { bearer: "JIRA_TOKEN" },
      allow_all: true,
    });
  });

  it("rejects non-JSON schema objects", () => {
    const document = fixtureDocument();
    const first = (
      document["tools"] as Array<Record<string, unknown>>
    )[0] as Record<string, unknown>;
    const properties = (first["inputSchema"] as Record<string, unknown>)[
      "properties"
    ] as Record<string, unknown>;
    (properties["issueIdOrKey"] as Record<string, unknown>)["default"] =
      new Date();
    const result = ToolDefinitionsSchema.safeParse(document);
    expect(result.success).toBe(false);
    expect(messages(result)).toContain("JSON values");
  });

  it("preserves separator checks through malformed percent escapes", () => {
    expect(() => validateSegment("safe%2F..%ZZ")).toThrow("non-traversing");
    expect(() => validateSegment("safe%5C..")).toThrow("non-traversing");
    expect(validateSegment("safe%ZZ")).toBe("safe%ZZ");
  });

  it("accepts generated identity-escape patterns", () => {
    const document = fixtureDocument();
    const first = (
      document["tools"] as Array<Record<string, unknown>>
    )[0] as Record<string, unknown>;
    const properties = (first["inputSchema"] as Record<string, unknown>)[
      "properties"
    ] as Record<string, unknown>;
    (properties["issueIdOrKey"] as Record<string, unknown>)["pattern"] =
      "^[A-Za-z0-9\\_\\-]+$";
    expect(ToolDefinitionsSchema.safeParse(document).success).toBe(true);
  });

  it("rejects duplicate YAML mapping keys", () => {
    const temporary = mkdtempSync(join(tmpdir(), "connector-definitions-"));
    try {
      const path = join(temporary, "duplicate.tool_defs.yaml");
      writeFileSync(
        path,
        "source:\n  name: one\nsource:\n  name: two\n",
        "utf8",
      );
      expect(() => loadToolDefs(path)).toThrow(
        "invalid compiled connector catalog",
      );
    } finally {
      rmSync(temporary, { recursive: true, force: true });
    }
  });
});
