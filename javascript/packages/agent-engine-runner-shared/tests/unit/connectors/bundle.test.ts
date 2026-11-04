import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import {
  BundledTool,
  ConnectorBundle,
  ConnectorBundleError,
  ToolDefinitionsSchema,
  loadToolDefs,
  type BundledConnector,
} from "../../../src/connectors/index.js";

const FIXTURE = fileURLToPath(
  new URL("./fixtures/jira.tool_defs.yaml", import.meta.url),
);

function bundleFor(overrides: { auth?: Record<string, unknown> } = {}): {
  connector: BundledConnector;
  tools: BundledTool[];
} {
  const document = structuredClone(loadToolDefs(FIXTURE)) as unknown as Record<
    string,
    unknown
  >;
  if (overrides.auth) document["auth"] = overrides.auth;
  const definitions = ToolDefinitionsSchema.parse(document);
  const connector: BundledConnector = {
    name: definitions.source.name,
    definitions,
  };
  return {
    connector,
    tools: definitions.tools.map(
      (operation) => new BundledTool(connector, operation),
    ),
  };
}

describe("connector runtime bundle views", () => {
  it("returns detached schemas and the declared network host", () => {
    const { tools } = bundleFor();
    expect(tools[0]?.schema).toEqual({
      name: "jiradc_getComments",
      description: "Returns comments for an issue.",
      inputSchema: {
        type: "object",
        properties: { issueIdOrKey: { type: "string" } },
        required: ["issueIdOrKey"],
      },
    });
    expect(tools[0]?.network).toEqual(["jira.example"]);
    expect(tools[0]?.schema).not.toBe(tools[0]?.schema);
    const network = tools[0]?.network;
    network?.push("mutated.example");
    expect(tools[0]?.network).toEqual(["jira.example"]);
  });

  it("projects shared AER and Tool Pod registration metadata", () => {
    expect(bundleFor().tools[0]?.registrationMetadata).toEqual({
      name: "jiradc_getComments",
      description: "Returns comments for an issue.",
      is_local: false,
      provider: null,
      scopes: [],
      network: ["jira.example"],
      timeout_seconds: 30,
      redact_fields: [],
      connector: "jiradc",
    });
  });

  it("keeps delegated OAuth policy outside the model schema", () => {
    const { tools } = bundleFor({
      auth: { provider: "custom-github", scopes: ["repo"] },
    });
    for (const tool of tools) {
      expect(tool.registrationMetadata.provider).toBe("custom-github");
      expect(tool.registrationMetadata.scopes).toEqual(["repo"]);
      expect(tool.schema).not.toHaveProperty("provider");
      expect(tool.schema.inputSchema["properties"]).not.toHaveProperty(
        "authorization",
      );
    }
    const metadata = tools[0]?.registrationMetadata;
    metadata?.scopes.push("mutated");
    expect(tools[0]?.registrationMetadata.scopes).toEqual(["repo"]);
  });

  it("rejects name collisions before registration", () => {
    const { connector, tools } = bundleFor();
    const bundle = new ConnectorBundle([connector], tools);
    expect(() =>
      bundle.requireAvailableNames(new Set(["existing", "jiradc_addComment"])),
    ).toThrow(ConnectorBundleError);
    expect(() =>
      bundle.requireAvailableNames(new Set(["existing", "jiradc_addComment"])),
    ).toThrow("jiradc_addComment");
    expect(() =>
      new ConnectorBundle().requireAvailableNames(new Set(["anything"])),
    ).not.toThrow();
  });
});
