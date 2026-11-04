import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { App } from "../src/runtime.js";

const CATALOG = `
source:
  name: jira
  base_url: https://jira.example/rest
auth:
  bearer: JIRA_TOKEN
tools:
  - name: jiradc_getComments
    description: Returns comments for an issue.
    method: GET
    path: /api/2/issue/{issueIdOrKey}/comment
    params:
      path: [issueIdOrKey]
    inputSchema:
      type: object
      properties:
        issueIdOrKey: {type: string}
      required: [issueIdOrKey]
  - name: jiradc_addComment
    description: Adds a comment to an issue.
    method: POST
    path: /api/2/issue/{issueIdOrKey}/comment
    params:
      path: [issueIdOrKey]
      body: [body]
    inputSchema:
      type: object
      properties:
        issueIdOrKey: {type: string}
        body: {type: string}
      required: [issueIdOrKey, body]
`;

const TOOL_YAML = `
name: jira
tool_defs: jira.tool_defs.yaml
source:
  type: openapi
  base_url: https://jira.corp/rest
expose:
  allow:
    - jiradc_getComments
    - jiradc_addComment
auth:
  bearer: JIRA_TOKEN
`;

interface RuntimeMetadataView {
  runtime: {
    getToolMetadata(name: string): Record<string, unknown>;
  };
}

function writeWorkspace(root: string): void {
  const connector = join(root, "connectors/jira");
  mkdirSync(connector, { recursive: true });
  writeFileSync(join(connector, "jira.tool_defs.yaml"), CATALOG, "utf8");
  writeFileSync(join(connector, "tool.yaml"), TOOL_YAML, "utf8");
  writeFileSync(
    join(root, "agent.yaml"),
    [
      "name: connector-agent",
      "entrypoint: connectors.agent:app",
      "connectors:",
      "  - source: connectors/jira/tool.yaml",
      "",
    ].join("\n"),
    "utf8",
  );
}

describe("App — connector registration", () => {
  let root: string;

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), "langgraph-connectors-"));
    vi.stubEnv("RUNNER_MODE", "aer");
    vi.stubEnv("AGENTIC_AGENT_WORKDIR", root);
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    rmSync(root, { recursive: true, force: true });
  });

  it("registers connector tools as remote AER tools", async () => {
    writeWorkspace(root);
    const app = new App({ appName: "Connector Agent" });
    await app.ready();

    const definitions = app.getToolDefinitions();
    expect(definitions.map((definition) => definition.name)).toEqual([
      "jiradc_getComments",
      "jiradc_addComment",
    ]);
    expect(definitions[0]).toMatchObject({
      remote: true,
      network: ["jira.corp"],
      timeout_seconds: 30,
    });
    expect(definitions[0]?.callable).toBeTypeOf("function");
    expect(() => definitions[0]?.callable({ issueIdOrKey: "DEMO-1" })).toThrow(
      "connector tools execute only through the Tool Pod",
    );

    const metadata = (
      app as unknown as RuntimeMetadataView
    ).runtime.getToolMetadata("jiradc_getComments");
    expect(metadata).toMatchObject({
      is_local: false,
      connector: "jira",
      network: ["jira.corp"],
    });
  });

  it("rejects a connector name collision before registering any connector", async () => {
    writeWorkspace(root);
    const app = new App({ appName: "Connector Agent" });
    app.tool({ name: "jiradc_getComments" })(() => "user");
    await expect(app.ready()).rejects.toThrow("connector tool names collide");
    expect(
      app.getToolDefinitions().map((definition) => definition.name),
    ).toEqual(["jiradc_getComments"]);
  });

  it("does not register connectors in Tool Pod mode", async () => {
    writeWorkspace(root);
    vi.stubEnv("RUNNER_MODE", "tool");
    const app = new App({ appName: "Connector Agent" });
    await app.ready();
    expect(app.getToolDefinitions()).toEqual([]);
  });
});
