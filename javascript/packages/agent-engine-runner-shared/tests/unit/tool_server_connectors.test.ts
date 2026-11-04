import {
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { fileURLToPath } from "node:url";

import { afterEach, describe, expect, it, vi } from "vitest";

import { RuntimeAgentConfig } from "../../src/agent_config.js";
import { ConnectorExecutor } from "../../src/connectors/executor.js";
import { withConnectorEnvironment } from "../../src/connectors/index.js";
import { ToolServer } from "../../src/index.js";
import type { ITenantRuntime, ServerToolFn } from "../../src/index.js";
import type { ToolPodExecuteRequest } from "../../src/models.js";

const FIXTURE = fileURLToPath(
  new URL("./connectors/fixtures/jira.tool_defs.yaml", import.meta.url),
);

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

interface MockRuntime {
  tools: Record<string, ServerToolFn>;
  toolDefinitions: Record<string, Record<string, unknown>>;
  graphBuilder: null;
  getAgentConfig: () => RuntimeAgentConfig;
  getMongodbUri: () => null;
  getAgent: () => never;
}

function workspace(
  withConnectors = true,
  baseUrl = "https://jira.corp/rest",
): {
  root: string;
  config: RuntimeAgentConfig;
} {
  const root = mkdtempSync(join(tmpdir(), "tool-server-connectors-"));
  if (withConnectors) {
    const connector = join(root, "connectors/jira");
    mkdirSync(connector, { recursive: true });
    writeFileSync(
      join(connector, "jira.tool_defs.yaml"),
      readFileSync(FIXTURE),
    );
    writeFileSync(
      join(connector, "tool.yaml"),
      TOOL_YAML.replace("https://jira.corp/rest", baseUrl),
      "utf8",
    );
    writeFileSync(
      join(root, "agent.yaml"),
      [
        "name: connector-agent",
        "connectors:",
        "  - source: connectors/jira/tool.yaml",
      ].join("\n"),
      "utf8",
    );
  }
  return {
    root,
    config: new RuntimeAgentConfig({
      path: join(root, "agent.yaml"),
      ...(withConnectors && {
        connectors: [{ source: "connectors/jira/tool.yaml" }],
      }),
    }),
  };
}

function makeRuntime(config: RuntimeAgentConfig): MockRuntime {
  return {
    tools: {},
    toolDefinitions: {},
    graphBuilder: null,
    getAgentConfig: () => config,
    getMongodbUri: () => null,
    getAgent: () => {
      throw new Error("not used");
    },
  };
}

function makeServer(runtime: MockRuntime): ToolServer {
  return new ToolServer(runtime as unknown as ITenantRuntime);
}

describe("ToolServer connector registration", () => {
  const cleanups: string[] = [];
  afterEach(() => {
    vi.restoreAllMocks();
    for (const root of cleanups.splice(0)) {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("registers connector callables and metadata at startup", async () => {
    const { root, config } = workspace();
    cleanups.push(root);
    const runtime = makeRuntime(config);
    const server = makeServer(runtime);
    await server.onStartup();

    expect(Object.keys(runtime.tools).sort()).toEqual([
      "jiradc_addComment",
      "jiradc_getComments",
    ]);
    expect(runtime.toolDefinitions["jiradc_getComments"]).toMatchObject({
      name: "jiradc_getComments",
      is_local: false,
      provider: null,
      network: ["jira.corp"],
      connector: "jira",
    });
    await server.onShutdown();
    await expect(
      withConnectorEnvironment({}, () =>
        runtime.tools["jiradc_getComments"]?.({ issueIdOrKey: "DEMO-1" }),
      ),
    ).rejects.toThrow("connector executor is closed");
  });

  it("passes the ToolServer request environment to connector execution", async () => {
    const { root, config } = workspace();
    cleanups.push(root);
    const runtime = makeRuntime(config);
    const server = makeServer(runtime);
    await server.onStartup();
    const previous = process.env["JIRA_TOKEN"];
    process.env["JIRA_TOKEN"] = "tool-server-token";
    let observed: string | undefined;
    const executeSpy = vi
      .spyOn(ConnectorExecutor.prototype, "execute")
      .mockImplementation(async (_name, _arguments, options) => {
        observed = options?.env?.["JIRA_TOKEN"];
        return "ok";
      });

    try {
      const response = await (
        server as unknown as {
          doHandleExecute: (
            request: ToolPodExecuteRequest,
          ) => Promise<{ status: string }>;
        }
      ).doHandleExecute({
        execution_id: "exec-connector",
        tool_name: "jiradc_getComments",
        arguments: { issueIdOrKey: "DEMO-1" },
        session_id: "session-1",
        metadata: {},
      });
      expect(response.status).toBe("success");
      expect(executeSpy).toHaveBeenCalledOnce();
      expect(observed).toBe("tool-server-token");
    } finally {
      if (previous === undefined) delete process.env["JIRA_TOKEN"];
      else process.env["JIRA_TOKEN"] = previous;
      await server.onShutdown();
    }
  });

  it("keeps concurrent request environments isolated through ToolServer", async () => {
    const { root, config } = workspace();
    cleanups.push(root);
    const runtime = makeRuntime(config);
    const server = makeServer(runtime);
    await server.onStartup();
    const previous = process.env["JIRA_TOKEN"];
    const observed: Array<string | undefined> = [];
    let markFirstStarted: (() => void) | undefined;
    const firstStarted = new Promise<void>((resolve) => {
      markFirstStarted = resolve;
    });
    let releaseFirst: (() => void) | undefined;
    const firstRelease = new Promise<void>((resolve) => {
      releaseFirst = resolve;
    });
    const executeSpy = vi
      .spyOn(ConnectorExecutor.prototype, "execute")
      .mockImplementation(async (_name, _arguments, options) => {
        observed.push(options?.env?.["JIRA_TOKEN"]);
        if (observed.length === 1) {
          markFirstStarted?.();
          await firstRelease;
          return "first";
        }
        return "second";
      });
    const execute = (executionId: string) =>
      (
        server as unknown as {
          doHandleExecute: (request: ToolPodExecuteRequest) => Promise<{
            status: string;
            result?: string;
          }>;
        }
      ).doHandleExecute({
        execution_id: executionId,
        tool_name: "jiradc_getComments",
        arguments: { issueIdOrKey: "DEMO-1" },
        session_id: "session-1",
        metadata: {},
      });

    try {
      process.env["JIRA_TOKEN"] = "token-a";
      const first = execute("exec-first");
      await firstStarted;
      process.env["JIRA_TOKEN"] = "token-b";
      const second = execute("exec-second");
      await expect(second).resolves.toMatchObject({
        status: "success",
        result: "second",
      });
      releaseFirst?.();
      await expect(first).resolves.toMatchObject({
        status: "success",
        result: "first",
      });
      expect(observed).toEqual(["token-a", "token-b"]);
      expect(executeSpy).toHaveBeenCalledTimes(2);
    } finally {
      releaseFirst?.();
      if (previous === undefined) delete process.env["JIRA_TOKEN"];
      else process.env["JIRA_TOKEN"] = previous;
      await server.onShutdown();
    }
  });

  it("classifies provider errors through the shared Jira fixture", async () => {
    const provider = createServer((_request, response) => {
      response.writeHead(403, { "content-type": "application/json" });
      response.end('{"message":"provider denied"}');
    });
    await new Promise<void>((resolve) =>
      provider.listen(0, "127.0.0.1", resolve),
    );
    const address = provider.address() as AddressInfo;
    const { root, config } = workspace(
      true,
      `http://127.0.0.1:${address.port}`,
    );
    cleanups.push(root);
    const runtime = makeRuntime(config);
    const server = makeServer(runtime);
    const previous = process.env["JIRA_TOKEN"];
    process.env["JIRA_TOKEN"] = "token";

    try {
      await server.onStartup();
      const response = await (
        server as unknown as {
          doHandleExecute: (request: ToolPodExecuteRequest) => Promise<{
            status: string;
            error?: string;
            tool_api_error?: {
              classification?: string;
              http_status?: number;
              provider_type?: string | null;
            };
          }>;
        }
      ).doHandleExecute({
        execution_id: "exec-connector-error",
        tool_name: "jiradc_getComments",
        arguments: { issueIdOrKey: "DEMO-1" },
        session_id: "session-1",
        metadata: {},
      });

      expect(response.status).toBe("error");
      expect(response.error).toContain(
        "External API call failed: HTTP 403 AUTH_FAILED",
      );
      expect(response.tool_api_error).toMatchObject({
        classification: "AUTH_FAILED",
        http_status: 403,
        provider_type: null,
      });
    } finally {
      if (previous === undefined) delete process.env["JIRA_TOKEN"];
      else process.env["JIRA_TOKEN"] = previous;
      await server.onShutdown();
      await new Promise<void>((resolve, reject) =>
        provider.close((error) => (error ? reject(error) : resolve())),
      );
    }
  });

  it("does nothing when the agent has no connectors", async () => {
    const { root, config } = workspace(false);
    cleanups.push(root);
    const runtime = makeRuntime(config);
    const server = makeServer(runtime);
    await server.onStartup();
    expect(runtime.tools).toEqual({});
    await server.onShutdown();
  });

  it("rejects a connector name colliding with a user tool", async () => {
    const { root, config } = workspace();
    cleanups.push(root);
    const runtime = makeRuntime(config);
    runtime.tools["jiradc_getComments"] = () => "user";
    const server = makeServer(runtime);
    await expect(server.onStartup()).rejects.toThrow(
      "connector tool names collide",
    );
  });
});
