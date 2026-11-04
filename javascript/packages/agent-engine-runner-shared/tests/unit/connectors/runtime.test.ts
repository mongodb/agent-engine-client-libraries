import { createServer, type AddressInfo } from "node:http";
import { fileURLToPath } from "node:url";

import { afterEach, describe, expect, it } from "vitest";

import { runWithExecutionContext } from "../../../src/context.js";
import {
  BundledTool,
  ConnectorBundle,
  ConnectorToolRuntime,
  loadToolDefs,
  withConnectorEnvironment,
  type BundledConnector,
} from "../../../src/connectors/index.js";

const FIXTURE = fileURLToPath(
  new URL("./fixtures/jira.tool_defs.yaml", import.meta.url),
);

async function listen(
  handler: Parameters<typeof createServer>[0],
): Promise<{ baseUrl: string; close: () => Promise<void> }> {
  const server = createServer(handler);
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address() as AddressInfo;
  return {
    baseUrl: `http://127.0.0.1:${address.port}`,
    close: () =>
      new Promise<void>((resolve, reject) =>
        server.close((error) => (error ? reject(error) : resolve())),
      ),
  };
}

function bundleFor(
  baseUrl: string,
  auth: "none" | "oauth" | "bearer" = "none",
): {
  bundle: ConnectorBundle;
  connector: BundledConnector;
} {
  const definitions = structuredClone(loadToolDefs(FIXTURE));
  definitions.source.base_url = baseUrl;
  definitions.auth =
    auth === "none"
      ? null
      : auth === "oauth"
        ? { provider: "github", scopes: ["repo"] }
        : { bearer: "JIRA_TOKEN" };
  const first = definitions.tools[0];
  if (first === undefined) throw new Error("fixture has no tools");
  definitions.tools = [first];
  const connector: BundledConnector = { name: "jira", definitions };
  const tool = new BundledTool(connector, first);
  return { bundle: new ConnectorBundle([connector], [tool]), connector };
}

describe("ConnectorToolRuntime", () => {
  const cleanups: Array<() => Promise<void>> = [];
  afterEach(async () => {
    await Promise.all(cleanups.splice(0).map((cleanup) => cleanup()));
  });

  it("registers one callable per operation and executes it", async () => {
    const server = await listen((request, response) => {
      expect(request.url).toBe("/api/2/issue/DEMO-1/comment");
      expect(request.headers.authorization).toBeUndefined();
      response.writeHead(200, { "content-type": "application/json" });
      response.end('{"ok":true}');
    });
    cleanups.push(server.close);
    const { bundle } = bundleFor(server.baseUrl);
    const runtime = new ConnectorToolRuntime(bundle);
    cleanups.push(() => runtime.close());

    expect(runtime.registrations).toHaveLength(1);
    expect(runtime.registrations[0]?.metadata).toMatchObject({
      name: "jiradc_getComments",
      is_local: false,
      connector: "jira",
    });
    await expect(
      withConnectorEnvironment({}, () =>
        runtime.registrations[0]?.callable({ issueIdOrKey: "DEMO-1" }),
      ),
    ).resolves.toBe('{"ok":true}');
  });

  it("fails closed when no invocation environment is bound", async () => {
    const { bundle } = bundleFor("https://jira.example/rest");
    const runtime = new ConnectorToolRuntime(bundle);
    cleanups.push(() => runtime.close());
    const callable = runtime.registrations[0]?.callable;
    if (callable === undefined) throw new Error("fixture has no callable");

    await expect(callable({ issueIdOrKey: "DEMO-1" })).rejects.toThrow(
      'connector invocation environment is unavailable for tool "jiradc_getComments"',
    );
  });

  it("binds delegated authorization from the current execution", async () => {
    const server = await listen((request, response) => {
      expect(request.headers.authorization).toBe("Bearer alice-token");
      response.writeHead(200);
      response.end("ok");
    });
    cleanups.push(server.close);
    const { bundle } = bundleFor(server.baseUrl, "oauth");
    const runtime = new ConnectorToolRuntime(bundle);
    cleanups.push(() => runtime.close());

    await runWithExecutionContext(
      {
        executionId: "exec-1",
        wrapper: null,
        oeUrl: "",
        authorization: { token: "alice-token" },
      },
      async () => {
        await expect(
          withConnectorEnvironment({}, () =>
            runtime.registrations[0]?.callable({ issueIdOrKey: "DEMO-1" }),
          ),
        ).resolves.toBe("ok");
      },
    );
  });

  it("keeps each concurrent call's captured environment", async () => {
    const observed: string[] = [];
    let releaseFirst: (() => void) | undefined;
    const firstStarted = new Promise<void>((resolve) => {
      releaseFirst = resolve;
    });
    let firstResponseRelease: (() => void) | undefined;
    const firstResponse = new Promise<void>((resolve) => {
      firstResponseRelease = resolve;
    });
    const server = await listen((request, response) => {
      observed.push(String(request.headers.authorization));
      if (observed.length === 1) {
        releaseFirst?.();
        void firstResponse.then(() => {
          response.writeHead(200);
          response.end("first");
        });
        return;
      }
      response.writeHead(200);
      response.end("second");
    });
    cleanups.push(server.close);
    const { bundle } = bundleFor(server.baseUrl, "bearer");
    const runtime = new ConnectorToolRuntime(bundle);
    cleanups.push(() => runtime.close());
    const callable = runtime.registrations[0]?.callable;
    if (callable === undefined) throw new Error("fixture has no callable");

    const first = withConnectorEnvironment({ JIRA_TOKEN: "token-a" }, () =>
      callable({ issueIdOrKey: "DEMO-1" }),
    );
    await firstStarted;
    const second = withConnectorEnvironment({ JIRA_TOKEN: "token-b" }, () =>
      callable({ issueIdOrKey: "DEMO-1" }),
    );
    await expect(second).resolves.toBe("second");
    firstResponseRelease?.();
    await expect(first).resolves.toBe("first");
    expect(observed).toEqual(["Bearer token-a", "Bearer token-b"]);
  });

  it("closes every executor once and rejects later calls", async () => {
    const server = await listen((_request, response) => {
      response.writeHead(200);
      response.end("ok");
    });
    cleanups.push(server.close);
    const { bundle } = bundleFor(server.baseUrl);
    const runtime = new ConnectorToolRuntime(bundle);

    await runtime.close();
    await runtime.close();
    await expect(
      withConnectorEnvironment({}, () =>
        runtime.registrations[0]?.callable({ issueIdOrKey: "DEMO-1" }),
      ),
    ).rejects.toThrow("connector executor is closed");
  });
});
