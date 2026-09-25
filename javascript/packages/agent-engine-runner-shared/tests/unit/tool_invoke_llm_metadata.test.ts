/** Preparation uses startup configuration and never rebuilds with request credentials. */

import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { describe, test, expect, vi, beforeEach, afterEach } from "vitest";
import {
  ToolServer,
  registerLlm,
  getNamedLlm,
  hasNamedLlms,
  resetHooks,
  entrypointScope,
  type ITenantRuntime,
} from "../../src/index.js";
import { RuntimeAgentConfig } from "../../src/agent_config.js";
import type { LLMStreamChunk, LLMPodInvokeRequest } from "../../src/index.js";
import type * as MetadataModule from "../../src/server/metadata.js";

const hoisted = vi.hoisted(() => ({ metadataDir: "" }));

vi.mock("../../src/server/metadata.js", async (importOriginal) => {
  const actual = await importOriginal<typeof MetadataModule>();
  return {
    ...actual,
    // Production wrappers, redirected at the test's tmp metadata dir (the
    // default dir binds to /run/meta, which doesn't exist in tests).
    mergeMetadataEnv: () => actual.mergeMetadataEnv(hoisted.metadataDir),
    withMetadataEnv: <T>(fn: () => Promise<T>) =>
      actual.withMetadataEnv(fn, hoisted.metadataDir),
    withMetadataEnvGen: <T>(genFactory: () => AsyncGenerator<T>) =>
      actual.withMetadataEnvGen(genFactory, hoisted.metadataDir),
  };
});

interface ServerPrivates {
  onStartup: () => Promise<void>;
  doHandleInvokeLlm: (
    request: LLMPodInvokeRequest,
  ) => Promise<{ status: string }>;
  handleInvokeLlmStream: (
    request: LLMPodInvokeRequest,
  ) => AsyncGenerator<string>;
  invokeLlmStreamEvents: () => AsyncGenerator<string>;
  streamLlmChunks: (
    request: LLMPodInvokeRequest,
  ) => AsyncGenerator<LLMStreamChunk>;
}

function makeServer(entrypoint: () => unknown): ServerPrivates {
  const runtime = {
    tools: {},
    toolDefinitions: {},
    graphBuilder: { getAgent: entrypoint },
    getAgentConfig: () => new RuntimeAgentConfig(),
  };
  const server = new ToolServer(
    runtime as unknown as ITenantRuntime,
  ) as unknown as ServerPrivates;

  server.streamLlmChunks = async function* () {
    yield { content: "ok" } as LLMStreamChunk;
  };
  return server;
}

describe("prepared registry is independent of request metadata", () => {
  let seenKey: string | undefined;

  beforeEach(() => {
    resetHooks();
    seenKey = undefined;
    hoisted.metadataDir = mkdtempSync(join(tmpdir(), "meta-"));
    writeFileSync(join(hoisted.metadataDir, "FAKE_PROVIDER_KEY"), "shh");
  });

  afterEach(() => {
    rmSync(hoisted.metadataDir, { recursive: true, force: true });
    delete process.env["FAKE_PROVIDER_KEY"];
    resetHooks();
  });

  // What a user entrypoint does: read the provider key while constructing
  // the LLM client it registers.
  const entrypoint = () => {
    seenKey = process.env["FAKE_PROVIDER_KEY"];
    entrypointScope(() => registerLlm("primary", {}));
    return {};
  };

  test("non-streaming /invoke_llm", async () => {
    process.env["FAKE_PROVIDER_KEY"] = "startup-key";
    const build = vi.fn(entrypoint);
    const server = makeServer(build);
    await server.onStartup();
    expect(seenKey).toBe("startup-key");

    const response = await server.doHandleInvokeLlm({
      arguments: {},
    } as unknown as LLMPodInvokeRequest);

    expect(response.status).toBe("success");
    expect(seenKey).toBe("startup-key");
    expect(build).toHaveBeenCalledOnce();
    expect(hasNamedLlms()).toBe(true);
    // Unrestricted merge is permanent (no restore).
    expect(process.env["FAKE_PROVIDER_KEY"]).toBe("shh");
    delete process.env["FAKE_PROVIDER_KEY"];
  });

  test("streaming /invoke_llm/stream generator", async () => {
    process.env["FAKE_PROVIDER_KEY"] = "startup-key";
    const build = vi.fn(entrypoint);
    const server = makeServer(build);
    await server.onStartup();
    expect(seenKey).toBe("startup-key");

    const events: Array<{ content?: string; done?: boolean; error?: string }> =
      [];
    for await (const frame of server.handleInvokeLlmStream(
      {} as unknown as LLMPodInvokeRequest,
    )) {
      events.push(JSON.parse(frame.slice("data: ".length)));
    }

    expect(events[0]?.content).toBe("ok");
    expect(events.at(-1)?.done).toBe(true);
    expect(events.every((event) => event.error === undefined)).toBe(true);
    expect(seenKey).toBe("startup-key");
    expect(build).toHaveBeenCalledOnce();
    expect(process.env["FAKE_PROVIDER_KEY"]).toBe("shh");
    delete process.env["FAKE_PROVIDER_KEY"];
  });

  test("closing the stream closes the inner event generator", async () => {
    const server = makeServer(entrypoint);
    let closed = false;
    server.invokeLlmStreamEvents = async function* () {
      try {
        yield "data: first\n\n";
        yield "data: second\n\n";
      } finally {
        closed = true;
      }
    };

    const stream = server.handleInvokeLlmStream(
      {} as unknown as LLMPodInvokeRequest,
    );
    expect((await stream.next()).value).toBe("data: first\n\n");
    await stream.return(undefined);

    expect(closed).toBe(true);
  });

  test.each([
    { streaming: false, importTime: false },
    { streaming: true, importTime: false },
    { streaming: false, importTime: true },
    { streaming: true, importTime: true },
  ])(
    "failed warming recovers on requests ($streaming, $importTime)",
    async ({ streaming, importTime }) => {
      const fallback = {};
      const primary = {};
      if (importTime) entrypointScope(() => registerLlm("fallback", fallback));
      let recovered = false;
      const build = vi.fn(() => {
        entrypointScope(() => {
          if (!recovered) {
            registerLlm("partial", {});
            throw new Error("constructor dependency unavailable");
          }
          registerLlm("primary", primary);
        });
        return {};
      });
      const server = makeServer(build);
      await server.onStartup();
      expect(() => getNamedLlm("partial")).toThrow();
      if (importTime) expect(getNamedLlm("fallback")).toBe(fallback);
      server.streamLlmChunks = async function* () {
        expect(getNamedLlm("primary")).toBe(primary);
        yield { content: "recovered" } as LLMStreamChunk;
      };
      recovered = true;
      const invoke = async () => {
        const request = { arguments: {} } as unknown as LLMPodInvokeRequest;
        if (streaming) {
          const events = [];
          for await (const frame of server.handleInvokeLlmStream(request)) {
            events.push(JSON.parse(frame.slice("data: ".length)));
          }
          expect(events[0]?.content).toBe("recovered");
          expect(events.at(-1)?.done).toBe(true);
          expect(events.every((event) => event.error === undefined)).toBe(true);
        } else {
          expect((await server.doHandleInvokeLlm(request)).status).toBe(
            "success",
          );
        }
      };
      await Promise.all([invoke(), invoke()]);
      await invoke();
      expect(build).toHaveBeenCalledTimes(2);
    },
  );
});
