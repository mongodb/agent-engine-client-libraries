/**
 * Tests for the AER capability advertise flow (`advertiseCapabilities` /
 * `ensureOeRegistrations`).
 *
 * Mirrors Python's tests/unit/test_aer_capability_advertise.py. The TS AER
 * had no capability advertisement before this change; these tests cover the
 * whole net-new path: payload shape (matching the Python
 * `{workspace_id, org_id, project_id, language, framework, features}`
 * contract, including the SDK-injected `owner_callback_fallback: true`),
 * required startup registration, failure propagation, and
 * the invariant that `/execute` never runs before registration succeeds.
 *
 * TS-vs-Python: AERServer is built via `Object.create(prototype)` (skip
 * ctor); private methods/fields are reached via cast-through-unknown, same
 * pattern as aer_policy_denied.test.ts / aer_session_finish.test.ts.
 */

import { describe, test, expect, vi, beforeEach, afterEach } from "vitest";
import { mkdtempSync, writeFileSync, rmSync, realpathSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  AERServer,
  loadRuntimeAgentConfig,
  explicitFeatures,
  type ExecuteRequest,
  type RuntimeAgentConfig,
} from "../../src/index.js";

// ---------------------------------------------------------------------------
// Test scaffolding: per-test tmp dir + cwd swap for agent.yaml discovery.
// ---------------------------------------------------------------------------

let tmpDir: string;
let originalCwd: string;

beforeEach(() => {
  tmpDir = realpathSync(mkdtempSync(join(tmpdir(), "aer-capability-")));
  originalCwd = process.cwd();
  process.chdir(tmpDir);
});

afterEach(() => {
  process.chdir(originalCwd);
  rmSync(tmpDir, { recursive: true, force: true });
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});

function writeAgentYaml(content: string): void {
  writeFileSync(join(tmpDir, "agent.yaml"), content, "utf-8");
}

interface AerCapabilityPrivates {
  runtime: {
    orgId: string | null;
    projectId: string | null;
    graphBuilder?: unknown;
    getAgentConfig: () => RuntimeAgentConfig;
  };
  capabilitiesAdvertised: boolean;
  advertiseCapabilities: (oeUrl: string, workspaceId: string) => Promise<void>;
  ensureOeRegistrations: (oeUrl: string, workspaceId: string) => Promise<void>;
  onStartup: () => Promise<void>;
  doHandleExecute: (request: ExecuteRequest) => Promise<unknown>;
  executeViaAgentStream: (...args: unknown[]) => Promise<unknown>;
  sendStreamChunk: (...args: unknown[]) => Promise<void>;
  reportCallback: (...args: unknown[]) => Promise<void>;
  chunkSeq: Map<string, number>;
  ownerCallbackUrl: Map<string, string>;
}

function makeServer(
  agentConfig: RuntimeAgentConfig,
  overrides: Partial<AerCapabilityPrivates["runtime"]> = {},
): AerCapabilityPrivates {
  const server = Object.create(AERServer.prototype) as AerCapabilityPrivates;
  server.runtime = {
    orgId: "org-from-runtime",
    projectId: "proj-from-runtime",
    graphBuilder: {},
    getAgentConfig: () => agentConfig,
    ...overrides,
  };
  server.capabilitiesAdvertised = false;
  server.sendStreamChunk = vi.fn().mockResolvedValue(undefined);
  server.reportCallback = vi.fn().mockResolvedValue(undefined);
  server.chunkSeq = new Map();
  server.ownerCallbackUrl = new Map();
  return server;
}

function jsonResponse(status: number, body?: unknown): Response {
  return new Response(body === undefined ? null : JSON.stringify(body), {
    status,
  });
}

describe("AER capability advertise", () => {
  test("advertises language/framework/features from agent.yaml, with owner_callback_fallback merged in", async () => {
    writeAgentYaml(
      [
        "entrypoint: config.agent:app",
        "language: typescript",
        "framework: langgraph",
        "features:",
        "  durable_workflow: true",
        "  memory: false",
      ].join("\n"),
    );
    const cfg = loadRuntimeAgentConfig();
    const server = makeServer(cfg);

    let capturedUrl = "";
    let capturedBody: Record<string, unknown> | undefined;
    const fetchMock = vi.fn(async (url: string | URL, init?: RequestInit) => {
      capturedUrl = String(url);
      capturedBody = JSON.parse(String(init?.body)) as Record<string, unknown>;
      return jsonResponse(200);
    });
    vi.stubGlobal("fetch", fetchMock);

    await server.advertiseCapabilities("http://oe:8000", "ws-test-agent");

    expect(capturedUrl).toBe("http://oe:8000/agent/capabilities");
    expect(capturedBody).toMatchObject({
      workspace_id: "ws-test-agent",
      org_id: "org-from-runtime",
      project_id: "proj-from-runtime",
      language: "typescript",
      framework: "langgraph",
    });
    expect(capturedBody?.["features"]).toEqual({
      durable_workflow: true,
      memory: false,
      owner_callback_fallback: true,
    });
    expect(server.capabilitiesAdvertised).toBe(true);
  });

  test("omitted feature flags are not sent, but owner_callback_fallback is always injected", async () => {
    writeAgentYaml("entrypoint: config.agent:app\nfeatures:\n  memory: true\n");
    const cfg = loadRuntimeAgentConfig();
    const server = makeServer(cfg);

    let capturedBody: Record<string, unknown> | undefined;
    const fetchMock = vi.fn(async (_url: string | URL, init?: RequestInit) => {
      capturedBody = JSON.parse(String(init?.body)) as Record<string, unknown>;
      return jsonResponse(200);
    });
    vi.stubGlobal("fetch", fetchMock);

    await server.advertiseCapabilities("http://oe:8000", "ws-test");

    const features = capturedBody?.["features"] as Record<string, unknown>;
    expect(features).not.toHaveProperty("durable_workflow");
    expect(features["memory"]).toBe(true);
    expect(features["owner_callback_fallback"]).toBe(true);
  });

  test("owner_callback_fallback is injected without mutating the loaded agent config", async () => {
    writeAgentYaml(
      "entrypoint: config.agent:app\nlanguage: python\nframework: langgraph\n",
    );
    const cfg = loadRuntimeAgentConfig();
    const server = makeServer(cfg);

    expect(explicitFeatures(cfg.features)).toEqual({});

    let capturedBody: Record<string, unknown> | undefined;
    const fetchMock = vi.fn(async (_url: string | URL, init?: RequestInit) => {
      capturedBody = JSON.parse(String(init?.body)) as Record<string, unknown>;
      return jsonResponse(200);
    });
    vi.stubGlobal("fetch", fetchMock);

    await server.advertiseCapabilities("http://oe:8000", "ws-test");

    expect(Object.keys(capturedBody ?? {}).sort()).toEqual(
      [
        "workspace_id",
        "org_id",
        "project_id",
        "language",
        "framework",
        "features",
      ].sort(),
    );
    expect(capturedBody?.["features"]).toEqual({
      owner_callback_fallback: true,
    });
    // The agent config object itself is untouched by the injection.
    expect(explicitFeatures(cfg.features)).toEqual({});
  });

  test("rejects capability registration when org_id or project_id is missing", async () => {
    writeAgentYaml("entrypoint: config.agent:app\n");
    const cfg = loadRuntimeAgentConfig();
    const server = makeServer(cfg, { orgId: null });

    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      server.advertiseCapabilities("http://oe:8000", "ws-test"),
    ).rejects.toThrow("requires ORG_ID and PROJECT_ID");

    expect(fetchMock).not.toHaveBeenCalled();
    expect(server.capabilitiesAdvertised).toBe(false);
  });

  test("network error propagates", async () => {
    writeAgentYaml("entrypoint: config.agent:app\n");
    const cfg = loadRuntimeAgentConfig();
    const server = makeServer(cfg);

    const fetchMock = vi.fn(async () => {
      throw new Error("connection refused");
    });
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      server.advertiseCapabilities("http://oe:8000", "ws-test"),
    ).rejects.toThrow("connection refused");

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(server.capabilitiesAdvertised).toBe(false);
  });

  test("a terminal 404 rejects capability registration", async () => {
    writeAgentYaml("entrypoint: config.agent:app\n");
    const cfg = loadRuntimeAgentConfig();
    const server = makeServer(cfg);

    const textSpy = vi.fn(async () => "not found");
    const fetchMock = vi.fn().mockResolvedValue({
      status: 404,
      ok: false,
      text: textSpy,
    } as Response);
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      server.advertiseCapabilities("http://oe:8000", "ws-test"),
    ).rejects.toThrow("registration was rejected");
    expect(server.capabilitiesAdvertised).toBe(false);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(textSpy).not.toHaveBeenCalled();
  });

  test("cancels the unread response body so undici can release the connection", async () => {
    writeAgentYaml("entrypoint: config.agent:app\n");
    const cfg = loadRuntimeAgentConfig();
    const server = makeServer(cfg);

    let cancelled = false;
    const fetchMock = vi.fn(
      async () =>
        new Response(
          new ReadableStream({
            cancel() {
              cancelled = true;
            },
          }),
          { status: 200 },
        ),
    );
    vi.stubGlobal("fetch", fetchMock);

    await server.advertiseCapabilities("http://oe:8000", "ws-test");

    expect(server.capabilitiesAdvertised).toBe(true);
    expect(cancelled).toBe(true);
  });

  test("rejects capability advertisement redirects without retrying", async () => {
    writeAgentYaml("entrypoint: config.agent:app\n");
    const cfg = loadRuntimeAgentConfig();
    const server = makeServer(cfg);
    const fetchMock = vi.fn(async (_url: string | URL, init?: RequestInit) => {
      expect(init?.redirect).toBe("manual");
      return new Response(null, {
        status: 307,
        headers: { Location: "https://attacker.example/capture" },
      });
    });
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      server.advertiseCapabilities("http://oe:8000", "ws-test"),
    ).rejects.toThrow("registration was rejected");

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(server.capabilitiesAdvertised).toBe(false);
  });

  test("keeps an accepted advertisement when body cancellation fails", async () => {
    writeAgentYaml("entrypoint: config.agent:app\n");
    const cfg = loadRuntimeAgentConfig();
    const server = makeServer(cfg);
    vi.stubGlobal(
      "fetch",
      vi.fn(
        async () =>
          ({
            status: 200,
            body: {
              cancel: vi.fn(async () => Promise.reject(new Error("closed"))),
            },
          }) as unknown as Response,
      ),
    );

    await server.advertiseCapabilities("http://oe:8000", "ws-test");

    expect(server.capabilitiesAdvertised).toBe(true);
  });

  test("a 503 rejects capability registration", async () => {
    writeAgentYaml("entrypoint: config.agent:app\n");
    const cfg = loadRuntimeAgentConfig();
    const server = makeServer(cfg);

    const textSpy = vi.fn(async () => "unavailable");
    const fetchMock = vi.fn().mockResolvedValue({
      status: 503,
      ok: false,
      text: textSpy,
    } as Response);
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      server.advertiseCapabilities("http://oe:8000", "ws-test"),
    ).rejects.toThrow("registration was rejected");
    expect(server.capabilitiesAdvertised).toBe(false);
    expect(textSpy).not.toHaveBeenCalled();
  });

  test("onStartup advertises when OE_URL and APP_ID are both set", async () => {
    writeAgentYaml("entrypoint: config.agent:app\nlanguage: typescript\n");
    const cfg = loadRuntimeAgentConfig();
    const server = makeServer(cfg);
    vi.stubEnv("OE_URL", "http://oe:8000");
    vi.stubEnv("APP_ID", "ws-startup");

    let capturedUrl = "";
    const fetchMock = vi.fn(async (url: string | URL) => {
      capturedUrl = String(url);
      return jsonResponse(200);
    });
    vi.stubGlobal("fetch", fetchMock);

    await server.onStartup();

    expect(capturedUrl).toBe("http://oe:8000/agent/capabilities");
    expect(server.capabilitiesAdvertised).toBe(true);
  });

  test("onStartup rejects when OE_URL or APP_ID is unset", async () => {
    writeAgentYaml("entrypoint: config.agent:app\n");
    const cfg = loadRuntimeAgentConfig();
    const server = makeServer(cfg);
    vi.stubEnv("OE_URL", "");
    vi.stubEnv("APP_ID", "");

    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    await expect(server.onStartup()).rejects.toThrow(
      "requires OE_URL and APP_ID",
    );

    expect(fetchMock).not.toHaveBeenCalled();
    expect(server.capabilitiesAdvertised).toBe(false);
  });

  test("onStartup propagates a registration failure", async () => {
    writeAgentYaml("entrypoint: config.agent:app\n");
    const cfg = loadRuntimeAgentConfig();
    const server = makeServer(cfg);
    vi.stubEnv("OE_URL", "http://oe:8000");
    vi.stubEnv("APP_ID", "ws-startup");

    const fetchMock = vi.fn(async () => {
      throw new Error("network unreachable");
    });
    vi.stubGlobal("fetch", fetchMock);

    await expect(server.onStartup()).rejects.toThrow("network unreachable");

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(server.capabilitiesAdvertised).toBe(false);
  });

  test("onStartup preserves invalid TLS configuration errors", async () => {
    writeAgentYaml("entrypoint: config.agent:app\n");
    const cfg = loadRuntimeAgentConfig();
    const server = makeServer(cfg);
    vi.stubEnv("OE_URL", "https://oe:8443");
    vi.stubEnv("APP_ID", "ws-startup");
    for (const name of [
      "TLS_CERT_PATH",
      "TLS_KEY_PATH",
      "TLS_CA_CERT_PATH",
      "TLS_CERT_PEM",
      "TLS_KEY_PEM",
      "TLS_CA_CERT_PEM",
    ]) {
      vi.stubEnv(name, "");
    }

    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    await expect(server.onStartup()).rejects.toThrow(
      "HTTPS URL requires mTLS configuration",
    );

    expect(fetchMock).not.toHaveBeenCalled();
    expect(server.capabilitiesAdvertised).toBe(false);
  });

  test("successful advertisement is not re-sent on every /execute", async () => {
    writeAgentYaml("entrypoint: config.agent:app\nlanguage: typescript\n");
    const cfg = loadRuntimeAgentConfig();
    const server = makeServer(cfg, {
      graphBuilder: {},
    });
    vi.stubEnv("APP_ID", "ws-configured");
    (server.runtime as Record<string, unknown>)["getAgent"] = () => ({});

    const fetchMock = vi.fn(async () => jsonResponse(200));
    vi.stubGlobal("fetch", fetchMock);
    server.executeViaAgentStream = vi
      .fn()
      .mockResolvedValue({ content: "ok", messages: [] });

    const request = (executionId: string): ExecuteRequest =>
      ({
        execution_id: executionId,
        message: "hi",
        platform_api_url: "http://oe:8000",
        workspace_id: "ws-test",
        resume: false,
      }) as ExecuteRequest;

    await server.doHandleExecute(request("exec-1"));
    await server.doHandleExecute(request("exec-2"));

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0]?.[0]).toBe(
      "http://oe:8000/agent/capabilities",
    );
    expect(
      JSON.parse(String(fetchMock.mock.calls[0]?.[1]?.body))["workspace_id"],
    ).toBe("ws-configured");
    expect(server.capabilitiesAdvertised).toBe(true);
  });

  test("terminal registration failure during /execute prevents agent execution", async () => {
    writeAgentYaml("entrypoint: config.agent:app\nlanguage: typescript\n");
    const cfg = loadRuntimeAgentConfig();
    const server = makeServer(cfg, {
      graphBuilder: {},
    });
    vi.stubEnv("APP_ID", "ws-configured");
    (server.runtime as Record<string, unknown>)["getAgent"] = () => ({});

    const fetchMock = vi.fn(async (url: string | URL) => {
      if (String(url).endsWith("/agent/capabilities")) {
        return jsonResponse(404);
      }
      return jsonResponse(200);
    });
    vi.stubGlobal("fetch", fetchMock);
    server.executeViaAgentStream = vi
      .fn()
      .mockResolvedValue({ content: "ok", messages: [] });

    const request: ExecuteRequest = {
      execution_id: "exec-1",
      message: "hi",
      platform_api_url: "http://oe:8000",
      workspace_id: "ws-test",
      resume: false,
    } as ExecuteRequest;

    await expect(server.doHandleExecute(request)).rejects.toThrow(
      "registration was rejected",
    );
    expect(server.executeViaAgentStream).not.toHaveBeenCalled();
    expect(server.capabilitiesAdvertised).toBe(false);
  });

  test("request workspace cannot trigger advertisement without APP_ID", async () => {
    writeAgentYaml("entrypoint: config.agent:app\nlanguage: typescript\n");
    const cfg = loadRuntimeAgentConfig();
    const server = makeServer(cfg, {
      graphBuilder: {},
    });
    vi.stubEnv("APP_ID", "");
    (server.runtime as Record<string, unknown>)["getAgent"] = () => ({});

    const fetchMock = vi.fn(async () => jsonResponse(200));
    vi.stubGlobal("fetch", fetchMock);
    server.executeViaAgentStream = vi
      .fn()
      .mockResolvedValue({ content: "ok", messages: [] });

    await server.doHandleExecute({
      execution_id: "exec-untrusted-workspace",
      message: "hi",
      platform_api_url: "http://oe:8000",
      workspace_id: "ws-from-request",
      resume: false,
    } as ExecuteRequest);

    expect(
      fetchMock.mock.calls.some(([url]) =>
        String(url).endsWith("/agent/capabilities"),
      ),
    ).toBe(false);
    expect(server.capabilitiesAdvertised).toBe(false);
  });
});
