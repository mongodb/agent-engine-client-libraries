/**
 * `App.deepAgent()` and an already-registered orchestrator model.
 *
 * An agent that opts into per-execution builds has to register every `llmId`
 * before it reads invocation context, because the Tool Pod's context-free
 * discovery build only retains what was registered before the builder stopped.
 * `deepAgent()` registers the orchestrator itself under `__default__`, so an
 * author who has already registered it must not hit a duplicate id.
 *
 * Registering is not the whole contract. The model still has to reach the graph
 * wrapped, or its calls go straight to the provider and bypass OE approvals,
 * auditing, and guardrails. `app.llm()` registers the model it is given and
 * returns a *different* object, so an author who keeps its own raw instance and
 * passes that on is a supported shape and is covered here.
 *
 * These drive the public `App.deepAgent()` and inspect the model it hands to
 * the deep-agent factory, rather than calling the private resolver directly.
 * `createAgentEngineDeepAgent` is mocked so the assertions are about resolution
 * rather than about building a real graph.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { ChatOpenAI } from "@langchain/openai";

const { createAgentEngineDeepAgent } = vi.hoisted(() => ({
  createAgentEngineDeepAgent: vi.fn(),
}));
vi.mock("../src/deep_agent.js", () => ({ createAgentEngineDeepAgent }));

import {
  entrypointScope,
  resetLlmRegistry,
  snapshotLlmRegistry,
} from "@mongodb-js/agent-engine-runner-shared";
import { App } from "../src/runtime.js";
import { SecureWrappedLLM } from "../src/secure_llm.js";

let tmpDir: string;
let savedMode: string | undefined;
let savedConfigPath: string | undefined;

function makeApp(): App {
  const p = path.join(tmpDir, "agent.yaml");
  fs.writeFileSync(p, "name: deep-agent-demo\nfeatures:\n  deep_agent: true\n");
  process.env["AGENTIC_AGENT_CONFIG_PATH"] = p;
  return new App({ appName: "deep-agent-demo" });
}

function model(): ChatOpenAI {
  return new ChatOpenAI({ model: "gpt-4o", apiKey: "test-key" });
}

/** Register `raw` the way an author would, returning `app.llm()`'s result. */
function registerThroughAppLlm(app: App, raw: unknown): unknown {
  return entrypointScope(() => app.llm(raw as never));
}

/** Call `app.deepAgent(llm)` and return the model forwarded to the factory. */
function secureLlmForwardedToFactory(app: App, llm: unknown): unknown {
  createAgentEngineDeepAgent.mockClear();
  entrypointScope(() => app.deepAgent(llm as never));
  return createAgentEngineDeepAgent.mock.calls[0]?.[0];
}

/** The model a `SecureWrappedLLM` was built around. */
function wrappedModel(secure: unknown): unknown {
  return (secure as { llm: unknown }).llm;
}

beforeEach(() => {
  savedMode = process.env["RUNNER_MODE"];
  savedConfigPath = process.env["AGENTIC_AGENT_CONFIG_PATH"];
  process.env["RUNNER_MODE"] = "aer";
  resetLlmRegistry();
  createAgentEngineDeepAgent.mockReset();
  tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "deep-agent-orchestrator-"));
});

afterEach(() => {
  if (savedMode === undefined) delete process.env["RUNNER_MODE"];
  else process.env["RUNNER_MODE"] = savedMode;
  if (savedConfigPath === undefined)
    delete process.env["AGENTIC_AGENT_CONFIG_PATH"];
  else process.env["AGENTIC_AGENT_CONFIG_PATH"] = savedConfigPath;
  fs.rmSync(tmpDir, { recursive: true, force: true });
});

describe("App.deepAgent() orchestrator registration", () => {
  it("registers the orchestrator when the author has not", () => {
    const raw = model();

    const secure = secureLlmForwardedToFactory(makeApp(), raw);

    expect(secure).toBeInstanceOf(SecureWrappedLLM);
    expect(wrappedModel(secure)).toBe(raw);
    expect(snapshotLlmRegistry().get("__default__")).toBe(raw);
  });

  it("wraps a raw model the author already registered", () => {
    const app = makeApp();
    const raw = model();
    registerThroughAppLlm(app, raw);

    // A per-execution builder registers before it reads invocation context, so
    // the orchestrator is already in the registry by the time it builds. The
    // author is free to pass on the instance it registered.
    const secure = secureLlmForwardedToFactory(app, raw);

    expect(secure).toBeInstanceOf(SecureWrappedLLM);
    expect(wrappedModel(secure)).toBe(raw);
    // No second registration: the registry still holds the original model.
    expect(snapshotLlmRegistry().get("__default__")).toBe(raw);
  });

  it("reuses an already-wrapped model", () => {
    const app = makeApp();
    const raw = model();
    const wrapped = registerThroughAppLlm(app, raw);

    const secure = secureLlmForwardedToFactory(app, wrapped);

    expect(secure).toBe(wrapped);
    expect(snapshotLlmRegistry().get("__default__")).toBe(raw);
  });

  it("hands back the raw model in TOOL mode", () => {
    process.env["RUNNER_MODE"] = "tool";
    const app = makeApp();
    const raw = model();
    registerThroughAppLlm(app, raw);

    const secure = secureLlmForwardedToFactory(app, raw);

    expect(secure).toBe(raw);
    expect(snapshotLlmRegistry().get("__default__")).toBe(raw);
  });
});
