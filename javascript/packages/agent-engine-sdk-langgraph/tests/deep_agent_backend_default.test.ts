/**
 * `App.deepAgent()` backend defaulting.
 *
 * Verifies the acceptance criterion: deepAgent() defaults the backend to
 * AgentEngineToolPodBackend (so filesystem/shell ops are OE-audited), and a
 * caller-supplied backend still overrides. `createAgentEngineDeepAgent` is mocked
 * so we assert the forwarded backend argument without building a real graph.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { ChatOpenAI } from "@langchain/openai";
import type { AnyBackendProtocol } from "deepagents";

const { createAgentEngineDeepAgent } = vi.hoisted(() => ({
  createAgentEngineDeepAgent: vi.fn(),
}));
vi.mock("../src/deep_agent.js", () => ({ createAgentEngineDeepAgent }));

import {
  entrypointScope,
  resetLlmRegistry,
} from "@mongodb-js/agent-engine-runner-shared";
import { App } from "../src/runtime.js";
import { AgentEngineToolPodBackend } from "../src/backends/toolpod.js";

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

/** The backend is the second positional arg to createAgentEngineDeepAgent. */
function forwardedBackend(): unknown {
  return createAgentEngineDeepAgent.mock.calls[0]?.[1];
}

beforeEach(() => {
  savedMode = process.env["RUNNER_MODE"];
  savedConfigPath = process.env["AGENTIC_AGENT_CONFIG_PATH"];
  process.env["RUNNER_MODE"] = "aer";
  // The named-LLM registry is process-global; clear it so one test's
  // __default__ registration doesn't collide with the next.
  resetLlmRegistry();
  createAgentEngineDeepAgent.mockReset();
  tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "deep-agent-backend-"));
});

afterEach(() => {
  if (savedMode === undefined) delete process.env["RUNNER_MODE"];
  else process.env["RUNNER_MODE"] = savedMode;
  if (savedConfigPath === undefined)
    delete process.env["AGENTIC_AGENT_CONFIG_PATH"];
  else process.env["AGENTIC_AGENT_CONFIG_PATH"] = savedConfigPath;
  fs.rmSync(tmpDir, { recursive: true, force: true });
});

describe("App.deepAgent() backend defaulting", () => {
  it("defaults to AgentEngineToolPodBackend when none is supplied", () => {
    entrypointScope(() => makeApp().deepAgent(model()));
    expect(forwardedBackend()).toBeInstanceOf(AgentEngineToolPodBackend);
  });

  it("forwards a caller-supplied backend unchanged", () => {
    const custom = { id: "custom" } as unknown as AnyBackendProtocol;
    entrypointScope(() => makeApp().deepAgent(model(), { backend: custom }));
    expect(forwardedBackend()).toBe(custom);
  });
});
