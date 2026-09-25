import { describe, it, expect } from "vitest";
import { AgentEngineToolSandboxBackend } from "../src/backends/tool_sandbox.js";
import { AgentEngineToolPodBackend } from "../src/backends/toolpod.js";

describe("AgentEngineToolSandboxBackend", () => {
  it("is the same class as AgentEngineToolPodBackend", () => {
    expect(AgentEngineToolSandboxBackend).toBe(AgentEngineToolPodBackend);
  });

  it("has the correct id property", () => {
    const backend = new AgentEngineToolSandboxBackend();
    expect(backend.id).toBe("agent-engine-toolpod");
  });

  it("can be imported from index", async () => {
    const mod = await import("../src/index.js");
    expect(mod.AgentEngineToolSandboxBackend).toBeDefined();
    expect(mod.AgentEngineToolSandboxBackend).toBe(AgentEngineToolPodBackend);
  });
});
