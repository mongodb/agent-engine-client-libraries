import { afterEach, describe, expect, it, vi } from "vitest";
import { createDeepAgent } from "deepagents";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { createAgentEngineDeepAgent } from "../src/deep_agent.js";

vi.mock("deepagents", () => ({
  createDeepAgent: vi.fn((params: unknown) => params),
}));

describe("skill path resolution", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("preserves an absolute path when a base directory is configured", () => {
    const absolutePath = "/tmp/skills";

    createAgentEngineDeepAgent({} as BaseChatModel, undefined, {
      skills: [absolutePath],
      skillsBaseDir: "/tmp/agent",
    });

    expect(createDeepAgent).toHaveBeenLastCalledWith(
      expect.objectContaining({ skills: [absolutePath] }),
    );
  });

  it("preserves a relative path when no base directory is configured", () => {
    createAgentEngineDeepAgent({} as BaseChatModel, undefined, {
      skills: ["skills"],
    });

    expect(createDeepAgent).toHaveBeenLastCalledWith(
      expect.objectContaining({ skills: ["skills"] }),
    );
  });
});
