import { afterEach, beforeEach, describe, expect, it } from "vitest";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import type { BaseChatModelParams } from "@langchain/core/language_models/chat_models";
import {
  AIMessage,
  HumanMessage,
  type BaseMessage,
} from "@langchain/core/messages";
import type { ChatResult } from "@langchain/core/outputs";
import type { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import type { Runnable } from "@langchain/core/runnables";
import { FilesystemBackend } from "deepagents";

import { createAgentEngineDeepAgent } from "../src/deep_agent.js";

class RecordingChatModel extends BaseChatModel {
  prompts: BaseMessage[][] = [];
  private taskSpawned = false;

  constructor(
    params: BaseChatModelParams = {},
    private readonly spawnSubagent = false,
  ) {
    super(params);
  }

  _llmType(): string {
    return "recording-chat-model";
  }

  override bindTools(): Runnable {
    return this as unknown as Runnable;
  }

  async _generate(
    messages: BaseMessage[],
    _options: this["ParsedCallOptions"],
    _runManager?: CallbackManagerForLLMRun,
  ): Promise<ChatResult> {
    this.prompts.push(messages);
    if (this.spawnSubagent && !this.taskSpawned) {
      this.taskSpawned = true;
      const message = new AIMessage({
        content: "",
        tool_calls: [
          {
            id: "call_reviewer",
            name: "task",
            args: {
              subagent_type: "reviewer",
              description: "Review the change.",
            },
            type: "tool_call",
          },
        ],
      });
      return { generations: [{ message, text: "" }] };
    }
    const message = new AIMessage({ content: "done" });
    return { generations: [{ message, text: "done" }] };
  }
}

let root: string;

beforeEach(() => {
  root = fs.mkdtempSync(path.join(os.tmpdir(), "skills-runtime-"));
});

afterEach(() => {
  fs.rmSync(root, { recursive: true, force: true });
});

describe("createAgentEngineDeepAgent skills", () => {
  it("discovers a skill from a relative parent source directory", async () => {
    const skillsDir = path.join(root, "skills");
    const skillDir = path.join(skillsDir, "security-review");
    fs.mkdirSync(skillDir, { recursive: true });
    fs.writeFileSync(
      path.join(skillDir, "SKILL.md"),
      "---\nname: security-review\ndescription: Security review instructions\n---\n\n# Security review\n",
    );
    const model = new RecordingChatModel();
    const agent = createAgentEngineDeepAgent(
      model,
      new FilesystemBackend({ rootDir: root, virtualMode: false }),
      { skills: ["skills"], skillsBaseDir: root },
    );

    await agent.invoke({ messages: [new HumanMessage("Review this change")] });

    const prompt = model.prompts[0]
      ?.map((message) => JSON.stringify(message.content))
      .join("\n");
    expect(prompt).toContain("## Skills System");
    expect(prompt).toContain("Security review instructions");
    expect(prompt).toContain(path.join(skillDir, "SKILL.md"));
  });

  it("discovers skills configured on a subagent", async () => {
    const skillsDir = path.join(root, "skills");
    const skillDir = path.join(skillsDir, "specialist");
    fs.mkdirSync(skillDir, { recursive: true });
    fs.writeFileSync(
      path.join(skillDir, "SKILL.md"),
      "---\nname: specialist\ndescription: Specialist instructions\n---\n\n# Specialist\n",
    );
    const model = new RecordingChatModel({}, true);
    const agent = createAgentEngineDeepAgent(
      model,
      new FilesystemBackend({ rootDir: root, virtualMode: false }),
      {
        subagents: [
          {
            name: "reviewer",
            description: "Reviews changes",
            systemPrompt: "Review the change.",
            model,
            skills: ["skills"],
          },
        ],
        skillsBaseDir: root,
      },
    );

    await agent.invoke({ messages: [new HumanMessage("Review this change")] });

    const prompts = model.prompts.map((messages) =>
      messages.map((message) => JSON.stringify(message.content)).join("\n"),
    );
    expect(
      prompts.some((prompt) => prompt.includes("Specialist instructions")),
    ).toBe(true);
    expect(
      prompts.some((prompt) =>
        prompt.includes(path.join(skillDir, "SKILL.md")),
      ),
    ).toBe(true);
  });
});
