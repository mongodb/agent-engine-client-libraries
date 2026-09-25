/**
 * Integration coverage for `App.deepAgent()` — the deep-agent / subagent
 * orchestration path.
 *
 * Demonstrates the acceptance criterion (a TS orchestrator wiring in a TS
 * sub-agent) at construction level, plus the two guardrails that keep subagent
 * LLM calls auditable: the `features.deep_agent` gate and string-model
 * rejection. A full deployed orchestrator→subagent run is exercised out of band
 * via `agentengine dev up` (see docs/runner/deep-agents.md) since it needs the
 * runner-base image and a cluster.
 */

import { afterEach, beforeEach, describe, expect, it } from "vitest";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { ChatOpenAI } from "@langchain/openai";
import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import type { BaseChatModelParams } from "@langchain/core/language_models/chat_models";
import { AIMessage, type BaseMessage } from "@langchain/core/messages";
import type { ChatResult } from "@langchain/core/outputs";
import type { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import type { Runnable } from "@langchain/core/runnables";
import {
  getNamedLlm,
  resetLlmRegistry,
} from "@mongodb-js/agent-engine-runner-shared";
import type { AnySubAgent, SubAgent } from "deepagents";

import { App } from "../../src/runtime.js";
import { createAgentEngineDeepAgent } from "../../src/deep_agent.js";

/**
 * Minimal tool-calling fake chat model. LangChain's shipped fakes cannot emit
 * `tool_calls`, which the deepagents `task` dispatch needs. On its first call it
 * spawns the `researcher` sub-agent via the `task` tool; every later call (the
 * sub-agent's own turn, and the parent's wrap-up) returns a plain final
 * message. `bindTools` is a no-op that preserves the scripted behavior.
 */
class ScriptedToolModel extends BaseChatModel {
  calls = 0;
  private taskSpawned = false;

  constructor(params: BaseChatModelParams = {}) {
    super(params);
  }

  _llmType(): string {
    return "scripted-tool-fake";
  }

  override bindTools(): Runnable {
    return this as unknown as Runnable;
  }

  async _generate(
    _messages: BaseMessage[],
    _options: this["ParsedCallOptions"],
    _runManager?: CallbackManagerForLLMRun,
  ): Promise<ChatResult> {
    this.calls += 1;
    if (!this.taskSpawned) {
      this.taskSpawned = true;
      const message = new AIMessage({
        content: "",
        tool_calls: [
          {
            id: "call_task_1",
            name: "task",
            args: {
              subagent_type: "researcher",
              description: "Investigate the topic.",
            },
            type: "tool_call",
          },
        ],
      });
      return { generations: [{ message, text: "" }] };
    }
    const message = new AIMessage({ content: "RESEARCH_DONE" });
    return { generations: [{ message, text: "RESEARCH_DONE" }] };
  }
}

let savedMode: string | undefined;
let savedConfigPath: string | undefined;
let savedSkillsDir: string | undefined;
let tmpDir: string;

/** Write an agent.yaml with the given features block; return its path. */
function writeAgentYaml(features: string): string {
  const p = path.join(tmpDir, "agent.yaml");
  fs.writeFileSync(p, `name: deep-agent-demo\n${features}\n`);
  return p;
}

function orchestratorModel(): ChatOpenAI {
  // Constructed only — never invoked — so no network/key is needed.
  return new ChatOpenAI({ model: "gpt-4o", apiKey: "test-key" });
}

beforeEach(() => {
  savedMode = process.env["RUNNER_MODE"];
  savedConfigPath = process.env["AGENTIC_AGENT_CONFIG_PATH"];
  savedSkillsDir = process.env["AGENTIC_SKILLS_DIR"];
  delete process.env["AGENTIC_SKILLS_DIR"];
  process.env["RUNNER_MODE"] = "aer";
  // The named-LLM registry is process-global; clear it so one test's
  // registration cannot collide with the next.
  resetLlmRegistry();
  tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "deep-agent-demo-"));
});

afterEach(() => {
  if (savedMode === undefined) delete process.env["RUNNER_MODE"];
  else process.env["RUNNER_MODE"] = savedMode;
  if (savedConfigPath === undefined)
    delete process.env["AGENTIC_AGENT_CONFIG_PATH"];
  else process.env["AGENTIC_AGENT_CONFIG_PATH"] = savedConfigPath;
  if (savedSkillsDir === undefined) delete process.env["AGENTIC_SKILLS_DIR"];
  else process.env["AGENTIC_SKILLS_DIR"] = savedSkillsDir;
  fs.rmSync(tmpDir, { recursive: true, force: true });
});

describe("App.deepAgent()", () => {
  it("throws when features.deep_agent is not enabled", () => {
    process.env["AGENTIC_AGENT_CONFIG_PATH"] = writeAgentYaml(
      "features:\n  deep_agent: false",
    );
    const app = new App({ appName: "deep-agent-demo" });
    expect(() => app.deepAgent(orchestratorModel())).toThrow(
      /requires 'features.deep_agent: true'/,
    );
  });

  it("rejects a string-model subagent even with the flag on", () => {
    process.env["AGENTIC_AGENT_CONFIG_PATH"] = writeAgentYaml(
      "features:\n  deep_agent: true",
    );
    const app = new App({ appName: "deep-agent-demo" });
    const badSub = {
      name: "researcher",
      description: "research specialist",
      systemPrompt: "Investigate and report.",
      model: "openai:gpt-4o",
    } as unknown as AnySubAgent;
    // Declare the entrypoint the way a real agent does; the framework's
    // getAgent() evaluates it (inside its internal entrypoint scope) and the
    // validation error propagates out.
    app.entrypoint(() =>
      app.deepAgent(orchestratorModel(), { subagents: [badSub] }),
    );
    expect(() => app.getAgent()).toThrow(/string model spec/);
  });

  it("builds an orchestrator graph with a wrapped sub-agent", () => {
    process.env["AGENTIC_AGENT_CONFIG_PATH"] = writeAgentYaml(
      "features:\n  deep_agent: true",
    );
    const app = new App({ appName: "deep-agent-demo" });
    // Real user shape: app.llm()/app.deepAgent() live inside the
    // @app.entrypoint builder, which the framework evaluates via getAgent().
    app.entrypoint(() => {
      const researcher: AnySubAgent = {
        name: "researcher",
        description: "research specialist",
        systemPrompt: "Investigate and report.",
        // app.llm() wraps the model so the sub-agent's calls route through OE.
        // A distinct llm_id keeps it out of the orchestrator's __default__ slot.
        model: app.llm(
          orchestratorModel(),
          "researcher",
        ) as unknown as SubAgent["model"],
      } as AnySubAgent;

      return app.deepAgent(orchestratorModel(), {
        subagents: [researcher],
        systemPrompt: "You are an orchestrator. Delegate research.",
      });
    });
    expect(app.getAgent()).toBeDefined();
  });

  it("runs the orchestration graph, spawns the sub-agent, and returns a result", async () => {
    // Exercised at the factory level with raw (unwrapped) fakes. `App.deepAgent`
    // wraps the model in `SecureWrappedLLM`, which requires a live
    // execution-context wrapper — that OE-audited path is verified by the
    // secure_llm tests and on the deployed stack. Here we prove the deepagents
    // orchestration our factory builds actually runs: the parent spawns the
    // sub-agent via the `task` tool and the graph completes with output.
    const parentModel = new ScriptedToolModel();
    const subModel = new ScriptedToolModel();
    const researcher = {
      name: "researcher",
      description: "research specialist",
      systemPrompt: "Investigate and report.",
      model: subModel,
    } as unknown as AnySubAgent;

    const graph = createAgentEngineDeepAgent(
      parentModel as unknown as never,
      undefined,
      {
        subagents: [researcher],
        systemPrompt: "You are an orchestrator. Delegate research.",
      },
    ) as unknown as {
      invoke: (input: unknown) => Promise<{ messages: BaseMessage[] }>;
    };

    const result = await graph.invoke({
      messages: [{ role: "user", content: "Go" }],
    });

    // The sub-agent ran (its model was invoked) and the graph produced output.
    expect(subModel.calls).toBeGreaterThan(0);
    expect(result.messages.length).toBeGreaterThan(0);
  });

  it("keeps subagent LLM registrations after wrapping the orchestrator", () => {
    // Regression: deepAgent() must not clear the registry. A subagent model
    // built inline as `app.llm(model, "researcher")` registers during argument
    // evaluation (before the method body runs); it has to survive so the Tool
    // Pod's /invoke_llm can resolve it by id.
    process.env["AGENTIC_AGENT_CONFIG_PATH"] = writeAgentYaml(
      "features:\n  deep_agent: true",
    );
    const app = new App({ appName: "deep-agent-demo" });
    app.entrypoint(() => {
      const researcher: AnySubAgent = {
        name: "researcher",
        description: "research specialist",
        systemPrompt: "Investigate and report.",
        model: app.llm(
          orchestratorModel(),
          "researcher",
        ) as unknown as SubAgent["model"],
      } as AnySubAgent;

      return app.deepAgent(orchestratorModel(), { subagents: [researcher] });
    });
    app.getAgent();

    // Both the subagent id and the orchestrator's __default__ resolve.
    expect(() => getNamedLlm("researcher")).not.toThrow();
    expect(() => getNamedLlm("__default__")).not.toThrow();
  });
});

/**
 * `AGENTIC_SKILLS_DIR` resolution (mirrors Python's test_runtime.py group). The
 * resolver is private; exercise it through a cast rather than building a graph
 * so each throw branch is asserted directly.
 */
describe("App resolveSkillsBaseDir()", () => {
  interface AppWithPrivates {
    resolveSkillsBaseDir(): string | undefined;
  }
  const resolver = (app: App): AppWithPrivates =>
    app as unknown as AppWithPrivates;

  it("defaults to the agent.yaml directory when no override is set", () => {
    const configPath = writeAgentYaml("features:\n  deep_agent: true");
    process.env["AGENTIC_AGENT_CONFIG_PATH"] = configPath;
    const app = new App({ appName: "deep-agent-demo" });
    expect(resolver(app).resolveSkillsBaseDir()).toBe(
      path.dirname(path.resolve(configPath)),
    );
  });

  it("narrows to a source-root-relative override", () => {
    const configPath = writeAgentYaml("features:\n  deep_agent: true");
    process.env["AGENTIC_AGENT_CONFIG_PATH"] = configPath;
    process.env["AGENTIC_SKILLS_DIR"] = "custom-skills";
    const app = new App({ appName: "deep-agent-demo" });
    expect(resolver(app).resolveSkillsBaseDir()).toBe(
      path.resolve(path.dirname(configPath), "custom-skills"),
    );
  });

  it("rejects an absolute AGENTIC_SKILLS_DIR", () => {
    process.env["AGENTIC_AGENT_CONFIG_PATH"] = writeAgentYaml(
      "features:\n  deep_agent: true",
    );
    process.env["AGENTIC_SKILLS_DIR"] = "/etc/skills";
    const app = new App({ appName: "deep-agent-demo" });
    expect(() => resolver(app).resolveSkillsBaseDir()).toThrow(
      /must be relative to the agent source root/,
    );
  });

  it("rejects a traversal above the agent source root", () => {
    process.env["AGENTIC_AGENT_CONFIG_PATH"] = writeAgentYaml(
      "features:\n  deep_agent: true",
    );
    process.env["AGENTIC_SKILLS_DIR"] = "../escape";
    const app = new App({ appName: "deep-agent-demo" });
    expect(() => resolver(app).resolveSkillsBaseDir()).toThrow(
      /must stay within the agent source root/,
    );
  });

  it("rejects a relative override when agent.yaml cannot be located", () => {
    // No config path and a cwd without agent.yaml → agentConfig.path is null.
    delete process.env["AGENTIC_AGENT_CONFIG_PATH"];
    process.env["AGENTIC_SKILLS_DIR"] = "skills";
    const app = new App({ appName: "deep-agent-demo" });
    // Only assert the throw when the harness cwd genuinely has no agent.yaml;
    // otherwise agentConfig.path is set and the override resolves normally.
    if (app.agentConfig.path === null) {
      expect(() => resolver(app).resolveSkillsBaseDir()).toThrow(
        /requires agent\.yaml to determine the agent source root/,
      );
    } else {
      expect(resolver(app).resolveSkillsBaseDir()).toBeDefined();
    }
  });
});
