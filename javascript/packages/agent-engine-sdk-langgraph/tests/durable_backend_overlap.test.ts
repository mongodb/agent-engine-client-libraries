/**
 * Durable Deep Agent backend operations that overlap: a model's parallel tool
 * calls, a large tool result the filesystem middleware evicts beside a sibling,
 * and parallel subagents loading their skills.
 *
 * The model, tool wrapper, backend, and Deep Agent run for real against a fake
 * OE that records activity outcomes by position and replays them on later
 * attempts, the same boundary a restarted runner sees.
 */

import { create } from "@bufbuild/protobuf";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { tool } from "@langchain/core/tools";
import type { AgentInput, RequestContext } from "@mongodb-js/agent-engine-sdk";
import {
  AttemptContextSchema,
  SecureLLMProxy,
  SecureToolWrapper,
  TenantScopeSchema,
  WorkflowIdentitySchema,
  runWithAttemptContext,
  runWithExecutionContext,
  type CompleteExecutionCommand,
} from "@mongodb-js/agent-engine-runner-shared";
import type { AnySubAgent } from "deepagents";
import { afterEach, describe, expect, it, vi } from "vitest";
import { z } from "zod";

import { LangGraphBaseAgent } from "../src/agent.js";
import { AgentEngineToolPodBackend } from "../src/backends/toolpod.js";
import {
  createAgentEngineDeepAgent,
  type CreateAgentEngineDeepAgentOptions,
} from "../src/deep_agent.js";
import { PlatformCheckpointer } from "../src/platform_checkpointer.js";
import { SecureWrappedLLM } from "../src/secure_llm.js";

const identity = create(WorkflowIdentitySchema, {
  tenantScope: create(TenantScopeSchema, {
    orgId: "org",
    projectId: "project",
    workspaceId: "workspace",
  }),
  sessionId: "session",
  executionId: "execution",
});
const input: AgentInput = { payload: { message: "start" } };
const baseContext = {
  sessionId: "session",
  workspaceId: "workspace",
} as RequestContext;

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

function attempt(attemptId: string, replay: boolean) {
  return create(AttemptContextSchema, {
    attemptId,
    fencingToken: replay ? 8n : 7n,
    replayMode: replay,
    workflowIdentity: identity,
  });
}

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

/** What each Tool Pod handler answers. */
const ABANDONED_MESSAGE =
  "The outcome of this tool call is unknown: the runtime was lost while it " +
  "was running, so it may or may not have taken effect. It was not run again.";

const HANDLER_RESULTS: Record<string, unknown> = {
  filesystem_ls: { entries: [{ path: "/a.txt", is_dir: false }] },
  filesystem_read: { content: "file body" },
  filesystem_write: { path: "/large_tool_results/call_big.txt" },
};

/**
 * OE fake: records activity outcomes by position, replays them, and answers
 * each remote tool only once every call in `heldUntilAllArrive` is in flight,
 * so a call that does not truly overlap its siblings hangs the test. List a
 * name once per call to wait for. Like OE, it settles an activity a lost runner
 * left started as failed instead of running it again.
 */
function fakeOe(heldUntilAllArrive: readonly string[] = []) {
  /** Every position admitted, as `step|path|ordinal` -> activity name. */
  const positions = new Map<string, string>();
  const dispatched: string[] = [];
  /** Each attempt's semantic input at a position, oldest first. */
  const semanticInputs = new Map<string, string[]>();
  const recorded = new Map<string, Record<string, unknown>>();
  const positionByActivity = new Map<string, string>();
  const waiting = [...heldUntilAllArrive];
  let release = () => {};
  const everyoneArrived = new Promise<void>((resolve) => {
    release = resolve;
  });
  if (waiting.length === 0) release();

  vi.stubGlobal(
    "fetch",
    vi.fn(async (request: string | URL | Request, init?: RequestInit) => {
      const url = String(request);
      const body = JSON.parse(String(init?.body)) as Record<string, unknown>;
      if (url.endsWith("/executor/activity/start")) {
        const position = body["position"] as {
          step_ordinal: string;
          operation_path?: { segments?: { name: string; ordinal: string }[] };
          activity_ordinal: string;
        };
        const path = (position.operation_path?.segments ?? [])
          .map((segment) => `${segment.name}/${segment.ordinal}`)
          .join("/");
        const label = `${position.step_ordinal}|${path}|${position.activity_ordinal}`;
        const name = String(body["activity_name"]);
        const previous = positions.get(label);
        // OE rejects a position whose activity changed; the fake must too, or
        // it would replay the wrong tool's result without failing.
        if (previous !== undefined && previous !== name) {
          throw new Error(`nondeterministic: ${label} ${previous} -> ${name}`);
        }
        positions.set(label, name);
        semanticInputs.set(label, [
          ...(semanticInputs.get(label) ?? []),
          JSON.stringify(body["semantic_input"]),
        ]);
        const provenance = {
          workflow_identity: body["workflow_identity"],
          attempt_id: body["attempt_id"],
          fencing_token: body["fencing_token"],
        };
        const outcome = recorded.get(label);
        if (outcome !== undefined) {
          return jsonResponse({ outcome: { ...outcome, ...provenance } });
        }
        if (previous !== undefined) {
          return jsonResponse({
            outcome: {
              ...provenance,
              activity_id: "abandoned",
              outcome_kind: "ACTIVITY_OUTCOME_KIND_FAILED",
              error: {
                code: "WORKFLOW_ERROR_CODE_OUTCOME_UNKNOWN",
                message: ABANDONED_MESSAGE,
              },
            },
          });
        }
        const activityId = `activity-${positionByActivity.size + 1}`;
        positionByActivity.set(activityId, label);
        return jsonResponse({
          activity_context: { ...provenance, activity_id: activityId },
        });
      }
      if (url.endsWith("/executor/activity/outcome")) {
        const label = positionByActivity.get(String(body["activity_id"]));
        if (label === undefined) throw new Error("unknown activity");
        recorded.set(label, body);
        return jsonResponse({});
      }
      if (url.endsWith("/tool/execute")) {
        const toolName = String(body["tool_name"]);
        if (body["is_local"] === true) {
          return jsonResponse({ proceed: true, route_to: "callback" });
        }
        dispatched.push(toolName);
        const arrived = waiting.indexOf(toolName);
        if (arrived !== -1) waiting.splice(arrived, 1);
        if (waiting.length === 0) release();
        await everyoneArrived;
        return jsonResponse({
          proceed: true,
          status: "success",
          result: HANDLER_RESULTS[toolName],
        });
      }
      if (url.endsWith("/tool/result")) return jsonResponse({});
      throw new Error(`unexpected URL: ${url}`);
    }),
  );
  return {
    positions,
    dispatched,
    semanticInputs,
    /**
     * Rewind to the moment a runner died with `name` in flight: it stays started
     * with no outcome, and nothing that began after it ever happened.
     */
    loseRunnerDuring(name: string): void {
      const labels = [...positions.keys()];
      const inFlight = labels.findIndex(
        (label) => positions.get(label) === name,
      );
      recorded.delete(labels[inFlight] as string);
      for (const label of labels.slice(inFlight + 1)) {
        positions.delete(label);
        recorded.delete(label);
      }
    },
  };
}

interface ScriptedCall {
  readonly id: string;
  readonly name: string;
  readonly args?: Record<string, unknown>;
}

/**
 * A durable model that makes `calls` on its first turn and then finishes. The
 * real LLM proxy records the turn, so each attempt preallocates the calls.
 */
function durableModel(calls: readonly ScriptedCall[]) {
  const toolResults: string[] = [];
  vi.spyOn(
    SecureLLMProxy.prototype as unknown as {
      streamLlm: (request: unknown) => AsyncGenerator<unknown>;
    },
    "streamLlm",
  ).mockImplementation(async function* (request: unknown) {
    const messages = (
      request as { messages: { role?: string; content?: unknown }[] }
    ).messages;
    if (messages.at(-1)?.role === "tool") {
      for (const message of messages) {
        if (message.role === "tool") toolResults.push(String(message.content));
      }
      yield { content: "finished" };
      return;
    }
    yield {
      toolCalls: calls.map((call, index) => ({
        id: call.id,
        name: call.name,
        args: JSON.stringify(call.args ?? {}),
        index,
      })),
    };
  });
  const holder = { wrapper: new SecureToolWrapper("http://oe", "execution") };
  const llm = new SecureWrappedLLM(
    {
      model: "test-model",
      stream: () => undefined,
    } as unknown as BaseChatModel,
    () => holder.wrapper,
  );
  return { llm, holder, toolResults };
}

function platformSaver() {
  const completeExecution = vi.fn(
    async (_command: CompleteExecutionCommand) => undefined,
  );
  const saver = new PlatformCheckpointer({
    native: null,
    clientFactory: () => ({
      finalizeStep: vi.fn(async () => []),
      completeExecution,
    }),
  });
  return { saver, completeExecution };
}

/**
 * Hold each backend tool until the one before it in `order` has been admitted,
 * so a replacement attempt can reach OE in the opposite order.
 */
function admitInOrder(order: readonly string[]): void {
  const original = SecureToolWrapper.prototype.executeTool;
  const admitted = new Map(
    order.map((name) => {
      let resolve = () => {};
      const promise = new Promise<void>((done) => {
        resolve = done;
      });
      return [name, { promise, resolve }] as const;
    }),
  );
  vi.spyOn(SecureToolWrapper.prototype, "executeTool").mockImplementation(
    async function (this: SecureToolWrapper, name, args, options) {
      const index = order.indexOf(name);
      const previous = order[index - 1];
      if (previous !== undefined) await admitted.get(previous)?.promise;
      const running = original.call(this, name, args, options);
      // The wrapper allocates a position before its first await, so the next
      // tool in order may start as soon as this call returned its promise.
      admitted.get(name)?.resolve();
      return running;
    },
  );
}

/** One attempt on a fresh wrapper, as a restarted runner would make it. */
async function runAttempt(
  agent: LangGraphBaseAgent,
  holder: { wrapper: SecureToolWrapper },
  current: ReturnType<typeof attempt>,
): Promise<void> {
  holder.wrapper = new SecureToolWrapper("http://oe", "execution");
  await runWithExecutionContext(
    {
      executionId: "execution",
      wrapper: holder.wrapper,
      oeUrl: "http://oe",
      userId: "user-1",
    },
    () =>
      runWithAttemptContext(current, () => agent.invoke(baseContext, input)),
  );
}

function deepAgent(
  llm: SecureWrappedLLM,
  saver: PlatformCheckpointer,
  options: Pick<CreateAgentEngineDeepAgentOptions, "tools" | "subagents"> = {},
): LangGraphBaseAgent {
  const graph = createAgentEngineDeepAgent(
    llm as unknown as BaseChatModel,
    new AgentEngineToolPodBackend(),
    {
      ...options,
      systemPrompt: "Use your tools.",
      checkpointer: saver as never,
    },
  );
  return new LangGraphBaseAgent(graph as never);
}

const PARALLEL_READS: readonly ScriptedCall[] = [
  { id: "call_ls", name: "ls", args: { path: "/" } },
  { id: "call_read", name: "read_file", args: { file_path: "/a.txt" } },
];

/** The admitted positions of `name`, without their step ordinal. */
function positionsOf(
  positions: ReadonlyMap<string, string>,
  name: string,
): string[] {
  return [...positions]
    .filter(([, activity]) => activity === name)
    .map(([label]) => label.split("|").slice(1).join("#"));
}

describe("durable Deep Agent backend operations that overlap", () => {
  it("runs a model's parallel file tools at positions from their own calls", async () => {
    const oe = fakeOe(["filesystem_ls", "filesystem_read"]);
    const { llm, holder, toolResults } = durableModel(PARALLEL_READS);
    const { saver, completeExecution } = platformSaver();

    await runAttempt(
      deepAgent(llm, saver),
      holder,
      attempt("attempt-1", false),
    );

    expect(completeExecution).toHaveBeenCalledTimes(1);
    // Each call's own result reaches the model.
    expect(toolResults).toHaveLength(2);
    expect(toolResults.join("\n")).toContain("/a.txt");
    expect(toolResults.join("\n")).toContain("file body");
    // The LLM activity takes ordinal 1, so the batch takes 2 and 3 in the
    // order the model listed its calls.
    expect(positionsOf(oe.positions, "filesystem_ls")).toEqual([
      "agent/1/tool:effects/2#1",
    ]);
    expect(positionsOf(oe.positions, "filesystem_read")).toEqual([
      "agent/1/tool:effects/3#1",
    ]);
  });

  it("sends the model the same tool message IDs when a replacement attempt rebuilds the turn", async () => {
    const oe = fakeOe();
    const { llm, holder } = durableModel(PARALLEL_READS);
    const { saver } = platformSaver();

    await runAttempt(
      deepAgent(llm, saver),
      holder,
      attempt("attempt-1", false),
    );
    await runAttempt(deepAgent(llm, saver), holder, attempt("attempt-2", true));

    // The LLM call after the tools carries the messages they built.
    const [first, replay] =
      [...oe.semanticInputs].find(
        ([label]) =>
          oe.positions.get(label) === "test-model" && label.endsWith("|4"),
      )?.[1] ?? [];
    const toolMessageIds = (input?: string) =>
      [...(input ?? "").matchAll(/"id":"(durable-message:[^"]+)"/g)].map(
        (match) => match[1],
      );
    expect(new Set(toolMessageIds(first)).size).toBe(2);
    expect(toolMessageIds(replay)).toEqual(toolMessageIds(first));
  });

  it("replays recorded results when a restarted runner reaches OE in the other order", async () => {
    const oe = fakeOe(["filesystem_ls", "filesystem_read"]);
    const { llm, holder } = durableModel(PARALLEL_READS);
    const { saver, completeExecution } = platformSaver();
    const agent = deepAgent(llm, saver);

    admitInOrder(["filesystem_ls", "filesystem_read"]);
    await runAttempt(agent, holder, attempt("attempt-1", false));
    vi.mocked(SecureToolWrapper.prototype.executeTool).mockRestore();

    admitInOrder(["filesystem_read", "filesystem_ls"]);
    await runAttempt(agent, holder, attempt("attempt-2", true));

    expect([...oe.dispatched].sort()).toEqual([
      "filesystem_ls",
      "filesystem_read",
    ]);
    expect(completeExecution).toHaveBeenCalledTimes(2);
  });

  it("tells the model not to retry an operation a lost runner left in flight", async () => {
    const oe = fakeOe(["filesystem_ls", "filesystem_read"]);
    const { llm, holder, toolResults } = durableModel(PARALLEL_READS);
    const { saver, completeExecution } = platformSaver();
    const agent = deepAgent(llm, saver);

    await runAttempt(agent, holder, attempt("attempt-1", false));
    oe.loseRunnerDuring("filesystem_read");
    toolResults.length = 0;
    await runAttempt(agent, holder, attempt("attempt-2", true));

    // Neither operation runs again: one replays, the other is settled as failed.
    expect([...oe.dispatched].sort()).toEqual([
      "filesystem_ls",
      "filesystem_read",
    ]);
    expect(completeExecution).toHaveBeenCalledTimes(2);
    expect(toolResults.join("\n")).toContain(
      `[NON-RETRYABLE] ${ABANDONED_MESSAGE}`,
    );
  });

  it("still refuses an operation inside a recorded tool's body", async () => {
    const oe = fakeOe();
    const { llm, holder } = durableModel([
      { id: "call_audited", name: "audited" },
    ]);
    const { saver } = platformSaver();
    const backend = new AgentEngineToolPodBackend();
    // On replay this body would not run, so OE would never see its operation
    // and would refuse the step.
    const audited = tool(
      (_args, config: { toolCall?: { id?: string } }) =>
        holder.wrapper.executeTool(
          "audited",
          {},
          {
            toolCallId: config.toolCall?.id,
            isLocal: true,
            localExecutor: () => backend.ls("/"),
          },
        ),
      {
        name: "audited",
        description: "A tool whose body uses the backend.",
        schema: z.object({}),
      },
    );

    await expect(
      runAttempt(
        deepAgent(llm, saver, { tools: [audited] }),
        holder,
        attempt("attempt-1", false),
      ),
    ).rejects.toThrow("this attempt already has an in-flight activity");
    expect(oe.dispatched).toEqual([]);
  });

  it("evicts a large tool result while a sibling's operation is in flight", async () => {
    const oe = fakeOe(["filesystem_write", "filesystem_ls"]);
    const { llm, holder } = durableModel([
      { id: "call_big", name: "big_report" },
      { id: "call_ls", name: "ls", args: { path: "/" } },
    ]);
    const { saver, completeExecution } = platformSaver();
    const bigReport = tool(async () => "x".repeat(100_000), {
      name: "big_report",
      description: "Return a report too large for the conversation.",
      schema: z.object({}),
    });

    await runAttempt(
      deepAgent(llm, saver, { tools: [bigReport] }),
      holder,
      attempt("attempt-1", false),
    );

    expect(completeExecution).toHaveBeenCalledTimes(1);
    expect(positionsOf(oe.positions, "filesystem_write")).toEqual([
      "agent/1/tool:effects/2#1",
    ]);
  });

  it("lets parallel subagents load their skills at the same time", async () => {
    const oe = fakeOe(["filesystem_ls", "filesystem_ls"]);
    const { llm, holder } = durableModel([
      {
        id: "task_a",
        name: "task",
        args: { subagent_type: "researcher", description: "first" },
      },
      {
        id: "task_b",
        name: "task",
        args: { subagent_type: "researcher", description: "second" },
      },
    ]);
    const { saver, completeExecution } = platformSaver();
    const researcher = {
      name: "researcher",
      description: "research specialist",
      systemPrompt: "Investigate and report.",
      model: llm,
      tools: [],
      skills: ["/skills/"],
    } as unknown as AnySubAgent;

    await runAttempt(
      deepAgent(llm, saver, { subagents: [researcher] }),
      holder,
      attempt("attempt-1", false),
    );

    expect(completeExecution).toHaveBeenCalledTimes(1);
    expect(positionsOf(oe.positions, "filesystem_ls").sort()).toEqual([
      "agent/1/researcher/1#1",
      "agent/1/researcher/2#1",
    ]);
  });

  it("keeps a lone tool call's operation at the position it always recorded", async () => {
    const oe = fakeOe();
    const { llm, holder } = durableModel([
      { id: "call_ls", name: "ls", args: { path: "/" } },
    ]);
    const { saver } = platformSaver();

    await runAttempt(
      deepAgent(llm, saver),
      holder,
      attempt("attempt-1", false),
    );

    // The LLM activity takes ordinal 1, its one tool call 2, and the operation
    // the next free ordinal: a turn recorded before this position existed
    // still replays.
    expect(positionsOf(oe.positions, "filesystem_ls")).toEqual(["agent/1#3"]);
  });
});
