/**
 * A platform failure during a tool call ends the turn.
 *
 * The stock ToolNode turns every error a tool throws into a ToolMessage, so
 * these tests run a real graph and check that the model never reads a
 * platform failure: the checkpoint gate refuses to commit the tool step, the
 * model gate refuses the next model call, and the adapter refuses to report a
 * turn that ends after its tools step as completed.
 */

import { create, toJson } from "@bufbuild/protobuf";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { AIMessage, ToolMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import {
  END,
  MemorySaver,
  MessagesAnnotation,
  START,
  StateGraph,
} from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import {
  ActivityContextSchema,
  AttemptContextSchema,
  SecureToolWrapper,
  TenantScopeSchema,
  ToolPlatformError,
  WorkflowIdentitySchema,
  getCurrentWrapper,
  runWithAttemptContext,
  runWithExecutionContext,
  type FinalizeStepCommand,
  type StepActivityEntry,
} from "@mongodb-js/agent-engine-runner-shared";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { z } from "zod";

import { LangGraphBaseAgent } from "../src/agent.js";
import { PlatformCheckpointer } from "../src/platform_checkpointer.js";
import { SecureWrappedLLM, type SecureLLMWrapper } from "../src/secure_llm.js";

const identity = create(WorkflowIdentitySchema, {
  tenantScope: create(TenantScopeSchema, {
    orgId: "org",
    projectId: "project",
    workspaceId: "workspace",
  }),
  sessionId: "session",
  executionId: "execution",
});

const attempt = create(AttemptContextSchema, {
  attemptId: "attempt-1",
  fencingToken: 7n,
  workflowIdentity: identity,
});

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function dispatchResponse(activityId: string): Response {
  const context = create(ActivityContextSchema, {
    workflowIdentity: identity,
    activityId,
    attemptId: attempt.attemptId,
    fencingToken: attempt.fencingToken,
  });
  return jsonResponse({
    activity_context: toJson(ActivityContextSchema, context, {
      useProtoFieldName: true,
    }),
  });
}

interface OeRequests {
  /** `tool_name` of every `/tool/execute` request; a model call is `invoke_llm`. */
  toolNames: string[];
  /** `kind` of every durable activity start. */
  activityKinds: string[];
  /** Tool calls only, excluding model calls. */
  readonly toolCalls: number;
}

/**
 * Stub OE: `/tool/execute` answers with `toolReply`, and the durable activity
 * endpoints accept every call. Records what each request asked for.
 */
function stubOe(toolReply: Record<string, unknown>): OeRequests {
  const requests: OeRequests = {
    toolNames: [],
    activityKinds: [],
    get toolCalls() {
      return this.toolNames.filter((name) => name !== "invoke_llm").length;
    },
  };
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
      const url = String(input);
      const body =
        typeof init?.body === "string"
          ? (JSON.parse(init.body) as Record<string, unknown>)
          : {};
      if (url.endsWith("/executor/activity/start")) {
        requests.activityKinds.push(String(body["kind"]));
        return dispatchResponse("issue-refund");
      }
      if (url.endsWith("/tool/execute")) {
        requests.toolNames.push(String(body["tool_name"]));
        return jsonResponse(toolReply);
      }
      if (url.endsWith("/executor/activity/outcome")) {
        return jsonResponse({});
      }
      throw new Error(`unexpected URL: ${url}`);
    }),
  );
  return requests;
}

/** No model request reached OE, neither a model call nor a durable model activity. */
function expectNoModelCall(oe: OeRequests): void {
  expect(oe.toolNames).not.toContain("invoke_llm");
  expect(oe.activityKinds.filter((kind) => /LLM/i.test(kind))).toEqual([]);
}

const issueRefund = tool(
  async () =>
    (getCurrentWrapper() as SecureToolWrapper).executeTool("issue_refund", {}),
  {
    name: "issue_refund",
    description: "Issue a refund.",
    schema: z.object({}),
  },
);

const toolCallInput = () => ({
  messages: [
    new AIMessage({
      content: "",
      id: "assistant-plan",
      tool_calls: [{ name: "issue_refund", args: {}, id: "tool-call-1" }],
    }),
  ],
});

/** tools -> respond, where `respond` is the graph's next step. */
function toolThenRespond(
  respond: (
    state: typeof MessagesAnnotation.State,
  ) => Promise<{ messages: AIMessage[] }>,
) {
  return new StateGraph(MessagesAnnotation)
    .addNode("tools", new ToolNode([issueRefund]))
    .addNode("respond", respond)
    .addEdge(START, "tools")
    .addEdge("tools", "respond")
    .addEdge("respond", END);
}

function durableSaver() {
  const finalizeStep = vi.fn(
    async (_command: FinalizeStepCommand): Promise<StepActivityEntry[]> => [],
  );
  const saver = new PlatformCheckpointer({
    native: new MemorySaver(),
    clientFactory: () => ({
      finalizeStep,
      completeExecution: vi.fn(async () => undefined),
    }),
  });
  return { saver, finalizeStep };
}

function inExecution<T>(wrapper: SecureToolWrapper, fn: () => Promise<T>) {
  return runWithExecutionContext(
    { executionId: "execution", wrapper, oeUrl: "http://oe" },
    fn,
  );
}

beforeEach(() => {
  if (!process.env["RUNNER_MODE"]) process.env["RUNNER_MODE"] = "aer";
});

afterEach(() => {
  vi.unstubAllGlobals();
});

/**
 * A `respond` node making the next model call through SecureWrappedLLM, which
 * sends it to OE as an `invoke_llm` request.
 */
function modelRespond() {
  const inner = {
    bindTools: vi.fn(),
    invoke: vi.fn(),
    model: "gpt-4o",
  } as unknown as BaseChatModel;
  const llm = new SecureWrappedLLM(
    inner,
    () => getCurrentWrapper() as SecureLLMWrapper | null,
  );
  return async () => ({
    messages: [(await llm.invoke("next")) as AIMessage],
  });
}

/** plan -> tools -> END: the turn's last step is the tool call. */
function planThenTools() {
  return new StateGraph(MessagesAnnotation)
    .addNode("plan", async () => toolCallInput())
    .addNode("tools", new ToolNode([issueRefund]))
    .addEdge(START, "plan")
    .addEdge("plan", "tools")
    .addEdge("tools", END);
}

const turn = {
  payload: { message: "refund the order", session_id: "session" },
};

describe("a platform failure during a tool call", () => {
  it("ends a durable turn without a model call or a committed tool step", async () => {
    const oe = stubOe({ proceed: true, route_to: "elsewhere" });
    const respond = modelRespond();
    const { saver, finalizeStep } = durableSaver();
    const toolCallsAtFinalize: number[] = [];
    finalizeStep.mockImplementation(async () => {
      toolCallsAtFinalize.push(oe.toolCalls);
      return [];
    });
    const graph = toolThenRespond(respond).compile({
      checkpointer: saver as never,
    });
    const wrapper = new SecureToolWrapper("http://oe", "execution");

    const error = await inExecution(wrapper, () =>
      runWithAttemptContext(attempt, () =>
        graph.invoke(toolCallInput(), {
          configurable: { thread_id: "thread" },
        }),
      ),
    ).catch((caught: unknown) => caught);

    expect(error).toBeInstanceOf(ToolPlatformError);
    expect(oe.toolNames).toEqual(["issue_refund"]);
    expectNoModelCall(oe);
    // LangGraph.js saves checkpoints in the background, so the next node may
    // start before the save is refused; the step holding the failure is
    // still never committed.
    expect(toolCallsAtFinalize.every((calls) => calls === 0)).toBe(true);
  });

  it("ends a native turn without a model call or a saved tool step", async () => {
    const oe = stubOe({ proceed: true, route_to: "elsewhere" });
    const native = new MemorySaver();
    const checkpointer = new PlatformCheckpointer({ native });
    const graph = toolThenRespond(modelRespond()).compile({
      checkpointer: checkpointer as never,
    });
    const wrapper = new SecureToolWrapper("http://oe", "execution");
    const config = { configurable: { thread_id: "thread" } };

    const error = await inExecution(wrapper, () =>
      graph.invoke(toolCallInput(), config),
    ).catch((caught: unknown) => caught);

    expect(error).toBeInstanceOf(ToolPlatformError);
    expect((error as Error).message).toMatch(/Unexpected OE tool route/);
    expect(oe.toolNames).toEqual(["issue_refund"]);
    expectNoModelCall(oe);
    // A retry resumes from the last saved step and restores the pending
    // writes of the tasks that finished in it; neither may hold the failure.
    const saved = await native.getTuple(config);
    const messages = (saved?.checkpoint.channel_values["messages"] ??
      []) as unknown[];
    expect(messages.some((message) => message instanceof ToolMessage)).toBe(
      false,
    );
    expect(
      (saved?.pendingWrites ?? []).some(
        ([, , value]) => value instanceof ToolMessage,
      ),
    ).toBe(false);

    // The retry runs under a fresh execution, so the latch no longer stands
    // in the way; the model must still never read the original failure.
    stubOe({ proceed: true, status: "success", result: "refunded" });
    const seen: string[] = [];
    const retry = toolThenRespond(async (state) => {
      for (const message of state.messages) {
        if (message instanceof ToolMessage) seen.push(String(message.content));
      }
      return { messages: [new AIMessage({ content: "done" })] };
    }).compile({ checkpointer: checkpointer as never });

    await inExecution(new SecureToolWrapper("http://oe", "execution"), () =>
      retry.invoke(null, config),
    );

    expect(seen.length).toBeGreaterThan(0);
    expect(seen.join("\n")).not.toMatch(/Unexpected OE tool route/);
  });

  it.each([
    [
      "invoked",
      async (agent: LangGraphBaseAgent) => {
        await agent.invoke({} as never, turn);
      },
    ],
    [
      "streamed",
      async (agent: LangGraphBaseAgent) => {
        for await (const _event of agent.stream({} as never, turn)) {
          // drain
        }
      },
    ],
  ])(
    "ends a %s native turn whose last step is the tool call",
    async (_mode, run) => {
      stubOe({ proceed: true, route_to: "elsewhere" });
      // No checkpointer, so no save can refuse the step: only the adapter
      // stands between the reported failure and a completed turn.
      const agent = new LangGraphBaseAgent(planThenTools().compile() as never);
      const wrapper = new SecureToolWrapper("http://oe", "execution");

      const error = await inExecution(wrapper, () => run(agent)).catch(
        (caught: unknown) => caught,
      );

      expect(error).toBeInstanceOf(ToolPlatformError);
    },
  );

  it("leaves the tool's own failure to ToolNode, which reports it to the model", async () => {
    stubOe({ proceed: true, status: "error", error: "refund service down" });
    const seen: string[] = [];
    const respond = vi.fn(async (state: typeof MessagesAnnotation.State) => {
      const last = state.messages.at(-1);
      if (last instanceof ToolMessage) seen.push(String(last.content));
      return { messages: [new AIMessage({ content: "sorry" })] };
    });
    const { saver, finalizeStep } = durableSaver();
    const graph = toolThenRespond(respond).compile({
      checkpointer: saver as never,
    });
    const wrapper = new SecureToolWrapper("http://oe", "execution");

    await inExecution(wrapper, () =>
      runWithAttemptContext(attempt, () =>
        graph.invoke(toolCallInput(), {
          configurable: { thread_id: "thread" },
        }),
      ),
    );

    expect(seen).toHaveLength(1);
    expect(seen[0]).toContain("refund service down");
    expect(finalizeStep).toHaveBeenCalled();
    expect(() => wrapper.throwIfPlatformFailed()).not.toThrow();
  });
});
