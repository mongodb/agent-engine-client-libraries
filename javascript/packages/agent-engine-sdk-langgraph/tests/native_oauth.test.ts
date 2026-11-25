import { AIMessage, ToolMessage } from "@langchain/core/messages";
import { tool as lcTool } from "@langchain/core/tools";
import {
  Command,
  END,
  MemorySaver,
  MessagesAnnotation,
  START,
  StateGraph,
  interrupt,
  isGraphBubbleUp,
} from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import {
  ExternalAPICallError,
  SecureToolWrapper,
  ToolPlatformError,
  createSecureToolFunction,
  registerSuspendHandler,
  resetHooks,
  runWithExecutionContext,
} from "@mongodb-js/agent-engine-runner-shared";
import { afterEach, describe, expect, it, vi } from "vitest";
import { z } from "zod";

import {
  withDurableToolResultIdentity,
  withOAuthTask,
} from "../src/durable_tools.js";

type Transport = "invoke" | "stream";

interface PendingInterrupt {
  readonly id: string;
  readonly value: Record<string, unknown>;
}

const AUTH_FAILED = {
  provider_type: "github",
  classification: "AUTH_FAILED",
  http_status: 401,
  retryable: false,
};

function fakeOe(
  connected: string[] = [],
  failing: string[] = [],
  toolApiError?: typeof AUTH_FAILED,
) {
  const consented = new Set(connected);
  const providerCalls: string[] = [];
  const requests: string[] = [];
  const steps: number[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (_url: unknown, init?: RequestInit) => {
      const body = JSON.parse(String(init?.body)) as {
        arguments: { owner: string };
        step_number: number;
      };
      steps.push(body.step_number);
      const { owner } = body.arguments;
      requests.push(owner);
      const reply = consented.has(owner)
        ? (providerCalls.push(owner),
          failing.includes(owner)
            ? {
                proceed: true,
                status: "error",
                error: `GitHub has no user ${owner}`,
                ...(toolApiError ? { tool_api_error: toolApiError } : {}),
                duration_ms: 1,
              }
            : {
                proceed: true,
                status: "success",
                result: JSON.stringify({ repos: [`${owner}-repo`] }),
                duration_ms: 1,
              })
        : {
            proceed: false,
            status: "suspend",
            latest_step_number: body.step_number,
            elicitation: {
              elicitation_id: `consent-${owner}`,
              authorization_url: `https://idp.example/authorize?owner=${owner}`,
              message: "Connect GitHub",
              created: true,
            },
          };
      return new Response(JSON.stringify(reply), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }),
  );
  return {
    consent: (owner: string) => consented.add(owner),
    providerCalls,
    requests,
    steps,
  };
}

function oauthTool(name = "list_repositories") {
  const secure = createSecureToolFunction(
    { invoke: () => Promise.reject(new Error("OE runs provider tools")) },
    name,
    false,
    {
      isLocal: false,
      provider: "github",
      scopes: ["repo"],
      responseFormat: "content_and_artifact",
      isFrameworkControlFlow: isGraphBubbleUp,
    },
  );
  return lcTool(
    withDurableToolResultIdentity(withOAuthTask(secure, name), name) as never,
    {
      name,
      description: `${name} tool`,
      schema: z.object({ owner: z.string() }),
      responseFormat: "content_and_artifact",
    },
  );
}

function toolCalls(...owners: string[]): AIMessage {
  return new AIMessage({
    content: "",
    tool_calls: owners.map((owner) => ({
      id: `call-${owner}`,
      name: "list_repositories",
      args: { owner },
    })),
  });
}

const wrappers = new Map<string, SecureToolWrapper>();

async function run(
  graph: ReturnType<typeof compileTools>,
  input: unknown,
  threadId: string,
  transport: Transport,
): Promise<{
  messages: ToolMessage[];
  interrupts: PendingInterrupt[];
}> {
  const config = {
    configurable: {
      thread_id: threadId,
      request_context: { sessionId: threadId },
    },
  };
  const wrapper =
    wrappers.get(threadId) ??
    new SecureToolWrapper("http://oe:8000", `exec-${threadId}`);
  wrappers.set(threadId, wrapper);
  await runWithExecutionContext(
    { executionId: `exec-${threadId}`, wrapper, oeUrl: "http://oe:8000" },
    async () => {
      if (transport === "invoke") {
        await graph.invoke(input as never, config);
        return;
      }
      for await (const _ of await graph.stream(input as never, config));
    },
  );
  const state = await graph.getState(config);
  return {
    messages: (state.values as { messages: unknown[] }).messages.filter(
      (message): message is ToolMessage => message instanceof ToolMessage,
    ),
    interrupts: state.tasks.flatMap(
      (task) => task.interrupts as unknown as PendingInterrupt[],
    ),
  };
}

function compileTools(checkpointer = new MemorySaver()) {
  return new StateGraph(MessagesAnnotation)
    .addNode("tools", new ToolNode([oauthTool()]))
    .addEdge(START, "tools")
    .addEdge("tools", END)
    .compile({ checkpointer });
}

class ChildWriteFailingSaver extends MemorySaver {
  constructor(private readonly failing: () => boolean) {
    super();
  }

  private check(config: { configurable?: Record<string, unknown> }): void {
    const namespace = String(config.configurable?.["checkpoint_ns"] ?? "");
    if (this.failing() && namespace.includes("delegated_oauth")) {
      throw new Error("checkpoint write failed");
    }
  }

  override async put(
    ...args: Parameters<MemorySaver["put"]>
  ): ReturnType<MemorySaver["put"]> {
    this.check(args[0]);
    return super.put(...args);
  }

  override async putWrites(
    ...args: Parameters<MemorySaver["putWrites"]>
  ): ReturnType<MemorySaver["putWrites"]> {
    this.check(args[0]);
    return super.putWrites(...args);
  }
}

function onlyInterrupt(interrupts: PendingInterrupt[]): PendingInterrupt {
  expect(interrupts).toHaveLength(1);
  return interrupts[0] as PendingInterrupt;
}

function contents(messages: ToolMessage[]): Record<string, unknown> {
  return Object.fromEntries(
    messages.map((message) => [message.tool_call_id, message.content]),
  );
}

describe("delegated OAuth on native checkpoints", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    resetHooks();
  });

  it.each([
    { transport: "invoke", consented: true },
    { transport: "stream", consented: true },
    { transport: "invoke", consented: false },
    { transport: "stream", consented: false },
  ] as const)(
    "continues after consent with the API result, or fails before the provider ($transport, consented: $consented)",
    async ({ transport, consented }) => {
      registerSuspendHandler(interrupt as never);
      const oe = fakeOe();
      const graph = compileTools();
      const thread = `continue-${transport}-${consented}`;

      const paused = await run(
        graph,
        { messages: [toolCalls("alice")] },
        thread,
        transport,
      );
      const pending = onlyInterrupt(paused.interrupts);
      expect(pending.value).toMatchObject({
        kind: "oauth_authorization",
        suspend_reason: "authorization_required",
        suspend_context: { elicitation_id: "consent-alice" },
      });

      if (consented) oe.consent("alice");
      const resumed = await run(
        graph,
        new Command({ resume: { [pending.id]: { confirmed: true } } }),
        thread,
        transport,
      );

      expect(resumed.interrupts).toEqual([]);
      expect(contents(resumed.messages)).toEqual({
        "call-alice": consented
          ? JSON.stringify({ repos: ["alice-repo"] })
          : expect.stringContaining(
              "OAuth authorization is still required after continuation",
            ),
      });
      expect(oe.providerCalls).toEqual(consented ? ["alice"] : []);
      expect(new Set(oe.steps).size).toBe(oe.steps.length);
    },
  );

  it("returns a call's stored result when its node runs again, and keeps the consent answer from a later wait", async () => {
    registerSuspendHandler(interrupt as never);
    const oe = fakeOe();
    const repositories = oauthTool();
    const answers: unknown[] = [];
    const graph = new StateGraph(MessagesAnnotation)
      .addNode("ask", async () => {
        const result = (await repositories.invoke({
          id: "call-alice",
          name: "list_repositories",
          args: { owner: "alice" },
          type: "tool_call",
        })) as ToolMessage;
        answers.push(interrupt({ question: "Share these repositories?" }));
        return { messages: [result] };
      })
      .addEdge(START, "ask")
      .addEdge("ask", END)
      .compile({ checkpointer: new MemorySaver() });
    const thread = "later-wait";

    const consent = onlyInterrupt(
      (await run(graph as never, { messages: [] }, thread, "stream"))
        .interrupts,
    );
    oe.consent("alice");
    const question = onlyInterrupt(
      (
        await run(
          graph as never,
          new Command({ resume: { [consent.id]: { confirmed: true } } }),
          thread,
          "stream",
        )
      ).interrupts,
    );
    expect(question.id).not.toBe(consent.id);
    expect(question.value).toEqual({ question: "Share these repositories?" });
    expect(answers).toEqual([]);

    const done = await run(
      graph as never,
      new Command({ resume: { [question.id]: "yes" } }),
      thread,
      "stream",
    );

    expect(answers).toEqual(["yes"]);
    expect(contents(done.messages)).toEqual({
      "call-alice": JSON.stringify({ repos: ["alice-repo"] }),
    });
    expect(oe.providerCalls).toEqual(["alice"]);
  });

  it("ends the turn instead of handing the model a retryable error when the call's checkpoint cannot be saved after the provider answered", async () => {
    registerSuspendHandler(interrupt as never);
    const oe = fakeOe(["alice"]);
    const graph = compileTools(
      new ChildWriteFailingSaver(() => oe.providerCalls.length > 0),
    );
    const thread = "child-write-fails";

    await run(graph, { messages: [toolCalls("alice")] }, thread, "invoke");

    expect(oe.providerCalls).toEqual(["alice"]);
    expect(() => wrappers.get(thread)?.throwIfPlatformFailed()).toThrow(
      ToolPlatformError,
    );
  });

  it("does not repeat a parallel call that completed before its sibling's consent", async () => {
    registerSuspendHandler(interrupt as never);
    const oe = fakeOe(["alice"]);
    const graph = compileTools();
    const thread = "parallel-completed";

    const paused = await run(
      graph,
      { messages: [toolCalls("alice", "bob")] },
      thread,
      "stream",
    );
    const pending = onlyInterrupt(paused.interrupts);
    expect(pending.value).toMatchObject({
      suspend_context: { elicitation_id: "consent-bob" },
    });
    expect(oe.providerCalls).toEqual(["alice"]);

    oe.consent("bob");
    const resumed = await run(
      graph,
      new Command({ resume: { [pending.id]: { confirmed: true } } }),
      thread,
      "stream",
    );

    expect(contents(resumed.messages)).toEqual({
      "call-alice": JSON.stringify({ repos: ["alice-repo"] }),
      "call-bob": JSON.stringify({ repos: ["bob-repo"] }),
    });
    expect(oe.providerCalls).toEqual(["alice", "bob"]);
  });

  it("reports a parallel call's own failure again without dispatching it after its sibling's consent", async () => {
    registerSuspendHandler(interrupt as never);
    fakeOe(["ghost"], ["ghost"]);
    const wrapper = new SecureToolWrapper("http://oe:8000", "exec-reference");
    const direct = await runWithExecutionContext(
      { executionId: "exec-reference", wrapper, oeUrl: "http://oe:8000" },
      () =>
        new StateGraph(MessagesAnnotation)
          .addNode("tools", new ToolNode([oauthTool()]))
          .addEdge(START, "tools")
          .addEdge("tools", END)
          .compile()
          .invoke(
            { messages: [toolCalls("ghost")] },
            { configurable: { request_context: { sessionId: "reference" } } },
          ),
    );
    const directFailure = contents(
      direct.messages.filter(
        (message): message is ToolMessage => message instanceof ToolMessage,
      ),
    )["call-ghost"];
    expect(directFailure).toContain("GitHub has no user ghost");

    const oe = fakeOe(["ghost"], ["ghost"]);
    const graph = compileTools();
    const thread = "parallel-failed";

    const pending = onlyInterrupt(
      (
        await run(
          graph,
          { messages: [toolCalls("ghost", "bob")] },
          thread,
          "stream",
        )
      ).interrupts,
    );
    expect(oe.requests.filter((owner) => owner === "ghost")).toEqual(["ghost"]);

    oe.consent("bob");
    const resumed = await run(
      graph,
      new Command({ resume: { [pending.id]: { confirmed: true } } }),
      thread,
      "stream",
    );

    expect(contents(resumed.messages)).toEqual({
      "call-ghost": directFailure,
      "call-bob": JSON.stringify({ repos: ["bob-repo"] }),
    });
    expect(oe.requests.filter((owner) => owner === "ghost")).toEqual(["ghost"]);
  });

  it("rethrows a recorded call's classified failure with its type when its node runs again", async () => {
    registerSuspendHandler(interrupt as never);
    const oe = fakeOe(["alice"], ["alice"], AUTH_FAILED);
    const repositories = oauthTool();
    const failures: unknown[] = [];
    const graph = new StateGraph(MessagesAnnotation)
      .addNode("calls", async () => {
        await repositories
          .invoke({
            id: "call-alice",
            name: "list_repositories",
            args: { owner: "alice" },
            type: "tool_call",
          })
          .catch((error: unknown) => failures.push(error));
        interrupt("continue");
        return {};
      })
      .addEdge(START, "calls")
      .addEdge("calls", END)
      .compile({ checkpointer: new MemorySaver() });
    const thread = "classified-failure";

    const pending = onlyInterrupt(
      (await run(graph as never, { messages: [] }, thread, "stream"))
        .interrupts,
    );
    await run(
      graph as never,
      new Command({ resume: { [pending.id]: "go" } }),
      thread,
      "stream",
    );

    expect(failures).toHaveLength(2);
    for (const failure of failures) {
      expect(failure).toBeInstanceOf(ExternalAPICallError);
      expect(failure).toMatchObject({
        message: "Tool execution error: GitHub has no user alice",
        tool_api_error: AUTH_FAILED,
      });
    }
    expect(oe.requests).toEqual(["alice"]);
  });

  it("asks parallel calls for consent one after another, each answer waking only its own call", async () => {
    registerSuspendHandler(interrupt as never);
    const oe = fakeOe();
    const graph = compileTools();
    const thread = "parallel-consents";

    const first = onlyInterrupt(
      (
        await run(
          graph,
          { messages: [toolCalls("alice", "bob")] },
          thread,
          "stream",
        )
      ).interrupts,
    );
    const firstOwner = first.value["suspend_context"] as {
      elicitation_id: string;
    };
    const [answered, waiting] =
      firstOwner.elicitation_id === "consent-alice"
        ? ["alice", "bob"]
        : ["bob", "alice"];
    oe.consent(answered);
    const second = onlyInterrupt(
      (
        await run(
          graph,
          new Command({ resume: { [first.id]: { confirmed: true } } }),
          thread,
          "stream",
        )
      ).interrupts,
    );
    expect(second.id).not.toBe(first.id);
    expect(second.value["suspend_context"]).toMatchObject({
      elicitation_id: `consent-${waiting}`,
    });
    expect(oe.providerCalls).toEqual([answered]);

    oe.consent(waiting);

    const done = await run(
      graph,
      new Command({ resume: { [second.id]: { confirmed: true } } }),
      thread,
      "stream",
    );

    expect(done.interrupts).toEqual([]);
    expect(contents(done.messages)).toEqual({
      "call-alice": JSON.stringify({ repos: ["alice-repo"] }),
      "call-bob": JSON.stringify({ repos: ["bob-repo"] }),
    });
    expect(oe.providerCalls).toEqual([answered, waiting]);
  });

  it.each([
    { slot: "tool-call ID", changed: "arguments", recorded: "result" },
    { slot: "tool-call ID", changed: "arguments", recorded: "consent wait" },
    { slot: "position", changed: "tool", recorded: "result" },
    { slot: "position", changed: "tool", recorded: "consent wait" },
  ])(
    "fails before any provider call when a $slot slot that recorded its $recorded gets other $changed",
    async ({ slot, changed, recorded }) => {
      registerSuspendHandler(interrupt as never);
      const oe = fakeOe(recorded === "result" ? ["alice", "bob"] : []);
      const repositories = oauthTool();
      const issues = oauthTool("list_issues");
      let rerun = false;
      const graph = new StateGraph(MessagesAnnotation)
        .addNode("calls", async () => {
          if (slot === "tool-call ID") {
            await repositories.invoke({
              id: "call-1",
              name: "list_repositories",
              args: { owner: rerun ? "bob" : "alice" },
              type: "tool_call",
            });
          } else {
            const order = rerun
              ? [issues, repositories]
              : [repositories, issues];
            for (const call of order) await call.invoke({ owner: "alice" });
          }
          interrupt("continue");
          return {};
        })
        .addEdge(START, "calls")
        .addEdge("calls", END)
        .compile({ checkpointer: new MemorySaver() });
      const thread = `changed-${slot}-${recorded}`;

      const pending = onlyInterrupt(
        (await run(graph as never, { messages: [] }, thread, "stream"))
          .interrupts,
      );
      const calls = [...oe.providerCalls];
      rerun = true;
      oe.consent("alice");
      oe.consent("bob");

      await expect(
        run(
          graph as never,
          new Command({ resume: { [pending.id]: { confirmed: true } } }),
          thread,
          "stream",
        ),
      ).rejects.toThrow(
        changed === "arguments" ? "arguments changed" : "identity changed",
      );
      expect(oe.providerCalls).toEqual(calls);
    },
  );

  it("runs directly in a graph without a checkpointer", async () => {
    const oe = fakeOe(["alice"]);
    const graph = new StateGraph(MessagesAnnotation)
      .addNode("tools", new ToolNode([oauthTool()]))
      .addEdge(START, "tools")
      .addEdge("tools", END)
      .compile();
    const wrapper = new SecureToolWrapper("http://oe:8000", "exec-stateless");

    const result = await runWithExecutionContext(
      { executionId: "exec-stateless", wrapper, oeUrl: "http://oe:8000" },
      () =>
        graph.invoke(
          { messages: [toolCalls("alice")] },
          { configurable: { request_context: { sessionId: "stateless" } } },
        ),
    );

    expect(
      contents(
        result.messages.filter(
          (message): message is ToolMessage => message instanceof ToolMessage,
        ),
      ),
    ).toEqual({ "call-alice": JSON.stringify({ repos: ["alice-repo"] }) });
    expect(oe.providerCalls).toEqual(["alice"]);
  });
});
