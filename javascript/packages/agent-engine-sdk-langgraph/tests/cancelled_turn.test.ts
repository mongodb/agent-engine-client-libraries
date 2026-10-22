/** Cancelled-turn checkpoint commit: the drain-teardown write that keeps a
 * stopped turn visible in playground history without making it resumable.
 * Mirrors Python's tests/test_cancelled_turn.py. */

import { describe, expect, test } from "vitest";
import type {
  Checkpoint,
  CheckpointMetadata,
  CheckpointTuple,
} from "@langchain/langgraph-checkpoint";
import {
  AIMessage,
  HumanMessage,
  type BaseMessage,
} from "@langchain/core/messages";
import type { RunnableConfig } from "@langchain/core/runnables";
import {
  END,
  MemorySaver,
  MessagesAnnotation,
  Overwrite,
  Send,
  START,
  StateGraph,
  type CompiledStateGraph,
} from "@langchain/langgraph";
import type { AgentInput, RequestContext } from "@mongodb-js/agent-engine-sdk";

/** The branch helpers take the framework-widened graph type; the compiled
 * concrete graph is the same object at runtime. */
const asBranchGraph = (graph: unknown): CompiledStateGraph<unknown, unknown> =>
  graph as CompiledStateGraph<unknown, unknown>;

import { LangGraphBaseAgent } from "../src/agent.js";
import {
  CANCELLED_TURN_MESSAGE_ID_METADATA_KEY,
  CANCELLED_TURN_MESSAGE_IDS_METADATA_KEY,
  CANCELLED_TURN_METADATA_KEY,
  commitCancelledTurnCheckpoint,
} from "../src/cancelled_turn.js";
import {
  completedRootCheckpoint,
  copyCheckpoint,
} from "../src/checkpoint_branch.js";

interface RecordedPut {
  config: RunnableConfig;
  checkpoint: Checkpoint;
  // Widened: the commit's marker keys ride the metadata beyond the
  // CheckpointMetadata shape, and the assertions index them.
  metadata: CheckpointMetadata & Record<string, unknown>;
}

/** Duck-typed native saver: serves one canned tip tuple, records puts. */
class RecordingSaver {
  puts: RecordedPut[] = [];
  writeCalls: unknown[] = [];

  constructor(private readonly latest?: CheckpointTuple) {}

  async getTuple(
    _config: RunnableConfig,
  ): Promise<CheckpointTuple | undefined> {
    // After a put, the tip IS the last-put checkpoint — a real saver
    // reflects its own writes, and the commit hook reads the tip back for
    // the coordinate.
    const last = this.puts[this.puts.length - 1];
    if (last !== undefined) {
      return {
        config: last.config,
        checkpoint: last.checkpoint,
        metadata: last.metadata,
      } as unknown as CheckpointTuple;
    }
    return this.latest;
  }

  async put(
    config: RunnableConfig,
    checkpoint: Checkpoint,
    metadata: CheckpointMetadata,
  ): Promise<RunnableConfig> {
    this.puts.push({
      config,
      checkpoint,
      metadata: metadata as CheckpointMetadata & Record<string, unknown>,
    });
    return config;
  }

  async putWrites(...args: unknown[]): Promise<void> {
    this.writeCalls.push(args);
  }

  getNextVersion(current: number | undefined): number {
    return current === undefined ? 1 : current + 1;
  }
}

function makeTuple(args: {
  checkpointId: string;
  messages: unknown[];
  channelVersions: Record<string, number | string>;
  versionsSeen: Record<string, Record<string, number | string>>;
  step: number;
}): CheckpointTuple {
  return {
    config: {
      configurable: {
        thread_id: "t",
        checkpoint_ns: "",
        checkpoint_id: args.checkpointId,
      },
    },
    checkpoint: {
      v: 4,
      id: args.checkpointId,
      ts: "2026-09-28T00:00:00.000Z",
      channel_values: { messages: args.messages },
      channel_versions: args.channelVersions,
      versions_seen: args.versionsSeen,
    },
    metadata: { source: "loop", step: args.step, parents: {} },
  };
}

function kwargsOf(message: unknown): Record<string, unknown> {
  return ((message as { additional_kwargs?: Record<string, unknown> })
    .additional_kwargs ?? {}) as Record<string, unknown>;
}

/** The commit writes exactly one checkpoint; tests read it positionally. */
function onlyPut(saver: RecordingSaver): RecordedPut {
  const [put] = saver.puts;
  if (put === undefined) throw new Error("expected one recorded put");
  return put;
}

describe("commitCancelledTurnCheckpoint", () => {
  test("identical consecutive prompts, cancel before input persistence", async () => {
    // Turn 1 "try again" committed; turn 2 "try again" cancelled before its
    // input checkpoint landed. The captured id is absent from the stored
    // messages, so the commit injects a NEW opener message under that id and
    // leaves the completed first turn unmarked — a content comparison would
    // wrongly pick h1 here.
    const latest = makeTuple({
      checkpointId: "c2",
      messages: [
        new HumanMessage({ content: "try again", id: "h1" }),
        new AIMessage({ content: "done", id: "a1" }),
      ],
      channelVersions: { messages: 1 },
      versionsSeen: { node_a: { messages: 1 } },
      step: 1,
    });
    const saver = new RecordingSaver(latest);

    const messageId = await commitCancelledTurnCheckpoint(saver, {
      threadId: "t",
      userMessage: "try again",
      turnMessageId: "input:turn-2",
      nodeNames: ["node_a"],
    });

    expect(messageId).toBe("input:turn-2");
    const { checkpoint, metadata } = onlyPut(saver);
    const stored = checkpoint.channel_values["messages"] as BaseMessage[];
    expect(stored.map((m) => m.id)).toEqual(["h1", "a1", "input:turn-2"]);
    expect(kwargsOf(stored[0])).toEqual({}); // completed turn 1 must not be marked
    expect(checkpoint.channel_versions["messages"]).toBe(2); // bumped for the append
    expect(metadata[CANCELLED_TURN_MESSAGE_ID_METADATA_KEY]).toBe(
      "input:turn-2",
    );
  });

  test("identical consecutive prompts, after input persistence", async () => {
    // Both turns sent "try again" and turn 2's input DID land. The captured
    // id picks h2 — h1 stays unmarked and no duplicate is appended.
    const latest = makeTuple({
      checkpointId: "c9",
      messages: [
        new HumanMessage({ content: "try again", id: "h1" }),
        new AIMessage({ content: "ok", id: "a1" }),
        new HumanMessage({ content: "try again", id: "h2" }),
      ],
      channelVersions: { messages: 2 },
      versionsSeen: { node_a: { messages: 1 } },
      step: 4,
    });
    const saver = new RecordingSaver(latest);

    const messageId = await commitCancelledTurnCheckpoint(saver, {
      threadId: "t",
      userMessage: "try again",
      turnMessageId: "h2",
      nodeNames: ["node_a"],
    });

    expect(messageId).toBe("h2");
    const { config, checkpoint, metadata } = onlyPut(saver);
    expect(config.configurable?.["checkpoint_id"]).toBe("c9"); // parent chain
    expect(checkpoint.id).not.toBe("c9");
    const stored = checkpoint.channel_values["messages"] as BaseMessage[];
    expect(stored.map((m) => m.id)).toEqual(["h1", "a1", "h2"]);
    // Channel versions unchanged — nothing ran — but every node now reads as
    // caught up, so the tip presents no pending work.
    expect(checkpoint.channel_versions["messages"]).toBe(2);
    expect(checkpoint.versions_seen["node_a"]?.["messages"]).toBe(2);
    expect(metadata.step).toBe(5);
    expect(metadata.source).toBe("loop");
    expect(metadata[CANCELLED_TURN_MESSAGE_ID_METADATA_KEY]).toBe("h2"); // h1 unmarked
    // The canned tip's own state must not be mutated through the copy.
    expect(latest.checkpoint.versions_seen["node_a"]?.["messages"]).toBe(1);
  });

  test("matches the turn by captured id, not content", async () => {
    // A prepareInput hook rewrote the message content: the payload text must
    // not be consulted — the captured id alone picks the turn opener.
    const latest = makeTuple({
      checkpointId: "c9",
      messages: [
        new HumanMessage({ content: "original phrasing", id: "h1" }),
        new AIMessage({ content: "first reply", id: "a1" }),
        new HumanMessage({ content: "rewritten by the hook", id: "h2" }),
      ],
      channelVersions: { messages: 2 },
      versionsSeen: { node_a: { messages: 2 } },
      step: 4,
    });
    const saver = new RecordingSaver(latest);

    const messageId = await commitCancelledTurnCheckpoint(saver, {
      threadId: "t",
      userMessage: "original phrasing",
      turnMessageId: "h2",
      nodeNames: ["node_a"],
    });

    expect(messageId).toBe("h2");
    const stored = onlyPut(saver).checkpoint.channel_values[
      "messages"
    ] as BaseMessage[];
    expect(stored.map((m) => m.id)).toEqual(["h1", "a1", "h2"]); // no append
    expect(
      onlyPut(saver).metadata[CANCELLED_TURN_MESSAGE_ID_METADATA_KEY],
    ).toBe("h2");
  });

  test("no captured id appends a fresh opener for an identical prompt", async () => {
    // turnMessageId=null means the turn's input was never built (a cancel
    // during preparation). An identical earlier prompt must NOT be treated as
    // this turn's opener: append a fresh message and record that id, leaving
    // the earlier turn untouched.
    const latest = makeTuple({
      checkpointId: "c9",
      messages: [
        new HumanMessage({ content: "stop this", id: "h1" }),
        new AIMessage({ content: "older reply", id: "a1" }),
      ],
      channelVersions: { messages: 2 },
      versionsSeen: { node_a: { messages: 2 } },
      step: 4,
    });
    const saver = new RecordingSaver(latest);

    const messageId = await commitCancelledTurnCheckpoint(saver, {
      threadId: "t",
      userMessage: "stop this",
      turnMessageId: null,
      nodeNames: ["node_a"],
    });

    expect(messageId).toMatch(/^msg-cancelled-/);
    const stored = onlyPut(saver).checkpoint.channel_values[
      "messages"
    ] as BaseMessage[];
    expect(stored.map((m) => m.id)).toEqual(["h1", "a1", messageId]);
    expect(
      onlyPut(saver).metadata[CANCELLED_TURN_MESSAGE_ID_METADATA_KEY],
    ).toBe(messageId);
  });

  test("no captured id injects with a fresh marker id", async () => {
    const latest = makeTuple({
      checkpointId: "c2",
      messages: [
        new HumanMessage({ content: "older turn", id: "h1" }),
        new AIMessage({ content: "older reply", id: "a1" }),
      ],
      channelVersions: { messages: 1 },
      versionsSeen: { node_a: { messages: 1 } },
      step: 1,
    });
    const saver = new RecordingSaver(latest);

    const messageId = await commitCancelledTurnCheckpoint(saver, {
      threadId: "t",
      userMessage: "never landed",
      turnMessageId: null,
      nodeNames: ["node_a"],
    });

    expect(messageId).toMatch(/^msg-cancelled-/);
    const { checkpoint, metadata } = onlyPut(saver);
    const stored = checkpoint.channel_values["messages"] as BaseMessage[];
    expect(stored.map((m) => m.id).slice(0, 2)).toEqual(["h1", "a1"]);
    expect(stored[2]).toBeInstanceOf(HumanMessage);
    expect(stored[2]?.content).toBe("never landed");
    expect(stored[2]?.id).toBe(messageId);
    expect(checkpoint.channel_versions["messages"]).toBe(2); // bumped for the append
    expect(metadata[CANCELLED_TURN_MESSAGE_ID_METADATA_KEY]).toBe(messageId);
  });

  test("the commit does not mutate the shared source message", async () => {
    // The opener is recorded in the checkpoint's metadata, not on the
    // message; the deserialized source message object is shared read state
    // and must stay untouched.
    const sourceOpener = new HumanMessage({ content: "stop this", id: "h2" });
    const latest = makeTuple({
      checkpointId: "c9",
      messages: [
        new HumanMessage({ content: "earlier", id: "h1" }),
        new AIMessage({ content: "done earlier", id: "a1" }),
        sourceOpener,
      ],
      channelVersions: { messages: 2 },
      versionsSeen: { node_a: { messages: 2 } },
      step: 4,
    });
    const saver = new RecordingSaver(latest);

    await commitCancelledTurnCheckpoint(saver, {
      threadId: "t",
      userMessage: "stop this",
      turnMessageId: "h2",
      nodeNames: ["node_a"],
    });

    const [, , written] = onlyPut(saver).checkpoint.channel_values[
      "messages"
    ] as BaseMessage[];
    expect(kwargsOf(written)).toEqual({});
    expect(sourceOpener.additional_kwargs).toEqual({});
    expect(
      onlyPut(saver).metadata[CANCELLED_TURN_MESSAGE_ID_METADATA_KEY],
    ).toBe("h2");
  });

  test("a node that never ran is terminal-ized", async () => {
    // node_b never ran in this thread, so it has no versions_seen entry. Left
    // absent it would read as pending; the commit must catch it up over every
    // channel.
    const latest = makeTuple({
      checkpointId: "c9",
      messages: [new HumanMessage({ content: "stop this", id: "h2" })],
      channelVersions: { messages: 2 },
      versionsSeen: { node_a: { messages: 1 } },
      step: 4,
    });
    const saver = new RecordingSaver(latest);

    await commitCancelledTurnCheckpoint(saver, {
      threadId: "t",
      userMessage: "stop this",
      turnMessageId: "h2",
      nodeNames: ["node_a", "node_b"],
    });

    const writtenSeen = onlyPut(saver).checkpoint.versions_seen;
    expect(writtenSeen["node_a"]?.["messages"]).toBe(2);
    expect(writtenSeen["node_b"]).toEqual({ messages: 2 });
    expect(latest.checkpoint.versions_seen).toEqual({
      node_a: { messages: 1 },
    });
  });

  test("empty thread commits the user message as the first checkpoint", async () => {
    const saver = new RecordingSaver(undefined);

    const messageId = await commitCancelledTurnCheckpoint(saver, {
      threadId: "t",
      userMessage: "hi there",
      turnMessageId: "input:turn-1",
      nodeNames: ["node_a"],
    });

    expect(messageId).toBe("input:turn-1");
    expect(saver.puts).toHaveLength(1);
    const { config, checkpoint, metadata } = onlyPut(saver);
    expect(config.configurable).toEqual({
      thread_id: "t",
      checkpoint_ns: "",
      checkpoint_id: undefined,
    });
    const messages = checkpoint.channel_values["messages"] as HumanMessage[];
    expect(messages).toHaveLength(1);
    expect(messages[0]?.content).toBe("hi there");
    expect(messages[0]?.id).toBe("input:turn-1");
    expect(checkpoint.channel_versions["messages"]).toBe(1);
    expect(checkpoint.versions_seen).toEqual({ node_a: { messages: 1 } });
    expect(metadata[CANCELLED_TURN_METADATA_KEY]).toBe(true);
    expect(metadata[CANCELLED_TURN_MESSAGE_ID_METADATA_KEY]).toBe(
      "input:turn-1",
    );
    expect(metadata.step).toBe(0);
  });

  test("empty message terminal-izes without a turn opener", async () => {
    // Resume-cancel shape: no fresh user message. The marker tip still lands
    // (terminal-ized, so the cancelled super-step never looks resumable) but
    // no message is injected and no turn message id is recorded.
    const latest = makeTuple({
      checkpointId: "c9",
      messages: [
        new HumanMessage({ content: "original question", id: "h1" }),
        new AIMessage({ content: "partial", id: "a1" }),
      ],
      channelVersions: { messages: 1 },
      versionsSeen: { node_a: { messages: 0 } },
      step: 4,
    });
    const saver = new RecordingSaver(latest);

    const messageId = await commitCancelledTurnCheckpoint(saver, {
      threadId: "t",
      userMessage: "",
      turnMessageId: null,
      nodeNames: ["node_a", "node_b"],
    });

    expect(messageId).toBeNull();
    const { checkpoint, metadata } = onlyPut(saver);
    const stored = checkpoint.channel_values["messages"] as BaseMessage[];
    expect(stored.map((m) => m.id)).toEqual(["h1", "a1"]);
    expect(stored.map(kwargsOf)).toEqual([{}, {}]);
    expect(checkpoint.versions_seen["node_a"]?.["messages"]).toBe(1);
    expect(checkpoint.versions_seen["node_b"]).toEqual({ messages: 1 });
    expect(metadata[CANCELLED_TURN_METADATA_KEY]).toBe(true);
    expect(metadata[CANCELLED_TURN_MESSAGE_ID_METADATA_KEY]).toBeUndefined();
  });

  test("the commit never attaches pending writes", async () => {
    const saver = new RecordingSaver(undefined);
    await commitCancelledTurnCheckpoint(saver, {
      threadId: "t",
      userMessage: "hi",
      turnMessageId: null,
      nodeNames: [],
    });
    expect(saver.writeCalls).toEqual([]);
  });
});

/** One reply node compiled over the saver — the real-run hook-test graph. */
function replyGraph(saver: MemorySaver) {
  return new StateGraph(MessagesAnnotation)
    .addNode("reply", async () => ({
      messages: [new AIMessage({ content: "reply done", id: "a-1" })],
    }))
    .addEdge(START, "reply")
    .addEdge("reply", END)
    .compile({ checkpointer: saver });
}

/** Messages on the thread's tip checkpoint (real saver read). */
async function tipMessages(saver: MemorySaver): Promise<BaseMessage[]> {
  const tuple = await saver.getTuple({
    configurable: { thread_id: "sess-1:ws-1" },
  });
  return (
    (tuple?.checkpoint.channel_values["messages"] as
      | BaseMessage[]
      | undefined) ?? []
  );
}

/** Metadata on the thread's tip checkpoint (real saver read). */
async function tipMetadata(
  saver: MemorySaver,
): Promise<Record<string, unknown>> {
  const tuple = await saver.getTuple({
    configurable: { thread_id: "sess-1:ws-1" },
  });
  return (tuple?.metadata ?? {}) as Record<string, unknown>;
}

/** A → B with a compile-time interrupt before B, plus the call log. */
function twoNodeGraph(saver: MemorySaver) {
  const calls: string[] = [];
  const graph = new StateGraph(MessagesAnnotation)
    .addNode("node_a", async () => {
      calls.push("a");
      return {
        messages: [
          new AIMessage({ content: "a done", id: `a-${calls.length}` }),
        ],
      };
    })
    .addNode("node_b", async () => {
      calls.push("b");
      return { messages: [new AIMessage({ content: "b done", id: "b-1" })] };
    })
    .addEdge(START, "node_a")
    .addEdge("node_a", "node_b")
    .addEdge("node_b", END)
    .compile({ checkpointer: saver, interruptBefore: ["node_b"] });
  return { graph, calls };
}

/** Router dispatching its tool call as a Send — the deep agents' routing.
 * The worker signals `started` on entry and blocks until `release()`; later
 * entries pass through (the gate stays open once released). */
function sendGraph(saver: MemorySaver) {
  const calls: string[] = [];
  let startedResolve: () => void = () => {};
  const started = new Promise<void>((resolve) => {
    startedResolve = resolve;
  });
  let entries = 0;
  let releaseWorker: () => void = () => {};
  const graph = new StateGraph(MessagesAnnotation)
    .addNode("router", async () => ({
      messages: [new AIMessage({ content: "dispatch", id: "r-1" })],
    }))
    .addNode("worker", async () => {
      calls.push("w");
      entries += 1;
      // Only the first dispatch (the cancelled one) blocks — the pod torn down
      // mid-tool. Later dispatches (a fresh turn) run straight through.
      if (entries === 1) {
        startedResolve();
        await new Promise<void>((resolve) => {
          releaseWorker = resolve;
        });
      }
      return {
        messages: [new AIMessage({ content: "tool done", id: "tool-1" })],
      };
    })
    .addEdge(START, "router")
    .addConditionalEdges("router", () => [
      new Send("worker", [{ role: "user", content: "tool payload" }]),
    ])
    .compile({ checkpointer: saver });
  return {
    graph,
    calls,
    started,
    release: () => {
      releaseWorker();
    },
  };
}

describe("real LangGraph graphs", () => {
  test("two-node graph: synthetic tip is terminal and never resumes", async () => {
    // Cancel with a never-run downstream node: the synthetic tip must report
    // no pending work and never schedule the cancelled node on resume; a
    // follow-up turn starts fresh.
    const saver = new MemorySaver();
    const { graph, calls } = twoNodeGraph(saver);

    const config: RunnableConfig = { configurable: { thread_id: "t" } };
    await graph.invoke(
      { messages: [new HumanMessage({ content: "go", id: "input:1" })] },
      config,
    );
    expect(calls).toEqual(["a"]); // stopped before node_b ever ran

    const resolved = await commitCancelledTurnCheckpoint(saver, {
      threadId: "t",
      userMessage: "go",
      turnMessageId: "input:1",
      nodeNames: Object.keys(graph.getGraph().nodes),
    });
    expect(resolved).toBe("input:1");

    const markerTip = await saver.getTuple(config);
    expect(
      (markerTip?.metadata as Record<string, unknown>)[
        CANCELLED_TURN_MESSAGE_ID_METADATA_KEY
      ],
    ).toBe("input:1");

    const state = await graph.getState(config);
    expect(state.next).toEqual([]);
    expect(state.tasks).toEqual([]);

    // Resume-without-input must not schedule the cancelled node_b.
    await graph.invoke(null, config);
    expect(calls).toEqual(["a"]);

    // A follow-up turn starts fresh: node_a runs once for it.
    // (interruptBefore is compile-time, so this invoke also parks before
    // node_b — resuming THAT real pending interrupt is what runs node_b, in
    // contrast to the synthetic tip above, where resume was a no-op.)
    await graph.invoke(
      { messages: [new HumanMessage({ content: "again", id: "input:2" })] },
      config,
    );
    expect(calls).toEqual(["a", "a"]);
    await graph.invoke(null, config);
    expect(calls).toEqual(["a", "a", "b"]);
    const finalState = await graph.getState(config);
    expect(finalState.next).toEqual([]);
  });

  test("stop during a dispatched Send tool leaves no pending work", async () => {
    // The deep agents dispatch tool calls as Send packets, and the checkpoint
    // committed while a dispatched tool executes carries the queued packet.
    // The marker must not present it as pending: fork preflight accepts the
    // tip and the cancelled payload never runs on resume.
    const saver = new MemorySaver();
    const { graph, calls, started, release } = sendGraph(saver);

    const config: RunnableConfig = { configurable: { thread_id: "t" } };
    const run = graph.invoke(
      { messages: [new HumanMessage({ content: "go", id: "input:1" })] },
      config,
    );
    await started; // dispatched tool executing — the cancel window

    // Precondition: the latest committed checkpoint queues the Send.
    const latest = await saver.getTuple(config);
    const queued = latest?.checkpoint.channel_values["__pregel_tasks"];
    expect(queued).toBeTruthy();

    // The pod is torn down mid-tool: the blocked invocation never commits
    // again, so the marker written now is provably the thread's tip.
    const resolved = await commitCancelledTurnCheckpoint(saver, {
      threadId: "t",
      userMessage: "go",
      turnMessageId: "input:1",
      nodeNames: Object.keys(graph.getGraph().nodes),
    });
    expect(resolved).toBe("input:1");

    // The marker IS the source tip, and presents no pending work.
    const markerSource = await completedRootCheckpoint(asBranchGraph(graph), {
      threadId: "t",
      historyId: null,
    });
    const tip = await saver.getTuple(config);
    expect(tip?.checkpoint.id).toBe(markerSource.checkpointId);
    const state = await graph.getState(config);
    expect(state.next).toEqual([]);
    expect(state.tasks).toEqual([]);

    // Resume-without-input runs none of the cancelled work (the worker logged
    // its dispatch once, when the Send fired before the cancel).
    const dispatched = calls.length;
    await graph.invoke(null, config);
    expect(calls.length).toBe(dispatched);

    // Fork the marker tip, then fork the first copy: the stopped opener and
    // authoritative messages survive both copies, nothing left pending.
    await copyCheckpoint(
      asBranchGraph(graph),
      { configurable: { thread_id: "t-fork" } },
      markerSource,
    );
    const copied = await graph.getState({
      configurable: { thread_id: "t-fork" },
    });
    expect(copied.next).toEqual([]);
    expect(copied.tasks).toEqual([]);
    const copiedMessages = copied.values["messages"] as BaseMessage[];
    expect(copiedMessages.map((m) => m.content)).toEqual(["go", "dispatch"]);
    const copiedTip = await saver.getTuple({
      configurable: { thread_id: "t-fork" },
    });
    expect(
      (copiedTip?.metadata as Record<string, unknown>)[
        CANCELLED_TURN_METADATA_KEY
      ],
    ).toBe(true);
    // The source opener id rides onto the copy so the reader still classifies
    // the forked opener as stopped.
    expect(
      (copiedTip?.metadata as Record<string, unknown>)[
        CANCELLED_TURN_MESSAGE_ID_METADATA_KEY
      ],
    ).toBe(copiedMessages[0]?.id);

    const secondSource = await completedRootCheckpoint(asBranchGraph(graph), {
      threadId: "t-fork",
      historyId: null,
    });
    await copyCheckpoint(
      asBranchGraph(graph),
      { configurable: { thread_id: "t-fork-2" } },
      secondSource,
    );
    const copied2 = await graph.getState({
      configurable: { thread_id: "t-fork-2" },
    });
    expect(copied2.next).toEqual([]);
    expect(copied2.tasks).toEqual([]);
    const copied2Messages = copied2.values["messages"] as BaseMessage[];
    expect(copied2Messages.map((m) => m.content)).toEqual(["go", "dispatch"]);
    const copied2Tip = await saver.getTuple({
      configurable: { thread_id: "t-fork-2" },
    });
    expect(
      (copied2Tip?.metadata as Record<string, unknown>)[
        CANCELLED_TURN_METADATA_KEY
      ],
    ).toBe(true);

    // Neither copy redispatches the cancelled work on a null-input resume...
    await graph.invoke(null, { configurable: { thread_id: "t-fork" } });
    await graph.invoke(null, { configurable: { thread_id: "t-fork-2" } });
    expect(calls.length).toBe(dispatched);

    // ...and a fresh turn on the copied stopped tip dispatches anew.
    await graph.invoke(
      { messages: [new HumanMessage({ content: "again", id: "input:2" })] },
      { configurable: { thread_id: "t-fork" } },
    );
    expect(calls.length).toBe(dispatched + 1);

    // Drain the torn-down invocation so nothing is left pending.
    release();
    await run.catch(() => undefined);
  });

  test("fork reproduces a stopped tip and later history", async () => {
    // Forking from a cancelled-turn marker tip — and from later history built
    // on one — reproduces the authoritative state, marker tip included.
    const saver = new MemorySaver();
    const { graph } = twoNodeGraph(saver);
    const config: RunnableConfig = { configurable: { thread_id: "t" } };
    await graph.invoke(
      { messages: [new HumanMessage({ content: "go", id: "input:1" })] },
      config,
    );
    await commitCancelledTurnCheckpoint(saver, {
      threadId: "t",
      userMessage: "go",
      turnMessageId: "input:1",
      nodeNames: Object.keys(graph.getGraph().nodes),
    });

    const source = await completedRootCheckpoint(asBranchGraph(graph), {
      threadId: "t",
      historyId: null,
    });
    await copyCheckpoint(
      asBranchGraph(graph),
      { configurable: { thread_id: "t-fork" } },
      source,
    );
    const copied = await graph.getState({
      configurable: { thread_id: "t-fork" },
    });
    expect(copied.next).toEqual([]);
    expect(copied.tasks).toEqual([]);
    const sourceMessages = (await graph.getState(config)).values[
      "messages"
    ] as BaseMessage[];
    const copiedMessages = copied.values["messages"] as BaseMessage[];
    expect(copiedMessages.map((m) => m.content)).toEqual(
      sourceMessages.map((m) => m.content),
    );
    const copiedTip = await saver.getTuple({
      configurable: { thread_id: "t-fork" },
    });
    expect(
      (copiedTip?.metadata as Record<string, unknown>)[
        CANCELLED_TURN_METADATA_KEY
      ],
    ).toBe(true);

    // A follow-up turn on the source, then fork again: the full later history
    // survives.
    await graph.invoke(
      { messages: [new HumanMessage({ content: "again", id: "input:2" })] },
      config,
    );
    await graph.invoke(null, config); // past the compile-time interrupt
    const source2 = await completedRootCheckpoint(asBranchGraph(graph), {
      threadId: "t",
      historyId: null,
    });
    await copyCheckpoint(
      asBranchGraph(graph),
      { configurable: { thread_id: "t-fork-2" } },
      source2,
    );
    const copied2 = await graph.getState({
      configurable: { thread_id: "t-fork-2" },
    });
    const copied2Messages = copied2.values["messages"] as BaseMessage[];
    expect(copied2Messages.map((m) => m.content)).toEqual([
      "go",
      "a done",
      "again",
      "a done",
      "b done",
    ]);
    // Replay rebuilds checkpoints and drops the per-turn marker metadata, so
    // the fork re-records the source's cancelled openers on the copied tip —
    // the earlier stopped opener keeps its id and stays classified on the fork.
    const copied2Tip = await saver.getTuple({
      configurable: { thread_id: "t-fork-2" },
    });
    const copied2Metadata = copied2Tip?.metadata as Record<string, unknown>;
    expect(copied2Metadata[CANCELLED_TURN_METADATA_KEY]).toBe(true);
    expect(copied2Messages[0]?.id).toBe("input:1");
    expect(copied2Metadata[CANCELLED_TURN_MESSAGE_IDS_METADATA_KEY]).toEqual([
      "input:1",
    ]);
  });

  test("fork from a first turn cancelled before any checkpoint", async () => {
    // The marker tip is the thread's ONLY checkpoint — plain replay reports
    // "no checkpoint history"; the synthesized superstep makes it forkable.
    const saver = new MemorySaver();
    const { graph } = twoNodeGraph(saver);
    await commitCancelledTurnCheckpoint(saver, {
      threadId: "t",
      userMessage: "never ran",
      turnMessageId: "input:1",
      nodeNames: Object.keys(graph.getGraph().nodes),
    });

    const source = await completedRootCheckpoint(asBranchGraph(graph), {
      threadId: "t",
      historyId: null,
    });
    await copyCheckpoint(
      asBranchGraph(graph),
      { configurable: { thread_id: "t-fork" } },
      source,
    );
    const copied = await graph.getState({
      configurable: { thread_id: "t-fork" },
    });
    const messages = copied.values["messages"] as BaseMessage[];
    expect(messages.map((m) => m.content)).toEqual(["never ran"]);
    const copiedTip = await saver.getTuple({
      configurable: { thread_id: "t-fork" },
    });
    expect(
      (copiedTip?.metadata as Record<string, unknown>)[
        CANCELLED_TURN_METADATA_KEY
      ],
    ).toBe(true);
    // The source opener id rides onto the copy so the reader still classifies
    // the forked opener as stopped.
    expect(
      (copiedTip?.metadata as Record<string, unknown>)[
        CANCELLED_TURN_MESSAGE_ID_METADATA_KEY
      ],
    ).toBe(messages[0]?.id);
  });

  test("second identical prompt cancelled during preparation", async () => {
    // Turn 2 repeats turn 1's prompt and is cancelled during preparation (its
    // input was never built, so no id was captured). The commit must append a
    // separate stopped opener — the earlier opener and its completed steps
    // stay untouched.
    const saver = new MemorySaver();
    const { graph, calls } = twoNodeGraph(saver);
    const config: RunnableConfig = { configurable: { thread_id: "t" } };
    await graph.invoke(
      { messages: [new HumanMessage({ content: "go", id: "input:1" })] },
      config,
    );
    await graph.invoke(null, config); // past the compile-time interrupt
    expect(calls).toEqual(["a", "b"]); // turn 1 completed fully

    const resolved = await commitCancelledTurnCheckpoint(saver, {
      threadId: "t",
      userMessage: "go",
      turnMessageId: null, // prep-phase cancel: input never built
      nodeNames: Object.keys(graph.getGraph().nodes),
    });
    expect(resolved).toMatch(/^msg-cancelled-/);

    const state = await graph.getState(config);
    expect(state.next).toEqual([]);
    expect(state.tasks).toEqual([]);
    const messages = state.values["messages"] as BaseMessage[];
    expect(messages.map((m) => m.content)).toEqual([
      "go",
      "a done",
      "b done",
      "go",
    ]);
    expect(messages[0]?.id).toBe("input:1"); // turn 1's opener stays unmarked
    expect(messages[3]?.id).toBe(resolved);
    const markerTip = await saver.getTuple(config);
    expect(
      (markerTip?.metadata as Record<string, unknown>)[
        CANCELLED_TURN_MESSAGE_ID_METADATA_KEY
      ],
    ).toBe(resolved);
  });

  test("a forked stopped tip is itself forkable", async () => {
    // Two consecutive forks of an early-cancelled first turn: the first copy's
    // terminal-ized tip carries the marker, so the second fork replays it as a
    // marker too — stopped opener preserved, nothing left pending.
    const saver = new MemorySaver();
    const { graph } = twoNodeGraph(saver);
    await commitCancelledTurnCheckpoint(saver, {
      threadId: "t",
      userMessage: "never ran",
      turnMessageId: "input:1",
      nodeNames: Object.keys(graph.getGraph().nodes),
    });

    const source = await completedRootCheckpoint(asBranchGraph(graph), {
      threadId: "t",
      historyId: null,
    });
    await copyCheckpoint(
      asBranchGraph(graph),
      { configurable: { thread_id: "t-fork" } },
      source,
    );
    const tip = await saver.getTuple({
      configurable: { thread_id: "t-fork" },
    });
    expect(
      (tip?.metadata as Record<string, unknown> | undefined)?.[
        CANCELLED_TURN_METADATA_KEY
      ],
    ).toBe(true);

    const second = await completedRootCheckpoint(asBranchGraph(graph), {
      threadId: "t-fork",
      historyId: null,
    });
    await copyCheckpoint(
      asBranchGraph(graph),
      { configurable: { thread_id: "t-fork-2" } },
      second,
    );
    const copied = await graph.getState({
      configurable: { thread_id: "t-fork-2" },
    });
    expect(copied.next).toEqual([]);
    expect(copied.tasks).toEqual([]);
    const messages = copied.values["messages"] as BaseMessage[];
    expect(messages.map((m) => m.content)).toEqual(["never ran"]);
    const copiedTip = await saver.getTuple({
      configurable: { thread_id: "t-fork-2" },
    });
    expect(
      (copiedTip?.metadata as Record<string, unknown>)[
        CANCELLED_TURN_METADATA_KEY
      ],
    ).toBe(true);
  });

  test("agent hook commits the turn by captured identity after a real run", async () => {
    // Full path: invoke builds the input (id assigned at build), the prepare
    // step stashes it, and the teardown hook finds the message by that id —
    // a duplicate-content prompt would defeat content matching.
    const saver = new MemorySaver();
    const graph = new StateGraph(MessagesAnnotation)
      .addNode("reply", async () => ({
        messages: [new AIMessage({ content: "hi", id: "a-1" })],
      }))
      .addEdge(START, "reply")
      .addEdge("reply", END)
      .compile({ checkpointer: saver });
    const agent = new LangGraphBaseAgent(graph as never);
    const ctx = {
      sessionId: "sess-1",
      workspaceId: "ws-1",
    } as RequestContext;
    const input: AgentInput = { payload: { message: "stop this" } };

    await agent.invoke(ctx, input); // completes the turn, stashes the input id
    const committed = await agent.commitCancelledTurn(ctx, input);

    expect(committed).not.toBeNull();
    const tuple = await saver.getTuple({
      configurable: { thread_id: "sess-1:ws-1" },
    });
    const messages = tuple?.checkpoint.channel_values["messages"] as
      | BaseMessage[]
      | undefined;
    expect(messages).toHaveLength(2); // no duplicate user message appended
    const opener = messages?.[0];
    expect(opener?.id).toMatch(/^input:/);
    expect(
      (tuple?.metadata as Record<string, unknown>)[
        CANCELLED_TURN_MESSAGE_ID_METADATA_KEY
      ],
    ).toBe(opener?.id);
  });

  test("agent hook records a hook-rewritten opener by its assigned id", async () => {
    // A prepareInput hook rewrites content and drops ids. The run assigns and
    // captures the opener's id at build time, so the cancel commit records
    // the hook's own persisted message — never a duplicate raw-prompt append.
    const saver = new MemorySaver();
    const graph = new StateGraph(MessagesAnnotation)
      .addNode("reply", async () => ({
        messages: [new AIMessage({ content: "hi", id: "a-1" })],
      }))
      .addEdge(START, "reply")
      .addEdge("reply", END)
      .compile({ checkpointer: saver });
    const agent = new LangGraphBaseAgent(graph as never, [], (input) => ({
      messages: [
        new HumanMessage({
          content: `wrapped: ${String((input.payload as { message?: unknown })?.message ?? "")}`,
        }),
      ],
    }));
    const ctx = {
      sessionId: "sess-1",
      workspaceId: "ws-1",
    } as RequestContext;
    const input: AgentInput = { payload: { message: "original" } };

    await agent.invoke(ctx, input);
    const committed = await agent.commitCancelledTurn(ctx, input);

    expect(committed).not.toBeNull();
    const tuple = await saver.getTuple({
      configurable: { thread_id: "sess-1:ws-1" },
    });
    const messages = tuple?.checkpoint.channel_values["messages"] as
      | BaseMessage[]
      | undefined;
    expect(messages).toHaveLength(2); // hook's opener + reply — no duplicate
    const opener = messages?.[0];
    expect(opener?.content).toBe("wrapped: original");
    expect(opener?.id).toMatch(/^input:/);
    expect(
      (tuple?.metadata as Record<string, unknown>)[
        CANCELLED_TURN_MESSAGE_ID_METADATA_KEY
      ],
    ).toBe(opener?.id);
  });

  // A hook may return supported message-likes instead of BaseMessages: the
  // input-build boundary converts them, so the id the graph persists is the
  // id the teardown commit captured — one opener recorded before the
  // completed step, never a duplicate raw-prompt append.
  test("agent hook tuple opener is captured and recorded after a real run", async () => {
    const saver = new MemorySaver();
    const agent = new LangGraphBaseAgent(
      replyGraph(saver) as never,
      [],
      () => ({ messages: [["human", "stop this"]] }) as never,
    );
    const ctx = { sessionId: "sess-1", workspaceId: "ws-1" } as RequestContext;
    const input: AgentInput = { payload: { message: "stop this" } };

    await agent.invoke(ctx, input);
    expect(await agent.commitCancelledTurn(ctx, input)).not.toBeNull();

    const messages = await tipMessages(saver);
    expect(messages.map((m) => m.content)).toEqual(["stop this", "reply done"]);
    expect(messages[0]?.id).toMatch(/^input:/);
    expect(
      (await tipMetadata(saver))[CANCELLED_TURN_MESSAGE_ID_METADATA_KEY],
    ).toBe(messages[0]?.id);
  });

  test("agent hook role-dict opener keeps a given id", async () => {
    const saver = new MemorySaver();
    const agent = new LangGraphBaseAgent(
      replyGraph(saver) as never,
      [],
      () =>
        ({
          messages: [{ role: "user", content: "stop this", id: "hook-1" }],
        }) as never,
    );
    const ctx = { sessionId: "sess-1", workspaceId: "ws-1" } as RequestContext;
    const input: AgentInput = { payload: { message: "stop this" } };

    await agent.invoke(ctx, input);
    expect(await agent.commitCancelledTurn(ctx, input)).not.toBeNull();

    const messages = await tipMessages(saver);
    expect(messages.map((m) => m.content)).toEqual(["stop this", "reply done"]);
    expect(messages[0]?.id).toBe("hook-1");
    expect(
      (await tipMetadata(saver))[CANCELLED_TURN_MESSAGE_ID_METADATA_KEY],
    ).toBe("hook-1");
  });

  test("agent hook role-dict opener without an id gets the captured id", async () => {
    const saver = new MemorySaver();
    const agent = new LangGraphBaseAgent(
      replyGraph(saver) as never,
      [],
      () => ({ messages: [{ role: "user", content: "stop this" }] }) as never,
    );
    const ctx = { sessionId: "sess-1", workspaceId: "ws-1" } as RequestContext;
    const input: AgentInput = { payload: { message: "stop this" } };

    await agent.invoke(ctx, input);
    expect(await agent.commitCancelledTurn(ctx, input)).not.toBeNull();

    const messages = await tipMessages(saver);
    expect(messages).toHaveLength(2); // no duplicate opener
    expect(messages[0]?.id).toMatch(/^input:/);
    expect(
      (await tipMetadata(saver))[CANCELLED_TURN_MESSAGE_ID_METADATA_KEY],
    ).toBe(messages[0]?.id);
  });

  test("agent hook Overwrite opener replaces history and is recorded", async () => {
    // Overwrite survives the build-time conversion: the hook's list replaces
    // prior history wholesale, and the new opener is what the cancel commits.
    const saver = new MemorySaver();
    const ctx = { sessionId: "sess-1", workspaceId: "ws-1" } as RequestContext;
    await new LangGraphBaseAgent(replyGraph(saver) as never).invoke(ctx, {
      payload: { message: "first" },
    });

    const agent = new LangGraphBaseAgent(
      replyGraph(saver) as never,
      [],
      () =>
        ({
          messages: new Overwrite([
            new HumanMessage({ content: "fresh slate" }),
          ]),
        }) as never,
    );
    const hookedInput: AgentInput = {
      payload: { message: "ignored by the hook" },
    };
    await agent.invoke(ctx, hookedInput);
    expect(await agent.commitCancelledTurn(ctx, hookedInput)).not.toBeNull();

    const messages = await tipMessages(saver);
    expect(messages.map((m) => m.content)).toEqual([
      "fresh slate",
      "reply done",
    ]);
    expect(messages[0]?.id).toMatch(/^input:/);
    expect(
      (await tipMetadata(saver))[CANCELLED_TURN_MESSAGE_ID_METADATA_KEY],
    ).toBe(messages[0]?.id);
  });

  test("agent hook serialized __overwrite__ opener is captured and recorded", async () => {
    // The serialized wire shape crosses a serialization boundary: same
    // replacement semantics through the build-boundary conversion — prior
    // history replaced, one recorded opener, no duplicate.
    const saver = new MemorySaver();
    const ctx = { sessionId: "sess-1", workspaceId: "ws-1" } as RequestContext;
    await new LangGraphBaseAgent(replyGraph(saver) as never).invoke(ctx, {
      payload: { message: "first" },
    });

    const agent = new LangGraphBaseAgent(
      replyGraph(saver) as never,
      [],
      () =>
        ({
          messages: { __overwrite__: [["human", "fresh slate"]] },
        }) as never,
    );
    const hookedInput: AgentInput = {
      payload: { message: "ignored by the hook" },
    };
    await agent.invoke(ctx, hookedInput);
    expect(await agent.commitCancelledTurn(ctx, hookedInput)).not.toBeNull();

    const messages = await tipMessages(saver);
    expect(messages.map((m) => m.content)).toEqual([
      "fresh slate",
      "reply done",
    ]);
    expect(messages[0]?.id).toMatch(/^input:/);
    expect(
      (await tipMetadata(saver))[CANCELLED_TURN_MESSAGE_ID_METADATA_KEY],
    ).toBe(messages[0]?.id);
  });
});
