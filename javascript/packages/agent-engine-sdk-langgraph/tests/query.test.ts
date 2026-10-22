import { AIMessage, HumanMessage } from "@langchain/core/messages";
import type { MongoDBSaver } from "@langchain/langgraph-checkpoint-mongodb";
import type { MongoClient } from "mongodb";
import { describe, expect, test } from "vitest";
import {
  CANCELLED_TURN_MESSAGE_ID_METADATA_KEY,
  CANCELLED_TURN_MESSAGE_IDS_METADATA_KEY,
  CANCELLED_TURN_METADATA_KEY,
} from "../src/cancelled_turn.js";
import { LangGraphQueryPlugin } from "../src/query.js";

function emptyCursor() {
  return {
    [Symbol.asyncIterator]: async function* () {},
    close: async () => {},
  };
}

function fakeClient(): MongoClient {
  return {
    db: () => ({
      collection: () => ({ find: () => emptyCursor() }),
    }),
  } as unknown as MongoClient;
}

function fakeSaver(message: unknown): MongoDBSaver {
  return {
    checkpointCollectionName: "checkpoints",
    checkpointWritesCollectionName: "checkpoint_writes",
    list: async function* () {
      yield {
        checkpoint: {
          ts: "2026-01-02T03:04:05.000Z",
          channel_values: { messages: [message] },
        },
      };
    },
  } as unknown as MongoDBSaver;
}

/** Saver stub that records the thread_id of every `list` call. */
function recordingSaver(requestedThreadIds: string[]): MongoDBSaver {
  return {
    checkpointCollectionName: "checkpoints",
    checkpointWritesCollectionName: "checkpoint_writes",
    list: async function* (config: { configurable?: { thread_id?: string } }) {
      requestedThreadIds.push(config.configurable?.thread_id ?? "");
      // No checkpoints — the assertion is on which keys were queried.
      yield* [];
    },
  } as unknown as MongoDBSaver;
}

describe("LangGraphQueryPlugin", () => {
  test("round-trips tool call args into a JSON-safe shape", async () => {
    const message = new AIMessage({
      content: "",
      tool_calls: [
        {
          id: "tc-json",
          name: "my_tool",
          args: {
            city: "Kyoto",
            visited_at: new Date("2026-01-02T03:04:05.000Z"),
          },
          type: "tool_call",
        },
      ],
    } as unknown as ConstructorParameters<typeof AIMessage>[0]);
    const plugin = new LangGraphQueryPlugin({
      saver: fakeSaver(message),
      client: fakeClient(),
      dbName: "unused",
    });

    const result = await plugin.getMessagesForSession("sess-json");

    expect(result.messages[0]?.tool_calls?.[0]?.args).toEqual({
      city: "Kyoto",
      visited_at: "2026-01-02T03:04:05.000Z",
    });
  });

  test("queries only the workspace-scoped key when a scope is known", async () => {
    const requested: string[] = [];
    const plugin = new LangGraphQueryPlugin({
      saver: recordingSaver(requested),
      client: fakeClient(),
      dbName: "unused",
      workspaceIdResolver: () => "ws-1",
    });

    await plugin.getMessagesForSession("sess-1");

    // The bare unscoped key must never be queried once a scope exists —
    // it is readable/writable by every workspace on the shared store.
    expect(requested).toEqual(["sess-1:ws-1"]);
  });

  test("fails closed only when a resolver signals a genuinely unresolvable scope", async () => {
    const requested: string[] = [];
    const plugin = new LangGraphQueryPlugin({
      saver: recordingSaver(requested),
      client: fakeClient(),
      dbName: "unused",
      // A custom resolver returning null says "I cannot decide" — unlike "",
      // which is a deliberate explicitly-unscoped scope.
      workspaceIdResolver: () => null,
    });

    await expect(plugin.getMessagesForSession("sess-1")).rejects.toThrow(
      /workspace scope unavailable/,
    );
    expect(requested).toEqual([]);
  });

  test("an explicitly unscoped runtime (resolver returns '') reads bare keys", async () => {
    // Local dev / tests: no APP_ID, so writes are bare-keyed and reads must
    // match — this is the normal `agentengine dev up` history path.
    const requested: string[] = [];
    const plugin = new LangGraphQueryPlugin({
      saver: recordingSaver(requested),
      client: fakeClient(),
      dbName: "unused",
      workspaceIdResolver: () => "",
    });

    await plugin.getMessagesForSession("sess-1");

    expect(requested).toEqual(["sess-1"]);
  });
});

// ---------------------------------------------------------------------------
// Cancelled-turn outcome (mirror of Python test_query.py's outcome section)
//
// The write path records the turn-opening message id in the cancelled-turn
// checkpoint's metadata (see cancelled_turn.ts). The reader classifies a
// message as stopped only when its id matches that metadata — never from an
// agent-authorable message kwarg.
// ---------------------------------------------------------------------------

function plainTuple(
  ts: string,
  messages: unknown[],
  metadata?: Record<string, unknown>,
) {
  return {
    checkpoint: { ts, channel_values: { messages } },
    config: { configurable: { thread_id: "t", checkpoint_ns: "" } },
    metadata: { source: "loop", step: 1, parents: {}, ...metadata },
  };
}

/**
 * Saver + client stubs whose checkpoint collection exposes the tuples'
 * metadata, mirroring the blob the JS `MongoDBSaver` stores — the
 * cancelled-turn lookup decodes it per checkpoint.
 */
function tupleHarness(tuples: unknown[]): {
  saver: MongoDBSaver;
  client: MongoClient;
  cursorCloses: { count: number };
} {
  const cursorCloses = { count: 0 };
  const saver = {
    checkpointCollectionName: "checkpoints",
    checkpointWritesCollectionName: "checkpoint_writes",
    serde: {
      loadsTyped: async (_type: string, data: unknown) =>
        JSON.parse(String(data)) as unknown,
    },
    list: async function* () {
      yield* tuples;
    },
  } as unknown as MongoDBSaver;
  const client = {
    db: () => ({
      collection: () => ({
        find: () => {
          const docs = tuples
            .map(
              (tuple) =>
                (tuple as { metadata?: Record<string, unknown> }).metadata,
            )
            .filter((metadata): metadata is Record<string, unknown> =>
              Boolean(metadata),
            )
            .map((metadata) => ({
              type: "json",
              metadata: { value: () => JSON.stringify(metadata) },
            }));
          return {
            [Symbol.asyncIterator]: async function* () {
              yield* docs;
            },
            close: async () => {
              cursorCloses.count += 1;
            },
          };
        },
      }),
    }),
  } as unknown as MongoClient;
  return { saver, client, cursorCloses };
}

/**
 * Client whose checkpoint cursor never yields: `next()` hangs until the lookup's
 * abort signal fires, then rejects. Records find calls, aborts, and closes so a
 * test can prove the lookup aborts a stalled read and releases the cursor.
 */
function stalledMarkerClient(record: {
  findCalls: number;
  aborted: number;
  closed: number;
}): MongoClient {
  return {
    db: () => ({
      collection: () => ({
        find: (_query: unknown, options?: { signal?: AbortSignal }) => {
          record.findCalls += 1;
          const signal = options?.signal;
          return {
            [Symbol.asyncIterator]: () => ({
              next: () =>
                new Promise<void>((_resolve, reject) => {
                  const onAbort = () => {
                    record.aborted += 1;
                    reject(signal?.reason);
                  };
                  if (signal?.aborted) {
                    onAbort();
                    return;
                  }
                  signal?.addEventListener("abort", onAbort, { once: true });
                }),
            }),
            close: async () => {
              record.closed += 1;
            },
          };
        },
      }),
    }),
  } as unknown as MongoClient;
}

function outcomes(result: {
  messages: Array<{ outcome?: unknown }>;
}): unknown[] {
  return result.messages.map((m) => m.outcome ?? "");
}

/** Saver + client stubs rich enough for the summaries path. */
function summariesHarness(args: {
  messages: unknown[];
  writesPreview?: string;
}): { saver: MongoDBSaver; client: MongoClient } {
  const saver = {
    checkpointCollectionName: "checkpoints",
    checkpointWritesCollectionName: "checkpoint_writes",
    serde: {
      loadsTyped: async () => [
        new HumanMessage({ content: args.writesPreview ?? "", id: "h9" }),
      ],
    },
    list: async function* () {
      yield {
        checkpoint: {
          ts: "2026-01-02T03:04:05.000Z",
          channel_values: { messages: args.messages },
        },
      };
    },
  } as unknown as MongoDBSaver;
  const client = {
    db: () => ({
      collection: (name: string) => ({
        aggregate: () => ({
          toArray: async () =>
            name === "checkpoints"
              ? [
                  {
                    _id: "s1",
                    latest_ts: new Date("2026-01-02T03:04:05.000Z"),
                    first_ts: new Date("2026-01-02T03:04:05.000Z"),
                  },
                ]
              : args.writesPreview === undefined
                ? []
                : [{ _id: "s1", writes: [{ type: "t", value: "v" }] }],
        }),
      }),
    }),
  } as unknown as MongoClient;
  return { saver, client };
}

describe("LangGraphQueryPlugin cancelled-turn outcome", () => {
  test("a marked opener gets the cancelled outcome", async () => {
    const { saver, client } = tupleHarness([
      plainTuple(
        "2026-01-02T03:04:06.000Z",
        [
          new HumanMessage({ content: "first", id: "h1" }),
          new AIMessage({ content: "first reply", id: "a1" }),
          new HumanMessage({ content: "stop this", id: "h2" }),
        ],
        {
          [CANCELLED_TURN_METADATA_KEY]: true,
          [CANCELLED_TURN_MESSAGE_ID_METADATA_KEY]: "h2",
        },
      ),
      plainTuple("2026-01-02T03:04:05.000Z", [
        new HumanMessage({ content: "first", id: "h1" }),
        new AIMessage({ content: "first reply", id: "a1" }),
      ]),
    ]);
    const plugin = new LangGraphQueryPlugin({
      saver,
      client,
      dbName: "unused",
    });

    const result = await plugin.getMessagesForSession("s1");

    expect(outcomes(result)).toEqual(["", "", "cancelled"]);
  });

  test("a marker on an older checkpoint still classifies the opener", async () => {
    // The marker lives on an older checkpoint in the read window, not on the
    // tip — the reader scans the whole window, so an old cancelled turn keeps
    // its status while later turns are served unmarked.
    const { saver, client } = tupleHarness([
      plainTuple("2026-01-02T03:04:07.000Z", [
        new HumanMessage({ content: "first", id: "h1" }),
        new AIMessage({ content: "first reply", id: "a1" }),
        new HumanMessage({ content: "stop this", id: "h2" }),
        new HumanMessage({ content: "try again", id: "h3" }),
        new AIMessage({ content: "ok this time", id: "a3" }),
      ]),
      plainTuple(
        "2026-01-02T03:04:06.000Z",
        [
          new HumanMessage({ content: "first", id: "h1" }),
          new AIMessage({ content: "first reply", id: "a1" }),
          new HumanMessage({ content: "stop this", id: "h2" }),
        ],
        {
          [CANCELLED_TURN_METADATA_KEY]: true,
          [CANCELLED_TURN_MESSAGE_ID_METADATA_KEY]: "h2",
        },
      ),
    ]);
    const plugin = new LangGraphQueryPlugin({
      saver,
      client,
      dbName: "unused",
    });

    const result = await plugin.getMessagesForSession("s1");

    expect(outcomes(result)).toEqual(["", "", "cancelled", "", ""]);
    // The reader leaves additional_kwargs untouched.
    expect(result.messages.map((m) => m.additional_kwargs ?? null)).toEqual([
      null,
      null,
      null,
      null,
      null,
    ]);
  });

  test("unmarked checkpoints carry no cancelled outcome", async () => {
    const { saver, client } = tupleHarness([
      plainTuple("2026-01-02T03:04:05.000Z", [
        new HumanMessage({ content: "hi", id: "h1" }),
      ]),
    ]);
    const plugin = new LangGraphQueryPlugin({
      saver,
      client,
      dbName: "unused",
    });

    const result = await plugin.getMessagesForSession("s1");

    expect(outcomes(result)).toEqual([""]);
  });

  test("a hand-written kwarg without marker metadata does not classify", async () => {
    // The stopped classification is platform-owned: a `cancelled_turn` entry
    // in a message's additional_kwargs — something an agent could write —
    // must never mark a turn cancelled on its own.
    const { saver, client } = tupleHarness([
      plainTuple("2026-01-02T03:04:05.000Z", [
        new HumanMessage({ content: "first", id: "h1" }),
        new HumanMessage({
          content: "stop this",
          id: "h2",
          additional_kwargs: { cancelled_turn: true },
        }),
      ]),
    ]);
    const plugin = new LangGraphQueryPlugin({
      saver,
      client,
      dbName: "unused",
    });

    const result = await plugin.getMessagesForSession("s1");

    expect(outcomes(result)).toEqual(["", ""]);
    // The kwarg still passes through verbatim — it just classifies nothing.
    expect(result.messages.map((m) => m.additional_kwargs ?? null)).toEqual([
      null,
      { cancelled_turn: true },
    ]);
  });

  test("a marker older than the message window still classifies", async () => {
    // The lookup is decoupled from the 500-checkpoint message window, so a
    // stopped turn keeps its status even when the marker checkpoint sits far
    // outside it.
    const fillers = Array.from({ length: 600 }, (_, index) =>
      plainTuple("2026-01-02T03:01:00.000Z", [
        new HumanMessage({ content: "f", id: `f${index}` }),
      ]),
    );
    const tip = plainTuple("2026-01-02T03:05:00.000Z", [
      new HumanMessage({ content: "stop this", id: "h2" }),
      new AIMessage({ content: "later reply", id: "a9" }),
    ]);
    const marker = plainTuple(
      "2026-01-02T03:00:00.000Z",
      [new HumanMessage({ content: "stop this", id: "h2" })],
      {
        [CANCELLED_TURN_METADATA_KEY]: true,
        [CANCELLED_TURN_MESSAGE_ID_METADATA_KEY]: "h2",
      },
    );
    const { saver, client } = tupleHarness([tip, ...fillers, marker]);
    const plugin = new LangGraphQueryPlugin({
      saver,
      client,
      dbName: "unused",
    });

    const result = await plugin.getMessagesForSession("s1");

    expect(outcomes(result)).toEqual(["cancelled", ""]);
  });

  test("carried cancelled openers from a fork classify every opener", async () => {
    // A fork re-records the source thread's whole set of cancelled-opener ids
    // on its copied tip, because replay rebuilds checkpoints and drops the
    // per-turn marker metadata. The reader resolves every carried id.
    const { saver, client } = tupleHarness([
      plainTuple(
        "2026-01-02T03:04:05.000Z",
        [
          new HumanMessage({ content: "stop one", id: "h2" }),
          new HumanMessage({ content: "stop two", id: "h4" }),
          new AIMessage({ content: "later", id: "a9" }),
        ],
        {
          [CANCELLED_TURN_METADATA_KEY]: true,
          [CANCELLED_TURN_MESSAGE_IDS_METADATA_KEY]: ["h2", "h4"],
        },
      ),
    ]);
    const plugin = new LangGraphQueryPlugin({
      saver,
      client,
      dbName: "unused",
    });

    const result = await plugin.getMessagesForSession("s1");

    expect(outcomes(result)).toEqual(["cancelled", "cancelled", ""]);
  });

  test("the cancelled-turn scan releases its cursor", async () => {
    // The lookup bounds its work with a database-side maxTimeMS and closes the
    // cursor on exit, so an expired read stops consuming the source rather than
    // leaving a whole-thread traversal running.
    const { saver, client, cursorCloses } = tupleHarness([
      plainTuple("2026-01-02T03:04:05.000Z", [
        new HumanMessage({ content: "hi", id: "h1" }),
      ]),
    ]);
    const plugin = new LangGraphQueryPlugin({
      saver,
      client,
      dbName: "unused",
    });

    await plugin.getMessagesForSession("s1");

    expect(cursorCloses.count).toBeGreaterThan(0);
  });

  test("a stalled cancelled-turn cursor is aborted at the budget and released", async () => {
    // The lookup runs under an absolute client-side deadline; aborting the
    // signal interrupts the pending read and rejects the cursor, so expiry
    // stops the work and releases the cursor rather than abandoning it.
    const record = { findCalls: 0, aborted: 0, closed: 0 };
    const { saver } = tupleHarness([
      plainTuple("2026-01-02T03:04:05.000Z", [
        new HumanMessage({ content: "stop this", id: "h2" }),
        new AIMessage({ content: "later reply", id: "a9" }),
      ]),
    ]);
    const plugin = new LangGraphQueryPlugin({
      saver,
      client: stalledMarkerClient(record),
      dbName: "unused",
      cancelledTurnReadBudgetMs: 50,
    });

    const result = await plugin.getMessagesForSession("s1");

    // The transcript still returns; the stalled lookup degrades to unclassified.
    expect(outcomes(result)).toEqual(["", ""]);
    expect(record.aborted).toBeGreaterThan(0);
    expect(record.closed).toBeGreaterThan(0);
  });

  test("the lookup receives only the remainder of the session budget", async () => {
    // A message read that consumes the whole session budget leaves the lookup
    // nothing, so it is skipped rather than adding a fresh upstream timeout.
    const record = { findCalls: 0, aborted: 0, closed: 0 };
    const { saver } = tupleHarness([
      plainTuple("2026-01-02T03:04:05.000Z", [
        new HumanMessage({ content: "hi", id: "h1" }),
      ]),
    ]);
    const plugin = new LangGraphQueryPlugin({
      saver,
      client: stalledMarkerClient(record),
      dbName: "unused",
      sessionReadBudgetMs: 0,
      cancelledTurnReadBudgetMs: 3_000,
    });

    const result = await plugin.getMessagesForSession("s1");

    expect(outcomes(result)).toEqual([""]);
    expect(record.findCalls).toBe(0);
  });
});

describe("LangGraphQueryPlugin summary previews", () => {
  test("preview falls back to authoritative messages when no writes preview exists", async () => {
    // A first prompt rescued by a cancelled-turn commit exists only in the
    // synthetic tip's channel values — the writes-derived preview pipeline has
    // nothing for it, so the preview comes from the authoritative list the
    // count already loaded.
    const { saver, client } = summariesHarness({
      messages: [
        new HumanMessage({ content: "rescued first prompt", id: "h1" }),
      ],
    });
    const plugin = new LangGraphQueryPlugin({
      saver,
      client,
      dbName: "unused",
    });

    const result = await plugin.getSummariesForSessions(["s1"]);

    expect(result.sessions[0]?.message_count).toBe(1);
    expect(result.sessions[0]?.first_message_preview).toBe(
      "rescued first prompt",
    );
  });

  test("the writes-derived preview wins over the fallback", async () => {
    // Normal sessions keep the checkpoint_writes-derived preview — the
    // authoritative-list fallback only fills sessions the pipeline missed.
    const { saver, client } = summariesHarness({
      messages: [new HumanMessage({ content: "from checkpoints", id: "h1" })],
      writesPreview: "from writes",
    });
    const plugin = new LangGraphQueryPlugin({
      saver,
      client,
      dbName: "unused",
    });

    const result = await plugin.getSummariesForSessions(["s1"]);

    expect(result.sessions[0]?.first_message_preview).toBe("from writes");
  });
});
