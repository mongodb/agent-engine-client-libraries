/**
 * Tests for App.resolveCheckpointDbName() / checkpointer() DB selection —
 * the async per-project store-DB resolution that runs in run() before the
 * checkpointer/query plugin are wired, plus the exact
 * CHECKPOINT_DB_NAME override used by builders that call checkpointer()
 * before run().
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const resolveStoreDbNameMock = vi.fn();
const closeMock = vi.fn();
const dbMock = vi.fn();
const mongoDBSaverCtor = vi.fn();
const registerQueryPluginMock = vi.fn();

vi.mock("mongodb", () => ({
  MongoClient: class FakeMongoClient {
    close = closeMock;
    db = dbMock;
  },
}));

vi.mock("@langchain/langgraph-checkpoint-mongodb", () => ({
  MongoDBSaver: class FakeMongoDBSaver {
    dbName: string;
    checkpointCollectionName = "checkpoints";
    checkpointWritesCollectionName = "checkpoint_writes";
    constructor(opts: { dbName: string }) {
      mongoDBSaverCtor(opts);
      this.dbName = opts.dbName;
    }
    putWrites = vi.fn();
  },
}));

vi.mock("@mongodb-js/agent-engine-runner-shared", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return {
    ...actual,
    resolveStoreDbName: resolveStoreDbNameMock,
    registerQueryPlugin: registerQueryPluginMock,
  };
});

const { App } = await import("../src/runtime.js");

type PrivateApp = {
  resolveCheckpointDbName: () => Promise<void>;
  resolvedCheckpointDb: string | null;
  registerQueryPlugin: () => void;
  checkpointDbName: () => string;
};

describe("App.resolveCheckpointDbName()", () => {
  let savedMode: string | undefined;
  let savedUri: string | undefined;
  let savedCheckpointDb: string | undefined;

  beforeEach(() => {
    savedMode = process.env["RUNNER_MODE"];
    savedUri = process.env["MONGODB_URI"];
    savedCheckpointDb = process.env["CHECKPOINT_DB_NAME"];
    process.env["RUNNER_MODE"] = "aer";
    process.env["MONGODB_URI"] = "mongodb://localhost:27017";
    delete process.env["CHECKPOINT_DB_NAME"];
    resolveStoreDbNameMock.mockReset();
    closeMock.mockReset();
    mongoDBSaverCtor.mockReset();
    registerQueryPluginMock.mockReset();
    dbMock.mockReset();
    dbMock.mockReturnValue({
      collection: vi.fn().mockReturnValue({}),
    });
  });

  afterEach(() => {
    if (savedMode === undefined) delete process.env["RUNNER_MODE"];
    else process.env["RUNNER_MODE"] = savedMode;
    if (savedUri === undefined) delete process.env["MONGODB_URI"];
    else process.env["MONGODB_URI"] = savedUri;
    if (savedCheckpointDb === undefined)
      delete process.env["CHECKPOINT_DB_NAME"];
    else process.env["CHECKPOINT_DB_NAME"] = savedCheckpointDb;
  });

  it("stores the resolved scoped DB name and closes the probe client", async () => {
    resolveStoreDbNameMock.mockResolvedValue("mdb_agentic_store_proj1");
    const app = new App({ appName: "Test Agent" }) as unknown as PrivateApp;

    await app.resolveCheckpointDbName();

    expect(resolveStoreDbNameMock).toHaveBeenCalledTimes(1);
    expect(app.resolvedCheckpointDb).toBe("mdb_agentic_store_proj1");
    expect(closeMock).toHaveBeenCalledTimes(1);
  });

  it("uses CHECKPOINT_DB_NAME exactly and skips project scoping", async () => {
    process.env["CHECKPOINT_DB_NAME"] = "external_checkpoints_agent";
    const app = new App({ appName: "Test Agent" }) as unknown as PrivateApp;

    await app.resolveCheckpointDbName();

    expect(resolveStoreDbNameMock).not.toHaveBeenCalled();
    expect(app.resolvedCheckpointDb).toBe("external_checkpoints_agent");
    expect(closeMock).not.toHaveBeenCalled();
  });

  it("fails closed and still closes the client on resolution failure", async () => {
    resolveStoreDbNameMock.mockRejectedValue(new Error("cluster unreachable"));
    const app = new App({ appName: "Test Agent" }) as unknown as PrivateApp;

    await expect(app.resolveCheckpointDbName()).rejects.toThrow(
      "cluster unreachable",
    );

    expect(app.resolvedCheckpointDb).toBeNull();
    expect(closeMock).toHaveBeenCalledTimes(1);
  });

  it("passes CHECKPOINT_DB_NAME to MongoDBSaver before resolveCheckpointDbName", () => {
    // Agent builders call app.checkpointer() from the entrypoint before run()
    // resolves the per-project store — the override must still win.
    process.env["CHECKPOINT_DB_NAME"] = "external_checkpoints_agent";
    const app = new App({ appName: "Test Agent" });

    app.checkpointer();

    expect(resolveStoreDbNameMock).not.toHaveBeenCalled();
    expect(mongoDBSaverCtor).toHaveBeenCalledWith(
      expect.objectContaining({ dbName: "external_checkpoints_agent" }),
    );
  });

  it("registers the query plugin against the same CHECKPOINT_DB_NAME", async () => {
    process.env["CHECKPOINT_DB_NAME"] = "external_checkpoints_agent";
    const app = new App({ appName: "Test Agent" }) as unknown as PrivateApp;

    await app.resolveCheckpointDbName();
    app.registerQueryPlugin();

    expect(registerQueryPluginMock).toHaveBeenCalledTimes(1);
    expect(dbMock).toHaveBeenCalledWith("external_checkpoints_agent");
    expect(app.checkpointDbName()).toBe("external_checkpoints_agent");
  });
});
