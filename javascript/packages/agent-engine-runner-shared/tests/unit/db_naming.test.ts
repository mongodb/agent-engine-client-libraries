/**
 * Tests for per-project store-DB resolution (db_naming + db_config).
 *
 * Mirrors the memory-server / OE / runner-shared(Python) suites — kept in sync
 * deliberately. The runner/AER resolves the same algorithm as the OE
 * so writers and readers converge on the same database.
 */

import { afterEach, beforeEach, describe, expect, it } from "vitest";

import {
  getStoreDbName,
  resetStoreDbCache,
  resolveStoreDbName,
} from "../../src/db_config.js";
import {
  projectScopingRequired,
  resolveEffectiveDb,
  resolveProjectScopedDb,
  type ListsDatabaseNames,
} from "../../src/db_naming.js";

const PROJ = "proj1";

describe("resolveProjectScopedDb", () => {
  const cases: Array<[string, string, string[], string]> = [
    [
      "mdb_agentic_store",
      "",
      ["mdb_agentic_store", "mdb_agentic_store_proj1"],
      "mdb_agentic_store",
    ],
    ["custom_store", "", [], "custom_store"],
    ["mdb_agentic_store_proj1", PROJ, [], "mdb_agentic_store_proj1"],
    [
      "mdb_agentic_store",
      PROJ,
      ["mdb_agentic_store_proj1"],
      "mdb_agentic_store_proj1",
    ],
    [
      "mdb_agentic_store",
      PROJ,
      ["mdb_agentic_store", "admin"],
      "mdb_agentic_store_proj1",
    ],
    ["mdb_agentic_store", PROJ, ["admin"], "mdb_agentic_store_proj1"],
    [
      "mdb_agentic_store",
      PROJ,
      ["mdb_agentic_store", "mdb_agentic_store_proj1"],
      "mdb_agentic_store_proj1",
    ],
    [
      "mdb_agentic_store",
      PROJ,
      ["mdb_agentic_store_other"],
      "mdb_agentic_store_proj1",
    ],
    [
      "mdb_agentic_store_proj1_extra",
      PROJ,
      [],
      "mdb_agentic_store_proj1_extra_proj1",
    ],
  ];
  it.each(cases)(
    "resolve(%s, %s) -> %s",
    (base, projectId, existing, expected) => {
      expect(resolveProjectScopedDb(base, projectId, existing)).toBe(expected);
    },
  );
});

class FakeClient implements ListsDatabaseNames {
  calls = 0;
  constructor(private names: string[]) {}
  async listDatabaseNames(): Promise<string[]> {
    this.calls += 1;
    return this.names;
  }
}

class FailingClient implements ListsDatabaseNames {
  async listDatabaseNames(): Promise<string[]> {
    throw new Error("cluster unreachable");
  }
}

describe("resolveEffectiveDb", () => {
  it("uses scoped when scoped database exists", async () => {
    expect(
      await resolveEffectiveDb(
        new FakeClient(["mdb_agentic_store_proj1"]),
        "mdb_agentic_store",
        PROJ,
      ),
    ).toBe("mdb_agentic_store_proj1");
  });

  it("adopts the previous unscoped default during private preview", async () => {
    expect(
      await resolveEffectiveDb(
        new FakeClient(["mdb_agentic_store"]),
        "mdb_store",
        PROJ,
        { legacyBases: ["mdb_agentic_store"] },
      ),
    ).toBe("mdb_agentic_store");
  });

  it("fails closed on list failure", async () => {
    await expect(
      resolveEffectiveDb(new FailingClient(), "mdb_agentic_store", PROJ),
    ).rejects.toThrow(/database discovery failed/);
  });

  it("adopts the previous scoped default", async () => {
    const projectId = "0123456789abcdef01234567";
    await expect(
      resolveEffectiveDb(
        new FakeClient(["mdb_agentic_store", `mdb_agentic_store_${projectId}`]),
        "mdb_store",
        projectId,
        { legacyBases: ["mdb_agentic_store"] },
      ),
    ).resolves.toBe(`mdb_agentic_store_${projectId}`);
  });

  it("returns unscoped for empty project when not required", async () => {
    expect(
      await resolveEffectiveDb(new FakeClient([]), "mdb_agentic_store", "", {
        required: false,
      }),
    ).toBe("mdb_agentic_store");
  });

  it("throws for empty project when required", async () => {
    await expect(
      resolveEffectiveDb(new FakeClient([]), "mdb_agentic_store", "", {
        required: true,
      }),
    ).rejects.toThrow(/PROJECT_ID is empty/);
  });
});

describe("projectScopingRequired", () => {
  const prev = process.env["REQUIRE_PROJECT_SCOPED_DB"];
  afterEach(() => {
    if (prev === undefined) delete process.env["REQUIRE_PROJECT_SCOPED_DB"];
    else process.env["REQUIRE_PROJECT_SCOPED_DB"] = prev;
  });
  it("reads REQUIRE_PROJECT_SCOPED_DB", () => {
    delete process.env["REQUIRE_PROJECT_SCOPED_DB"];
    expect(projectScopingRequired()).toBe(false);
    process.env["REQUIRE_PROJECT_SCOPED_DB"] = "true";
    expect(projectScopingRequired()).toBe(true);
  });
});

// resolveStoreDbName drives a real MongoClient via `client.db().admin()
// .listDatabases()`, so the fake must expose that shape (not listDatabaseNames).
function fakeMongoClient(
  names: string[],
  counter?: { n: number },
): Parameters<typeof resolveStoreDbName>[0] {
  return {
    db: () => ({
      admin: () => ({
        listDatabases: async () => {
          if (counter) counter.n += 1;
          return { databases: names.map((name) => ({ name })) };
        },
      }),
    }),
  } as unknown as Parameters<typeof resolveStoreDbName>[0];
}

describe("resolveStoreDbName (db_config)", () => {
  const prevProj = process.env["PROJECT_ID"];
  const prevBase = process.env["MDB_AGENTIC_STORE_DB"];

  beforeEach(() => {
    delete process.env["MDB_AGENTIC_STORE_DB"];
    delete process.env["REQUIRE_PROJECT_SCOPED_DB"];
    resetStoreDbCache();
  });
  afterEach(() => {
    if (prevProj === undefined) delete process.env["PROJECT_ID"];
    else process.env["PROJECT_ID"] = prevProj;
    if (prevBase === undefined) delete process.env["MDB_AGENTIC_STORE_DB"];
    else process.env["MDB_AGENTIC_STORE_DB"] = prevBase;
    resetStoreDbCache();
  });

  it("uses an Atlas Flex-compatible default base", () => {
    const projectId = "0123456789abcdef01234567";
    expect(getStoreDbName()).toBe("mdb_store");
    expect(
      Buffer.byteLength(`${getStoreDbName()}_${projectId}`),
    ).toBeLessThanOrEqual(38);
  });

  it("memoizes the resolved name per base", async () => {
    process.env["PROJECT_ID"] = "proj1";
    const counter = { n: 0 };
    const c = fakeMongoClient(["mdb_agentic_store_proj1"], counter);
    expect(await resolveStoreDbName(c)).toBe("mdb_agentic_store_proj1");
    expect(await resolveStoreDbName(c)).toBe("mdb_agentic_store_proj1");
    expect(counter.n).toBe(1);
  });

  it("uses a custom env override exactly", async () => {
    process.env["PROJECT_ID"] = "proj1";
    process.env["MDB_AGENTIC_STORE_DB"] = "custom_store";
    expect(getStoreDbName()).toBe("custom_store");
    const counter = { n: 0 };
    expect(await resolveStoreDbName(fakeMongoClient([], counter))).toBe(
      "custom_store",
    );
    expect(counter.n).toBe(0);
  });

  it("bypasses the cached default for an explicit override", async () => {
    process.env["PROJECT_ID"] = "proj1";
    const counter = { n: 0 };
    const client = fakeMongoClient(["mdb_agentic_store_proj1"], counter);
    expect(await resolveStoreDbName(client)).toBe("mdb_agentic_store_proj1");

    process.env["MDB_AGENTIC_STORE_DB"] = "mdb_store";
    expect(await resolveStoreDbName(client)).toBe("mdb_store");
    expect(counter.n).toBe(1);
  });

  it("uses a programmatic override exactly", async () => {
    process.env["PROJECT_ID"] = "proj1";
    const counter = { n: 0 };
    expect(
      await resolveStoreDbName(
        fakeMongoClient(["mdb_agentic_store_proj1"], counter),
        "custom_store",
      ),
    ).toBe("custom_store");
    expect(counter.n).toBe(0);
  });

  it("treats an empty programmatic override as the managed default", async () => {
    process.env["PROJECT_ID"] = "proj1";
    const counter = { n: 0 };
    expect(
      await resolveStoreDbName(
        fakeMongoClient(["mdb_agentic_store_proj1"], counter),
        "",
      ),
    ).toBe("mdb_agentic_store_proj1");
    expect(counter.n).toBe(1);
  });
});
