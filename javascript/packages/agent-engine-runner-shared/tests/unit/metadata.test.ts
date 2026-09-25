/**
 * Tests for the metadata secret reader and env applicator.
 *
 * Mirrors runner-shared/tests/unit/test_metadata.py.
 */

import { describe, test, expect, beforeEach, afterEach } from "vitest";
import { mkdtempSync, rmSync, writeFileSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  mergeMetadataEnv,
  readMetadataSecrets,
  withMetadataEnv,
  withMetadataEnvGen,
} from "../../src/server/metadata.js";

let tmpDir: string;

beforeEach(() => {
  tmpDir = mkdtempSync(join(tmpdir(), "metadata-"));
});

afterEach(() => {
  rmSync(tmpDir, { recursive: true, force: true });
});

describe("readMetadataSecrets", () => {
  test("reads files as key-value pairs", () => {
    writeFileSync(join(tmpDir, "STRIPE_KEY"), "sk_live_123");
    writeFileSync(join(tmpDir, "OPENAI_API_KEY"), "sk-openai-456");

    expect(readMetadataSecrets(tmpDir)).toEqual({
      STRIPE_KEY: "sk_live_123",
      OPENAI_API_KEY: "sk-openai-456",
    });
  });

  test("returns empty when dir missing", () => {
    expect(readMetadataSecrets(join(tmpDir, "nonexistent"))).toEqual({});
  });

  test("skips subdirectories", () => {
    writeFileSync(join(tmpDir, "SECRET"), "val");
    mkdirSync(join(tmpDir, "subdir"));

    expect(readMetadataSecrets(tmpDir)).toEqual({ SECRET: "val" });
  });

  test("returns empty for empty dir", () => {
    expect(readMetadataSecrets(tmpDir)).toEqual({});
  });
});

describe("withMetadataEnv", () => {
  test("injects secrets and restores", async () => {
    writeFileSync(join(tmpDir, "NEW_SECRET"), "secret_value");
    expect(process.env["NEW_SECRET"]).toBeUndefined();

    await withMetadataEnv(async () => {
      expect(process.env["NEW_SECRET"]).toBe("secret_value");
    }, tmpDir);

    expect(process.env["NEW_SECRET"]).toBeUndefined();
  });

  test("restores overwritten keys", async () => {
    process.env["EXISTING_KEY"] = "original";
    writeFileSync(join(tmpDir, "EXISTING_KEY"), "overwritten");

    try {
      await withMetadataEnv(async () => {
        expect(process.env["EXISTING_KEY"]).toBe("overwritten");
      }, tmpDir);

      expect(process.env["EXISTING_KEY"]).toBe("original");
    } finally {
      delete process.env["EXISTING_KEY"];
    }
  });

  test("no-op when dir missing", async () => {
    process.env["SENTINEL"] = "keep";
    try {
      await withMetadataEnv(async () => {}, join(tmpDir, "nonexistent"));
      expect(process.env["SENTINEL"]).toBe("keep");
    } finally {
      delete process.env["SENTINEL"];
    }
  });

  test("no-op when dir empty", async () => {
    process.env["SENTINEL"] = "keep";
    try {
      await withMetadataEnv(async () => {}, tmpDir);
      expect(process.env["SENTINEL"]).toBe("keep");
    } finally {
      delete process.env["SENTINEL"];
    }
  });

  test("restores on exception", async () => {
    writeFileSync(join(tmpDir, "TEMP_SECRET"), "temp");
    expect(process.env["TEMP_SECRET"]).toBeUndefined();

    await expect(
      withMetadataEnv(async () => {
        expect(process.env["TEMP_SECRET"]).toBe("temp");
        throw new Error("boom");
      }, tmpDir),
    ).rejects.toThrow("boom");

    expect(process.env["TEMP_SECRET"]).toBeUndefined();
  });

  test("reinstates keys deleted during the call", async () => {
    process.env["PRE_EXISTING"] = "keep-me";
    try {
      await withMetadataEnv(async () => {
        delete process.env["PRE_EXISTING"];
      }, tmpDir);
      expect(process.env["PRE_EXISTING"]).toBe("keep-me");
    } finally {
      delete process.env["PRE_EXISTING"];
    }
  });

  test("removes keys added during the call", async () => {
    writeFileSync(join(tmpDir, "META_KEY"), "meta_val");

    await withMetadataEnv(async () => {
      process.env["TOOL_ADDED_KEY"] = "should_vanish";
    }, tmpDir);

    expect(process.env["META_KEY"]).toBeUndefined();
    expect(process.env["TOOL_ADDED_KEY"]).toBeUndefined();
  });
});

describe("mergeMetadataEnv", () => {
  test("overwrites idempotently without restore", () => {
    writeFileSync(join(tmpDir, "PERSIST_SECRET"), "keep-me");
    expect(process.env["PERSIST_SECRET"]).toBeUndefined();

    mergeMetadataEnv(tmpDir);
    mergeMetadataEnv(tmpDir);
    expect(process.env["PERSIST_SECRET"]).toBe("keep-me");

    writeFileSync(join(tmpDir, "PERSIST_SECRET"), "rotated");
    mergeMetadataEnv(tmpDir);
    expect(process.env["PERSIST_SECRET"]).toBe("rotated");
    delete process.env["PERSIST_SECRET"];
  });
});

describe("overlapping applied-env regions fail closed", () => {
  test("a secret-bearing region overlapping another secret region throws before mutating env", async () => {
    const dirA = join(tmpDir, "a");
    const dirB = join(tmpDir, "b");
    mkdirSync(dirA);
    mkdirSync(dirB);
    writeFileSync(join(dirA, "SECRET_A"), "value_a");
    writeFileSync(join(dirB, "SECRET_B"), "value_b");

    // Without the guard this interleaving is the fail-closed repro: B enters
    // while A's secrets are applied, snapshots them as baseline, and B's
    // restore re-installs SECRET_A into process.env permanently.
    await withMetadataEnv(async () => {
      expect(process.env["SECRET_A"]).toBe("value_a");
      await expect(withMetadataEnv(async () => {}, dirB)).rejects.toThrow(
        "overlaps a concurrent secret-bearing region",
      );
      // The rejected region must not have touched the live secrets.
      expect(process.env["SECRET_A"]).toBe("value_a");
      expect(process.env["SECRET_B"]).toBeUndefined();
    }, dirA);

    expect(process.env["SECRET_A"]).toBeUndefined();
    expect(process.env["SECRET_B"]).toBeUndefined();
  });

  test("a secret-bearing region overlapping an open secret stream throws", async () => {
    const dirA = join(tmpDir, "a");
    const dirB = join(tmpDir, "b");
    mkdirSync(dirA);
    mkdirSync(dirB);
    writeFileSync(join(dirA, "STREAM_SECRET"), "stream_val");
    writeFileSync(join(dirB, "PLAIN_SECRET"), "plain_val");

    async function* source(): AsyncGenerator<string> {
      yield "a";
      yield "b";
    }

    const gen = withMetadataEnvGen(() => source(), dirA);
    const first = await gen.next();
    expect(first.value).toBe("a");
    expect(process.env["STREAM_SECRET"]).toBe("stream_val");

    // The stream region is open: a secret-bearing call must not snapshot it.
    await expect(withMetadataEnv(async () => {}, dirB)).rejects.toThrow(
      "overlaps a concurrent secret-bearing region",
    );
    expect(process.env["STREAM_SECRET"]).toBe("stream_val");
    expect(process.env["PLAIN_SECRET"]).toBeUndefined();

    // After the stream closes, the same call succeeds with only its secrets.
    for await (const item of gen) void item;
    expect(process.env["STREAM_SECRET"]).toBeUndefined();
    await withMetadataEnv(async () => {
      expect(process.env["PLAIN_SECRET"]).toBe("plain_val");
      expect(process.env["STREAM_SECRET"]).toBeUndefined();
    }, dirB);
    expect(process.env["PLAIN_SECRET"]).toBeUndefined();
  });

  test("a NUL-containing secret value does not wedge later regions", async () => {
    const dirBad = join(tmpDir, "bad");
    const dirOk = join(tmpDir, "ok");
    mkdirSync(dirBad);
    mkdirSync(dirOk);
    // Metadata files are opaque bytes; depending on the Node version and
    // platform, assigning a NUL-containing value to process.env either
    // throws or silently truncates. Neither outcome may unbalance the
    // overlap accounting (which would reject every later region) or leave
    // residue in the shared environment.
    writeFileSync(join(dirBad, "GOOD_KEY"), "fine");
    writeFileSync(join(dirBad, "NUL_KEY"), "bad\u0000value");
    writeFileSync(join(dirOk, "OK_SECRET"), "ok_val");

    await withMetadataEnv(async () => {}, dirBad).catch(() => {});
    expect(process.env["GOOD_KEY"]).toBeUndefined();
    expect(process.env["NUL_KEY"]).toBeUndefined();

    // Later regions still work and restore cleanly.
    await withMetadataEnv(async () => {
      expect(process.env["OK_SECRET"]).toBe("ok_val");
    }, dirOk);
    expect(process.env["OK_SECRET"]).toBeUndefined();
  });

  test("overlapping secret-less restores do not resurrect callback-added env state", async () => {
    // Region A's callback sets a request-scoped env value; region B overlaps
    // and snapshots it. A's restore removes it — B's restore must not
    // blanket-reassign its stale snapshot and reintroduce A's value.
    const emptyA = join(tmpDir, "empty-a");
    const emptyB = join(tmpDir, "empty-b");
    mkdirSync(emptyA);
    mkdirSync(emptyB);

    let releaseA: () => void = () => {};
    const aGate = new Promise<void>((resolve) => {
      releaseA = resolve;
    });
    let releaseB: () => void = () => {};
    const bGate = new Promise<void>((resolve) => {
      releaseB = resolve;
    });

    // A enters and sets a request-scoped key (runs synchronously up to the
    // first await), then B enters and snapshots the env *with* that key.
    const callA = withMetadataEnv(async () => {
      process.env["REQUEST_SCOPED"] = "a-value";
      await aGate;
    }, emptyA);
    const callB = withMetadataEnv(async () => {
      await bGate;
    }, emptyB);

    releaseA();
    await callA;
    // A's restore removed its callback-added key.
    expect(process.env["REQUEST_SCOPED"]).toBeUndefined();

    releaseB();
    await callB;
    // B's snapshot carried REQUEST_SCOPED, but B must not re-add it.
    expect(process.env["REQUEST_SCOPED"]).toBeUndefined();
  });

  test("overlapping restore does not reassert an overwritten pre-existing value", async () => {
    // A (entered clean) overwrites a pre-existing key; B overlaps and
    // snapshots A's request-scoped value. A's restore reverts to baseline —
    // B's tainted restore must not write A's value back.
    const emptyA = join(tmpDir, "empty-a");
    const emptyB = join(tmpDir, "empty-b");
    mkdirSync(emptyA);
    mkdirSync(emptyB);
    process.env["BASELINE_KEY"] = "baseline";
    try {
      let releaseA: () => void = () => {};
      const aGate = new Promise<void>((resolve) => {
        releaseA = resolve;
      });
      let releaseB: () => void = () => {};
      const bGate = new Promise<void>((resolve) => {
        releaseB = resolve;
      });

      const callA = withMetadataEnv(async () => {
        process.env["BASELINE_KEY"] = "a-value";
        await aGate;
      }, emptyA);
      const callB = withMetadataEnv(async () => {
        await bGate;
      }, emptyB);

      releaseA();
      await callA;
      expect(process.env["BASELINE_KEY"]).toBe("baseline");

      releaseB();
      await callB;
      expect(process.env["BASELINE_KEY"]).toBe("baseline");
    } finally {
      delete process.env["BASELINE_KEY"];
    }
  });

  test("secret-less regions may still overlap (cross-instance ToolServer contract)", async () => {
    // Regions with no secrets snapshot an env identical to the live one, so
    // restore is idempotent and overlap is harmless — separate ToolServer
    // instances rely on being able to run concurrently.
    const emptyA = join(tmpDir, "empty-a");
    const emptyB = join(tmpDir, "empty-b");
    mkdirSync(emptyA);
    mkdirSync(emptyB);

    const release = { resolve: () => {} };
    const gate = new Promise<void>((resolve) => {
      release.resolve = resolve;
    });
    const callA = withMetadataEnv(async () => {
      await gate;
      return "a";
    }, emptyA);
    const callB = withMetadataEnv(async () => {
      release.resolve();
      return "b";
    }, emptyB);

    await expect(Promise.all([callA, callB])).resolves.toEqual(["a", "b"]);
  });
});

describe("withMetadataEnvGen", () => {
  test("secrets present during iteration and restored after", async () => {
    writeFileSync(join(tmpDir, "GEN_SECRET"), "gen_val");
    expect(process.env["GEN_SECRET"]).toBeUndefined();

    async function* source(): AsyncGenerator<string> {
      expect(process.env["GEN_SECRET"]).toBe("gen_val");
      yield "a";
      expect(process.env["GEN_SECRET"]).toBe("gen_val");
      yield "b";
    }

    const seen: string[] = [];
    for await (const item of withMetadataEnvGen(() => source(), tmpDir)) {
      expect(process.env["GEN_SECRET"]).toBe("gen_val");
      seen.push(item);
    }

    expect(seen).toEqual(["a", "b"]);
    expect(process.env["GEN_SECRET"]).toBeUndefined();
  });

  test("restores when iteration is closed early", async () => {
    writeFileSync(join(tmpDir, "GEN_SECRET"), "gen_val");

    async function* source(): AsyncGenerator<string> {
      yield "a";
      yield "b";
    }

    for await (const item of withMetadataEnvGen(() => source(), tmpDir)) {
      expect(process.env["GEN_SECRET"]).toBe("gen_val");
      void item;
      break;
    }

    expect(process.env["GEN_SECRET"]).toBeUndefined();
  });
});
