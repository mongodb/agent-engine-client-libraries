import { describe, expect, it } from "vitest";

import {
  resolveCheckpointWorkspaceId,
  resetCheckpointWorkspaceState,
  noteCheckpointWireWorkspaceId,
  getCheckpointWorkspaceId,
} from "../../src/checkpoint_workspace.js";

describe("resolveCheckpointWorkspaceId", () => {
  it("prefers APP_ID over wire value", () => {
    const prev = process.env["APP_ID"];
    process.env["APP_ID"] = "ws-pod";
    try {
      expect(resolveCheckpointWorkspaceId("ws-other")).toBe("ws-pod");
    } finally {
      if (prev === undefined) delete process.env["APP_ID"];
      else process.env["APP_ID"] = prev;
    }
  });

  it("falls back to wire value when APP_ID is unset", () => {
    const prev = process.env["APP_ID"];
    delete process.env["APP_ID"];
    try {
      expect(resolveCheckpointWorkspaceId("ws-wire")).toBe("ws-wire");
    } finally {
      if (prev === undefined) delete process.env["APP_ID"];
      else process.env["APP_ID"] = prev;
    }
  });

  it("returns null when neither is set", () => {
    const prev = process.env["APP_ID"];
    delete process.env["APP_ID"];
    try {
      expect(resolveCheckpointWorkspaceId(null)).toBeNull();
    } finally {
      if (prev === undefined) delete process.env["APP_ID"];
      else process.env["APP_ID"] = prev;
    }
  });

  it("does not trust the wire workspace when managed scoping is required", () => {
    const prevApp = process.env["APP_ID"];
    const prevRequired = process.env["REQUIRE_PROJECT_SCOPED_DB"];
    delete process.env["APP_ID"];
    process.env["REQUIRE_PROJECT_SCOPED_DB"] = "true";
    try {
      expect(() => resolveCheckpointWorkspaceId("attacker-workspace")).toThrow(
        /APP_ID is not set/,
      );
    } finally {
      if (prevApp === undefined) delete process.env["APP_ID"];
      else process.env["APP_ID"] = prevApp;
      if (prevRequired === undefined)
        delete process.env["REQUIRE_PROJECT_SCOPED_DB"];
      else process.env["REQUIRE_PROJECT_SCOPED_DB"] = prevRequired;
    }
  });
});

describe("getCheckpointWorkspaceId", () => {
  it("uses stored wire fallback for query reads", () => {
    const prev = process.env["APP_ID"];
    delete process.env["APP_ID"];
    resetCheckpointWorkspaceState();
    try {
      noteCheckpointWireWorkspaceId("ws-wire");
      expect(getCheckpointWorkspaceId()).toBe("ws-wire");
    } finally {
      resetCheckpointWorkspaceState();
      if (prev === undefined) delete process.env["APP_ID"];
      else process.env["APP_ID"] = prev;
    }
  });

  it("returns empty string (explicitly unscoped) when no workspace can be resolved", () => {
    // Platform pods always carry APP_ID, so an absent scope means local dev /
    // tests, where checkpoints are bare-keyed by construction — reads must
    // match the write path, not fail closed.
    const prev = process.env["APP_ID"];
    const prevRequired = process.env["REQUIRE_PROJECT_SCOPED_DB"];
    delete process.env["APP_ID"];
    delete process.env["REQUIRE_PROJECT_SCOPED_DB"];
    resetCheckpointWorkspaceState();
    try {
      expect(getCheckpointWorkspaceId()).toBe("");
    } finally {
      if (prev === undefined) delete process.env["APP_ID"];
      else process.env["APP_ID"] = prev;
      if (prevRequired === undefined)
        delete process.env["REQUIRE_PROJECT_SCOPED_DB"];
      else process.env["REQUIRE_PROJECT_SCOPED_DB"] = prevRequired;
    }
  });

  it("fails closed when managed-runtime scoping is required", () => {
    const prev = process.env["APP_ID"];
    const prevRequired = process.env["REQUIRE_PROJECT_SCOPED_DB"];
    delete process.env["APP_ID"];
    process.env["REQUIRE_PROJECT_SCOPED_DB"] = "true";
    resetCheckpointWorkspaceState();
    try {
      expect(() => getCheckpointWorkspaceId()).toThrow(
        /checkpoint workspace scope is required/,
      );
    } finally {
      resetCheckpointWorkspaceState();
      if (prev === undefined) delete process.env["APP_ID"];
      else process.env["APP_ID"] = prev;
      if (prevRequired === undefined)
        delete process.env["REQUIRE_PROJECT_SCOPED_DB"];
      else process.env["REQUIRE_PROJECT_SCOPED_DB"] = prevRequired;
    }
  });
});
