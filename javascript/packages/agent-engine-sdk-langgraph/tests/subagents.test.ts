/**
 * Port of `agent-engine-sdk-langgraph/tests/test_subagents.py`.
 *
 * Covers validateSubagentTree — string-model rejection, model-instance
 * acceptance, compiled/remote spec skipping, nested-tree walking, and the
 * recursion depth cap.
 */

import { describe, expect, it } from "vitest";
import type { AnySubAgent } from "deepagents";

import {
  MAX_SUBAGENT_NESTING_DEPTH,
  validateSubagentTree,
} from "../src/subagents.js";

// A stand-in "model instance" — the validator only checks it is not a string.
const modelInstance = { _modelType: "fake" } as unknown;

function subAgent(overrides: Record<string, unknown>): AnySubAgent {
  return {
    name: "sub",
    description: "d",
    systemPrompt: "p",
    ...overrides,
  } as AnySubAgent;
}

describe("validateSubagentTree", () => {
  it("accepts undefined / empty", () => {
    expect(() => validateSubagentTree(undefined)).not.toThrow();
    expect(() => validateSubagentTree([])).not.toThrow();
  });

  it("accepts a model instance", () => {
    expect(() =>
      validateSubagentTree([subAgent({ model: modelInstance })]),
    ).not.toThrow();
  });

  it("accepts a spec with no model (inherits default)", () => {
    expect(() => validateSubagentTree([subAgent({})])).not.toThrow();
  });

  it("rejects a string model", () => {
    expect(() =>
      validateSubagentTree([
        subAgent({ name: "reviewer", model: "openai:gpt-4o" }),
      ]),
    ).toThrow(/string model spec.*bypasses OE routing/s);
  });

  it("skips a compiled subagent (runnable) without inspecting model", () => {
    const compiled = {
      name: "c",
      description: "d",
      runnable: {},
    } as unknown as AnySubAgent;
    expect(() => validateSubagentTree([compiled])).not.toThrow();
  });

  it("skips (warns on) a remote async subagent (graphId)", () => {
    const remote = {
      name: "r",
      description: "d",
      graphId: "remote-graph",
    } as unknown as AnySubAgent;
    expect(() => validateSubagentTree([remote])).not.toThrow();
  });

  it("walks nested subagents and rejects a deep string model", () => {
    const tree = [
      subAgent({
        model: modelInstance,
        subagents: [subAgent({ name: "deep", model: "openai:gpt-4o" })],
      }),
    ];
    expect(() => validateSubagentTree(tree)).toThrow(/SubAgent 'deep'/);
  });

  it("throws when nesting exceeds the depth cap", () => {
    // Build a chain one level deeper than the cap.
    let node = subAgent({ name: "leaf", model: modelInstance });
    for (let i = 0; i < MAX_SUBAGENT_NESTING_DEPTH + 1; i++) {
      node = subAgent({ model: modelInstance, subagents: [node] });
    }
    expect(() => validateSubagentTree([node])).toThrow(
      /nesting exceeds maximum recursion depth/,
    );
  });
});
