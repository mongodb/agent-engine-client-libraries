/** Mirrors sdk-core/tests/test_app.py. */

import { describe, expect, it } from "vitest";
import { BaseApp } from "../src/app.js";
import type { ToolDefinition } from "../src/index.js";

class ConcreteApp extends BaseApp {
  private readonly _toolDefs: ToolDefinition[] = [];

  constructor() {
    super("TestApp");
  }

  getToolDefinitions(): ToolDefinition[] {
    return this._toolDefs;
  }

  tools(): unknown[] {
    return [];
  }

  tool(..._args: unknown[]): unknown {
    return (fn: unknown) => fn;
  }

  entrypoint(fn: unknown): unknown {
    return fn;
  }
}

describe("BaseApp", () => {
  it("concrete subclass instantiates", () => {
    const app = new ConcreteApp();
    expect(app.name).toBe("TestApp");
  });

  it("wrapper attributes default to null", () => {
    const app = new ConcreteApp();
    expect(app.toolWrapper).toBeNull();
    expect(app.llmWrapper).toBeNull();
  });

  it("wrapper injection", () => {
    const app = new ConcreteApp();
    app.toolWrapper = "mock_tool_wrapper";
    app.llmWrapper = "mock_llm_wrapper";
    expect(app.toolWrapper).toBe("mock_tool_wrapper");
    expect(app.llmWrapper).toBe("mock_llm_wrapper");
  });

  it("getToolDefinitions returns empty list", () => {
    const app = new ConcreteApp();
    expect(app.getToolDefinitions()).toEqual([]);
  });
});
