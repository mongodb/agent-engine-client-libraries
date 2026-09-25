/** Base application class for framework-specific SDK integrations. */

import type { ToolDefinition } from "./models.js";

/**
 * Abstract base class for framework-specific SDK integrations.
 *
 * Each framework SDK (LangGraph, CrewAI, etc.) subclasses `BaseApp`
 * and implements the abstract methods using framework-native constructs.
 *
 * The platform runtime sets `toolWrapper` and `llmWrapper` on the
 * instance before calling `entrypoint()` so that tools and LLMs are
 * routed through the secure execution layer.
 */
export abstract class BaseApp {
  readonly name: string;
  toolWrapper: unknown = null;
  llmWrapper: unknown = null;

  constructor(name: string) {
    this.name = name;
  }

  /** Returns all registered tool definitions. Used by the OE to obtain tool execution information. */
  abstract getToolDefinitions(): ToolDefinition[];

  /** Returns a list of wrapped, framework-specific tools. */
  abstract tools(): unknown[];

  /** Decorator that registers a tool on this app. */
  abstract tool(...args: unknown[]): unknown;

  /** Decorator that registers the agent builder function. Called by runtimes to obtain a BaseAgent. */
  abstract entrypoint(fn: unknown): unknown;
}
