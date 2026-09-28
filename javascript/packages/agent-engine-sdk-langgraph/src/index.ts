// Copyright 2026 MongoDB, Inc.
// SPDX-License-Identifier: Apache-2.0

/**
 * LangChain SDK for Atlas Agent Engine.
 *
 * Port of `agent_engine_sdk_langgraph/__init__.py`.
 */

export { LangGraphBaseAgent } from "./agent.js";
export type { PrepareAgentInput, ResolveThreadId } from "./agent.js";
export { LangGraphQueryPlugin } from "./query.js";
export type { LangGraphQueryPluginParams } from "./query.js";
export { App } from "./runtime.js";
export type { DeepAgentOptions } from "./runtime.js";
export type { SessionFinishStatus } from "@mongodb-js/agent-engine-runner-shared";
export { getCallAbortSignal } from "@mongodb-js/agent-engine-runner-shared";
export { withCallInterruptSupport } from "./call_interrupt.js";
export {
  createAgentEngineDeepAgent,
  type CreateAgentEngineDeepAgentOptions,
} from "./deep_agent.js";
export {
  validateSubagentTree,
  MAX_SUBAGENT_NESTING_DEPTH,
} from "./subagents.js";
export { AgentEngineToolPodBackend } from "./backends/toolpod.js";
export { AgentEngineToolSandboxBackend } from "./backends/tool_sandbox.js";
