/**
 * AgentEngineToolSandboxBackend — canonical name for AgentEngineToolPodBackend.
 *
 * This module exists so agent authors can use the stable sandbox terminology
 * without depending on the legacy AER/ToolPod names. The implementation is
 * identical; both names refer to the same class.
 *
 * @example
 * ```typescript
 * // The package `exports` map only exposes the root, so import from there.
 * import { AgentEngineToolSandboxBackend } from "@mongodb-js/agent-engine-sdk-langgraph";
 * ```
 */

export { AgentEngineToolPodBackend as AgentEngineToolSandboxBackend } from "./toolpod.js";
