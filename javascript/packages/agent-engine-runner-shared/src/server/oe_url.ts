/**
 * Trusted resolution of the Orchestration Engine callback URL.
 *
 * Mirrors Python's `agent_engine_runner_shared.server.oe_url`. Both runtimes are parallel
 * implementations of the same server architecture, so this file and its
 * Python twin must be changed together.
 *
 * `/execute` accepts a `platform_api_url` field, and that value used to
 * become the base for every outbound call the runner made during an
 * execution: tool and LLM approval requests, stream chunks, terminal
 * results, node execution records. The runner then treats those responses
 * as authoritative — as both the real tool/LLM result and the policy
 * approval decision — so a caller-chosen URL could exfiltrate prompts and
 * checkpointed history while injecting fabricated tool output and forged
 * "approved" decisions back into the agent's reasoning, which are then
 * persisted into session state.
 *
 * The runner already knows where its OE is: ECP stamps `OE_URL` on every
 * runner component at deploy time. Preferring that over the request field
 * removes the callback-hijack and SSRF surface rather than trying to
 * validate an attacker-supplied string.
 */

import { getLogger } from "../logger.js";

const logger = getLogger("agent_engine_runner_shared.server.oe_url");

/**
 * Returns the OE base URL to use for callbacks during an execution.
 *
 * The runner's own `OE_URL` wins whenever it is configured. A differing
 * request value is discarded and logged rather than honoured — quietly
 * ignoring it would hide an attempted hijack.
 *
 * When `OE_URL` is unset the requested value is used. That is the local
 * `agentengine dev` and unit-test path, where no deploy-time environment exists
 * and the request originates from the developer's own stack; a warning is
 * emitted so the weaker configuration is visible.
 */
export function resolveOeUrl(
  requested: string,
  env: NodeJS.ProcessEnv = process.env,
): string {
  const configured = (env["OE_URL"] ?? "").trim();
  if (!configured) {
    logger.warn(
      "OE_URL is not configured; falling back to the caller-supplied " +
        "platform_api_url. Set OE_URL so the callback target cannot be " +
        "chosen by the request.",
    );
    return requested;
  }
  if (requested && requested !== configured) {
    logger.warn(
      `Discarding caller-supplied platform_api_url ${JSON.stringify(requested)}; ` +
        `using the configured OE_URL ${JSON.stringify(configured)}.`,
    );
  }
  return configured;
}
