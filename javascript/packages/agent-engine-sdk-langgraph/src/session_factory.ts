/** Select native or durable LangGraph execution once per invoke or stream. */

import type { CompiledStateGraph } from "@langchain/langgraph";
import type { AgentInput, RequestContext } from "@mongodb-js/agent-engine-sdk";
import {
  currentAttemptContext,
  type AttemptContext,
} from "@mongodb-js/agent-engine-runner-shared";

import { branchPointFromMetadata } from "./checkpoint_branch.js";
import { DurableSession, type DurableExecution } from "./durable_session.js";
import {
  type ExecutionSession,
  explicitCheckpointId,
  NativeSession,
  type SessionOptions,
} from "./execution_session.js";
import {
  PlatformCheckpointer,
  requireDurableExecutionId,
  UnsupportedDurableGraphError,
} from "./platform_checkpointer.js";

function validateDurableExecution(
  graph: CompiledStateGraph<unknown, unknown>,
  ctx: RequestContext,
  attempt: AttemptContext,
): DurableExecution {
  const checkpointer = (graph as { checkpointer?: unknown }).checkpointer;
  if (!(checkpointer instanceof PlatformCheckpointer)) {
    throw new UnsupportedDurableGraphError(
      "durable execution requires the platform checkpointer",
    );
  }
  const executionId = requireDurableExecutionId(attempt);
  if (attempt.replayMode !== true) {
    if (explicitCheckpointId(ctx) !== undefined) {
      throw new UnsupportedDurableGraphError(
        "explicit checkpoint targeting is not supported on durable_workflow sessions",
      );
    }
    if (branchPointFromMetadata(ctx.metadata) !== undefined) {
      throw new UnsupportedDurableGraphError(
        "checkpoint branching is not supported on durable_workflow sessions",
      );
    }
  }
  return { attempt, checkpointer, executionId };
}

export function executionSession(
  graph: CompiledStateGraph<unknown, unknown>,
  ctx: RequestContext,
  input: AgentInput,
  options: SessionOptions,
): ExecutionSession {
  const attempt = currentAttemptContext();
  return attempt === null
    ? new NativeSession(graph, ctx, input, options)
    : new DurableSession(
        graph,
        ctx,
        input,
        options,
        validateDurableExecution(graph, ctx, attempt),
      );
}
