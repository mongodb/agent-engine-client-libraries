/** Stable identities for messages emitted by durable LangGraph nodes. */

import { createHash } from "node:crypto";

import {
  RemoveMessage,
  coerceMessageLikeToMessage,
  type BaseMessage,
  type BaseMessageLike,
} from "@langchain/core/messages";
import { RunnableLambda, type Runnable } from "@langchain/core/runnables";
import {
  Overwrite,
  isCommand,
  messagesStateReducer,
  type Command,
  type CompiledStateGraph,
} from "@langchain/langgraph";
import {
  currentAttemptContext,
  currentOperationPath,
  currentStepOrdinal,
} from "@mongodb-js/agent-engine-runner-shared";

const installedNodes = new WeakSet<object>();

function durableMessageId(
  nodeName: string,
  outputPath: readonly (string | number)[],
): string {
  const attempt = currentAttemptContext();
  if (attempt === null) {
    throw new Error("durable message identity requires an active attempt");
  }
  const { executionId } = attempt.workflowIdentity as { executionId: string };
  const operationPath = currentOperationPath().segments.map((segment) => [
    segment.name,
    segment.ordinal.toString(),
  ]);
  const producer = JSON.stringify([operationPath, nodeName, outputPath]);
  const producerHash = createHash("sha256").update(producer).digest("hex");
  return `durable-message:${executionId}:${currentStepOrdinal()}:${producerHash}`;
}

function assignBaseMessageId(
  message: BaseMessage,
  nodeName: string,
  path: readonly (string | number)[],
): void {
  if (
    (message.id === null || message.id === undefined) &&
    !RemoveMessage.isInstance(message)
  ) {
    const id = durableMessageId(nodeName, path);
    message.id = id;
    message.lc_kwargs.id = id;
  }
}

function assignMessageId(
  value: unknown,
  nodeName: string,
  path: readonly (string | number)[],
): BaseMessage {
  const message = coerceMessageLikeToMessage(value as BaseMessageLike);
  assignBaseMessageId(message, nodeName, path);
  return message;
}

function assignMessageUpdateIds(
  value: unknown,
  nodeName: string,
  path: readonly (string | number)[],
): unknown {
  if (Overwrite.isInstance(value) && "__overwrite__" in value) {
    const overwrite = value as unknown as Record<string, unknown> & {
      __overwrite__: unknown;
    };
    const normalized = assignMessageUpdateIds(
      overwrite.__overwrite__,
      nodeName,
      [...path, "__overwrite__"],
    );
    return value instanceof Overwrite
      ? new Overwrite(normalized)
      : { ...overwrite, __overwrite__: normalized };
  }
  if (Array.isArray(value)) {
    return value.map((item, index) =>
      assignMessageId(item, nodeName, [...path, index]),
    );
  }
  return assignMessageId(value, nodeName, path);
}

function assignStateUpdateIds(
  value: unknown,
  nodeName: string,
  path: readonly (string | number)[],
  messageChannels: ReadonlySet<string>,
): unknown {
  if (value === null || typeof value !== "object") return value;

  const update = value as Record<string, unknown>;
  for (const channel of messageChannels) {
    if (channel in update) {
      update[channel] = assignMessageUpdateIds(update[channel], nodeName, [
        ...path,
        channel,
      ]);
    }
  }
  return update;
}

function withDurableMessageIdentity(
  output: unknown,
  nodeName: string,
  messageChannels: ReadonlySet<string>,
): unknown {
  if (currentAttemptContext() === null) return output;
  if (isCommand(output)) {
    const command = output as Command;
    if (
      command.update !== null &&
      typeof command.update === "object" &&
      !Array.isArray(command.update)
    ) {
      assignStateUpdateIds(
        command.update,
        nodeName,
        ["update"],
        messageChannels,
      );
    }
    // Tuple-pair and implicit-root updates remain LangGraph-owned shapes.
    return command;
  }
  return assignStateUpdateIds(output, nodeName, [], messageChannels);
}

type CompiledNode = { bound: Runnable<unknown, unknown> };
type GraphWithNodes = {
  nodes?: Record<string, CompiledNode>;
  channels?: Record<string, { operator?: unknown }>;
  getSubgraphs?: (
    namespace?: string,
    recurse?: boolean,
  ) => Iterable<[string, unknown]> | undefined;
};

/** Stamp missing durable message IDs before named LangGraph reducers run. */
export function installDurableMessageIdentity(
  graph: CompiledStateGraph<unknown, unknown>,
): void {
  const source = graph as unknown as GraphWithNodes;
  const discovered = source.getSubgraphs?.call(graph, undefined, true);
  const graphs: unknown[] = [
    ...(discovered === undefined
      ? []
      : [...discovered].map(([, subgraph]) => subgraph)),
    graph,
  ];
  for (const compiled of graphs) {
    const { channels, nodes } = compiled as GraphWithNodes;
    if (nodes === undefined) continue;
    const messageChannels = new Set(
      Object.entries(channels ?? {})
        .filter(
          ([name, channel]) =>
            name !== "__root__" && channel.operator === messagesStateReducer,
        )
        .map(([name]) => name),
    );
    if (messageChannels.size === 0) continue;
    for (const [nodeName, node] of Object.entries(nodes)) {
      if (nodeName.startsWith("__") || installedNodes.has(node)) continue;
      node.bound = node.bound.pipe(
        new RunnableLambda({
          func: (output: unknown) =>
            withDurableMessageIdentity(output, nodeName, messageChannels),
        }),
      );
      installedNodes.add(node);
    }
  }
}
