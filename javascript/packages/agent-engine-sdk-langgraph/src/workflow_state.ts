/** LangGraph application-channel translation for OE durable state. */

import { create } from "@bufbuild/protobuf";
import { isBaseMessage, type BaseMessage } from "@langchain/core/messages";
import {
  StateSnapshotSchema,
  type StateSnapshot,
} from "@mongodb-js/agent-engine-runner-shared";

import { requireJsonObject } from "./workflow_json.js";
import { WorkflowMessageCodec } from "./workflow_message.js";

/** LangGraph reserves these channels for execution mechanics, not app state. */
function isControlChannel(name: string): boolean {
  return (
    name.startsWith("__") ||
    name.startsWith("branch:") ||
    name.startsWith("start:")
  );
}

/** Read and validate the conventional LangGraph messages channel. */
function workflowMessages(value: unknown): BaseMessage[] {
  if (value === undefined) return [];
  if (!Array.isArray(value)) {
    throw new TypeError("LangGraph messages channel must be an array");
  }
  return value.map((message, index) => {
    if (!isBaseMessage(message)) {
      throw new TypeError(`LangGraph messages[${index}] is not a BaseMessage`);
    }
    return message;
  });
}

/** Convert LangGraph application channels into the OE-owned durable snapshot. */
export function channelValuesToStateSnapshot(
  channelValues: Record<string, unknown>,
  includeLegacySource = false,
): StateSnapshot {
  const properties: Record<string, unknown> = {};
  for (const name of Object.keys(channelValues).sort()) {
    if (name === "messages" || isControlChannel(name)) continue;
    // LangGraph materializes optional channels as undefined until they receive
    // a value. JSON has no undefined value; omitting the absent channel
    // reconstructs the same application state.
    if (channelValues[name] === undefined) continue;
    properties[name] = channelValues[name];
  }

  return create(StateSnapshotSchema, {
    properties: requireJsonObject(properties, "LangGraph application state"),
    messageEncodingVersion: 1,
    messages: workflowMessages(channelValues["messages"]).map((message) =>
      WorkflowMessageCodec.fromLangChainMessage(message, includeLegacySource),
    ),
  });
}

/** Restore OE durable state into the channels expected by LangGraph. */
export function stateSnapshotToChannelValues(
  snapshot: StateSnapshot,
): Record<string, unknown> {
  const properties = requireJsonObject(
    snapshot.properties ?? {},
    "previous workflow state",
  );
  for (const name of Object.keys(properties).sort()) {
    if (name === "messages" || isControlChannel(name)) {
      throw new TypeError(
        `previous workflow state contains reserved LangGraph channel "${name}"`,
      );
    }
  }
  const channelValues: Record<string, unknown> = {
    ...properties,
  };
  if (snapshot.messages.length > 0) {
    channelValues["messages"] = snapshot.messages.map((message, index) =>
      WorkflowMessageCodec.toLangChainMessage(message, `messages[${index}]`),
    );
  }
  return channelValues;
}
