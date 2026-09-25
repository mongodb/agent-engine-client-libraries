/** LangChain message translation for the OE durable workflow contract. */

import { create } from "@bufbuild/protobuf";
import {
  AIMessage,
  type BaseMessage,
  HumanMessage,
  isBaseMessage,
  mapStoredMessageToChatMessage,
  type StoredMessage,
  SystemMessage,
  ToolMessage,
} from "@langchain/core/messages";
import type { Message, Role } from "@mongodb-js/agent-engine-sdk";
import {
  MessageRole,
  WorkflowMessageSchema,
  type WorkflowMessage,
} from "@mongodb-js/agent-engine-runner-shared";

import { lcToPlatformMessage } from "./messages.js";
import {
  jsonValueFromProto,
  protoValueFromUnknown,
  serializeJsonObject,
  serializeJsonObjectArray,
} from "./workflow_json.js";

const ROLE_TO_WORKFLOW: Readonly<Record<Role, MessageRole>> = {
  user: MessageRole.USER,
  assistant: MessageRole.ASSISTANT,
  tool: MessageRole.TOOL,
  system: MessageRole.SYSTEM,
};

const ROLE_FROM_WORKFLOW: Readonly<Partial<Record<MessageRole, Role>>> = {
  [MessageRole.USER]: "user",
  [MessageRole.ASSISTANT]: "assistant",
  [MessageRole.TOOL]: "tool",
  [MessageRole.SYSTEM]: "system",
};

function definedToolCallFields(
  toolCall: Record<string, unknown>,
): Record<string, unknown> {
  return Object.fromEntries(
    Object.entries(toolCall).filter(([, value]) => value !== undefined),
  );
}

function durableContent(
  message: BaseMessage,
  platformContent: unknown,
): unknown {
  // LangGraph can transiently place a direct ToolMessage result inside another
  // message's content tuple. That framework-only object is not JSON state, so
  // keep the adapter's established platform normalization for this one shape.
  if (Array.isArray(message.content) && message.content.some(isBaseMessage)) {
    return platformContent;
  }
  return message.content;
}

/** Owns LangChain-specific serialization and validation at the workflow edge. */
export class WorkflowMessageCodec {
  static fromLangChainMessage(
    message: BaseMessage,
    includeLegacySource = false,
  ): WorkflowMessage {
    const platformMessage = lcToPlatformMessage(message);
    const artifactMetadata = message.additional_kwargs["artifacts"];
    const rawPlatformArtifacts =
      artifactMetadata === undefined ? [] : artifactMetadata;
    const rawInvalidToolCalls =
      message.type === "ai"
        ? ((message as AIMessage).invalid_tool_calls ?? [])
        : [];
    const additionalKwargs = { ...platformMessage.additionalKwargs };
    delete additionalKwargs["artifacts"];
    delete additionalKwargs["tool_calls"];

    return create(WorkflowMessageSchema, {
      role: ROLE_TO_WORKFLOW[platformMessage.role],
      content: protoValueFromUnknown(
        durableContent(message, platformMessage.content),
        "message content",
      ),
      toolCalls: serializeJsonObjectArray(
        (platformMessage.toolCalls ?? []).map((toolCall) =>
          definedToolCallFields(toolCall as unknown as Record<string, unknown>),
        ),
        "message tool_calls",
      ),
      invalidToolCalls: serializeJsonObjectArray(
        rawInvalidToolCalls.map((toolCall) =>
          definedToolCallFields(toolCall as unknown as Record<string, unknown>),
        ),
        "message invalid_tool_calls",
      ),
      ...(Object.keys(additionalKwargs).length > 0 && {
        additionalKwargs: serializeJsonObject(
          additionalKwargs,
          "message additional_kwargs",
        ),
      }),
      ...(platformMessage.responseMetadata !== undefined && {
        responseMetadata: serializeJsonObject(
          platformMessage.responseMetadata,
          "message response_metadata",
        ),
      }),
      platformArtifacts: serializeJsonObjectArray(
        rawPlatformArtifacts,
        "message additional_kwargs.artifacts",
      ),
      ...(includeLegacySource && {
        sourceMessage: protoValueFromUnknown(
          this.canonicalSourceMessage(message, platformMessage),
          "message source",
        ),
        artifacts: serializeJsonObjectArray(
          rawPlatformArtifacts,
          "message artifacts",
        ),
      }),
      ...(platformMessage.toolCallId !== undefined && {
        toolCallId: platformMessage.toolCallId,
      }),
      ...(platformMessage.name !== undefined && {
        name: platformMessage.name,
      }),
      ...(platformMessage.id !== undefined && { id: platformMessage.id }),
      ...(platformMessage.isError !== undefined && {
        isError: platformMessage.isError,
      }),
      // Checkpoint serializers may materialize an omitted artifact as null;
      // both shapes mean that no machine-readable tool result exists.
      ...(message instanceof ToolMessage &&
        message.artifact !== undefined &&
        message.artifact !== null && {
          toolArtifact: protoValueFromUnknown(
            message.artifact,
            "message tool_artifact",
          ),
        }),
    });
  }

  static toLangChainMessage(
    message: WorkflowMessage,
    path = "message",
  ): BaseMessage {
    const role = ROLE_FROM_WORKFLOW[message.role];
    if (role === undefined) {
      throw new TypeError(
        `${path}: previous workflow message has an unspecified role`,
      );
    }
    if (message.sourceMessage !== undefined) {
      return this.fromSourceMessage(message, role, path);
    }
    const content = jsonValueFromProto(message.content);
    if (typeof content !== "string" && !Array.isArray(content)) {
      throw new TypeError(
        `${path}: previous workflow message content is malformed`,
      );
    }
    const additionalKwargs = {
      ...message.additionalKwargs,
      ...(message.platformArtifacts.length > 0 && {
        artifacts: message.platformArtifacts,
      }),
    };
    const baseFields = {
      content: content as never,
      additional_kwargs: additionalKwargs,
      response_metadata: message.responseMetadata ?? {},
      ...(message.name !== undefined && { name: message.name }),
      ...(message.id !== undefined && { id: message.id }),
    };

    switch (message.role) {
      case MessageRole.USER:
        return new HumanMessage(baseFields);
      case MessageRole.ASSISTANT:
        return new AIMessage({
          ...baseFields,
          tool_calls: message.toolCalls as never,
          invalid_tool_calls: message.invalidToolCalls as never,
        });
      case MessageRole.TOOL:
        if (message.toolCallId === undefined || message.toolCallId === "") {
          throw new TypeError(
            `${path}: previous workflow message cannot be reconstructed`,
          );
        }
        return new ToolMessage({
          ...baseFields,
          tool_call_id: message.toolCallId,
          status: message.isError === true ? "error" : "success",
          ...(message.toolArtifact !== undefined && {
            artifact: jsonValueFromProto(message.toolArtifact),
          }),
        });
      case MessageRole.SYSTEM:
        return new SystemMessage(baseFields);
      default:
        throw new TypeError(
          `${path}: previous workflow message has an unspecified role`,
        );
    }
  }

  private static canonicalSourceMessage(
    message: BaseMessage,
    platformMessage: Message,
  ): StoredMessage {
    const storedMessage = message.toDict();
    if (storedMessage.type !== "ai") {
      return JSON.parse(JSON.stringify(storedMessage)) as StoredMessage;
    }

    const data = { ...storedMessage.data } as Record<string, unknown>;
    const additionalKwargs =
      data["additional_kwargs"] !== null &&
      typeof data["additional_kwargs"] === "object" &&
      !Array.isArray(data["additional_kwargs"])
        ? { ...(data["additional_kwargs"] as Record<string, unknown>) }
        : {};
    delete additionalKwargs["tool_calls"];
    delete data["tool_call_chunks"];
    delete data["usage_metadata"];
    data["additional_kwargs"] = additionalKwargs;
    data["tool_calls"] = (platformMessage.toolCalls ?? []).map((toolCall) =>
      toolCall.toLangchainDict(),
    );
    storedMessage.data = data as unknown as StoredMessage["data"];
    return JSON.parse(JSON.stringify(storedMessage)) as StoredMessage;
  }

  private static fromSourceMessage(
    message: WorkflowMessage,
    role: Role,
    path: string,
  ): BaseMessage {
    const source = jsonValueFromProto(message.sourceMessage);
    if (
      source === null ||
      Array.isArray(source) ||
      typeof source !== "object"
    ) {
      throw new TypeError(`${path}: previous workflow message is malformed`);
    }
    try {
      const converted = mapStoredMessageToChatMessage(
        source as unknown as StoredMessage,
      );
      const restored = lcToPlatformMessage(converted);
      if (restored.role !== role) {
        throw new TypeError(
          `${path}: previous workflow message role conflicts with source_message`,
        );
      }
      if (message.id !== undefined && restored.id !== message.id) {
        throw new TypeError(
          `${path}: previous workflow message id conflicts with source_message`,
        );
      }
      if (
        message.toolCallId !== undefined &&
        restored.toolCallId !== message.toolCallId
      ) {
        throw new TypeError(
          `${path}: previous workflow tool_call_id conflicts with source_message`,
        );
      }
      return converted;
    } catch (error) {
      if (error instanceof TypeError && error.message.startsWith(`${path}:`)) {
        throw error;
      }
      throw new TypeError(
        `${path}: previous workflow message cannot be reconstructed`,
        { cause: error },
      );
    }
  }
}
