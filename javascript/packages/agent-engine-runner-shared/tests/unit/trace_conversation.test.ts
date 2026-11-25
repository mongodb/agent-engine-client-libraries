import { describe, test, expect, vi } from "vitest";

vi.mock("../../src/logger.js", () => ({
  getLogger: () => ({
    info: vi.fn(),
    error: vi.fn(),
    warn: vi.fn(),
    debug: vi.fn(),
    trace: vi.fn(),
    fatal: vi.fn(),
    log: vi.fn(),
  }),
  setupLogging: vi.fn(),
}));

import {
  BasicTracerProvider,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import type { ReadableSpan, Span } from "@opentelemetry/sdk-trace-base";
import type {
  IExportTraceServiceResponse,
  ISerializer,
} from "@opentelemetry/otlp-transformer";

import { ExecutionContextSpanProcessor } from "../../src/tracing/setup.js";

import { PlatformSpanExporter } from "../../src/index.js";
import {
  ATTR_INPUT_MESSAGES_OFFSET,
  ATTR_INPUT_MESSAGES_RECORDED_BY,
  ATTR_INPUT_MESSAGES_STREAM,
  ATTR_INPUT_MESSAGES_TOTAL,
  RECORDED_BY_CALLER,
  ConversationDeltaTracker,
  RUNTIME_MODE_ATTRIBUTE,
  markInputRecordedByCaller,
  runAsCallerRecordedConversation,
} from "../../src/tracing/conversation.js";

const TRACE_ID = "12345678123456781234567812345678";

function messages(...contents: string[]): Record<string, unknown> {
  const attributes: Record<string, unknown> = {};
  contents.forEach((content, index) => {
    attributes[`llm.input_messages.${index}.message.role`] = "user";
    attributes[`llm.input_messages.${index}.message.content`] = content;
  });
  return attributes;
}

function storedContents(attributes: Record<string, unknown>): string[] {
  return Object.entries(attributes)
    .filter(
      ([key]) =>
        key.startsWith("llm.input_messages.") &&
        key.endsWith(".message.content"),
    )
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([, value]) => String(value));
}

describe("ConversationDeltaTracker", () => {
  test("first call keeps the whole conversation", () => {
    const tracker = new ConversationDeltaTracker();
    const trimmed = tracker.trim(messages("hi"), TRACE_ID, "aer");

    expect(storedContents(trimmed)).toEqual(["hi"]);
    expect(trimmed[ATTR_INPUT_MESSAGES_OFFSET]).toBe(0);
    expect(trimmed[ATTR_INPUT_MESSAGES_TOTAL]).toBe(1);
    expect(trimmed[ATTR_INPUT_MESSAGES_STREAM]).toBe("aer");
  });

  test("later call keeps only the new messages", () => {
    const tracker = new ConversationDeltaTracker();
    tracker.trim(messages("hi"), TRACE_ID, "aer");

    const trimmed = tracker.trim(
      messages("hi", "there", "again"),
      TRACE_ID,
      "aer",
    );

    expect(storedContents(trimmed)).toEqual(["there", "again"]);
    expect(trimmed[ATTR_INPUT_MESSAGES_OFFSET]).toBe(1);
    expect(trimmed[ATTR_INPUT_MESSAGES_TOTAL]).toBe(3);
    expect(trimmed["llm.input_messages.0.message.content"]).toBe("there");
  });

  test("repeated call stores no messages", () => {
    const tracker = new ConversationDeltaTracker();
    tracker.trim(messages("hi", "there"), TRACE_ID, "aer");

    const trimmed = tracker.trim(messages("hi", "there"), TRACE_ID, "aer");

    expect(storedContents(trimmed)).toEqual([]);
    expect(trimmed[ATTR_INPUT_MESSAGES_OFFSET]).toBe(2);
    expect(trimmed[ATTR_INPUT_MESSAGES_TOTAL]).toBe(2);
  });

  test("rewritten history stores the full list", () => {
    const tracker = new ConversationDeltaTracker();
    tracker.trim(messages("hi", "there"), TRACE_ID, "aer");

    const trimmed = tracker.trim(messages("hi", "different"), TRACE_ID, "aer");

    expect(storedContents(trimmed)).toEqual(["hi", "different"]);
    expect(trimmed[ATTR_INPUT_MESSAGES_OFFSET]).toBe(0);
  });

  test("shorter history stores the full list and resets", () => {
    const tracker = new ConversationDeltaTracker();
    tracker.trim(messages("hi", "there"), TRACE_ID, "aer");

    const trimmed = tracker.trim(messages("hi"), TRACE_ID, "aer");

    expect(storedContents(trimmed)).toEqual(["hi"]);
    expect(trimmed[ATTR_INPUT_MESSAGES_OFFSET]).toBe(0);
  });

  test("streams do not share progress", () => {
    const tracker = new ConversationDeltaTracker();
    tracker.trim(messages("hi"), TRACE_ID, "aer");

    const toolPod = tracker.trim(messages("hi"), TRACE_ID, "tool");

    expect(storedContents(toolPod)).toEqual(["hi"]);
    expect(toolPod[ATTR_INPUT_MESSAGES_OFFSET]).toBe(0);
  });

  test("traces do not share progress", () => {
    const tracker = new ConversationDeltaTracker();
    tracker.trim(messages("hi"), TRACE_ID, "aer");

    const otherTrace = tracker.trim(
      messages("hi", "there"),
      "f".repeat(32),
      "aer",
    );

    expect(storedContents(otherTrace)).toEqual(["hi", "there"]);
    expect(otherTrace[ATTR_INPUT_MESSAGES_OFFSET]).toBe(0);
  });

  test("spans without messages are untouched", () => {
    const tracker = new ConversationDeltaTracker();
    const attributes = { "tool.parameters": "{}" };

    expect(tracker.trim(attributes, TRACE_ID, "aer")).toEqual(attributes);
  });

  test("drops the serialized prompt along with the messages", () => {
    const tracker = new ConversationDeltaTracker();
    const trimmed = tracker.trim(
      {
        ...messages("hi", "there"),
        "input.value": '{"messages": [["user", "hi"], ["user", "there"]]}',
        "input.mime_type": "application/json",
      },
      TRACE_ID,
      "aer",
    );

    expect(trimmed["input.value"]).toBeUndefined();
    expect(trimmed["input.mime_type"]).toBeUndefined();
    expect(storedContents(trimmed)).toEqual(["hi", "there"]);
  });

  test("a copy does not share progress with the original", () => {
    const tracker = new ConversationDeltaTracker();
    tracker.trim(messages("hi"), TRACE_ID, "aer");

    const staged = tracker.copy();
    staged.trim(messages("hi", "there"), TRACE_ID, "aer");

    const after = tracker.trim(messages("hi", "there"), TRACE_ID, "aer");
    expect(after[ATTR_INPUT_MESSAGES_OFFSET]).toBe(1);
  });

  test("bounds the number of tracked conversations", () => {
    const tracker = new ConversationDeltaTracker(2);
    for (let index = 0; index < 3; index += 1) {
      tracker.trim(messages("hi"), index.toString(16).padStart(32, "0"), "aer");
    }

    const resent = tracker.trim(messages("hi"), "0".repeat(32), "aer");
    expect(resent[ATTR_INPUT_MESSAGES_OFFSET]).toBe(0);
  });
});

function recordingExporter(): {
  exporter: PlatformSpanExporter;
  stored: Array<Record<string, unknown>>;
} {
  const stored: Array<Record<string, unknown>> = [];
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();
  const serializer: ISerializer<ReadableSpan[], IExportTraceServiceResponse> = {
    serializeRequest: (spans) =>
      encoder.encode(JSON.stringify(spans.map((span) => span.attributes))),
    deserializeResponse: () => ({}),
  };
  const fetchImpl = async (_url: string, init: RequestInit) => {
    stored.push(...JSON.parse(decoder.decode(init.body as Uint8Array)));
    return {
      status: 200,
      headers: { get: () => null },
      arrayBuffer: async () => new ArrayBuffer(0),
    } as unknown as Response;
  };
  const exporter = new PlatformSpanExporter("http://oe:8000", {
    fetchImpl: fetchImpl as unknown as typeof fetch,
    tlsOptions: () => undefined,
    serializer,
  });
  return { exporter, stored };
}

describe("PlatformSpanExporter caller-recorded prompts", () => {
  function makeSpan(attributes: Record<string, unknown>): ReadableSpan {
    return {
      spanContext: () => ({
        traceId: TRACE_ID,
        spanId: "1234567812345678",
        traceFlags: 1,
        isRemote: false,
      }),
      name: "ChatOpenAI",
      kind: 0,
      startTime: [1, 0],
      endTime: [2, 0],
      status: { code: 1 },
      attributes,
      resource: {
        attributes: {
          "service.name": "my-agent",
          [RUNTIME_MODE_ATTRIBUTE]: "aer",
        },
      },
      events: [],
    } as unknown as ReadableSpan;
  }

  function exportAndWait(
    exporter: PlatformSpanExporter,
    spans: ReadableSpan[],
  ) {
    return new Promise<void>((resolve) => {
      exporter.export(spans, () => resolve());
    });
  }

  function callerRecorded(
    attributes: Record<string, unknown>,
  ): Record<string, unknown> {
    return {
      ...attributes,
      [ATTR_INPUT_MESSAGES_RECORDED_BY]: RECORDED_BY_CALLER,
    };
  }

  test("drops a prompt the caller records", async () => {
    const { exporter, stored } = recordingExporter();

    await exportAndWait(exporter, [
      makeSpan(
        callerRecorded({
          ...messages("hi", "there"),
          "input.value": '{"messages": [["user", "hi"]]}',
          "input.mime_type": "application/json",
        }),
      ),
    ]);

    const attributes = stored[0] ?? {};
    expect(storedContents(attributes)).toEqual([]);
    expect(attributes["input.value"]).toBeUndefined();
    expect(attributes["input.mime_type"]).toBeUndefined();
    expect(attributes[ATTR_INPUT_MESSAGES_RECORDED_BY]).toBe(
      RECORDED_BY_CALLER,
    );
    expect(attributes[ATTR_INPUT_MESSAGES_OFFSET]).toBeUndefined();
  });

  test("a dropped prompt does not advance the tracked conversation", async () => {
    const { exporter, stored } = recordingExporter();

    await exportAndWait(exporter, [
      makeSpan(callerRecorded(messages("hi", "there"))),
    ]);
    await exportAndWait(exporter, [
      makeSpan(messages("asked a provider directly")),
    ]);

    const direct = stored[1] ?? {};
    expect(storedContents(direct)).toEqual(["asked a provider directly"]);
    expect(direct[ATTR_INPUT_MESSAGES_OFFSET]).toBe(0);
  });
});

describe("ExecutionContextSpanProcessor", () => {
  async function storedModelAttributes(
    callerRecords: boolean,
  ): Promise<Record<string, unknown>> {
    const { exporter, stored } = recordingExporter();
    const provider = new BasicTracerProvider({
      spanProcessors: [
        new ExecutionContextSpanProcessor(),
        new SimpleSpanProcessor(exporter),
      ],
    });
    const tracer = provider.getTracer("test");

    const start = () => tracer.startSpan("ChatOpenAI");
    const span = callerRecords
      ? runAsCallerRecordedConversation(start)
      : start();
    span.setAttribute("llm.input_messages.0.message.role", "user");
    span.setAttribute("llm.input_messages.0.message.content", "hi");
    span.setAttribute("input.value", '{"messages": [["user", "hi"]]}');
    span.end();

    await vi.waitFor(() => expect(stored).toHaveLength(1));
    return stored[0] ?? {};
  }

  test("stores no prompt for a span the processor marked", async () => {
    const stored = await storedModelAttributes(true);

    expect(storedContents(stored)).toEqual([]);
    expect(stored["input.value"]).toBeUndefined();
    expect(stored[ATTR_INPUT_MESSAGES_RECORDED_BY]).toBe(RECORDED_BY_CALLER);
  });

  test("keeps the prompt when no forwarding route marked the span", async () => {
    const stored = await storedModelAttributes(false);

    expect(storedContents(stored)).toEqual(["hi"]);
    expect(stored[ATTR_INPUT_MESSAGES_RECORDED_BY]).toBeUndefined();
  });
});

describe("caller-recorded conversation marker", () => {
  test("marks spans only inside the scoped block", () => {
    const outside = {
      setAttribute: vi.fn(),
    } as unknown as Span;
    markInputRecordedByCaller(outside);
    expect(outside.setAttribute).not.toHaveBeenCalled();

    const inside = {
      setAttribute: vi.fn(),
    } as unknown as Span;
    runAsCallerRecordedConversation(() => markInputRecordedByCaller(inside));
    expect(inside.setAttribute).toHaveBeenCalledWith(
      ATTR_INPUT_MESSAGES_RECORDED_BY,
      RECORDED_BY_CALLER,
    );
  });

  test("the marker ends with the block", () => {
    const span = { setAttribute: vi.fn() } as unknown as Span;
    runAsCallerRecordedConversation(() => undefined);
    markInputRecordedByCaller(span);

    expect(span.setAttribute).not.toHaveBeenCalled();
  });
});
