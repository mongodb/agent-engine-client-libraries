/**
 * Capture the structured JSON records the runner ships, for tests that assert
 * record attribution. Same mechanism as `setupCapture()` in
 * `structured_logging.test.ts`: a capturing `write` is installed before
 * `installStructuredLogging` so the appender snapshots it.
 */

import log4js from "log4js";

import { installStructuredLogging } from "../../src/structured_logging.js";

const ORIGINAL_WRITE_SLOT = Symbol.for(
  "agent_engine_runner_shared.original_write",
);

export interface JsonRecord {
  message: string;
  executionId: string | null;
  sessionId: string | null;
  source: string;
  origin?: string;
  traceId?: string;
}

export interface JsonLogCapture {
  records: JsonRecord[];
  ids: (prefix: string) => Set<string | null>;
  restore: () => void;
}

function clearSlots(): void {
  for (const stream of [process.stdout, process.stderr]) {
    delete (stream as unknown as Record<symbol, unknown>)[ORIGINAL_WRITE_SLOT];
  }
}

export function captureJsonLogs(): JsonLogCapture {
  const realStdout = process.stdout.write.bind(process.stdout);
  const realStderr = process.stderr.write.bind(process.stderr);
  const records: JsonRecord[] = [];
  let buffer = "";
  const write = ((chunk: unknown, ...rest: unknown[]): boolean => {
    buffer += Buffer.isBuffer(chunk) ? chunk.toString("utf-8") : String(chunk);
    let nl = buffer.indexOf("\n");
    while (nl !== -1) {
      const line = buffer.slice(0, nl);
      buffer = buffer.slice(nl + 1);
      try {
        records.push(JSON.parse(line) as JsonRecord);
      } catch {
        // Not a structured record.
      }
      nl = buffer.indexOf("\n");
    }
    const cb = rest.find((a) => typeof a === "function");
    if (typeof cb === "function") (cb as () => void)();
    return true;
  }) as NodeJS.WriteStream["write"];

  clearSlots();
  process.stdout.write = write;
  process.stderr.write = write;
  installStructuredLogging({ level: "debug", mode: "tool" });

  return {
    records,
    ids: (prefix) => {
      const matching = records.filter((r) => r.message.startsWith(prefix));
      if (matching.length === 0) {
        throw new Error(`no record starting with ${JSON.stringify(prefix)}`);
      }
      return new Set(matching.map((r) => r.executionId));
    },
    restore: () => {
      process.stdout.write = realStdout;
      process.stderr.write = realStderr;
      clearSlots();
      log4js.shutdown(() => {});
    },
  };
}
