/**
 * Process-level crash handling for the structured-logging pipeline.
 *
 * Exceptions use `uncaughtExceptionMonitor`: Node keeps printing and exiting, so
 * this never owns termination, and monitors run before any handler the agent
 * installed. Rejections need a real listener — a monitor stops seeing them once
 * the agent registers its own handler.
 */

import type { Logger } from "log4js";

import {
  ACTIVE_ROOT_LOGGER,
  UNCAUGHT_HANDLERS_REGISTERED,
} from "./constants.js";

/** @internal — exported only for tests; not part of the public API. */
export const FALLBACK_EXIT_MS = 5000;

function currentLogger(): Logger {
  return (globalThis as unknown as Record<symbol, unknown>)[
    ACTIVE_ROOT_LOGGER
  ] as Logger;
}

function toError(reason: unknown): Error {
  if (reason instanceof Error) return reason;
  const err = new Error(String(reason), { cause: reason });
  err.name = "NonErrorRejection";
  return err;
}

function logCrash(err: Error, label: string): void {
  currentLogger().error(err, `${label}: ${err.name}: ${err.message}`);
}

function exitSoon(): void {
  if (process.stdout.writableLength === 0) {
    process.exit(1);
  } else {
    process.stdout.once("drain", () => process.exit(1));
  }
}

function handleFatalRejection(err: Error): void {
  logCrash(err, "Unhandled promise rejection");
  process.stderr.write((err.stack ?? String(err)) + "\n");

  // Only own termination when nothing else is listening — another listener may
  // be deliberately recovering. The timer backstops exitSoon()'s drain wait.
  if (process.listenerCount("unhandledRejection") <= 1) {
    process.exitCode = 1;
    exitSoon();
    setTimeout(() => process.exit(1), FALLBACK_EXIT_MS).unref();
  }
}

/** Install the crash listeners. Idempotent — registered once per process. */
export function installUncaughtExceptionHandlers(rootLogger: Logger): void {
  const g = globalThis as unknown as Record<symbol, unknown>;
  g[ACTIVE_ROOT_LOGGER] = rootLogger;
  if (g[UNCAUGHT_HANDLERS_REGISTERED]) return;
  g[UNCAUGHT_HANDLERS_REGISTERED] = true;

  process.on("uncaughtExceptionMonitor", (value) => {
    // Typed `Error`, but Node forwards whatever was thrown.
    logCrash(toError(value as unknown), "Uncaught exception");
  });

  process.on("unhandledRejection", (reason: unknown) => {
    handleFatalRejection(toError(reason));
  });
}
