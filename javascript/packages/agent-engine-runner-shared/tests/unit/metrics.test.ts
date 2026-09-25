/**
 * Unit tests for logToolResult in metrics.ts.
 *
 * Mirrors Python's tests/unit/test_metrics.py.
 *
 * Python uses logger.log(logging.INFO, ...) and asserts on call_args[0][0].
 * TS uses separate logger.info(...) / logger.error(...) methods — assertion
 * shape changes accordingly (which method was called, not the level argument).
 */

import { describe, test, expect, vi, beforeEach } from "vitest";

// Mock the logger module BEFORE metrics.ts imports it.
// vi.hoisted ensures `mocks` exists when the hoisted vi.mock factory runs.
const mocks = vi.hoisted(() => ({
  logger: {
    info: vi.fn(),
    error: vi.fn(),
    warn: vi.fn(),
    debug: vi.fn(),
    trace: vi.fn(),
    fatal: vi.fn(),
    log: vi.fn(),
  },
}));

vi.mock("../../src/logger.js", () => ({
  getLogger: () => mocks.logger,
  setupLogging: vi.fn(),
}));

import { logToolResult } from "../../src/metrics.js";

describe("logToolResult", () => {
  beforeEach(() => {
    mocks.logger.info.mockClear();
    mocks.logger.error.mockClear();
  });

  test.each(["success", "suspend", "cached"])(
    "logs at INFO level for status=%s",
    (status) => {
      // test_log_tool_result_info_statuses
      logToolResult({
        executionId: "exec-1",
        stepNumber: 1,
        toolName: "my_tool",
        status,
        durationMs: 10.0,
      });

      expect(mocks.logger.info).toHaveBeenCalledOnce();
      expect(mocks.logger.error).not.toHaveBeenCalled();
    },
  );

  test.each(["error", "timeout", "unknown"])(
    "logs at ERROR level for status=%s",
    (status) => {
      // test_log_tool_result_error_statuses
      logToolResult({
        executionId: "exec-1",
        stepNumber: 1,
        toolName: "my_tool",
        status,
        durationMs: 10.0,
      });

      expect(mocks.logger.error).toHaveBeenCalledOnce();
      expect(mocks.logger.info).not.toHaveBeenCalled();
    },
  );
});
