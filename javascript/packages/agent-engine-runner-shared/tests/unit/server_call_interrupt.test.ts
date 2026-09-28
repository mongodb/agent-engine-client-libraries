/**
 * Per-call interrupt receiver (POST /interrupt/call), TypeScript side.
 *
 * Mirrors Python's tests/unit/test_call_interrupt.py — the two runner
 * runtimes are parallel implementations, so the twins must stay
 * behaviourally identical. The cross-language request/response vectors live
 * in `client-libraries/test-fixtures/interrupt-call/contract.json` and are
 * pinned by server_call_interrupt_contract.test.ts.
 *
 * TS-vs-Python divergence (mirrors the drain twin boundary): a promise has
 * no cancellation, so "signal the call" aborts that call's AbortController,
 * which reaches only work with a signal channel. Track-only work (plain tool
 * functions) answers not_cancellable rather than claiming abandonment.
 */

import { describe, expect, test } from "vitest";

import {
  CALL_INTERRUPT_REASON,
  DrainRegistry,
} from "../../src/server/drain.js";

describe("abortCall", () => {
  test("signals only the named step", () => {
    const registry = new DrainRegistry();
    const controller3 = new AbortController();
    const controller4 = new AbortController();
    registry.beginWork("exec-1", controller3, 3);
    registry.beginWork("exec-1", controller4, 4);

    expect(registry.abortCall("exec-1", 3)).toBe("interrupted");
    expect(controller3.signal.aborted).toBe(true);
    expect(controller3.signal.reason).toBe(CALL_INTERRUPT_REASON);
    expect(controller4.signal.aborted).toBe(false);
  });

  test("never latches admission — a later step of the same execution proceeds", () => {
    const registry = new DrainRegistry();
    registry.beginWork("exec-1", new AbortController(), 3);

    registry.abortCall("exec-1", 3);

    const followUp = new AbortController();
    expect(() => registry.beginWork("exec-1", followUp, 4)).not.toThrow();
    expect(followUp.signal.aborted).toBe(false);
  });

  test("unknown execution is not_found", () => {
    const registry = new DrainRegistry();
    expect(registry.abortCall("exec-x", 3)).toBe("not_found");
  });

  test("unknown step is not_found and the live call is untouched", () => {
    const registry = new DrainRegistry();
    const controller = new AbortController();
    registry.beginWork("exec-1", controller, 3);

    expect(registry.abortCall("exec-1", 9)).toBe("not_found");
    expect(controller.signal.aborted).toBe(false);
  });

  test("a settled call is already_settled", () => {
    const registry = new DrainRegistry();
    const controller = new AbortController();
    registry.beginWork("exec-1", controller, 3);
    registry.endWork("exec-1", controller, 3);

    expect(registry.abortCall("exec-1", 3)).toBe("already_settled");
  });

  test("a claimed settlement reads already_settled and never aborts", () => {
    // Once the result report begins, a late abort is observed, not honored:
    // the call's controller stays live.
    const registry = new DrainRegistry();
    const controller = new AbortController();
    registry.beginWork("exec-1", controller, 3);

    registry.claimSettlement("exec-1", 3);
    expect(registry.abortCall("exec-1", 3)).toBe("already_settled");
    expect(controller.signal.aborted).toBe(false);
  });

  test("a repeat abort re-reads the same outcome", () => {
    const registry = new DrainRegistry();
    registry.beginWork("exec-1", new AbortController(), 3);

    expect(registry.abortCall("exec-1", 3)).toBe("interrupted");
    expect(registry.abortCall("exec-1", 3)).toBe("interrupted");
  });

  test("track-only work with no signal channel is not_cancellable", () => {
    // Plain tool functions receive no AbortSignal: registered without a
    // controller, they can only be waited out.
    const registry = new DrainRegistry();
    registry.beginWork("exec-1", undefined, 3);

    expect(registry.abortCall("exec-1", 3)).toBe("not_cancellable");
  });

  test("a repeat abort of track-only work stays not_cancellable", () => {
    // The missing-controller answer is stable across repeats — a repeat must
    // never flip to interrupted (matches the Python receiver).
    const registry = new DrainRegistry();
    registry.beginWork("exec-1", undefined, 3);

    expect(registry.abortCall("exec-1", 3)).toBe("not_cancellable");
    expect(registry.abortCall("exec-1", 3)).toBe("not_cancellable");
  });
});
