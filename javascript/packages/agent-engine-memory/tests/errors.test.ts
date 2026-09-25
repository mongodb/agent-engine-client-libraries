/*
 * Mirrors runner ../agent-engine-sdk-memory/tests/test_apikey_errors.py
 *
 * Ported cases:
 * - MemoryAPIError exposes status/code/responseText; defaults are null
 * - message is CODE-ANNOTATED ("forbidden (code=DENIED)")
 * - toString includes message + code
 * - the 5 API subclasses are instanceof MemoryAPIError
 * - MemoryIdentityError is NOT instanceof MemoryAPIError, but IS instanceof Error
 * - MemoryNotSupportedError is instanceof MemoryClientError and NOT MemoryAPIError
 * - MemoryRouteNotFoundError is instanceof MemoryBadRequestError
 * - each subclass carries a distinct .name
 *
 * - MemoryNotSupportedError preserves an underlying error as `cause`
 *   (mirrors Python's `raise ... from exc` chaining)
 *
 * Adaptation notes:
 * - Python's MemoryAPIError.message is the bare message; the TS port folds the
 *   code into the Error message ("<msg> (code=<code>)"). This asserts the
 *   annotated form, which is the actual TS behavior.
 */

import { describe, expect, it } from "vitest";

import {
  MemoryAPIError,
  MemoryAuthError,
  MemoryBadRequestError,
  MemoryClientError,
  MemoryConnectionError,
  MemoryIdentityError,
  MemoryNotProvisionedError,
  MemoryNotSupportedError,
  MemoryRouteNotFoundError,
  MemoryServerError,
} from "../src/errors.js";

describe("MemoryAPIError base", () => {
  it("exposes status/code/responseText and a code-annotated message", () => {
    const err = new MemoryAPIError("forbidden", {
      status: 403,
      code: "DENIED",
      responseText: "raw",
    });
    expect(err.message).toBe("forbidden (code=DENIED)");
    expect(err.status).toBe(403);
    expect(err.code).toBe("DENIED");
    expect(err.responseText).toBe("raw");
  });

  it("defaults status/code/responseText to null and omits the annotation", () => {
    const err = new MemoryAPIError("boom");
    expect(err.status).toBeNull();
    expect(err.code).toBeNull();
    expect(err.responseText).toBeNull();
    expect(err.message).toBe("boom");
  });

  it("toString includes both the message and the code", () => {
    const text = new MemoryAPIError("forbidden", { code: "DENIED" }).toString();
    expect(text).toContain("forbidden");
    expect(text).toContain("DENIED");
  });
});

describe("error hierarchy", () => {
  it.each([
    MemoryAuthError,
    MemoryNotProvisionedError,
    MemoryBadRequestError,
    MemoryServerError,
    MemoryConnectionError,
  ])("%p is an instance of MemoryAPIError", (Cls) => {
    const err = new Cls("msg", { status: 400, code: "X" });
    expect(err).toBeInstanceOf(MemoryAPIError);
  });

  it("MemoryIdentityError is an Error but not a MemoryAPIError", () => {
    const err = new MemoryIdentityError("bad identity");
    expect(err).toBeInstanceOf(Error);
    expect(err).not.toBeInstanceOf(MemoryAPIError);
  });

  it("MemoryNotSupportedError is a client error, not an API error", () => {
    const err = new MemoryNotSupportedError("unsupported in this mode");
    expect(err).toBeInstanceOf(MemoryClientError);
    expect(err).not.toBeInstanceOf(MemoryAPIError);
    expect(err.message).toBe("unsupported in this mode");
  });

  it("MemoryNotSupportedError preserves an underlying error as cause", () => {
    const underlying = new MemoryBadRequestError("Not Found", { status: 404 });
    const err = new MemoryNotSupportedError("unsupported", {
      cause: underlying,
    });
    expect(err.cause).toBe(underlying);
  });

  it("MemoryRouteNotFoundError subclasses MemoryBadRequestError", () => {
    expect(new MemoryRouteNotFoundError("404")).toBeInstanceOf(
      MemoryBadRequestError,
    );
  });

  it("each subclass carries a distinct .name", () => {
    const names = [
      new MemoryAPIError("m").name,
      new MemoryAuthError("m").name,
      new MemoryNotProvisionedError("m").name,
      new MemoryBadRequestError("m").name,
      new MemoryRouteNotFoundError("m").name,
      new MemoryServerError("m").name,
      new MemoryConnectionError("m").name,
      new MemoryClientError("m").name,
      new MemoryNotSupportedError("m").name,
      new MemoryIdentityError("m").name,
    ];
    expect(new Set(names).size).toBe(names.length);
  });
});
