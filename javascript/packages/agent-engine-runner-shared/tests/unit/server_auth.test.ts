/**
 * Unit tests for runner request authentication.
 *
 * Mirrors Python's tests/unit/test_server_auth.py — the two runner runtimes
 * are parallel implementations, so the twins must stay behaviourally
 * identical.
 */

import { describe, test, expect } from "vitest";
import fastify from "fastify";
import {
  bearerToken,
  isUnauthenticatedPath,
  MissingRunnerAuthTokenError,
  registerAuthHook,
  tokensEqual,
} from "../../src/server/auth.js";

async function appWithAuth(env: Record<string, string>) {
  const app = fastify({ logger: false });
  app.get("/health", async () => ({ status: "ok" }));
  app.post("/execute", async () => ({ ok: true }));
  registerAuthHook(app, "aer", env as NodeJS.ProcessEnv);
  await app.ready();
  return app;
}

describe("runner request authentication", () => {
  test("parses the bearer token", () => {
    expect(bearerToken("Bearer s3cret")).toBe("s3cret");
    // RFC 7235 makes the scheme case-insensitive.
    expect(bearerToken("bearer s3cret")).toBe("s3cret");
    expect(bearerToken("s3cret")).toBeNull();
    expect(bearerToken("Basic s3cret")).toBeNull();
    expect(bearerToken("Bearer ")).toBeNull();
    expect(bearerToken("")).toBeNull();
    expect(bearerToken(undefined)).toBeNull();
  });

  test("compares tokens exactly", () => {
    expect(tokensEqual("s3cret", "s3cret")).toBe(true);
    expect(tokensEqual("s3cre", "s3cret")).toBe(false);
    expect(tokensEqual("s3cretX", "s3cret")).toBe(false);
    expect(tokensEqual("", "s3cret")).toBe(false);
  });

  test("leaves only health paths unauthenticated", () => {
    // Kubelet probes cannot present a token and these carry no tenant data.
    for (const path of ["/", "/health", "/healthz", "/ready", "/readyz"]) {
      expect(isUnauthenticatedPath(path)).toBe(true);
    }
    expect(isUnauthenticatedPath("/execute")).toBe(false);
    expect(isUnauthenticatedPath("/invoke_llm")).toBe(false);
    expect(isUnauthenticatedPath("/query/sessions")).toBe(false);
    // /metrics carries tenant-derived labels (tool_name, model_name) and is
    // not a scrape target, so it is gated with everything else.
    expect(isUnauthenticatedPath("/metrics")).toBe(false);
  });

  test("requires a token on /execute when configured", async () => {
    const app = await appWithAuth({ RUNNER_AUTH_TOKEN: "s3cret" });

    expect(
      (await app.inject({ method: "POST", url: "/execute" })).statusCode,
    ).toBe(401);
    expect(
      (
        await app.inject({
          method: "POST",
          url: "/execute",
          headers: { authorization: "Bearer wrong" },
        })
      ).statusCode,
    ).toBe(401);
    expect(
      (
        await app.inject({
          method: "POST",
          url: "/execute",
          headers: { authorization: "s3cret" },
        })
      ).statusCode,
    ).toBe(401);

    const ok = await app.inject({
      method: "POST",
      url: "/execute",
      headers: { authorization: "Bearer s3cret" },
    });
    expect(ok.statusCode).toBe(200);
    expect(ok.json()).toEqual({ ok: true });

    await app.close();
  });

  test("leaves health reachable without a token", async () => {
    const app = await appWithAuth({ RUNNER_AUTH_TOKEN: "s3cret" });
    expect(
      (await app.inject({ method: "GET", url: "/health" })).statusCode,
    ).toBe(200);
    await app.close();
  });

  test("does not enforce when no token is configured", async () => {
    // The runner image also runs under `agentengine dev up`, where no
    // deploy-time secret exists; a startup warning covers that case.
    for (const env of [
      {},
      { RUNNER_AUTH_TOKEN: "" },
      { RUNNER_AUTH_TOKEN: "   " },
      { RUNNER_AUTH_REQUIRED: "" },
      { RUNNER_AUTH_REQUIRED: "false" },
      { RUNNER_AUTH_REQUIRED: "0" },
      { RUNNER_AUTH_REQUIRED: "yes" },
    ]) {
      const app = await appWithAuth(env as Record<string, string>);
      expect(
        (await app.inject({ method: "POST", url: "/execute" })).statusCode,
      ).toBe(200);
      await app.close();
    }
  });

  test("required auth without a token refuses to start", () => {
    // A deployment that mandates auth must fail loudly on a provisioning
    // gap instead of silently serving unauthenticated routes.
    for (const required of ["true", "TRUE", "1"]) {
      const app = fastify({ logger: false });
      expect(() =>
        registerAuthHook(app, "aer", {
          RUNNER_AUTH_REQUIRED: required,
        } as NodeJS.ProcessEnv),
      ).toThrow(MissingRunnerAuthTokenError);
    }
  });

  test("required auth with a token enforces", async () => {
    const app = await appWithAuth({
      RUNNER_AUTH_TOKEN: "s3cret",
      RUNNER_AUTH_REQUIRED: "true",
    });
    expect(
      (await app.inject({ method: "POST", url: "/execute" })).statusCode,
    ).toBe(401);
    expect(
      (
        await app.inject({
          method: "POST",
          url: "/execute",
          headers: { authorization: "Bearer s3cret" },
        })
      ).statusCode,
    ).toBe(200);
    await app.close();
  });
});
