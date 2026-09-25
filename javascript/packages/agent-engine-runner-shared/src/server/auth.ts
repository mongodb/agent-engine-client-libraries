/**
 * Bearer-token authentication for Runner SDK servers.
 *
 * Mirrors Python's `agent_engine_runner_shared.server.auth`. Both runtimes are parallel
 * implementations of the same server architecture, so this file and its
 * Python twin must be changed together.
 *
 * Runner servers registered every route — `/execute`, `/invoke_llm`, the AER
 * session-message reads — with no authentication at all, relying on network
 * reachability. The only legitimate caller is the Orchestration Engine, so
 * this is service-to-service auth using the same bearer-token shape as the
 * operator's gRPC control plane.
 *
 * Enforcement is conditional on `RUNNER_AUTH_TOKEN` being configured. The
 * runner image also runs under `agentengine dev up` on a developer's laptop,
 * where no deploy-time secret exists and hard-failing would break local
 * development outright; an unconfigured server logs a warning at startup so
 * the weaker posture is visible rather than silent.
 *
 * Deployments that provision the token can additionally set
 * `RUNNER_AUTH_REQUIRED=true` to make a missing token fatal at startup: the
 * server throws {@link MissingRunnerAuthTokenError} instead of serving
 * unauthenticated, so a provisioning gap crash-loops loudly rather than
 * silently weakening auth. The signal is opt-in so shipping this code
 * changes no running behavior until token provisioning is in place.
 */

import { createHash, timingSafeEqual } from "node:crypto";
import type { FastifyInstance, FastifyReply, FastifyRequest } from "fastify";
import { getLogger } from "../logger.js";

const logger = getLogger("agent_engine_runner_shared.server.auth");

/**
 * Routes reachable without a token.
 *
 * Liveness and readiness probes originate from the kubelet and cannot
 * present one, and these responses carry no tenant data. Everything else —
 * including every credential-bearing and data-returning route — is gated.
 *
 * `/metrics` is deliberately NOT here: it returns `Metrics.getAll()`, whose
 * counter names carry tenant-derived labels (`tool_name`, `model_name`), so
 * an unauthenticated reader could enumerate which tools and models an agent
 * uses. It is not a Prometheus scrape target — nothing in the platform reads
 * it, so gating it costs nothing.
 */
const UNAUTHENTICATED_PATHS = new Set([
  "/",
  "/health",
  "/healthz",
  "/ready",
  "/readyz",
]);

export function isUnauthenticatedPath(pathname: string): boolean {
  const path = (pathname || "").split("?")[0];
  return UNAUTHENTICATED_PATHS.has(path ?? "");
}

/**
 * Constant-time comparison, so a caller cannot recover the configured token
 * byte by byte from response timing.
 *
 * Both values are hashed first so the comparison is always over two 32-byte
 * buffers: `timingSafeEqual` throws on length mismatch, and short-circuiting
 * on length would otherwise leak the configured token's length.
 */
export function tokensEqual(presented: string, expected: string): boolean {
  const a = createHash("sha256").update(presented, "utf8").digest();
  const b = createHash("sha256").update(expected, "utf8").digest();
  return timingSafeEqual(a, b);
}

/** Extracts the bearer token from an Authorization header value. */
export function bearerToken(headerValue: string | undefined): string | null {
  if (!headerValue) {
    return null;
  }
  const prefix = "bearer ";
  if (headerValue.length <= prefix.length) {
    return null;
  }
  if (headerValue.slice(0, prefix.length).toLowerCase() !== prefix) {
    return null;
  }
  return headerValue.slice(prefix.length);
}

/**
 * Thrown at startup when `RUNNER_AUTH_REQUIRED=true` is set but no auth
 * token is configured.
 */
export class MissingRunnerAuthTokenError extends Error {
  constructor(modeName: string) {
    super(
      `${modeName} server refusing to start without request authentication: ` +
        `RUNNER_AUTH_REQUIRED is set but RUNNER_AUTH_TOKEN is not, so any ` +
        `caller that could reach this port would be able to execute tools ` +
        `and read session data. Provision RUNNER_AUTH_TOKEN, or unset ` +
        `RUNNER_AUTH_REQUIRED to accept running unauthenticated.`,
    );
    this.name = "MissingRunnerAuthTokenError";
  }
}

/**
 * Registers the authentication hook on `app`, if a token is configured.
 *
 * Returns whether enforcement is active, so the caller can surface it.
 * Throws {@link MissingRunnerAuthTokenError} when `RUNNER_AUTH_REQUIRED` is
 * set to `true` (or `1`) but no token is configured, so a deployment that
 * mandates auth fails loudly instead of serving unauthenticated.
 */
export function registerAuthHook(
  app: FastifyInstance,
  modeName: string,
  env: NodeJS.ProcessEnv = process.env,
): boolean {
  const expected = (env["RUNNER_AUTH_TOKEN"] ?? "").trim();
  if (!expected) {
    const required = (env["RUNNER_AUTH_REQUIRED"] ?? "").trim().toLowerCase();
    if (required === "1" || required === "true") {
      throw new MissingRunnerAuthTokenError(modeName);
    }
    logger.warn(
      `${modeName} server is running without request authentication: ` +
        `RUNNER_AUTH_TOKEN is not set, so any caller that can reach this ` +
        `port can execute tools and read session data.`,
    );
    return false;
  }

  app.addHook(
    "onRequest",
    async (request: FastifyRequest, reply: FastifyReply) => {
      // Fastify's request.url is the raw request-line URL; it never consults
      // the Host header, so it cannot be corrupted the way starlette's
      // Host-derived request.url.path can (see the Python twin). No fix
      // is needed here — this comment exists so the difference isn't mistaken
      // for an oversight.
      if (isUnauthenticatedPath(request.url)) {
        return;
      }
      const presented = bearerToken(request.headers.authorization);
      if (presented === null || !tokensEqual(presented, expected)) {
        // Returning the reply is the documented way to halt the lifecycle;
        // relying on the `reply.sent` short-circuit is version-dependent.
        return reply.code(401).send({ detail: "unauthorized" });
      }
    },
  );

  return true;
}
