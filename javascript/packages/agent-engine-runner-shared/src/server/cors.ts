/**
 * CORS origin allowlist resolution for Runner SDK servers.
 *
 * Mirrors Python's `agent_engine_runner_shared.server.cors`. Both runtimes are parallel
 * implementations of the same server architecture, so this file and its
 * Python twin must be changed together.
 *
 * The default is deny-all rather than `*`. Runner servers register
 * credential-bearing routes (`/execute`, `/invoke_llm`, AER session reads),
 * so a wildcard default meant any web page a developer visited could drive
 * tool execution and read stored conversation history once `agentengine dev`
 * published the ports locally.
 */

import { getLogger } from "../logger.js";

const logger = getLogger("agent_engine_runner_shared.server.cors");

/** Result of resolving the configured origin allowlist. */
export interface CorsPolicy {
  /**
   * Value for `@fastify/cors`'s `origin` option. `false` disables
   * cross-origin access entirely (the deny-all default).
   */
  origin: false | string[] | "*";
  /**
   * Whether to send `Access-Control-Allow-Credentials`. Never true
   * alongside a wildcard origin — that combination lets any site issue
   * credentialed requests and read the responses.
   */
  credentials: boolean;
}

/** A DNS name, an IPv4 literal, or a bracketed IPv6 literal. */
const HOST_PATTERN = /^(?:[a-z0-9.-]+|\[[0-9a-f:.]+\])$/;

/**
 * Normalizes `scheme://host[:port]` into the form browsers put in the
 * `Origin` header — lowercased scheme and host, default port dropped — so
 * an operator-written `HTTPS://A.com:443` still matches `https://a.com`.
 *
 * Returns `null` for anything an `Origin` header never carries (a path,
 * userinfo, a query, a non-http(s) scheme, wildcards inside the host), so
 * the entry is reported at startup rather than silently never matching.
 * Kept behaviourally identical to the Python twin's `_normalize_origin`.
 */
export function normalizeOrigin(origin: string): string | null {
  // The literal `null` origin sandboxed iframes and `file://` documents send.
  if (origin === "null") {
    return "null";
  }
  let parsed: URL;
  try {
    parsed = new URL(origin);
  } catch {
    return null;
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    return null;
  }
  // `new URL` normalizes an empty path to "/", so that is the only path an
  // origin-shaped value may have.
  if (parsed.pathname !== "/" && parsed.pathname !== "") {
    return null;
  }
  if (parsed.username || parsed.password || parsed.search || parsed.hash) {
    return null;
  }
  // WHATWG parsing permits `*` in a host, which no browser will ever send.
  if (!HOST_PATTERN.test(parsed.hostname)) {
    return null;
  }
  // `URL.origin` already lowercases and strips the scheme's default port.
  return parsed.origin;
}

/**
 * Resolves the CORS policy from a raw `CORS_ALLOWED_ORIGINS` value.
 *
 * Entries are trimmed before validation: an operator who writes
 * `a.com, b.com` reasonably expects both to be honoured, and the previous
 * behaviour of comparing the untrimmed `" b.com"` meant a configured
 * origin silently never matched — an operator could believe they had
 * restricted origins when they had not.
 *
 * Malformed entries are dropped and logged at warn level for the same
 * reason: silent non-matching is the failure mode worth eliminating.
 */
export function resolveCorsPolicy(raw: string | undefined): CorsPolicy {
  const entries = (raw ?? "")
    .split(",")
    .map((entry) => entry.trim())
    .filter((entry) => entry.length > 0);

  if (entries.length === 0) {
    // Distinguished from the misconfiguration warnings below: an unset value
    // used to mean "allow every origin", so a deployment that never set it is
    // exactly the one whose cross-origin access this change removes.
    logger.warn(
      "CORS_ALLOWED_ORIGINS is unset — denying all cross-origin requests. " +
        "Set it to a comma-separated list of origins to allow any.",
    );
    return { origin: false, credentials: false };
  }

  if (entries.includes("*")) {
    // Explicit opt-in only, and never with credentials. Kept as an escape
    // hatch for local experimentation; production origins must be listed.
    logger.warn(
      "CORS_ALLOWED_ORIGINS contains '*' — allowing all origins WITHOUT " +
        "credentials. List explicit origins to enable credentialed requests.",
    );
    return { origin: "*", credentials: false };
  }

  const valid: string[] = [];
  for (const entry of entries) {
    const normalized = normalizeOrigin(entry);
    if (normalized !== null) {
      if (!valid.includes(normalized)) {
        valid.push(normalized);
      }
    } else {
      logger.warn(
        `Ignoring malformed CORS origin ${JSON.stringify(entry)} — expected ` +
          `scheme://host[:port] with no trailing path.`,
      );
    }
  }

  if (valid.length === 0) {
    return { origin: false, credentials: false };
  }
  return { origin: valid, credentials: true };
}
