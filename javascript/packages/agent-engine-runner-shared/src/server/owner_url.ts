/**
 * Validation of a request-supplied replica-specific OE owner callback URL.
 *
 * Mirrors Python's `agent_engine_runner_shared.server.owner_url`. Both runtimes are parallel
 * implementations of the same server architecture, so this file and its Python
 * twin must be changed together.
 *
 * Dark-ship owner-callback fallback: the OE may stamp a `*_owner_url`
 * onto the execute / tool request naming the specific OE replica that owns the
 * execution. The runner delivers stream chunks, terminal callbacks and tool
 * results straight to that replica and falls back to the trusted service URL
 * when that owner-specific callback attempt is unusable.
 *
 * That owner URL is request-supplied, which reopens the callback-hijack / SSRF
 * surface that `resolveOeUrl` (see `oe_url.ts`) exists to close. Unlike the OE
 * base URL — where the runner's own deploy-time `OE_URL` is simply preferred —
 * the owner URL has no trusted counterpart to fall back to, so it is accepted
 * only when it is provably the headless-service address of one replica *behind
 * the already-trusted service URL*: identical scheme, identical port (explicit
 * or scheme default), and a host of exactly
 * `<one-dns-label>.<service-label>-headless.<rest>` when the service host is
 * `<service-label>.<rest>`. Anything else is logged and discarded — never an
 * error — and the caller keeps using the trusted service URL.
 */

import { getLogger } from "../logger.js";

const logger = getLogger("agent_engine_runner_shared.server.owner_url");

// WHATWG URL protocols carry a trailing colon.
const DEFAULT_PORTS: Record<string, number> = { "http:": 80, "https:": 443 };

// A single DNS label: alphanumerics plus interior hyphens. A dashed pod IP such
// as `10-1-2-3` — the label the OE actually stamps — qualifies. WHATWG URL
// lowercases the host, so a lowercase-only pattern is sufficient.
const DNS_LABEL = /^[a-z0-9]([a-z0-9-]*[a-z0-9])?$/;

/** Stable, non-secret URL summary safe for warning logs. */
function describeUrlForLog(url: URL): string {
  return JSON.stringify(
    url.port === ""
      ? `${url.protocol}//${url.hostname}`
      : `${url.protocol}//${url.hostname}:${url.port}`,
  );
}

/** Effective port for `url` — explicit if present, else the scheme default. */
function effectivePort(url: URL): number | null {
  if (url.port !== "") return Number(url.port);
  return DEFAULT_PORTS[url.protocol] ?? null;
}

/** Canonical bare origin for `url`, omitting scheme-default ports. */
function canonicalOrigin(url: URL): string {
  const port = effectivePort(url);
  const defaultPort = DEFAULT_PORTS[url.protocol] ?? null;
  return port === null || port === defaultPort
    ? `${url.protocol}//${url.hostname}`
    : `${url.protocol}//${url.hostname}:${port}`;
}

function hasAsciiControl(value: string): boolean {
  for (let i = 0; i < value.length; i += 1) {
    const code = value.charCodeAt(i);
    if (code <= 0x1f || code === 0x7f) return true;
  }
  return false;
}

/** Split a host on its first dot into `[label, rest]`, or null if there is none. */
function partitionHost(host: string): [string, string] | null {
  const idx = host.indexOf(".");
  if (idx === -1) return null;
  return [host.slice(0, idx), host.slice(idx + 1)];
}

/**
 * Return the canonical bare origin of `ownerUrl` when it is a valid replica of
 * `serviceUrl`, else `null`.
 *
 * `null` means "no usable owner URL"; the caller sends to the trusted
 * `serviceUrl` instead. A malformed or forged owner URL is logged at WARNING
 * and discarded rather than thrown — a rejected value must degrade to the safe
 * default, never fail the callback.
 */
export function resolveOwnerUrl(
  ownerUrl: string | null | undefined,
  serviceUrl: string,
): string | null {
  if (!ownerUrl) return null;
  if (hasAsciiControl(ownerUrl)) {
    logger.warn(
      "Discarding owner URL (contains control characters); " +
        "using configured service URL",
    );
    return null;
  }

  let owner: URL;
  let service: URL;
  try {
    owner = new URL(ownerUrl);
    service = new URL(serviceUrl);
  } catch {
    logger.warn(
      "Discarding unparseable owner URL; " + "using configured service URL",
    );
    return null;
  }

  const serviceSummary = describeUrlForLog(service);

  const reject = (reason: string): void => {
    logger.warn(
      `Discarding owner URL (${reason}); ` +
        `using service URL ${serviceSummary}`,
    );
  };

  // The OE stamps a bare origin. Credentials, a path, a query, or a fragment
  // are never expected and would ride along when a callback path is appended,
  // so treat their presence as a forged value. WHATWG URL yields a "/" pathname
  // for a bare origin, so "" and "/" are the only acceptable paths.
  if (
    owner.username ||
    owner.password ||
    (owner.pathname !== "" && owner.pathname !== "/") ||
    owner.search ||
    owner.hash
  ) {
    reject("not a bare origin");
    return null;
  }

  if (owner.protocol !== service.protocol) {
    reject("scheme mismatch");
    return null;
  }

  const ownerPort = effectivePort(owner);
  const servicePort = effectivePort(service);
  if (ownerPort === null || ownerPort !== servicePort) {
    reject("port mismatch");
    return null;
  }

  const ownerHost = owner.hostname;
  const serviceHost = service.hostname;
  if (!ownerHost || !serviceHost) {
    reject("missing host");
    return null;
  }

  const servicePart = partitionHost(serviceHost);
  if (servicePart === null || !servicePart[0] || !servicePart[1]) {
    reject("service host is not <label>.<rest>");
    return null;
  }
  const [serviceLabel, serviceRest] = servicePart;

  // partitionHost() splits on the first dot, so this guarantees exactly one
  // leading label; the exact-match on the remainder enforces the full shape.
  const ownerPart = partitionHost(ownerHost);
  if (ownerPart === null || !DNS_LABEL.test(ownerPart[0])) {
    reject("owner host is not <label>.<rest>");
    return null;
  }
  if (ownerPart[1] !== `${serviceLabel}-headless.${serviceRest}`) {
    reject("owner host is not a headless replica of the service host");
    return null;
  }

  return canonicalOrigin(owner);
}
