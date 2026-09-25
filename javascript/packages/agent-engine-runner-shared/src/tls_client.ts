/**
 * TLS client configuration for HTTPS connections with mTLS support.
 *
 * This module provides utilities for configuring fetch requests with mTLS
 * (mutual TLS) when connecting to platform services over HTTPS. When TLS
 * certificate paths are available via environment variables, the client will
 * present its certificate to the server for authentication.
 *
 * Uses Undici's Agent (Node.js built-in fetch is backed by Undici) for proper
 * dispatcher configuration with client certificates.
 */

import { readFileSync, statSync } from "node:fs";
import { context } from "@opentelemetry/api";
import {
  suppressTracing,
  W3CTraceContextPropagator,
} from "@opentelemetry/core";
import { Agent as UndiciAgent } from "undici";
import { getLogger } from "./logger.js";

const logger = getLogger("agent_engine_runner_shared.tls_client");
const traceContextPropagator = new W3CTraceContextPropagator();

/** Carry the active trace across a platform RPC without recording transport. */
export function fetchPlatform(
  url: string,
  init: RequestInit,
  fetchImpl: typeof fetch = fetch,
): Promise<Response> {
  const activeContext = context.active();
  const headers = {
    ...((init.headers as Record<string, string> | undefined) ?? {}),
  };
  traceContextPropagator.inject(activeContext, headers, {
    set(carrier, key, value) {
      carrier[key] = value;
    },
  });

  return context.with(suppressTracing(activeContext), () =>
    fetchImpl(url, { ...init, headers }),
  );
}

/**
 * Cached Undici agents per base URL for connection pooling.
 * Key is the normalized base URL (scheme + host).
 */
const agentCache = new Map<string, UndiciAgent>();

// Owner URLs are replica-specific and change as OE pods roll. Bound the
// origin cache so a long-lived runner cannot retain every historical replica.
export const TLS_AGENT_CACHE_MAX_ENTRIES = 32;

/**
 * Explicit connect budget for long-running calls.
 *
 * Undici's own default (10s) already bounds the handshake, but a call that may
 * legitimately run for ten minutes should not depend on an undefended library
 * default to fail fast when the OE is simply unreachable.
 */
const CONNECT_TIMEOUT_MS = 10_000;

/**
 * Agents for calls with an explicit deadline, keyed by `baseUrl|deadlineMs`.
 * Separate from `agentCache` so a long-deadline agent is never handed to a
 * caller that wanted the defaults.
 */
const longCallAgentCache = new Map<string, UndiciAgent>();

/**
 * Cached certificate file mtimes per base URL for rotation detection.
 * Key is the normalized base URL (scheme + host).
 * Value is [cert_mtime, key_mtime, ca_mtime] in milliseconds, or null if using PEM env vars.
 */
const agentMtimeCache = new Map<string, [number, number, number] | null>();

/**
 * Get certificate file modification times for rotation detection.
 *
 * Only container mode (TLS_*_PATH) supports rotation detection via mtime.
 * VM mode (TLS_*_PEM) cannot detect rotation - the env var is static for
 * the pod's lifetime, so rotation requires pod restart.
 *
 * @returns [cert_mtime, key_mtime, ca_mtime] in milliseconds, or null if not using file-based certs
 */
function getCertMtimes(): [number, number, number] | null {
  try {
    const certPath = process.env.TLS_CERT_PATH;
    const keyPath = process.env.TLS_KEY_PATH;
    const caPath = process.env.TLS_CA_CERT_PATH;

    // Only track mtimes if using file-based certs (container mode)
    if (!certPath || !keyPath || !caPath) {
      return null;
    }

    return [
      statSync(certPath).mtimeMs,
      statSync(keyPath).mtimeMs,
      statSync(caPath).mtimeMs,
    ];
  } catch {
    // Cert files don't exist or aren't readable - not using TLS
    return null;
  }
}

/**
 * Get certificate content from environment.
 *
 * For container mode, reads from TLS_*_PATH file.
 * For VM mode, reads TLS_*_PEM (raw PEM from SecretKeyRef) directly.
 *
 * @param certType - One of "CERT", "KEY", or "CA_CERT"
 * @returns Certificate content as string, or empty string if not available
 */
function getCertContentFromEnv(certType: string): string {
  // Check for direct path first (container mode)
  const pathEnv = `TLS_${certType}_PATH`;
  const certPath = process.env[pathEnv];
  if (certPath) {
    try {
      return readFileSync(certPath, "utf-8");
    } catch (err) {
      throw new Error(`Failed to read ${pathEnv} from ${certPath}: ${err}`, {
        cause: err,
      });
    }
  }

  // Check for raw PEM content (VM mode - SecretKeyRef injects decoded value)
  // Return directly without writing to disk - Undici Agent accepts cert strings
  const pemEnv = `TLS_${certType}_PEM`;
  const pemContent = process.env[pemEnv];
  if (!pemContent) {
    return "";
  }

  return pemContent;
}

/**
 * mTLS connect options for an https:// URL, or null for http://.
 *
 * Fail-closed: an https URL without all three certificate sources throws rather
 * than silently falling back to an unauthenticated connection. Shared by both
 * agent builders so the policy and its message live in one place.
 */
function getConnectOptionsForURL(
  url: string,
): { cert: string; key: string; ca: string; rejectUnauthorized: true } | null {
  if (!url.startsWith("https://")) {
    return null;
  }
  const cert = getCertContentFromEnv("CERT");
  const key = getCertContentFromEnv("KEY");
  const ca = getCertContentFromEnv("CA_CERT");
  if (!cert || !key || !ca) {
    throw new Error(
      `HTTPS URL requires mTLS configuration (fail-closed policy): ` +
        `all three env vars must be set when url uses https:// ` +
        `(TLS_CERT_PATH or TLS_CERT_PEM, TLS_KEY_PATH or TLS_KEY_PEM, ` +
        `TLS_CA_CERT_PATH or TLS_CA_CERT_PEM). ` +
        `For AER→OE communication, the OE HTTPS server (port 8443) requires ` +
        `client certificates. If TLS certificates are not available, use ` +
        `http:// in the URL.`,
    );
  }
  // Hostname verification is enabled by default (secure).
  return { cert, key, ca, rejectUnauthorized: true };
}

/**
 * Get fetch options configured for the given URL with mTLS support.
 *
 * If url uses HTTPS and TLS certificate environment variables are set
 * (TLS_CERT_PATH, TLS_KEY_PATH, TLS_CA_CERT_PATH), returns options with
 * an HTTPS agent configured for mTLS. For HTTP URLs or when certificates
 * are not available, returns undefined (use default fetch behavior).
 *
 * The HTTPS agent is cached per base URL for connection pooling.
 *
 * Performance: checks the agent cache BEFORE reading certificate files to avoid
 * blocking the event loop on filesystem I/O for every request. Certificate files
 * are only read once per base URL when creating the agent, not on every fetch.
 *
 * Certificate rotation: In container mode (TLS_*_PATH), automatically detects
 * cert-manager rotation by comparing file mtimes and invalidates the cached agent
 * so the next request picks up renewed certificates. In VM mode (TLS_*_PEM), the
 * env vars are static for the pod's lifetime, so rotation requires a pod restart.
 *
 * @param url - The target service URL (http:// or https://)
 * @returns Fetch options with agent, or undefined for HTTP/no-TLS
 * @throws Error if HTTPS is used but TLS configuration is incomplete
 */
export function getFetchOptionsWithTLS(url: string): RequestInit | undefined {
  // HTTP URLs don't need TLS configuration
  if (!url.startsWith("https://")) {
    return undefined;
  }

  // Extract base URL for cache key (scheme + host)
  const urlObj = new URL(url);
  const baseUrl = `${urlObj.protocol}//${urlObj.host}`;

  // Check if certs have been rotated (container mode only).
  // Cert-manager rotates certificates by updating the Secret volume mount.
  // Kubelet propagates the change to the pod's filesystem (up to ~60s delay).
  // We detect rotation by comparing file mtimes and invalidate the cached agent.
  const currentMtimes = getCertMtimes();
  const cachedMtimes = agentMtimeCache.get(baseUrl);

  if (
    currentMtimes !== null &&
    cachedMtimes !== undefined &&
    cachedMtimes !== null
  ) {
    // Both are file-based - compare mtimes for rotation
    if (
      currentMtimes[0] !== cachedMtimes[0] ||
      currentMtimes[1] !== cachedMtimes[1] ||
      currentMtimes[2] !== cachedMtimes[2]
    ) {
      logger.info(
        {
          base_url: baseUrl,
          old_mtimes: cachedMtimes,
          new_mtimes: currentMtimes,
        },
        "TLS certificates rotated, invalidating cached HTTPS agent",
      );
      const oldAgent = agentCache.get(baseUrl);
      if (oldAgent) {
        oldAgent.close();
      }
      agentCache.delete(baseUrl);
      agentMtimeCache.delete(baseUrl);
    }
  }

  // Check cache after rotation check
  let agent = agentCache.get(baseUrl);
  if (agent) {
    // Refresh insertion order so eviction behaves as an LRU cache.
    agentCache.delete(baseUrl);
    agentCache.set(baseUrl, agent);
    return { dispatcher: agent } as RequestInit;
  }

  // Cache miss: read certificates (fail-closed) and create a new agent.
  // Supports both direct paths (container mode) and PEM env content (VM mode).
  const connect = getConnectOptionsForURL(url);

  // Configure mTLS
  try {
    // Create Undici agent with client certificate.
    // Node.js's built-in fetch is backed by Undici, so we must use an Undici
    // Agent (not https.Agent) configured via the connect option.
    agent = new UndiciAgent({
      ...(connect ? { connect } : undefined),
      // Keep connections alive for connection pooling
      keepAliveTimeout: 60_000, // 60 seconds
      keepAliveMaxTimeout: 600_000, // 10 minutes
    });

    if (agentCache.size >= TLS_AGENT_CACHE_MAX_ENTRIES) {
      const oldestBaseUrl = agentCache.keys().next().value as
        | string
        | undefined;
      if (oldestBaseUrl !== undefined) {
        agentCache.get(oldestBaseUrl)?.close();
        agentCache.delete(oldestBaseUrl);
        agentMtimeCache.delete(oldestBaseUrl);
      }
    }

    // Cache the agent and current mtimes for reuse.
    agentCache.set(baseUrl, agent);
    agentMtimeCache.set(baseUrl, currentMtimes);

    logger.info(
      {
        base_url: baseUrl,
        has_cert: !!connect?.cert,
        has_key: !!connect?.key,
        has_ca: !!connect?.ca,
        cert_mtimes: currentMtimes,
      },
      "Configured HTTPS client with mTLS",
    );

    return { dispatcher: agent } as RequestInit;
  } catch (err) {
    throw new Error(
      `Failed to configure mTLS for HTTPS connection: ${err}. ` +
        `Check that TLS_*_PATH or TLS_*_PEM env vars are set correctly.`,
      { cause: err },
    );
  }
}

/**
 * Cleanup function to close all cached Undici agents.
 * Should be called during graceful shutdown.
 */
export function closeAllTLSAgents(): void {
  for (const [baseUrl, agent] of agentCache.entries()) {
    agent.close();
    logger.debug({ base_url: baseUrl }, "Closed TLS agent");
  }
  agentCache.clear();
  agentMtimeCache.clear();
}

/**
 * Fetch options for a call that may legitimately run for minutes.
 *
 * Node's fetch is backed by Undici, whose default `headersTimeout` of 300s
 * applies *independently* of any AbortSignal the caller passes. Since the OE
 * holds `/tool/execute` open until the tool returns, response headers do not
 * arrive until then — so without raising it here, a longer per-call deadline is
 * silently capped at 300s and surfaces as `UND_ERR_HEADERS_TIMEOUT` rather than
 * as the caller's own timeout. The connect budget stays short and is set
 * explicitly rather than inherited.
 *
 * Applies to http:// as well as https://, since the cap is Undici's, not TLS's.
 *
 * @param url - The target service URL
 * @param deadlineMs - The call's own deadline; headers/body budgets match it
 */
export function getFetchOptionsForLongCall(
  url: string,
  deadlineMs: number,
): RequestInit {
  const urlObj = new URL(url);
  const baseUrl = `${urlObj.protocol}//${urlObj.host}`;
  const cacheKey = `${baseUrl}|${deadlineMs}`;

  const cached = longCallAgentCache.get(cacheKey);
  if (cached) {
    return { dispatcher: cached } as RequestInit;
  }

  const connect = getConnectOptionsForURL(url);

  const agent = new UndiciAgent({
    ...(connect ? { connect } : undefined),
    connectTimeout: CONNECT_TIMEOUT_MS,
    headersTimeout: deadlineMs,
    bodyTimeout: deadlineMs,
    keepAliveTimeout: 60_000,
    keepAliveMaxTimeout: 600_000,
  });

  longCallAgentCache.set(cacheKey, agent);
  return { dispatcher: agent } as RequestInit;
}
