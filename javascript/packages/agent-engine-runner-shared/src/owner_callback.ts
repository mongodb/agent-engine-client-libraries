/**
 * Owner-callback pre-attempt shared by every owner-preferred OE POST.
 *
 * Policy (mirrored by `agent_engine_runner_shared/owner_callback.py` and the async
 * `_owner_delivered` in `agent_engine_runner_shared/server/http_retry.py` — keep in sync):
 * the replica-specific owner URL gets exactly one best-effort POST before the
 * trusted service URL. Any owner failure — a transport error, a redirect
 * (never followed: the owner URL is only validated as a headless replica of
 * the service origin, so a 3xx could forward the body elsewhere), or any
 * other non-2xx — falls back to the service path. The owner is never retried
 * and this helper never throws. Response bodies are never read or logged
 * (externally influenced) and are cancelled so undici can reuse the
 * connection.
 */

import { fetchPlatform } from "./tls_client.js";
import { discardResponseBody } from "./server/http_retry.js";

export async function ownerDelivered(
  ownerUrl: string,
  init: RequestInit | (() => RequestInit),
  log: (message: string) => void,
): Promise<boolean> {
  let failure: string;
  try {
    // Build URL-specific TLS options inside the best-effort boundary. Reading
    // rotated certificate files can fail before fetch starts and must still
    // degrade to the trusted service path.
    const requestInit = typeof init === "function" ? init() : init;
    // fetchPlatform, not bare fetch: the owner attempt must carry the same
    // W3C trace-context headers and tracing suppression as the service path.
    const response = await fetchPlatform(ownerUrl, {
      ...requestInit,
      redirect: "manual",
    });
    await discardResponseBody(response);
    if (response.ok) return true;
    failure = `HTTP ${response.status}`;
  } catch (e) {
    failure = e instanceof Error ? e.message : String(e);
  }
  log(`owner URL ${ownerUrl} unusable (${failure}); falling back to service`);
  return false;
}
