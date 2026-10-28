/**
 * Count tools in a single JSON-RPC response or a batch of responses.
 *
 * The SDK transport accepts JSON-RPC batches, so counting only a top-level
 * `result` would let every batched page pass the runner's tool cap at zero
 * tools. Kept outside the package barrel: this is an implementation detail of
 * the fetch seam, not part of the runner's public surface.
 */

export interface ToolPage {
  id: unknown;
  count: number;
}

export function toolsFromPayload(payload: unknown): ToolPage | undefined {
  const messages = Array.isArray(payload) ? payload : [payload];
  let count = 0;
  let id: unknown;
  let found = false;

  for (const message of messages) {
    if (message === null || typeof message !== "object") continue;
    const record = message as {
      id?: unknown;
      result?: { tools?: unknown };
    };
    if (!Array.isArray(record.result?.tools)) continue;
    count += record.result.tools.length;
    id = record.id;
    found = true;
  }

  return found ? { id, count } : undefined;
}
