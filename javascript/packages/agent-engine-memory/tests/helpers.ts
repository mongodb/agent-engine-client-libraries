/** Shared test helpers: a scriptable fake fetch. */

import type { FetchLike } from "../src/transport.js";

export interface RecordedCall {
  url: string;
  method: string;
  body: unknown;
  headers: Record<string, string>;
}

export interface FakeResponse {
  status?: number;
  json?: unknown;
  text?: string;
}

/**
 * Build a fake fetch that returns queued responses in order and records calls.
 * A queued entry may be an Error to simulate a network failure.
 */
export function fakeFetch(responses: Array<FakeResponse | Error>): {
  fetchImpl: FetchLike;
  calls: RecordedCall[];
} {
  const calls: RecordedCall[] = [];
  let i = 0;
  const fetchImpl: FetchLike = async (url, init) => {
    const body =
      typeof init.body === "string"
        ? JSON.parse(init.body)
        : (init.body ?? null);
    calls.push({
      url,
      method: init.method ?? "GET",
      body,
      headers: (init.headers as Record<string, string>) ?? {},
    });
    const next = responses[Math.min(i, responses.length - 1)];
    i++;
    if (next instanceof Error) {
      throw next;
    }
    const text =
      next.text ?? (next.json !== undefined ? JSON.stringify(next.json) : "");
    return new Response(text, {
      status: next.status ?? 200,
      headers: { "Content-Type": "application/json" },
    });
  };
  return { fetchImpl, calls };
}
