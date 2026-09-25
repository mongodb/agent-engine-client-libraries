/** HTTP client for the platform event API. */

import type { BranchRef, Event } from "../models.js";
import { EventResponseSchema, EventsResponseSchema } from "../api/events.js";

/**
 * Percent-encode an id for use as a single URL path segment, refusing bare
 * dot segments: encodeURIComponent leaves '.'/'..' untouched, and the URL
 * parser would collapse them into a parent-path rewrite of this authenticated
 * request.
 */
function encodeIdSegment(id: string): string {
  if (id === "." || id === "..") {
    throw new Error(`Invalid id: bare dot segment ${JSON.stringify(id)}`);
  }
  return encodeURIComponent(id);
}

export class EventClient {
  private readonly oeUrl: string;
  private readonly timeout: number;

  constructor(oeUrl: string, timeout = 30) {
    this.oeUrl = oeUrl.replace(/\/$/, "");
    this.timeout = timeout;
  }

  async appendEvent(
    sessionId: string,
    actorId: string,
    payload: Record<string, unknown>,
    parentEventId?: string,
    metadata?: Record<string, unknown>,
    branch?: BranchRef,
  ): Promise<Event> {
    const body: Record<string, unknown> = { actor_id: actorId, payload };
    if (parentEventId !== undefined) body["parent_event_id"] = parentEventId;
    if (metadata !== undefined) body["metadata"] = metadata;
    if (branch !== undefined) body["branch"] = branch;

    const resp = await fetch(
      `${this.oeUrl}/v1/sessions/${encodeIdSegment(sessionId)}/events`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(this.timeout * 1000),
      },
    );
    if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${await resp.text()}`);
    return EventResponseSchema.parse(await resp.json()).event;
  }

  async listEvents(
    sessionId: string,
    branch?: string,
    metadataFilters?: Record<string, string>,
  ): Promise<Event[]> {
    const params = new URLSearchParams();
    if (branch !== undefined) params.set("branch", branch);
    if (metadataFilters !== undefined) {
      for (const [key, val] of Object.entries(metadataFilters)) {
        params.set(`metadata.${key}`, val);
      }
    }
    const qs = params.size > 0 ? `?${params.toString()}` : "";
    const resp = await fetch(
      `${this.oeUrl}/v1/sessions/${encodeIdSegment(sessionId)}/events${qs}`,
      {
        signal: AbortSignal.timeout(this.timeout * 1000),
      },
    );
    if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${await resp.text()}`);
    return EventsResponseSchema.parse(await resp.json()).events;
  }

  async getEvent(sessionId: string, eventId: string): Promise<Event | null> {
    const resp = await fetch(
      `${this.oeUrl}/v1/sessions/${encodeIdSegment(sessionId)}/events/${encodeIdSegment(eventId)}`,
      {
        signal: AbortSignal.timeout(this.timeout * 1000),
      },
    );
    if (resp.status === 404) return null;
    if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${await resp.text()}`);
    return EventResponseSchema.parse(await resp.json()).event;
  }
}
