/** API response schemas for the platform event store. */

import { z } from "zod";
import { EventSchema } from "../models.js";

export const EventResponseSchema = z.looseObject({
  event: EventSchema,
});
export type EventResponse = z.infer<typeof EventResponseSchema>;

export const EventsResponseSchema = z.looseObject({
  events: z.array(EventSchema),
});
export type EventsResponse = z.infer<typeof EventsResponseSchema>;
