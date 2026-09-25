/**
 * Canonical models for the Memory API (TypeScript port).
 *
 * These zod schemas define the response types the Memory Server returns and the
 * facade surfaces. They are the TypeScript counterpart of the Python
 * `agent_engine_sdk_memory.models` module — only the shapes the unified `Memory`
 * facade actually returns are ported here.
 */

import { z } from "zod";

export type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue };

// =============================================================================
// Result models (operation results)
// =============================================================================

export const WriteTurnResultSchema = z.object({
  id: z.string(),
  session_id: z.string(),
  turn_seq: z.number().int(),
  acknowledged: z.boolean().default(true),
  has_embedding: z.boolean().default(false),
});
export type WriteTurnResult = z.infer<typeof WriteTurnResultSchema>;

export const CreateSemanticResultSchema = z.object({
  id: z.string(),
  label: z.string(),
  has_embedding: z.boolean(),
  acknowledged: z.boolean().default(true),
});
export type CreateSemanticResult = z.infer<typeof CreateSemanticResultSchema>;

export const CreateEpisodicResultSchema = z.object({
  id: z.string(),
  title: z.string(),
  has_embedding: z.boolean(),
  acknowledged: z.boolean().default(true),
});
export type CreateEpisodicResult = z.infer<typeof CreateEpisodicResultSchema>;

export const CreateTaxonomicResultSchema = z.object({
  id: z.string(),
  domain: z.string(),
  term: z.string(),
  has_embedding: z.boolean(),
  acknowledged: z.boolean().default(true),
});
export type CreateTaxonomicResult = z.infer<typeof CreateTaxonomicResultSchema>;

export const CreateProceduralResultSchema = z.object({
  id: z.string(),
  procedure: z.string(),
  has_embedding: z.boolean(),
  acknowledged: z.boolean().default(true),
});
export type CreateProceduralResult = z.infer<
  typeof CreateProceduralResultSchema
>;

export const TagScalarSchema = z.union([z.string(), z.number(), z.boolean()]);
export type TagScalar = z.infer<typeof TagScalarSchema>;

export const CustomMemorySaveResultSchema = z.object({
  id: z.string(),
  type: z.string(),
  tags: z.record(z.string(), TagScalarSchema),
  has_embedding: z.boolean(),
});
export type CustomMemorySaveResult = z.infer<
  typeof CustomMemorySaveResultSchema
>;

export const RetrievedCustomMemorySchema = z.object({
  id: z.string(),
  type: z.string(),
  content: z.string(),
  tags: z.record(z.string(), TagScalarSchema),
  contextual_metadata: z.record(z.string(), z.unknown()).nullish(),
  user_id: z.string().nullish(),
  agent_id: z.string().nullish(),
});
export type RetrievedCustomMemory = z.infer<typeof RetrievedCustomMemorySchema>;

export const CustomMemoryRetrieveResultSchema = z.object({
  results: z.array(RetrievedCustomMemorySchema),
  count: z.number(),
});
export type CustomMemoryRetrieveResult = z.infer<
  typeof CustomMemoryRetrieveResultSchema
>;

// =============================================================================
// Retrieval enums
// =============================================================================

/** Source type for memory chunks. */
export const MemorySource = {
  STM: "stm",
  EPISODIC: "episodic",
  SEMANTIC: "semantic",
  TAXONOMIC: "taxonomic",
  PROCEDURAL: "procedural",
} as const;
export type MemorySource = (typeof MemorySource)[keyof typeof MemorySource];

/**
 * A memory source that `Memory.search` can query.
 *
 * Mirrors `MemorySource` minus `stm` (short-term turns are not a search target).
 * Callers may pass either the constant or its string value.
 */
export const SearchSource = {
  SEMANTIC: "semantic",
  EPISODIC: "episodic",
  TAXONOMIC: "taxonomic",
  PROCEDURAL: "procedural",
} as const;
export type SearchSource = (typeof SearchSource)[keyof typeof SearchSource];

const SEARCH_SOURCE_VALUES = new Set<string>(Object.values(SearchSource));

/** Narrow an arbitrary string to a `SearchSource`, throwing on an unknown value. */
export function toSearchSource(value: string): SearchSource {
  if (!SEARCH_SOURCE_VALUES.has(value)) {
    throw new RangeError(`unknown search source: ${value}`);
  }
  return value as SearchSource;
}

// =============================================================================
// Retrieval models (memory chunks and context)
// =============================================================================

const MemorySourceSchema = z.enum([
  MemorySource.STM,
  MemorySource.EPISODIC,
  MemorySource.SEMANTIC,
  MemorySource.TAXONOMIC,
  MemorySource.PROCEDURAL,
]);

/**
 * Unified representation of memory from any source (STM, episodic, semantic, …).
 *
 * `timestamp` is coerced to a `Date`; a missing value on the wire is a server
 * contract violation and surfaces as a schema error at parse time.
 *
 * `similarity_score` is on the scale of the retrieval mode that produced the
 * chunk: vector similarity for semantic mode, text relevance for text mode, and
 * the fused rank-fusion score for hybrid mode. Comparable between chunks
 * retrieved the same way; not comparable across modes, or across sources
 * searched differently. Null when the chunk did not come from a search.
 */
export const MemoryChunkSchema = z.object({
  id: z.string(),
  content: z.string(),
  source: MemorySourceSchema,
  timestamp: z.coerce.date(),
  embedding: z.array(z.number()).nullish(),
  similarity_score: z.number().nullish(),
  metadata: z.record(z.string(), z.unknown()).nullish(),
});
export type MemoryChunk = z.infer<typeof MemoryChunkSchema>;

export const ContextMetadataSchema = z.object({
  token_count: z.number().int().default(0),
  memory_counts: z.record(z.string(), z.number()).default({}),
  timing: z.record(z.string(), z.number()).default({}),
});
export type ContextMetadata = z.infer<typeof ContextMetadataSchema>;

/**
 * Final response model for context building. `formatted_context` is a string or
 * a list of message dicts depending on the server's format style.
 */
export const ContextResponseSchema = z.object({
  formatted_context: z.union([
    z.string(),
    z.array(z.record(z.string(), z.unknown())),
  ]),
  metadata: ContextMetadataSchema.default(() => ({
    token_count: 0,
    memory_counts: {},
    timing: {},
  })),
  selected_memories: z.array(MemoryChunkSchema).nullish(),
});
export type ContextResponse = z.infer<typeof ContextResponseSchema>;
