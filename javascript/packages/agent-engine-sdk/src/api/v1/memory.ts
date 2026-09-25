/**
 * Canonical types for the Memory Server HTTP API.
 *
 * Single source of truth for result shapes returned by the memory server.
 * Consumed by MemoryClient and by framework SDK adapters (e.g. sdk-langgraph).
 */

import { z } from "zod";
import { DateFromStringSchema } from "../../models.js";

// =============================================================================
// Result models
//
// All response schemas use `z.looseObject` so unknown fields added by the
// server are preserved on the parsed result, rather than silently dropped.
// Request/config schemas (ContextConfigSchema) use strict object semantics.
// =============================================================================

export const WriteTurnResultSchema = z.looseObject({
  id: z.string(),
  session_id: z.string(),
  turn_seq: z.number(),
  acknowledged: z.boolean(),
});
export type WriteTurnResult = z.infer<typeof WriteTurnResultSchema>;

export const CreateSemanticResultSchema = z.looseObject({
  id: z.string(),
  label: z.string(),
  has_embedding: z.boolean(),
  acknowledged: z.boolean(),
});
export type CreateSemanticResult = z.infer<typeof CreateSemanticResultSchema>;

export const BulkCreateSemanticResultSchema = z.looseObject({
  created_count: z.number(),
  skipped_count: z.number(),
  created_ids: z.array(z.string()),
  skipped_labels: z.array(z.string()),
  has_embeddings: z.boolean(),
  acknowledged: z.boolean(),
});
export type BulkCreateSemanticResult = z.infer<
  typeof BulkCreateSemanticResultSchema
>;

export const CreateEpisodicResultSchema = z.looseObject({
  id: z.string(),
  title: z.string(),
  has_embedding: z.boolean(),
  acknowledged: z.boolean(),
});
export type CreateEpisodicResult = z.infer<typeof CreateEpisodicResultSchema>;

export const CreateTaxonomicResultSchema = z.looseObject({
  id: z.string(),
  domain: z.string(),
  term: z.string(),
  has_embedding: z.boolean(),
  acknowledged: z.boolean(),
});
export type CreateTaxonomicResult = z.infer<typeof CreateTaxonomicResultSchema>;

export const CreateProceduralResultSchema = z.looseObject({
  id: z.string(),
  procedure: z.string(),
  has_embedding: z.boolean(),
  acknowledged: z.boolean(),
});
export type CreateProceduralResult = z.infer<
  typeof CreateProceduralResultSchema
>;

export const CreateUserContextResultSchema = z.looseObject({
  id: z.string(),
  context_key: z.string(),
  acknowledged: z.boolean(),
});
export type CreateUserContextResult = z.infer<
  typeof CreateUserContextResultSchema
>;

export const CreateSnapshotResultSchema = z.looseObject({
  id: z.string(),
  session_id: z.string(),
  message_count: z.number(),
  has_embedding: z.boolean(),
  embedding_strategy: z.string().optional(),
  snapshot_reason: z.string(),
  acknowledged: z.boolean(),
});
export type CreateSnapshotResult = z.infer<typeof CreateSnapshotResultSchema>;

export const PromoteSnapshotResultSchema = z.looseObject({
  snapshot_id: z.string().optional(),
  session_id: z.string(),
  promoted_count: z.number(),
  reason: z.string(),
  has_embedding: z.boolean().default(false),
  acknowledged: z.boolean(),
});
export type PromoteSnapshotResult = z.infer<typeof PromoteSnapshotResultSchema>;

export const DeleteResultSchema = z.looseObject({
  deleted_count: z.number(),
  acknowledged: z.boolean(),
});
export type DeleteResult = z.infer<typeof DeleteResultSchema>;

export const InternalStateResultSchema = z.looseObject({
  session_id: z.string(),
  version: z.number(),
  token_count: z.number(),
  content_hash: z.string(),
  previous_version: z.number().optional(),
  was_noop: z.boolean(),
  acknowledged: z.boolean(),
});
export type InternalStateResult = z.infer<typeof InternalStateResultSchema>;

export const ISGenerationResultSchema = z.looseObject({
  updated: z.boolean(),
  version: z.number().optional(),
  token_count: z.number().optional(),
  content_hash: z.string().optional(),
  previous_version: z.number().optional(),
  was_noop: z.boolean(),
  coalesced: z.boolean(),
  skipped_reason: z.string().optional(),
  next_update_at: z.string().optional(),
  model_id: z.string().optional(),
  events_hash: z.string().optional(),
});
export type ISGenerationResult = z.infer<typeof ISGenerationResultSchema>;

export const TagScalarSchema = z.union([z.string(), z.number(), z.boolean()]);
export type TagScalar = z.infer<typeof TagScalarSchema>;

// Deliberate deviation from the module's looseObject convention: these
// custom-type schemas stay strict so the server's internal similarity ranking
// `score` never leaks into the public result shape. The trade-off is that
// future additive server fields are dropped rather than preserved.
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

export const MemorySourceSchema = z.enum([
  "stm",
  "episodic",
  "semantic",
  "taxonomic",
  "procedural",
]);
export type MemorySource = z.infer<typeof MemorySourceSchema>;
export const MemorySource = {
  STM: "stm" as const,
  EPISODIC: "episodic" as const,
  SEMANTIC: "semantic" as const,
  TAXONOMIC: "taxonomic" as const,
  PROCEDURAL: "procedural" as const,
} as const;

export const FormatStyleSchema = z.enum(["openai", "claude", "jinja2"]);
export type FormatStyle = z.infer<typeof FormatStyleSchema>;
export const FormatStyle = {
  OPENAI: "openai" as const,
  CLAUDE: "claude" as const,
  JINJA2: "jinja2" as const,
} as const;

export const ModelTypeSchema = z.enum([
  "openai",
  "claude",
  "anthropic",
  "gemini",
  "generic",
]);
export type ModelType = z.infer<typeof ModelTypeSchema>;
export const ModelType = {
  OPENAI: "openai" as const,
  CLAUDE: "claude" as const,
  ANTHROPIC: "anthropic" as const,
  GEMINI: "gemini" as const,
  GENERIC: "generic" as const,
} as const;

// =============================================================================
// Retrieval models
// =============================================================================

/** Unified representation of a retrieved memory chunk across all memory types. */
export const MemoryChunkSchema = z.looseObject({
  id: z.string(),
  content: z.string(),
  source: MemorySourceSchema,
  timestamp: DateFromStringSchema,
  embedding: z.array(z.number()).optional(),
  similarity_score: z.number().optional(),
  metadata: z.record(z.string(), z.unknown()).optional(),
});
export type MemoryChunk = z.infer<typeof MemoryChunkSchema>;

const lowerCaseStrings = (v: unknown): unknown =>
  typeof v === "string" ? v.toLowerCase() : v;
const dedupSources = (arr: MemorySource[]): MemorySource[] =>
  Array.from(new Set(arr));

/** Configuration for context building requests. */
export const ContextConfigSchema = z
  .object({
    max_tokens: z.number().int().min(1).default(16000),
    token_buffer: z.number().int().min(0).default(500),
    format_style: z
      .preprocess(lowerCaseStrings, FormatStyleSchema)
      .default(FormatStyle.OPENAI),
    model_type: z
      .preprocess(lowerCaseStrings, ModelTypeSchema)
      .default(ModelType.OPENAI),
    similarity_threshold: z.number().min(0).max(1).default(0),
    ranking_weights: z.record(z.string(), z.number()).default({}),
    enabled_sources: z
      .array(MemorySourceSchema)
      .transform(dedupSources)
      .default([MemorySource.EPISODIC, MemorySource.SEMANTIC]),
    include_memories: z.boolean().default(false),
  })
  .refine((data) => data.enabled_sources.length > 0, {
    message: "enabled_sources cannot be empty",
    path: ["enabled_sources"],
  });
export type ContextConfig = z.infer<typeof ContextConfigSchema>;

/** Metadata about a context build response (token counts, per-source counts, timing). */
export const ContextMetadataSchema = z.looseObject({
  token_count: z.number().int().min(0).default(0),
  memory_counts: z.record(z.string(), z.number()).default({}),
  timing: z.record(z.string(), z.number()).default({}),
});
export type ContextMetadata = z.infer<typeof ContextMetadataSchema>;

/** Response from a context build request containing formatted context and retrieval metadata. */
export const ContextResponseSchema = z.looseObject({
  formatted_context: z.union([
    z.string(),
    z.array(z.record(z.string(), z.unknown())),
  ]),
  metadata: ContextMetadataSchema,
  // Mirrors Python `list[MemoryChunk] | None` (default None): the server sends
  // `null` when include_memories=False (the only path any caller uses), so accept
  // null as well as undefined/array.
  selected_memories: z.array(MemoryChunkSchema).nullish(),
});
export type ContextResponse = z.infer<typeof ContextResponseSchema>;
