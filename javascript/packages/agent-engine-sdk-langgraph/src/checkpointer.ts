/**
 * Checkpointer hardening for the MongoDB saver.
 *
 * `@langchain/langgraph-checkpoint-mongodb`'s `putWrites` builds a bulk-write
 * from the pending writes and calls `collection.bulkWrite(operations)`
 * unconditionally. When a graph step produces no writes — which deep-agent
 * steps routinely do — `operations` is empty and MongoDB rejects it with
 * `MongoInvalidArgumentError: Invalid BulkOperation, Batch cannot be empty`,
 * surfacing as an AER 500. Persisting zero writes is a no-op, so we short-circuit
 * it. This affects any TypeScript agent using the Mongo checkpointer.
 */

import type { BaseCheckpointSaver } from "@langchain/langgraph-checkpoint";

/**
 * Wrap a checkpoint saver so `putWrites` skips empty write batches.
 *
 * Patches the instance in place and returns it. Idempotent and safe on any
 * `BaseCheckpointSaver`; only the empty-batch path is altered, every non-empty
 * call is forwarded unchanged.
 */
export function withEmptyBatchGuard<S extends BaseCheckpointSaver>(
  saver: S,
): S {
  const originalPutWrites = saver.putWrites.bind(saver);
  saver.putWrites = async (config, writes, taskId) => {
    // Only short-circuit the known empty-batch case. Anything else — including a
    // non-array, should the upstream contract change — is forwarded so a real
    // contract violation surfaces instead of silently dropping writes.
    if (Array.isArray(writes) && writes.length === 0) return;
    return originalPutWrites(config, writes, taskId);
  };
  return saver;
}
