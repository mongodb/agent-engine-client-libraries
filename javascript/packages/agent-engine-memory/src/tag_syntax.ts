/**
 * Syntax-only client-side checks for custom-type names and tag payloads.
 *
 * Mirrors the shape rules of the memory-server's tag canonicalization with
 * byte-identical messages, so a fail-fast client error reads the same as a
 * server rejection. Deliberately NOT mirrored (server-owned): the int64 range
 * guard, dotted+nested mixed-form detection, operator-expression detection,
 * key-count caps, declaration membership — those values reach the platform
 * intact, so the server's own rejection is the one the caller sees.
 *
 * Non-finite numbers are the exception: JSON.stringify rewrites them to null,
 * so the value never survives the wire for the server to judge. Rejecting here
 * is what makes the server's rule reachable at all.
 */

import { MemoryClientError } from "./errors.js";
import type { TagScalar } from "./models.js";

export type { TagScalar };
export type TagMap = Record<string, TagScalar | Record<string, TagScalar>>;

/** Mirror of pkg/memoryconfig/reserved_names.json (parity-tested in Python). */
export const RESERVED_TYPE_NAMES: ReadonlySet<string> = new Set([
  "semantic",
  "episodic",
  "taxonomic",
  "procedural",
  "snapshot",
  "entity",
  "preferences",
  "turn",
  "stm",
  "short_term",
]);

const MAX_TAG_KEY_DEPTH = 2;

/**
 * Mirror of the name_pattern in pkg/memoryconfig/reserved_names.json. A name
 * that cannot be declared can never exist, so checking it here turns an
 * unroutable request into a precise local error.
 */
const NAME_RE = /^[a-z][a-z0-9_]{0,63}$/;

export function validateMemoryType(memoryType: string): void {
  if (typeof memoryType !== "string" || memoryType.trim() === "") {
    throw new MemoryClientError("memory_type must be a non-empty string");
  }
  if (RESERVED_TYPE_NAMES.has(memoryType)) {
    throw new MemoryClientError(
      `'${memoryType}' is a built-in memory type and is not accepted by this operation`,
    );
  }
  if (!NAME_RE.test(memoryType)) {
    throw new MemoryClientError(
      `custom type name '${memoryType}' must match ${NAME_RE.source}`,
    );
  }
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return (
    typeof value === "object" &&
    value !== null &&
    !Array.isArray(value) &&
    Object.getPrototypeOf(value) === Object.prototype
  );
}

function checkPath(path: string): void {
  const segments = path.split(".");
  if (segments.some((segment) => segment === "")) {
    throw new MemoryClientError(`tag '${path}' must have a non-empty path`);
  }
  if (segments.length > MAX_TAG_KEY_DEPTH) {
    throw new MemoryClientError(
      `tag '${path}': nesting is at most one level deep`,
    );
  }
}

function checkScalar(path: string, value: unknown): void {
  if (typeof value === "string") {
    if (value.trim() === "") {
      throw new MemoryClientError(
        `tag '${path}' must have a non-empty string value`,
      );
    }
  } else if (typeof value === "number") {
    // JSON.stringify turns NaN and Infinity into null, which would store a
    // value the server never agreed to. Reject before that silent rewrite.
    if (!Number.isFinite(value)) {
      throw new MemoryClientError(`tag '${path}' must be a finite number`);
    }
  } else if (typeof value !== "boolean") {
    throw new MemoryClientError(
      `tag '${path}' must be a non-empty string, number, or boolean`,
    );
  }
}

/**
 * Yield [dotted path, value] pairs from either accepted wire form.
 *
 * Group-level rules are applied here, so the per-leaf checks that follow run
 * over one uniform sequence rather than once per wire form.
 */
function* flatten(tags: Record<string, unknown>): Generator<[string, unknown]> {
  for (const [key, value] of Object.entries(tags)) {
    if (!isPlainObject(value)) {
      yield [key, value];
      continue;
    }
    if (key.includes(".")) {
      throw new MemoryClientError(
        `tag '${key}': nesting is at most one level deep`,
      );
    }
    if (Object.keys(value).length === 0) {
      throw new MemoryClientError(
        `tag group '${key}' must contain at least one key`,
      );
    }
    for (const [subkey, subvalue] of Object.entries(value)) {
      if (subkey.includes(".") || isPlainObject(subvalue)) {
        throw new MemoryClientError(
          `tag '${key}': nesting is at most one level deep`,
        );
      }
      yield [`${key}.${subkey}`, subvalue];
    }
  }
}

export function validateTagSyntax(tags: Record<string, unknown>): void {
  for (const [path, value] of flatten(tags)) {
    checkPath(path);
    checkScalar(path, value);
  }
}
