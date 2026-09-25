/*
 * Mirrors runner ../agent-engine-sdk-memory/tests/test_package_contract.py
 * (PORTABLE SUBSET).
 *
 * Ported cases:
 * - the runtime (non-type) export key set equals the expected curated set
 * - "MemoryClient" is not re-exported from the package root
 * - MemoryRouteNotFoundError subclasses MemoryBadRequestError
 * - SearchSource and MemorySource enum shapes are exact
 *
 * Parked (no TS analog):
 * - import-slimness / denylist guards (Python subprocess fresh-import check has no
 *   TS equivalent here)
 *
 * Adaptation notes:
 * - Python asserts a frozen __all__; TS type-only exports are erased at runtime,
 *   so the expected set is derived from src/index.ts value/class/schema/error
 *   exports only (MemoryOptions, JsonValue, the *Result/Runtime types, etc. are
 *   type-only and absent from the namespace object).
 * - The enums are `as const` objects, not runtime-frozen, so exact shape is
 *   asserted via deep equality rather than Object.isFrozen.
 */

import { describe, expect, it } from "vitest";

import {
  MemoryBadRequestError,
  MemoryRouteNotFoundError,
} from "../src/errors.js";
import { MemorySource, SearchSource } from "../src/models.js";
import * as pkg from "../src/index.js";

const EXPECTED_RUNTIME_EXPORTS = new Set([
  "Memory",
  "MemorySource",
  "SearchSource",
  "toSearchSource",
  "WriteTurnResultSchema",
  "CreateSemanticResultSchema",
  "CreateEpisodicResultSchema",
  "CreateTaxonomicResultSchema",
  "CreateProceduralResultSchema",
  "CustomMemorySaveResultSchema",
  "RetrievedCustomMemorySchema",
  "CustomMemoryRetrieveResultSchema",
  "MemoryChunkSchema",
  "ContextMetadataSchema",
  "ContextResponseSchema",
  "hasAmbientIdentity",
  "resolveIdentity",
  "MemoryClientAdapter",
  "MemoryIdentityError",
  "MemoryClientError",
  "MemoryNotSupportedError",
  "MemoryAPIError",
  "MemoryAuthError",
  "MemoryNotProvisionedError",
  "MemoryBadRequestError",
  "MemoryRouteNotFoundError",
  "MemoryServerError",
  "MemoryConnectionError",
]);

describe("package boundary contract", () => {
  it("exposes exactly the curated runtime export set", () => {
    expect(new Set(Object.keys(pkg))).toEqual(EXPECTED_RUNTIME_EXPORTS);
  });

  it("does not re-export the low-level MemoryClient", () => {
    expect("MemoryClient" in pkg).toBe(false);
  });

  it("keeps MemoryRouteNotFoundError a subclass of MemoryBadRequestError", () => {
    expect(new MemoryRouteNotFoundError("x")).toBeInstanceOf(
      MemoryBadRequestError,
    );
  });

  it("freezes the SearchSource shape to the four searchable sources", () => {
    expect({ ...SearchSource }).toEqual({
      SEMANTIC: "semantic",
      EPISODIC: "episodic",
      TAXONOMIC: "taxonomic",
      PROCEDURAL: "procedural",
    });
  });

  it("freezes the MemorySource shape to the five sources", () => {
    expect({ ...MemorySource }).toEqual({
      STM: "stm",
      EPISODIC: "episodic",
      SEMANTIC: "semantic",
      TAXONOMIC: "taxonomic",
      PROCEDURAL: "procedural",
    });
  });
});
