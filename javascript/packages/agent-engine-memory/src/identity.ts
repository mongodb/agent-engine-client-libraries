/** Identity-resolution algorithm: call arg > bind context > runtime context. */

import { MemoryIdentityError } from "./errors.js";
import type { MemoryRequestContext } from "./transport.js";

const FIELDS = ["userId", "agentId", "sessionId"] as const;
type Field = (typeof FIELDS)[number];

export type ResolvedIdentity = Record<Field, string | null>;

function normalize(value: string | null | undefined): string | null {
  // A blank or whitespace-only value carries no identity; treating it as unset
  // keeps it from satisfying `required` or shadowing a bound value.
  if (value === null || value === undefined || value.trim() === "") {
    return null;
  }
  return value;
}

/**
 * Resolve identity fields by precedence: call arg > bind ctx > runtime ctx.
 *
 * Blank or whitespace-only values are treated as unset at every tier.
 *
 * Two optional guards drop tiers for one field each:
 *
 * `suppressRuntimeUserId` drops the runtime tier for `userId`. A read asking for
 * a broad visibility must not borrow the ambient principal as its filter. A
 * `userId` set by the caller or by bind is deliberate and still applies.
 *
 * `suppressInheritedSessionId` drops the bind and runtime tiers for `sessionId`,
 * so only an explicit call arg applies. A `sessionId` scopes conversation I/O,
 * not long-term search; episodic search matches it exactly and consolidated
 * episodic memory is stored without one, so an inherited session matches nothing.
 */
export function resolveIdentity(args: {
  callArgs: Partial<Record<Field, string | null | undefined>>;
  bindCtx?: MemoryRequestContext | null;
  runtimeCtx?: MemoryRequestContext | null;
  required?: readonly Field[];
  suppressRuntimeUserId?: boolean;
  suppressInheritedSessionId?: boolean;
}): ResolvedIdentity {
  const {
    callArgs,
    bindCtx = null,
    runtimeCtx = null,
    required = [],
    suppressRuntimeUserId = false,
    suppressInheritedSessionId = false,
  } = args;

  const resolved = {} as ResolvedIdentity;
  for (const field of FIELDS) {
    let value = normalize(callArgs[field]);
    const skipInherited = suppressInheritedSessionId && field === "sessionId";
    if (value === null && bindCtx !== null && !skipInherited) {
      value = normalize(bindCtx[field]);
    }
    const skipRuntime =
      skipInherited || (suppressRuntimeUserId && field === "userId");
    if (value === null && runtimeCtx !== null && !skipRuntime) {
      value = normalize(runtimeCtx[field]);
    }
    resolved[field] = value;
  }

  const missing = required.filter((field) => resolved[field] === null);
  if (missing.length > 0) {
    throw new MemoryIdentityError(
      `could not resolve required identity field(s): ${missing.join(", ")}`,
    );
  }
  return resolved;
}
