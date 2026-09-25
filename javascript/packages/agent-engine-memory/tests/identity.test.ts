import { describe, expect, it } from "vitest";

import { MemoryIdentityError } from "../src/errors.js";
import { resolveIdentity } from "../src/identity.js";

describe("resolveIdentity", () => {
  it("prefers call args over bind then runtime context", () => {
    const resolved = resolveIdentity({
      callArgs: { userId: "call" },
      bindCtx: { userId: "bind", sessionId: "bind-sess" },
      runtimeCtx: { userId: "rt", sessionId: "rt-sess", agentId: "rt-agent" },
    });
    expect(resolved.userId).toBe("call");
    expect(resolved.sessionId).toBe("bind-sess");
    expect(resolved.agentId).toBe("rt-agent");
  });

  it("treats blank/whitespace values as unset at every tier", () => {
    const resolved = resolveIdentity({
      callArgs: { userId: "   " },
      bindCtx: { userId: "" },
      runtimeCtx: { userId: "rt" },
    });
    expect(resolved.userId).toBe("rt");
  });

  it("throws MemoryIdentityError when a required field is unresolved", () => {
    expect(() =>
      resolveIdentity({ callArgs: {}, required: ["userId"] }),
    ).toThrow(MemoryIdentityError);
  });

  it("suppressRuntimeUserId drops only the runtime user_id tier", () => {
    const resolved = resolveIdentity({
      callArgs: {},
      runtimeCtx: { userId: "rt", sessionId: "rt-sess" },
      suppressRuntimeUserId: true,
    });
    expect(resolved.userId).toBeNull();
    expect(resolved.sessionId).toBe("rt-sess");
  });

  it("suppressInheritedSessionId keeps only an explicit session_id", () => {
    const inherited = resolveIdentity({
      callArgs: {},
      bindCtx: { sessionId: "bind-sess" },
      runtimeCtx: { sessionId: "rt-sess" },
      suppressInheritedSessionId: true,
    });
    expect(inherited.sessionId).toBeNull();

    const explicit = resolveIdentity({
      callArgs: { sessionId: "explicit" },
      bindCtx: { sessionId: "bind-sess" },
      suppressInheritedSessionId: true,
    });
    expect(explicit.sessionId).toBe("explicit");
  });
});
