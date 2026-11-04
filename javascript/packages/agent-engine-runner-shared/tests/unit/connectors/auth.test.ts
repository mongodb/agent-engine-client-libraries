import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { AuthSchema, authHeaderNames } from "../../../src/connectors/index.js";
import {
  formatZodIssues,
  loadAuthoringYaml,
} from "../../../src/connectors/definitions.js";

interface AuthCase {
  name: string;
  auth: Record<string, unknown>;
  error?: string;
}

// The Python suite asserts the same file and checks both copies are identical.
const CASES = loadAuthoringYaml(
  readFileSync(
    fileURLToPath(new URL("./fixtures/auth_cases.yaml", import.meta.url)),
    "utf8",
  ),
) as { valid: AuthCase[]; refused: AuthCase[] };

function refusal(block: unknown): string {
  const parsed = AuthSchema.safeParse(block);
  expect(parsed.success).toBe(false);
  return parsed.success ? "" : formatZodIssues(parsed.error);
}

describe("connector auth block", () => {
  it.each(CASES.valid.map((testCase) => [testCase.name, testCase]))(
    "accepts %s",
    (_name, testCase) => {
      expect(AuthSchema.parse(testCase.auth)).toMatchObject(testCase.auth);
    },
  );

  it.each(CASES.refused.map((testCase) => [testCase.name, testCase]))(
    "refuses %s with the shared message",
    (_name, testCase) => {
      // Python raises only the first refusal, so a second issue here would be
      // text the other runtime never gives, and could echo an unknown key.
      expect(AuthSchema.safeParse(testCase.auth).error?.issues).toHaveLength(1);
      expect(refusal(testCase.auth)).toContain(testCase.error);
    },
  );

  it.each([
    { type: "bearer", env: "ghp_SENTINELVALUE" },
    { bearer: "ghp_SENTINELVALUE", provider: "ghp_SENTINELVALUE" },
    { bearer: "ghp_SENTINELVALUE", scopes: ["ghp_SENTINELVALUE"] },
    { basic: { username: "ghp_SENTINELVALUE", password: "ghp_SENTINELVALUE" } },
    {
      api_keys: [
        { env: "ghp SENTINELVALUE", location: "header", name: "X-Key" },
      ],
    },
    { provider: "ghp SENTINELVALUE" },
    { bearer: "ghp SENTINELVALUE" },
    { bearer: "SERVICE_TOKEN", headers: { "X-Key": "ghp_SENTINELVALUE\n" } },
  ])("never echoes a value in a refusal (%#)", (block) => {
    // A secret pasted where a secret name belongs must not reach logs.
    expect(refusal(block)).not.toContain("SENTINELVALUE");
  });

  it.each([
    { bearer: 1 },
    { bearer: ["SERVICE_TOKEN"] },
    { basic: "SERVICE_USERNAME" },
    { basic: { username: 1, password: "SERVICE_PASSWORD" } },
    { api_keys: { env: "SERVICE_KEY", location: "header", name: "X-Key" } },
    { api_keys: [{ env: "SERVICE_KEY", location: "body", name: "key" }] },
    { provider: ["github"] },
    { provider: "github", scopes: "repo" },
  ])("refuses a malformed value (%#)", (block) => {
    expect(AuthSchema.safeParse(block).success).toBe(false);
  });

  it("treats a null key as absent", () => {
    const auth = AuthSchema.parse({
      bearer: null,
      provider: "github",
      scopes: null,
    });

    expect(auth.provider).toBe("github");
    expect(auth.bearer).toBeNull();
  });

  it.each([
    [{ bearer: "SERVICE_TOKEN" }, ["authorization"]],
    [
      { basic: { username: "SERVICE_USERNAME", password: "SERVICE_PASSWORD" } },
      ["authorization"],
    ],
    [
      { provider: "github", headers: { "X-Api-Version": "2" } },
      ["authorization", "x-api-version"],
    ],
    [
      {
        api_keys: [
          { env: "KEY", location: "header", name: "X-Key" },
          { env: "QUERY_KEY", location: "query", name: "key" },
        ],
      },
      ["x-key"],
    ],
  ])("reserves every credential header (%#)", (block, headers) => {
    // Model arguments can never set a header the credential owns.
    expect(authHeaderNames(AuthSchema.parse(block))).toEqual(new Set(headers));
  });
});
