/**
 * Verify agent-engine-runner-shared source has no LangChain / LangGraph / OpenInference
 * imports. Mirrors Python's tests/unit/test_no_lc_imports.py (which uses ast).
 *
 * Approach: scan src/ for static and dynamic ESM import specifiers via regex.
 * Pure ESM package, so no CJS require() check needed.
 *
 * Forbidden prefix mapping (Python → TS):
 *   langchain               → langchain, @langchain/
 *   langgraph               → langgraph, @langgraph/
 *   openinference           → openinference, @arizeai/openinference-
 *   agent_engine_sdk_langgraph    → (no TS analog)
 *   agent_engine_sdk_langgraph          → (removed Python namespace guard)
 */

import { describe, test, expect } from "vitest";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { fileURLToPath } from "node:url";

const RUNNER_SHARED_SRC = fileURLToPath(new URL("../../src/", import.meta.url));

const FORBIDDEN_PREFIXES = [
  "langchain",
  "@langchain/",
  "langgraph",
  "@langgraph/",
  "openinference",
  "@arizeai/openinference-",
] as const;

function walkTsFiles(dir: string): string[] {
  const out: string[] = [];
  for (const entry of readdirSync(dir)) {
    const p = join(dir, entry);
    if (statSync(p).isDirectory()) {
      out.push(...walkTsFiles(p));
    } else if (p.endsWith(".ts")) {
      out.push(p);
    }
  }
  return out;
}

/**
 * Extract every import specifier from TS source. Catches:
 *   import ... from '...'         (named, default, namespace, type-only)
 *   import '...'                  (side-effect)
 *   export ... from '...'         (re-export)
 *   await import('...')           (dynamic)
 */
function extractImportSpecifiers(
  src: string,
): Array<{ line: number; spec: string }> {
  const out: Array<{ line: number; spec: string }> = [];
  const patterns = [
    /^[ \t]*(?:import|export)[^\n;]*?['"]([^'"]+)['"]/gm,
    /\bimport\s*\(\s*['"]([^'"]+)['"]\s*\)/g,
  ];
  for (const re of patterns) {
    let m: RegExpExecArray | null;
    while ((m = re.exec(src)) !== null) {
      const line = src.slice(0, m.index).split("\n").length;
      out.push({ line, spec: m[1] });
    }
  }
  return out;
}

function isForbidden(spec: string): boolean {
  return FORBIDDEN_PREFIXES.some((p) => spec.startsWith(p));
}

describe("agent-engine-runner-shared source imports", () => {
  test("no LangChain / LangGraph / OpenInference imports", () => {
    const violations: string[] = [];
    for (const path of walkTsFiles(RUNNER_SHARED_SRC).sort()) {
      const src = readFileSync(path, "utf-8");
      for (const { line, spec } of extractImportSpecifiers(src)) {
        if (isForbidden(spec)) {
          const rel = relative(RUNNER_SHARED_SRC, path);
          violations.push(`${rel}:${line}: ${spec}`);
        }
      }
    }
    expect(
      violations,
      `Forbidden imports in agent-engine-runner-shared:\n${violations.join("\n")}`,
    ).toEqual([]);
  });
});
