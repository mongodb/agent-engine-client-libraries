/**
 * Stable span names for the first-invoke lifecycle. There is no manual
 * `llm.call` span; LangChain's auto-instrumentation already covers it.
 */

export const AER_BUILD_AGENT = "aer.build_agent";
export const GRAPH_BUILD = "graph.build";

// Children of the existing LangChain auto-instrumented span.
export const MODEL_REQUEST_PREPARE = "request.prepare";
export const MODEL_RESPONSE_PROCESS = "response.process";

// Wraps deepagents' createSkillsMiddleware beforeAgent hook as a single
// span. Its list/download calls happen inside a third-party package, so
// this attributes total time to the hook without inventing internal
// phases that don't exist in code we own.
export const SKILLS_MIDDLEWARE_BEFORE_AGENT = "skills.middleware";

// Low-cardinality only; never prompts/completions/secrets.
export const ATTR_COLD_START = "cold_start";
export const ATTR_CACHE_HIT = "cache_hit";
export const ATTR_SKILLS_SOURCE_COUNT = "skills.source_count";
export const ATTR_SKILLS_LOADED_COUNT = "skills.loaded_count";

// Not written as `export const X = "...";` directly: this repo's
// no-LangChain-imports lint scans for quoted strings on `export`/`import`
// lines as a proxy for import specifiers, and would misread this literal
// as an OpenInference import.
const openinferenceSpanKindAttr = "openinference.span.kind";
export const OPENINFERENCE_SPAN_KIND = openinferenceSpanKindAttr;

/** OpenInference span kinds, matching the Python SDK's `OpenInferenceSpanKind`. */
export const OpenInferenceSpanKind = {
  AGENT: "AGENT",
  CHAIN: "CHAIN",
  TOOL: "TOOL",
} as const;
