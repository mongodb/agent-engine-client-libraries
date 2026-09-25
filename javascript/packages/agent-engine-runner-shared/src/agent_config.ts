/**
 * Structured runtime helpers for reading `agent.yaml`.
 *
 * Lookup order is intentionally small and explicit:
 *   1. `configPath` argument: exact `agent.yaml` path, or a directory that
 *      contains it
 *   2. `AGENTIC_AGENT_CONFIG_PATH`: exact in-container file path baked into
 *      runtime images
 *   3. `AGENTIC_AGENT_WORKDIR`: agent working directory used by generated
 *      local dev stacks
 *   4. current working directory: supports tests and ad-hoc SDK usage
 */

import * as fs from "node:fs";
import * as nodePath from "node:path";

import * as yaml from "js-yaml";
import { z } from "zod";

import { isPlatformEnvVar } from "./utils.js";

// =============================================================================
// Constants
// =============================================================================

/** Pydantic's `Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]`. */
const NonEmptyString = z.string().trim().min(1);

// =============================================================================
// Feature flags
// =============================================================================

/**
 * Runtime feature flags from `agent.yaml`.
 *
 * Keep field names in parity with Python `AgentFeatureConfig` and the CLI
 * allowlist. Add a flag by adding a field here; `FeatureName` and runtime
 * feature access derive from it. `null` means "omitted", letting the runtime
 * fall back to legacy environment variables.
 */
export const AgentFeatureConfigSchema = z
  .object({
    memory: z.boolean().nullable().default(null),
    /**
     * Whether guardrails are enforced for this agent at runtime. `null` (the
     * default) means omitted, letting the runtime fall back to legacy behavior.
     */
    guardrails: z.boolean().nullable().default(null),
    /**
     * Whether the platform provisions playground UI for the agent
     * (null/true = provisioned, today's behavior). When false — e.g. for
     * non-chat agents with no conversation to preview — no playground is
     * built or served; callers use the invoke API directly. Read at
     * deploy/provisioning time only — no runtime effect.
     */
    playground: z.boolean().nullable().default(null),
    /**
     * Whether the agent uses the deep-agent (deepagents) harness. Gates the
     * Tool Pod's built-in filesystem + shell handler registration so tenants
     * that don't run deep agents get no filesystem/shell surface on their
     * Tool Pod. The key stays snake_case `deep_agent` because `agent.yaml` is
     * a cross-language artifact shared with the Python runtime and platform.
     */
    deep_agent: z.boolean().nullable().default(null),
    /**
     * Opt-in for author-defined streaming output shaping. When true the
     * adapter runs the registered output parser and emits `custom_event`
     * frames; off leaves the stream unchanged.
     */
    use_custom_parser: z.boolean().nullable().default(null),
    /**
     * Opt-in for OE-owned durable workflow. Omitted or false means the agent
     * stays on native checkpoints; only an explicit true opts in. When set,
     * OE uses this flag with the advertised language to sticky-assign the
     * session's workflow authority.
     */
    durable_workflow: z.boolean().nullable().default(null),
  })
  // Python: `extra="ignore"` — Zod default behaviour already drops unknown keys.
  .strip();
export type AgentFeatureConfig = z.infer<typeof AgentFeatureConfigSchema>;

/** Derived from `AgentFeatureConfigSchema` — do not maintain a parallel union. */
export type FeatureName = keyof AgentFeatureConfig;

/**
 * Return only the flags explicitly present in `agent.yaml` (omit unset /
 * `null` fields). Mirrors Python's `AgentFeatureConfig.explicit()` — used by
 * the AER's capability advertise to send OE only the flags the author
 * actually set, before SDK-injected defaults (e.g. `owner_callback_fallback`)
 * are layered on top.
 */
export function explicitFeatures(
  features: AgentFeatureConfig,
): Record<string, boolean> {
  const result: Record<string, boolean> = {};
  for (const [name, value] of Object.entries(features)) {
    if (typeof value === "boolean") result[name] = value;
  }
  return result;
}

// =============================================================================
// SecretsConfig
// =============================================================================

// Under `tools`, `invoke_llm` is a reserved key: the platform runs model calls
// in a tool pod, so the LLM provider key(s) are declared there rather than
// under a user-defined tool.
export const SecretsConfigSchema = z
  .object({
    // Parsed for compatibility; ignored at enforcement.
    disable_restriction: z.boolean().default(false),
    aer: z.array(z.string()).default([]),
    tools: z.record(z.string(), z.array(z.string())).default({}),
  })
  .strip();
export type SecretsConfig = z.infer<typeof SecretsConfigSchema>;

/**
 * Input shape callers may construct: every key is omittable, since Zod defaults
 * fill them in. The parsed {@link SecretsConfig} has them all present.
 */
export type SecretsConfigInput = z.input<typeof SecretsConfigSchema>;

// =============================================================================
// MCP config
// =============================================================================
//
// Port of `RuntimeMCPAuthConfig` / `RuntimeMCPServerConfig` / `RuntimeMCPConfig`
// in `agent_engine_runner_shared/agent_config.py`. Keep field names, defaults, and
// validation messages aligned with the Python side.

const FORBIDDEN_MCP_HEADER_KEYS: ReadonlySet<string> = new Set([
  "authorization",
  "cookie",
  "proxy-authorization",
]);

const MCP_AUTH_FIELD_NAMES = [
  "token_env",
  "redirect_uri",
  "client_name",
  "scope",
  "token_url",
  "client_id_env",
  "client_secret_env",
] as const;
type McpAuthFieldName = (typeof MCP_AUTH_FIELD_NAMES)[number];

const MCP_AUTH_REQUIRED_FIELDS: Record<string, readonly McpAuthFieldName[]> = {
  none: [],
  bearer_env: ["token_env"],
  oauth: [],
  client_credentials: ["client_id_env", "client_secret_env"],
};

const MCP_AUTH_ALLOWED_FIELDS: Record<string, readonly McpAuthFieldName[]> = {
  none: [],
  bearer_env: ["token_env"],
  oauth: ["redirect_uri", "client_name", "scope"],
  client_credentials: [
    "token_url",
    "client_id_env",
    "client_secret_env",
    "scope",
  ],
};

function tryParseUrl(value: string): URL | null {
  try {
    return new URL(value);
  } catch {
    return null;
  }
}

export const RuntimeMCPAuthConfigSchema = z
  .object({
    type: z
      .enum(["none", "bearer_env", "oauth", "client_credentials"])
      .default("none"),
    token_env: NonEmptyString.nullable().default(null),
    redirect_uri: NonEmptyString.nullable().default(null),
    client_name: NonEmptyString.nullable().default(null),
    scope: NonEmptyString.nullable().default(null),
    token_url: NonEmptyString.nullable().default(null),
    client_id_env: NonEmptyString.nullable().default(null),
    client_secret_env: NonEmptyString.nullable().default(null),
  })
  .strip()
  .superRefine((value, ctx) => {
    const configuredFields = MCP_AUTH_FIELD_NAMES.filter(
      (name) => value[name] !== null,
    );

    const missing = (MCP_AUTH_REQUIRED_FIELDS[value.type] ?? [])
      .filter((name) => !configuredFields.includes(name))
      .sort();
    if (missing.length > 0) {
      const required = missing.map((name) => `auth.${name}`).join(", ");
      ctx.addIssue({
        code: "custom",
        message: `mcp server auth.type ${value.type} requires fields: ${required}`,
      });
    }

    const allowed = MCP_AUTH_ALLOWED_FIELDS[value.type] ?? [];
    const unsupported = configuredFields
      .filter((name) => !allowed.includes(name))
      .sort();
    if (unsupported.length > 0) {
      const unsupportedStr = unsupported
        .map((name) => `auth.${name}`)
        .join(", ");
      ctx.addIssue({
        code: "custom",
        message: `mcp server auth.type ${value.type} does not support fields: ${unsupportedStr}`,
      });
    }

    if (value.type === "client_credentials" && value.token_url !== null) {
      const parsedTokenUrl = tryParseUrl(value.token_url);
      if (
        parsedTokenUrl === null ||
        parsedTokenUrl.protocol !== "https:" ||
        !parsedTokenUrl.hostname
      ) {
        ctx.addIssue({
          code: "custom",
          message: "mcp server auth.token_url must be an absolute https URL",
          path: ["token_url"],
        });
      }
    }

    for (const field of [
      "token_env",
      "client_id_env",
      "client_secret_env",
    ] as const) {
      const envVar = value[field];
      // These fields are secret-indirection paths read from raw process.env
      // and sent to the configured MCP server or token endpoint. Reject
      // platform-owned names so tenant YAML cannot redirect platform secrets
      // to outbound services.
      if (envVar !== null && isPlatformEnvVar(envVar)) {
        ctx.addIssue({
          code: "custom",
          message: "Please use a different environment variable name",
          path: [field],
        });
      }
    }
  });
export type RuntimeMCPAuthConfig = z.infer<typeof RuntimeMCPAuthConfigSchema>;

export const RuntimeMCPServerConfigSchema = z
  .object({
    transport: z.literal("streamable_http").default("streamable_http"),
    url: NonEmptyString,
    headers: z.record(z.string(), z.string()).default({}),
    auth: RuntimeMCPAuthConfigSchema.default(
      RuntimeMCPAuthConfigSchema.parse({}),
    ),
    allowed_tools: z.array(NonEmptyString).nullable().default(null),
    timeout_seconds: z.number().int().gt(0).default(30),
  })
  .strip()
  .superRefine((value, ctx) => {
    const parsedUrl = tryParseUrl(value.url);
    const validScheme =
      parsedUrl !== null &&
      (parsedUrl.protocol === "http:" || parsedUrl.protocol === "https:") &&
      !!parsedUrl.hostname;
    if (!validScheme) {
      ctx.addIssue({
        code: "custom",
        message: "mcp server url must be an absolute http(s) URL",
        path: ["url"],
      });
    } else if (parsedUrl?.protocol === "http:" && value.auth.type !== "none") {
      ctx.addIssue({
        code: "custom",
        message: "mcp server url must be https when auth.type is set",
        path: ["url"],
      });
    }

    const forbiddenHeaders = Object.keys(value.headers)
      .filter((header) => FORBIDDEN_MCP_HEADER_KEYS.has(header.toLowerCase()))
      .sort();
    if (forbiddenHeaders.length > 0) {
      ctx.addIssue({
        code: "custom",
        message:
          "mcp server headers must not include credential headers: " +
          `[${forbiddenHeaders.map((h) => `'${h}'`).join(", ")}]; use auth.token_env instead`,
        path: ["headers"],
      });
    }
  });
export type RuntimeMCPServerConfig = z.infer<
  typeof RuntimeMCPServerConfigSchema
>;

// A server name with zero alphanumeric characters (e.g. "---") sanitizes to
// an empty string in mcp_tools.ts's `makeMcpSdkToolName` (used to build the
// SDK-visible tool name), which only fails at discovery/call time. Reject it
// here instead so a malformed `mcp.servers` key fails fast at config load.
const MCP_SERVER_NAME_HAS_ALNUM_RE = /[a-zA-Z0-9]/;

export const RuntimeMCPConfigSchema = z
  .object({
    servers: z.record(NonEmptyString, RuntimeMCPServerConfigSchema).default({}),
  })
  .strip()
  .superRefine((value, ctx) => {
    for (const serverName of Object.keys(value.servers)) {
      if (!MCP_SERVER_NAME_HAS_ALNUM_RE.test(serverName)) {
        ctx.addIssue({
          code: "custom",
          message: `mcp server name '${serverName}' must contain alphanumeric characters`,
          path: ["servers", serverName],
        });
      }
    }
  });
export type RuntimeMCPConfig = z.infer<typeof RuntimeMCPConfigSchema>;

// =============================================================================
// `${VAR}` interpolation
// =============================================================================
//
// Port of `_interpolate_env_vars` / `_INTERPOLATABLE_PATHS` in
// `agent_engine_runner_shared/agent_config.py`. Intentionally narrow: only
// `mcp.servers.*.url` is interpolatable today. Adding a path is one array
// entry plus a happy-path test.
const INTERPOLATABLE_PATHS: readonly (readonly string[])[] = [
  ["mcp", "servers", "*", "url"],
];

// Pattern for a well-formed `${VAR}` substitution token.
const VAR_REFERENCE_SOURCE = String.raw`\$\{([A-Za-z_][A-Za-z0-9_]*)\}`;
// Captures any `${...}` token, well-formed or not, so malformed markers
// (`${VAR` unclosed, `${}` empty, `${1bad}` invalid identifier, `${VAR with
// spaces}`) can be rejected with a clear error instead of flowing through to
// the URL validator above.
const ANY_VAR_REFERENCE_SOURCE = String.raw`\$\{[^}]*\}?`;

function pathIsInterpolatable(path: readonly string[]): boolean {
  return INTERPOLATABLE_PATHS.some(
    (pattern) =>
      pattern.length === path.length &&
      pattern.every((segment, i) => segment === "*" || segment === path[i]),
  );
}

function interpolateEnvVars(
  data: unknown,
  envVars: Record<string, string> | undefined,
  path: readonly string[] = [],
): unknown {
  if (Array.isArray(data)) {
    return data.map((item, index) =>
      interpolateEnvVars(item, envVars, [...path, `[${index}]`]),
    );
  }
  if (data !== null && typeof data === "object") {
    const result: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(
      data as Record<string, unknown>,
    )) {
      result[key] = interpolateEnvVars(value, envVars, [...path, key]);
    }
    return result;
  }
  if (typeof data !== "string" || !pathIsInterpolatable(path)) {
    return data;
  }

  const yamlPath = path.join(".");

  // Reject malformed markers before the envVars branch so the error is the
  // same with or without a mapping.
  for (const marker of data.matchAll(
    new RegExp(ANY_VAR_REFERENCE_SOURCE, "g"),
  )) {
    if (!new RegExp(`^${VAR_REFERENCE_SOURCE}$`).test(marker[0])) {
      throw new Error(
        `agent.yaml at ${yamlPath} has malformed environment variable reference ` +
          `${JSON.stringify(marker[0])}; expected \${VAR} where VAR is a valid identifier`,
      );
    }
  }

  if (envVars === undefined) {
    if (new RegExp(VAR_REFERENCE_SOURCE).test(data)) {
      throw new Error(
        `agent.yaml at ${yamlPath} contains \${...} but environment variable ` +
          "interpolation is not enabled for this load. The runtime launcher " +
          "must pass envVars= to loadRuntimeAgentConfig().",
      );
    }
    return data;
  }

  const unset: string[] = [];
  const substituted = data.replace(
    new RegExp(VAR_REFERENCE_SOURCE, "g"),
    (_match, name: string) => {
      if (name in envVars) return envVars[name] ?? "";
      unset.push(name);
      return "";
    },
  );
  if (unset.length > 0) {
    // Report every unset name in one error so a tenant fixing a multi-var URL
    // doesn't have to redeploy once per missing var. Deduplicate while
    // preserving first-seen order for stable messages.
    const seen = new Set<string>();
    const names: string[] = [];
    for (const name of unset) {
      if (!seen.has(name)) {
        seen.add(name);
        names.push(`\${${name}}`);
      }
    }
    throw new Error(
      `agent.yaml at ${yamlPath} references unset environment variables: ${names.join(", ")}`,
    );
  }
  return substituted;
}

// =============================================================================
// _AgentFileConfig — internal raw-file shape
// =============================================================================

const AgentFileConfigSchema = z
  .object({
    entrypoint: NonEmptyString.optional(),
    language: z.string().nullable().optional(),
    framework: z.string().nullable().optional(),
    /**
     * `config` is mostly application-owned. The runtime exposes LLM hints
     * while keeping platform-owned keys out of the provider-specific
     * pass-through map. Python uses `dict[str, Any] | None = None`.
     */
    features: AgentFeatureConfigSchema.default({
      memory: null,
      guardrails: null,
      playground: null,
      deep_agent: null,
      use_custom_parser: null,
      durable_workflow: null,
    }),
    required_secrets: SecretsConfigSchema.default({
      disable_restriction: false,
      aer: [],
      tools: {},
    }),
    mcp: RuntimeMCPConfigSchema.default({ servers: {} }),
  })
  .strip();
type AgentFileConfig = z.infer<typeof AgentFileConfigSchema>;

// =============================================================================
// RuntimeAgentConfig
// =============================================================================

/**
 * Validated runtime view of `agent.yaml`.
 *
 * Class-based because Python uses computed `@property` accessors and helper
 * methods (`feature_enabled`, `configured_feature`). Properties are exposed as
 * TS getters under camelCase names; Python field names on
 * `AgentFeatureConfig` are preserved verbatim so wire/log parity is
 * maintained.
 */
export class RuntimeAgentConfig {
  readonly path: string | null;
  readonly entrypoint: string | null;
  /** Runtime language from agent.yaml; omitted means unset at advertise time. */
  readonly language: string | null;
  /** Application framework from agent.yaml; omitted means unset at advertise time. */
  readonly framework: string | null;
  readonly features: AgentFeatureConfig;
  readonly requiredSecrets: SecretsConfig;
  readonly mcp: RuntimeMCPConfig;

  constructor(
    init: {
      path?: string | null;
      entrypoint?: string | null;
      language?: string | null;
      framework?: string | null;
      features?: AgentFeatureConfig;
      // Input shape: callers may omit `restrict`; it is normalized below.
      requiredSecrets?: SecretsConfigInput;
      mcp?: RuntimeMCPConfig;
    } = {},
  ) {
    this.path = init.path ?? null;
    this.entrypoint = init.entrypoint ?? null;
    this.language = init.language ?? null;
    this.framework = init.framework ?? null;
    this.features = init.features ?? AgentFeatureConfigSchema.parse({});
    this.requiredSecrets = SecretsConfigSchema.parse(
      init.requiredSecrets ?? {},
    );
    this.mcp = init.mcp ?? RuntimeMCPConfigSchema.parse({});
  }

  /** Return a feature flag value, falling back to `defaultValue` when omitted. */
  featureEnabled(name: FeatureName, defaultValue = false): boolean {
    const value = this.configuredFeature(name);
    if (value === null) return defaultValue;
    return value;
  }

  /** Return the explicit feature value from `agent.yaml`, if it exists. */
  configuredFeature(name: FeatureName): boolean | null {
    // Normalize absent keys to null (Python getattr returns None) — a
    // RuntimeAgentConfig constructed with a plain object won't have schema
    // defaults applied.
    return this.features[name] ?? null;
  }
}

// =============================================================================
// Config file discovery + loading
// =============================================================================

function candidateLocations(configPath: string | undefined): string[] {
  if (configPath !== undefined) {
    // Explicit override from the caller. Accept the exact file path or the
    // directory that contains `agent.yaml`.
    return [configPath];
  }

  const candidates: string[] = [];
  for (const raw of [
    // Built/runtime images set the exact in-container `agent.yaml` path.
    process.env["AGENTIC_AGENT_CONFIG_PATH"],
    // Local dev stacks set the agent workspace directory.
    process.env["AGENTIC_AGENT_WORKDIR"],
    // Tests and ad-hoc usage often run directly from the agent directory.
    process.cwd(),
  ]) {
    if (!raw) continue;
    if (!candidates.includes(raw)) candidates.push(raw);
  }
  return candidates;
}

function configFileForLocation(location: string): string {
  if (nodePath.basename(location) === "agent.yaml") return location;
  return nodePath.join(location, "agent.yaml");
}

function discoverAgentConfigPath(
  configPath: string | undefined,
): string | null {
  for (const candidate of candidateLocations(configPath)) {
    const filePath = configFileForLocation(candidate);
    try {
      if (fs.statSync(filePath).isFile()) return filePath;
    } catch {
      // ENOENT or similar — try the next candidate.
    }
  }
  return null;
}

/**
 * Load and validate `agent.yaml` for runtime use.
 *
 * Missing files are treated as an empty config so unit tests and ad-hoc
 * SDK usage can still construct `App`/`TenantRuntime` outside generated
 * runtime environments. When a file exists, the runtime validates the
 * fields it owns directly (entrypoint/features) while passing the
 * application-owned `config` block through as raw data.
 *
 * `envVars` is the substitution mapping used to resolve `${VAR}` references
 * at allowlisted YAML paths (currently `mcp.servers.*.url`). Pass the
 * tenant-owned subset of the process environment — via `tenantEnvVars()` —
 * never raw `process.env`, so tenant `agent.yaml` cannot dereference platform
 * secrets. When `envVars` is `undefined` and a `${...}` reference is present
 * at an allowlisted path, the loader throws so the misconfiguration is
 * visible instead of falling through to `new URL()` with the literal string.
 */
export function loadRuntimeAgentConfig(
  configPath?: string,
  envVars?: Record<string, string>,
): RuntimeAgentConfig {
  const path = discoverAgentConfigPath(configPath);
  if (path === null) return new RuntimeAgentConfig();

  let text: string;
  try {
    text = fs.readFileSync(path, { encoding: "utf-8" });
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === "ENOENT") {
      return new RuntimeAgentConfig();
    }
    throw err;
  }

  let data: unknown;
  try {
    data = yaml.load(text) ?? {};
  } catch (err) {
    throw new Error(
      `Invalid YAML in agent config at ${path}: ${(err as Error).message}`,
      { cause: err },
    );
  }

  if (data === null || typeof data !== "object" || Array.isArray(data)) {
    throw new Error(`agent.yaml at ${path} must contain a top-level mapping`);
  }

  try {
    data = interpolateEnvVars(data, envVars);
  } catch (err) {
    throw new Error(
      `Invalid agent.yaml at ${path}: ${(err as Error).message}`,
      {
        cause: err,
      },
    );
  }

  let parsed: AgentFileConfig;
  try {
    parsed = AgentFileConfigSchema.parse(data);
  } catch (err) {
    throw new Error(
      `Invalid agent.yaml at ${path}: ${(err as Error).message}`,
      {
        cause: err,
      },
    );
  }

  const language =
    typeof parsed.language === "string" ? parsed.language.trim() : null;
  const framework =
    typeof parsed.framework === "string" ? parsed.framework.trim() : null;
  return new RuntimeAgentConfig({
    path,
    entrypoint:
      typeof parsed.entrypoint === "string" ? parsed.entrypoint.trim() : null,
    language: language || null,
    framework: framework || null,
    features: parsed.features,
    requiredSecrets: parsed.required_secrets,
    mcp: parsed.mcp,
  });
}
