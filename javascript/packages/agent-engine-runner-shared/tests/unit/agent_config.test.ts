/**
 * Tests for `loadRuntimeAgentConfig` and `TenantRuntime` agent.yaml integration.
 *
 * Mirrors Python's tests/unit/test_agent_config.py.
 *
 * TS-vs-Python adaptations:
 *   - Python `monkeypatch.chdir(tmp_path)` → TS `process.chdir(tmpDir)` with
 *     explicit save/restore in afterEach (Vitest has no equivalent helper).
 *   - Python `monkeypatch.setenv` → `vi.stubEnv` (auto-restored).
 *   - Path comparisons use `realpathSync(tmpDir)` to canonicalise — macOS
 *     resolves `/var/folders` to `/private/var/folders` via symlink, and
 *     `process.cwd()` after `process.chdir()` returns the canonical form.
 *     The returned object preserves snake_case wire fields (`base_url`).
 *   - TS exposes `featureEnabled(name, default)` where Python has
 *     `feature_enabled(name)`. Same API shape, default arg added in TS.
 *     No feature flags are currently defined, so the feature-resolution
 *     scaffolding has no live toggle to assert against.
 */

import { describe, test, expect, beforeEach, afterEach, vi } from "vitest";
import { mkdtempSync, writeFileSync, rmSync, realpathSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  loadRuntimeAgentConfig,
  RuntimeAgentConfig,
  TenantRuntime,
} from "../../src/index.js";
import { MemoryWriter } from "../../src/memory_writer.js";

// ---------------------------------------------------------------------------
// Test scaffolding: per-test tmp dir + cwd swap
// ---------------------------------------------------------------------------

let tmpDir: string;
let originalCwd: string;

beforeEach(() => {
  tmpDir = realpathSync(mkdtempSync(join(tmpdir(), "agent-config-")));
  originalCwd = process.cwd();
  process.chdir(tmpDir);
});

afterEach(() => {
  process.chdir(originalCwd);
  rmSync(tmpDir, { recursive: true, force: true });
  vi.unstubAllEnvs();
});

function writeAgentYaml(content: string): void {
  writeFileSync(join(tmpDir, "agent.yaml"), content, "utf-8");
}

// ---------------------------------------------------------------------------
// loadRuntimeAgentConfig — happy paths
// ---------------------------------------------------------------------------

describe("loadRuntimeAgentConfig — happy paths", () => {
  test("returns empty config when agent.yaml is missing", () => {
    // test_load_runtime_agent_config_returns_empty_when_missing
    const cfg = loadRuntimeAgentConfig();

    expect(cfg.path).toBeNull();
  });

  test("reads features.playground from agent.yaml", () => {
    // test_load_runtime_agent_config_reads_playground
    writeAgentYaml(
      [
        "name: non-chat-agent",
        "entrypoint: brief.agent:app",
        "features:",
        "  playground: false",
      ].join("\n"),
    );

    const cfg = loadRuntimeAgentConfig();

    expect(cfg.configuredFeature("playground")).toBe(false);
    expect(cfg.featureEnabled("playground", true)).toBe(false);
  });

  test("playground defaults to omitted", () => {
    // test_playground_defaults_to_omitted
    writeAgentYaml("entrypoint: chat.agent:app");

    const cfg = loadRuntimeAgentConfig();

    expect(cfg.configuredFeature("playground")).toBeNull();
    expect(cfg.featureEnabled("playground", true)).toBe(true);
  });

  test("reads language, framework, and features.durable_workflow from agent.yaml", () => {
    writeAgentYaml(
      [
        "entrypoint: config.agent:app",
        "language: typescript",
        "framework: langgraph",
        "features:",
        "  durable_workflow: true",
      ].join("\n"),
    );

    const cfg = loadRuntimeAgentConfig();

    expect(cfg.language).toBe("typescript");
    expect(cfg.framework).toBe("langgraph");
    expect(cfg.configuredFeature("durable_workflow")).toBe(true);
    expect(cfg.featureEnabled("durable_workflow")).toBe(true);
  });

  test("reads features.guardrails from agent.yaml", () => {
    writeAgentYaml(
      [
        "name: guarded-agent",
        "entrypoint: chat.agent:app",
        "features:",
        "  guardrails: true",
      ].join("\n"),
    );

    const cfg = loadRuntimeAgentConfig();

    expect(cfg.configuredFeature("guardrails")).toBe(true);
    expect(cfg.featureEnabled("guardrails")).toBe(true);
  });

  test("guardrails defaults to omitted", () => {
    writeAgentYaml("entrypoint: chat.agent:app");

    const cfg = loadRuntimeAgentConfig();

    expect(cfg.configuredFeature("guardrails")).toBeNull();
    expect(cfg.featureEnabled("guardrails")).toBe(false);
  });

  test("respects AGENTIC_AGENT_CONFIG_PATH env var override", () => {
    // test_load_runtime_agent_config_uses_explicit_env_path
    const agentDir = realpathSync(mkdtempSync(join(tmpDir, "agents-")));
    const agentYaml = join(agentDir, "agent.yaml");
    writeFileSync(
      agentYaml,
      ["name: support-agent", "entrypoint: support.agent:app"].join("\n"),
      "utf-8",
    );
    vi.stubEnv("AGENTIC_AGENT_CONFIG_PATH", agentYaml);

    const cfg = loadRuntimeAgentConfig();

    expect(cfg.path).toBe(agentYaml);
    expect(cfg.entrypoint).toBe("support.agent:app");
  });
});

// ---------------------------------------------------------------------------
// loadRuntimeAgentConfig — secrets
// ---------------------------------------------------------------------------

describe("loadRuntimeAgentConfig — secrets", () => {
  test("missing secrets block defaults to empty", () => {
    writeAgentYaml(
      ["name: no-secrets-agent", "entrypoint: app:main"].join("\n"),
    );

    const cfg = loadRuntimeAgentConfig();

    expect(cfg.requiredSecrets).toEqual({
      disable_restriction: false,
      aer: [],
      tools: {},
    });
  });

  test("RuntimeAgentConfig normalizes requiredSecrets with the flag omitted", () => {
    // Omission is a supported state, so the exported input shape must stay
    // constructible without it — requiring it would break consumers.
    const cfg = new RuntimeAgentConfig({
      requiredSecrets: { aer: [], tools: {} },
    });

    expect(cfg.requiredSecrets.disable_restriction).toBe(false);
  });

  test("disable_restriction: true parsed and keeps declarations", () => {
    writeAgentYaml(
      [
        "name: unrestricted-agent",
        "entrypoint: app:main",
        "required_secrets:",
        "  disable_restriction: true",
        "  aer:",
        "    - SECRET_A",
      ].join("\n"),
    );

    const cfg = loadRuntimeAgentConfig();

    expect(cfg.requiredSecrets.disable_restriction).toBe(true);
    expect(cfg.requiredSecrets.aer).toEqual(["SECRET_A"]);
  });

  test("disable_restriction: false keeps restriction enabled", () => {
    writeAgentYaml(
      [
        "name: restricted-agent",
        "entrypoint: app:main",
        "required_secrets:",
        "  disable_restriction: false",
      ].join("\n"),
    );

    expect(loadRuntimeAgentConfig().requiredSecrets.disable_restriction).toBe(
      false,
    );
  });

  test("disable_restriction unset keeps restriction enabled", () => {
    writeAgentYaml(
      [
        "name: default-agent",
        "entrypoint: app:main",
        "required_secrets:",
        "  aer:",
        "    - SECRET_A",
      ].join("\n"),
    );

    const cfg = loadRuntimeAgentConfig();

    expect(cfg.requiredSecrets.disable_restriction).toBe(false);
  });

  test("parses full secrets block with aer and tools", () => {
    writeAgentYaml(
      [
        "name: secrets-agent",
        "entrypoint: app:main",
        "required_secrets:",
        "  aer:",
        "    - SECRET_A",
        "    - SECRET_B",
        "  tools:",
        "    my_tool:",
        "      - SECRET_C",
        "    other_tool:",
        "      - SECRET_A",
      ].join("\n"),
    );

    const cfg = loadRuntimeAgentConfig();

    expect(cfg.requiredSecrets.aer).toEqual(["SECRET_A", "SECRET_B"]);
    expect(cfg.requiredSecrets.tools).toEqual({
      my_tool: ["SECRET_C"],
      other_tool: ["SECRET_A"],
    });
  });

  test("parses secrets with only aer", () => {
    writeAgentYaml(
      [
        "name: aer-only-agent",
        "entrypoint: app:main",
        "required_secrets:",
        "  aer:",
        "    - MY_KEY",
      ].join("\n"),
    );

    const cfg = loadRuntimeAgentConfig();

    expect(cfg.requiredSecrets.aer).toEqual(["MY_KEY"]);
    expect(cfg.requiredSecrets.tools).toEqual({});
  });

  test("parses secrets with only tools", () => {
    writeAgentYaml(
      [
        "name: tools-only-agent",
        "entrypoint: app:main",
        "required_secrets:",
        "  tools:",
        "    web_search:",
        "      - API_TOKEN",
      ].join("\n"),
    );

    const cfg = loadRuntimeAgentConfig();

    expect(cfg.requiredSecrets.aer).toEqual([]);
    expect(cfg.requiredSecrets.tools).toEqual({ web_search: ["API_TOKEN"] });
  });
});

// ---------------------------------------------------------------------------
// TenantRuntime — agent.yaml integration
// ---------------------------------------------------------------------------

describe("TenantRuntime — agent.yaml integration", () => {
  test("exposes agent.yaml read from the working directory", () => {
    vi.stubEnv("ENABLE_MEMORY", "false");
    writeAgentYaml(
      [
        "name: runtime-agent",
        "entrypoint: runtime.agent:app",
        "features:",
        "  memory: true",
      ].join("\n"),
    );

    const runtime = new TenantRuntime({ appName: "Runtime Agent" });

    expect(runtime.getAgentConfig().entrypoint).toBe("runtime.agent:app");
    expect(runtime.getAgentConfig().featureEnabled("memory")).toBe(true);
    expect(runtime.memoryWriter).toBeInstanceOf(MemoryWriter);
  });

  test("uses the legacy environment toggle when agent.yaml omits memory", () => {
    vi.stubEnv("ENABLE_MEMORY", "true");
    writeAgentYaml("entrypoint: runtime.agent:app");

    const runtime = new TenantRuntime({ appName: "Runtime Agent" });

    expect(runtime.memoryWriter).toBeInstanceOf(MemoryWriter);
  });
});

// ---------------------------------------------------------------------------
// MCP config — schema validation + ${VAR} interpolation
// ---------------------------------------------------------------------------

describe("MCP config — happy paths", () => {
  test("parses a minimal server with defaults", () => {
    writeAgentYaml(
      [
        "entrypoint: mcp.agent:app",
        "mcp:",
        "  servers:",
        "    github:",
        "      url: https://mcp.example.com/github",
      ].join("\n"),
    );

    const cfg = loadRuntimeAgentConfig();

    const server = cfg.mcp.servers["github"];
    expect(server?.transport).toBe("streamable_http");
    expect(server?.headers).toEqual({});
    expect(server?.auth.type).toBe("none");
    expect(server?.allowed_tools).toBeNull();
    expect(server?.timeout_seconds).toBe(30);
  });

  test("parses bearer_env auth", () => {
    writeAgentYaml(
      [
        "entrypoint: mcp.agent:app",
        "mcp:",
        "  servers:",
        "    github:",
        "      url: https://mcp.example.com/github",
        "      auth:",
        "        type: bearer_env",
        "        token_env: GITHUB_MCP_TOKEN",
      ].join("\n"),
    );

    const cfg = loadRuntimeAgentConfig();

    expect(cfg.mcp.servers["github"]?.auth).toMatchObject({
      type: "bearer_env",
      token_env: "GITHUB_MCP_TOKEN",
    });
  });
});

describe("MCP config — validation errors", () => {
  test("rejects a server name with no alphanumeric characters", () => {
    writeAgentYaml(
      [
        "entrypoint: mcp.agent:app",
        "mcp:",
        "  servers:",
        '    "---":',
        "      url: https://mcp.example.com/github",
      ].join("\n"),
    );

    expect(() => loadRuntimeAgentConfig()).toThrow(
      /mcp server name '---' must contain alphanumeric characters/,
    );
  });

  test("rejects bearer_env without token_env", () => {
    writeAgentYaml(
      [
        "entrypoint: mcp.agent:app",
        "mcp:",
        "  servers:",
        "    github:",
        "      url: https://mcp.example.com/github",
        "      auth:",
        "        type: bearer_env",
      ].join("\n"),
    );

    expect(() => loadRuntimeAgentConfig()).toThrow(/auth\.token_env/);
  });

  test("rejects fields unsupported by the configured auth type", () => {
    writeAgentYaml(
      [
        "entrypoint: mcp.agent:app",
        "mcp:",
        "  servers:",
        "    github:",
        "      url: https://mcp.example.com/github",
        "      auth:",
        "        type: bearer_env",
        "        token_env: GITHUB_MCP_TOKEN",
        "        redirect_uri: http://127.0.0.1:8765/callback",
      ].join("\n"),
    );

    expect(() => loadRuntimeAgentConfig()).toThrow(/does not support fields/);
  });

  test("rejects client_credentials missing required fields", () => {
    writeAgentYaml(
      [
        "entrypoint: mcp.agent:app",
        "mcp:",
        "  servers:",
        "    github:",
        "      url: https://mcp.example.com/github",
        "      auth:",
        "        type: client_credentials",
      ].join("\n"),
    );

    expect(() => loadRuntimeAgentConfig()).toThrow(
      /requires fields: auth\.client_id_env, auth\.client_secret_env/,
    );
  });

  test("rejects a non-https client_credentials token_url", () => {
    writeAgentYaml(
      [
        "entrypoint: mcp.agent:app",
        "mcp:",
        "  servers:",
        "    github:",
        "      url: https://mcp.example.com/github",
        "      auth:",
        "        type: client_credentials",
        "        client_id_env: GITHUB_CLIENT_ID",
        "        client_secret_env: GITHUB_CLIENT_SECRET",
        "        token_url: http://insecure.example.com/token",
      ].join("\n"),
    );

    expect(() => loadRuntimeAgentConfig()).toThrow(/absolute https URL/);
  });

  test("rejects a platform-owned env var name in token_env", () => {
    writeAgentYaml(
      [
        "entrypoint: mcp.agent:app",
        "mcp:",
        "  servers:",
        "    github:",
        "      url: https://mcp.example.com/github",
        "      auth:",
        "        type: bearer_env",
        "        token_env: OPENAI_API_KEY",
      ].join("\n"),
    );

    expect(() => loadRuntimeAgentConfig()).toThrow(
      /different environment variable name/,
    );
  });

  test("rejects a non-absolute http(s) url", () => {
    writeAgentYaml(
      [
        "entrypoint: mcp.agent:app",
        "mcp:",
        "  servers:",
        "    github:",
        "      url: not-a-url",
      ].join("\n"),
    );

    expect(() => loadRuntimeAgentConfig()).toThrow(/absolute http\(s\) URL/);
  });

  test("requires https when auth.type is set", () => {
    writeAgentYaml(
      [
        "entrypoint: mcp.agent:app",
        "mcp:",
        "  servers:",
        "    github:",
        "      url: http://mcp.example.com/github",
        "      auth:",
        "        type: bearer_env",
        "        token_env: GITHUB_MCP_TOKEN",
      ].join("\n"),
    );

    expect(() => loadRuntimeAgentConfig()).toThrow(/must be https/);
  });

  test("rejects credential headers", () => {
    writeAgentYaml(
      [
        "entrypoint: mcp.agent:app",
        "mcp:",
        "  servers:",
        "    github:",
        "      url: https://mcp.example.com/github",
        "      headers:",
        "        Authorization: Bearer abc123",
      ].join("\n"),
    );

    expect(() => loadRuntimeAgentConfig()).toThrow(/credential headers/);
  });
});

describe("MCP config — ${VAR} interpolation", () => {
  test("substitutes ${VAR} in mcp.servers.*.url when envVars is provided", () => {
    writeAgentYaml(
      [
        "entrypoint: mcp.agent:app",
        "mcp:",
        "  servers:",
        "    github:",
        "      url: https://${GITHUB_MCP_HOST}/mcp",
      ].join("\n"),
    );

    const cfg = loadRuntimeAgentConfig(undefined, {
      GITHUB_MCP_HOST: "mcp.example.com",
    });

    expect(cfg.mcp.servers["github"]?.url).toBe("https://mcp.example.com/mcp");
  });

  test("throws when ${VAR} is present but envVars is not passed", () => {
    writeAgentYaml(
      [
        "entrypoint: mcp.agent:app",
        "mcp:",
        "  servers:",
        "    github:",
        "      url: https://${GITHUB_MCP_HOST}/mcp",
      ].join("\n"),
    );

    expect(() => loadRuntimeAgentConfig()).toThrow(
      /environment variable interpolation is not enabled/,
    );
  });

  test("throws listing unset variable names", () => {
    writeAgentYaml(
      [
        "entrypoint: mcp.agent:app",
        "mcp:",
        "  servers:",
        "    github:",
        "      url: https://${GITHUB_MCP_HOST}/mcp",
      ].join("\n"),
    );

    expect(() => loadRuntimeAgentConfig(undefined, {})).toThrow(
      /references unset environment variables: \$\{GITHUB_MCP_HOST\}/,
    );
  });

  test("throws on a malformed ${...} marker", () => {
    writeAgentYaml(
      [
        "entrypoint: mcp.agent:app",
        "mcp:",
        "  servers:",
        "    github:",
        "      url: https://${1bad}/mcp",
      ].join("\n"),
    );

    expect(() => loadRuntimeAgentConfig(undefined, {})).toThrow(
      /malformed environment variable reference/,
    );
  });

  test("does not interpolate outside the allowlisted path", () => {
    writeAgentYaml(
      [
        "entrypoint: mcp.agent:app",
        "mcp:",
        "  servers:",
        "    tableau:",
        "      url: https://tableau.example.com/mcp",
        "      headers:",
        "        x-tenant: '${SOME_VAR}'",
      ].join("\n"),
    );

    const cfg = loadRuntimeAgentConfig(undefined, {});

    expect(cfg.mcp.servers.tableau.headers["x-tenant"]).toBe("${SOME_VAR}");
  });
});

describe("isPlatformEnvVar membership", () => {
  test("marks checkpoint and store DB names as platform-owned", async () => {
    // Mirrors Python test_is_platform_env_var_membership_checks for the
    // CHECKPOINT_DB_NAME deny-list entry (tenant YAML ${} interpolation).
    const { isPlatformEnvVar } = await import("../../src/utils.js");
    expect(isPlatformEnvVar("CHECKPOINT_DB_NAME")).toBe(true);
    expect(isPlatformEnvVar("MDB_AGENTIC_STORE_DB")).toBe(true);
    expect(isPlatformEnvVar("SHUTDOWN_GRACE_PERIOD_MS")).toBe(true);
    expect(isPlatformEnvVar("AGENT_ENGINE_ENVIRONMENT")).toBe(true);
    expect(isPlatformEnvVar("MY_API_TOKEN")).toBe(false);
  });
});
