import {
  mkdirSync,
  mkdtempSync,
  readFileSync,
  realpathSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { RuntimeAgentConfig } from "../../../src/agent_config.js";
import {
  ConnectorBundle,
  MaterializeError,
  materializeConnectors,
  resolveRuntimeBundle,
} from "../../../src/connectors/index.js";

const FIXTURE = fileURLToPath(
  new URL("./fixtures/jira.tool_defs.yaml", import.meta.url),
);

const TOOL_YAML = `
name: jira
tool_defs: jira.tool_defs.yaml
source:
  type: openapi
  base_url: https://jira.corp/rest
expose:
  allow:
    - jiradc_getComments
    - jiradc_addComment
auth:
  type: bearer
  env: JIRA_TOKEN
`;

function temporaryRoot(): string {
  return realpathSync(mkdtempSync(join(tmpdir(), "connector-materialize-")));
}

function writeAuthoring(
  root: string,
  toolYaml: string = TOOL_YAML,
  catalog: string | null = null,
  folder = "connectors/jira",
): string {
  const connectorDir = join(root, folder);
  mkdirSync(connectorDir, { recursive: true });
  writeFileSync(
    join(connectorDir, "jira.tool_defs.yaml"),
    catalog ?? readCatalog(),
    "utf8",
  );
  const path = join(connectorDir, "tool.yaml");
  writeFileSync(path, toolYaml, "utf8");
  return path;
}

function readCatalog(): string {
  return readFileSync(FIXTURE, "utf8");
}

describe("connector materialization", () => {
  it("materializes the selected operations and overlays the endpoint", () => {
    const root = temporaryRoot();
    try {
      writeAuthoring(root);
      const bundle = materializeConnectors(["connectors/jira/tool.yaml"], {
        root,
      });
      expect(bundle.connectors.map((connector) => connector.name)).toEqual([
        "jira",
      ]);
      expect(bundle.tools.map((tool) => tool.operation.name)).toEqual([
        "jiradc_getComments",
        "jiradc_addComment",
      ]);
      expect(bundle.tools[0]?.connector.definitions.source.base_url).toBe(
        "https://jira.corp/rest",
      );
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("selects an allow subset", () => {
    const root = temporaryRoot();
    try {
      writeAuthoring(root, TOOL_YAML.replace("    - jiradc_addComment\n", ""));
      const bundle = materializeConnectors(["connectors/jira/tool.yaml"], {
        root,
      });
      expect(bundle.tools.map((tool) => tool.operation.name)).toEqual([
        "jiradc_getComments",
      ]);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("subtracts disallow after allow_all", () => {
    const root = temporaryRoot();
    try {
      writeAuthoring(
        root,
        TOOL_YAML.replace(
          "expose:\n  allow:\n    - jiradc_getComments\n    - jiradc_addComment\n",
          "expose:\n  allow_all: true\n  disallow:\n    - jiradc_addComment\n",
        ),
      );
      const bundle = materializeConnectors(["connectors/jira/tool.yaml"], {
        root,
      });
      expect(bundle.tools.map((tool) => tool.operation.name)).toEqual([
        "jiradc_getComments",
      ]);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("fails closed without allow rules", () => {
    const root = temporaryRoot();
    try {
      writeAuthoring(
        root,
        TOOL_YAML.replace(
          "expose:\n  allow:\n    - jiradc_getComments\n    - jiradc_addComment\n",
          "expose: {}\n",
        ),
      );
      expect(() =>
        materializeConnectors(["connectors/jira/tool.yaml"], { root }),
      ).toThrow("fail-closed");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("rejects allow and allow_all together", () => {
    const root = temporaryRoot();
    try {
      writeAuthoring(
        root,
        TOOL_YAML.replace("  allow:\n", "  allow_all: true\n  allow:\n"),
      );
      expect(() =>
        materializeConnectors(["connectors/jira/tool.yaml"], { root }),
      ).toThrow("mutually exclusive");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("names unmatched allow rules", () => {
    const root = temporaryRoot();
    try {
      writeAuthoring(
        root,
        TOOL_YAML.replace("- jiradc_addComment", "- missing_operation"),
      );
      expect(() =>
        materializeConnectors(["connectors/jira/tool.yaml"], { root }),
      ).toThrow("allow rules match no catalog operation");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("applies delegated OAuth metadata outside the model schema", () => {
    const root = temporaryRoot();
    try {
      writeAuthoring(
        root,
        TOOL_YAML.replace(
          "  type: bearer\n  env: JIRA_TOKEN\n",
          "  type: oauth\n  provider_type: custom-github\n  scopes: [repo]\n",
        ),
      );
      const bundle = materializeConnectors(["connectors/jira/tool.yaml"], {
        root,
      });
      for (const tool of bundle.tools) {
        expect(tool.registrationMetadata.provider_type).toBe("custom-github");
        expect(tool.registrationMetadata.scopes).toEqual(["repo"]);
        expect(tool.schema).not.toHaveProperty("provider_type");
      }
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("treats auth none as no credential binding", () => {
    const root = temporaryRoot();
    try {
      writeAuthoring(
        root,
        TOOL_YAML.replace(
          "  type: bearer\n  env: JIRA_TOKEN\n",
          "  type: none\n",
        ),
      );
      const bundle = materializeConnectors(["connectors/jira/tool.yaml"], {
        root,
      });
      expect(bundle.connectors[0]?.definitions.auth).toBeNull();
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("requires a consumer-owned base URL", () => {
    const root = temporaryRoot();
    try {
      writeAuthoring(
        root,
        TOOL_YAML.replace("  base_url: https://jira.corp/rest\n", ""),
      );
      expect(() =>
        materializeConnectors(["connectors/jira/tool.yaml"], { root }),
      ).toThrow("source.base_url is required");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("rejects malformed disallow shapes", () => {
    const root = temporaryRoot();
    try {
      const base = TOOL_YAML.replace(
        "expose:\n  allow:\n    - jiradc_getComments\n    - jiradc_addComment\n" +
          "auth:\n  type: bearer\n  env: JIRA_TOKEN\n",
        "expose:\n  allow_all: true\n  disallow: ",
      );
      for (const value of ["false", "{}", "0", "jiradc_addComment"]) {
        writeAuthoring(root, `${base}${value}\n`);
        expect(() =>
          materializeConnectors(["connectors/jira/tool.yaml"], { root }),
        ).toThrow("expose.disallow must be a list");
      }
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("rejects non-string exposure entries", () => {
    const root = temporaryRoot();
    try {
      writeAuthoring(root, TOOL_YAML.replace("- jiradc_addComment", "- 1"));
      expect(() =>
        materializeConnectors(["connectors/jira/tool.yaml"], { root }),
      ).toThrow("entries must be non-empty operation names");
      writeAuthoring(
        root,
        TOOL_YAML.replace(
          "expose:\n  allow:\n    - jiradc_getComments\n    - jiradc_addComment\n",
          "expose:\n  allow_all: true\n  disallow:\n    - {}\n",
        ),
      );
      expect(() =>
        materializeConnectors(["connectors/jira/tool.yaml"], { root }),
      ).toThrow("entries must be non-empty operation names");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("rejects unmatched disallow rules and empty selections", () => {
    const root = temporaryRoot();
    try {
      writeAuthoring(
        root,
        TOOL_YAML.replace(
          "expose:\n  allow:\n    - jiradc_getComments\n    - jiradc_addComment\n",
          "expose:\n  allow_all: true\n  disallow: [missing]\n",
        ),
      );
      expect(() =>
        materializeConnectors(["connectors/jira/tool.yaml"], { root }),
      ).toThrow("disallow rules match no catalog operation");
      writeAuthoring(
        root,
        TOOL_YAML.replace(
          "expose:\n  allow:\n    - jiradc_getComments\n    - jiradc_addComment\n",
          "expose:\n  allow: [jiradc_getComments]\n  disallow: [jiradc_getComments]\n",
        ),
      );
      expect(() =>
        materializeConnectors(["connectors/jira/tool.yaml"], { root }),
      ).toThrow("select no operations");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("rejects connector-name traversal", () => {
    const root = temporaryRoot();
    try {
      writeAuthoring(root, TOOL_YAML.replace("name: jira", "name: ../evil"));
      expect(() =>
        materializeConnectors(["connectors/jira/tool.yaml"], { root }),
      ).toThrow("connector name must match");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("rejects malformed authoring auth", () => {
    const root = temporaryRoot();
    try {
      for (const auth of [
        "auth:\n  env: JIRA_TOKEN\n",
        "auth:\n  type: none\n  env: JIRA_TOKEN\n",
      ]) {
        writeAuthoring(
          root,
          TOOL_YAML.replace("auth:\n  type: bearer\n  env: JIRA_TOKEN\n", auth),
        );
        expect(() =>
          materializeConnectors(["connectors/jira/tool.yaml"], { root }),
        ).toThrow("invalid auth");
      }
      writeAuthoring(
        root,
        TOOL_YAML.replace(
          "auth:\n  type: bearer\n  env: JIRA_TOKEN\n",
          "auth: []\n",
        ),
      );
      expect(() =>
        materializeConnectors(["connectors/jira/tool.yaml"], { root }),
      ).toThrow("auth must be a mapping");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("rejects HTTPS and absolute catalog references", () => {
    const root = temporaryRoot();
    try {
      writeAuthoring(
        root,
        TOOL_YAML.replace(
          "tool_defs: jira.tool_defs.yaml",
          "tool_defs: https://example.test/catalog.yaml",
        ),
      );
      expect(() =>
        materializeConnectors(["connectors/jira/tool.yaml"], { root }),
      ).toThrow("HTTPS catalog references are not supported");
      writeAuthoring(
        root,
        TOOL_YAML.replace(
          "tool_defs: jira.tool_defs.yaml",
          "tool_defs: /tmp/catalog.yaml",
        ),
      );
      expect(() =>
        materializeConnectors(["connectors/jira/tool.yaml"], { root }),
      ).toThrow("local catalog path relative to this file");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("rejects a symlink that escapes the agent root", () => {
    const root = temporaryRoot();
    const outside = temporaryRoot();
    try {
      writeAuthoring(root);
      writeFileSync(join(outside, "outside.yaml"), readCatalog(), "utf8");
      symlinkSync(
        join(outside, "outside.yaml"),
        join(root, "connectors/jira/linked.yaml"),
      );
      writeAuthoring(
        root,
        TOOL_YAML.replace(
          "tool_defs: jira.tool_defs.yaml",
          "tool_defs: linked.yaml",
        ),
      );
      expect(() =>
        materializeConnectors(["connectors/jira/tool.yaml"], { root }),
      ).toThrow("resolves outside the agent workspace root");
    } finally {
      rmSync(root, { recursive: true, force: true });
      rmSync(outside, { recursive: true, force: true });
    }
  });

  it("allows in-root directories whose names begin with two dots", () => {
    const root = temporaryRoot();
    try {
      writeAuthoring(root, TOOL_YAML, null, "..catalog");
      const bundle = materializeConnectors(["..catalog/tool.yaml"], { root });
      expect(bundle.tools.map((tool) => tool.operation.name)).toEqual([
        "jiradc_getComments",
        "jiradc_addComment",
      ]);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("resolves YAML merge keys in authoring files", () => {
    const root = temporaryRoot();
    try {
      writeAuthoring(
        root,
        TOOL_YAML.replace(
          "auth:\n  type: bearer\n  env: JIRA_TOKEN\n",
          "shared_auth: &shared_auth\n  type: bearer\n  env: JIRA_TOKEN\nauth:\n  <<: *shared_auth\n",
        ),
      );
      const bundle = materializeConnectors(["connectors/jira/tool.yaml"], {
        root,
      });
      expect(bundle.connectors[0]?.definitions.auth).toMatchObject({
        type: "bearer",
        env: "JIRA_TOKEN",
      });
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("rejects duplicate connector names before tool registration", () => {
    const root = temporaryRoot();
    try {
      for (const [folder, toolName] of [
        ["connectors/a", "a_tool"],
        ["connectors/b", "b_tool"],
      ] as const) {
        const connectorDir = join(root, folder);
        mkdirSync(connectorDir, { recursive: true });
        writeFileSync(
          join(connectorDir, "catalog.yaml"),
          `source:\n  name: ${toolName}_source\n  base_url: https://jira.example/rest\ntools:\n  - name: ${toolName}\n    description: ${toolName}\n    method: GET\n    path: /api/2/issue\n    params: {}\n    inputSchema:\n      type: object\n      properties: {}\n`,
          "utf8",
        );
        writeFileSync(
          join(connectorDir, "tool.yaml"),
          `name: shared\ntool_defs: catalog.yaml\nsource:\n  base_url: https://jira.example/rest\nexpose:\n  allow_all: true\n`,
          "utf8",
        );
      }
      expect(() =>
        materializeConnectors(
          ["connectors/a/tool.yaml", "connectors/b/tool.yaml"],
          { root },
        ),
      ).toThrow("duplicate connector name 'shared'");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("rejects duplicate connector names and cross-connector tool collisions", () => {
    const root = temporaryRoot();
    try {
      writeAuthoring(root);
      writeAuthoring(
        root,
        TOOL_YAML.replace("name: jira", "name: jira-two"),
        readCatalog().replace("name: jiradc", "name: other"),
        "connectors/second",
      );
      expect(() =>
        materializeConnectors(
          ["connectors/jira/tool.yaml", "connectors/second/tool.yaml"],
          { root },
        ),
      ).toThrow("connector tool name collision");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});

describe("resolveRuntimeBundle", () => {
  it("is an empty no-op without connectors", () => {
    expect(resolveRuntimeBundle(new RuntimeAgentConfig())).toEqual(
      new ConnectorBundle(),
    );
  });

  it("materializes agent.yaml connectors", () => {
    const root = temporaryRoot();
    try {
      writeAuthoring(root);
      writeFileSync(
        join(root, "agent.yaml"),
        "name: connector-agent\nconnectors:\n  - source: connectors/jira/tool.yaml\n",
        "utf8",
      );
      const config = new RuntimeAgentConfig({
        path: join(root, "agent.yaml"),
        connectors: [{ source: "connectors/jira/tool.yaml" }],
      });
      const bundle = resolveRuntimeBundle(config);
      expect(bundle.tools.map((tool) => tool.operation.name)).toEqual([
        "jiradc_getComments",
        "jiradc_addComment",
      ]);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("rejects sources outside the agent.yaml directory", () => {
    const root = temporaryRoot();
    const outside = temporaryRoot();
    try {
      writeFileSync(join(outside, "tool.yaml"), TOOL_YAML, "utf8");
      const config = new RuntimeAgentConfig({
        path: join(root, "agent.yaml"),
        connectors: [{ source: join("..", basename(outside), "tool.yaml") }],
      });
      expect(() => resolveRuntimeBundle(config)).toThrow(
        "inside the agent.yaml directory",
      );
    } finally {
      rmSync(root, { recursive: true, force: true });
      rmSync(outside, { recursive: true, force: true });
    }
  });

  it("requires a located agent.yaml", () => {
    const config = new RuntimeAgentConfig({
      connectors: [{ source: "connectors/jira/tool.yaml" }],
    });
    expect(() => resolveRuntimeBundle(config)).toThrow("located agent.yaml");
  });
});

describe("materialize error type", () => {
  it("uses a named error for authoring failures", () => {
    const root = temporaryRoot();
    try {
      expect(() =>
        materializeConnectors(["missing/tool.yaml"], { root }),
      ).toThrow(MaterializeError);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});
