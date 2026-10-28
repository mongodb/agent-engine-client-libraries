/**
 * In-memory materialization of connector authoring files into a runtime bundle.
 *
 * Each process derives its own view from the same `agent.yaml` configuration:
 * AER registers schemas as ordinary remote tools, Tool Pod registers executor
 * callables. Nothing is written or shared between pods, and no network I/O
 * occurs: catalogs are local files resolved relative to their `tool.yaml`.
 *
 * Mirrors `agent_engine_runner_shared/connectors/materialize.py`.
 */

import { readFileSync, realpathSync, statSync } from "node:fs";
import { basename, dirname, isAbsolute, relative, resolve } from "node:path";

import type { RuntimeAgentConfig } from "../agent_config.js";
import {
  AuthSchema,
  ToolDefinitionsSchema,
  formatZodIssues,
  isRecord,
  loadAuthoringYaml,
  loadToolDefs,
  validateConnectorName,
  type Auth,
  type ToolDefinitions,
} from "./definitions.js";
import {
  BundledTool,
  ConnectorBundle,
  type BundledConnector,
} from "./bundle.js";

export class MaterializeError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "MaterializeError";
  }
}

function loadToolYaml(path: string): Record<string, unknown> {
  let text: string;
  try {
    text = readFileSync(path, "utf8");
  } catch (error) {
    throw new MaterializeError(
      `invalid connector authoring file ${path}: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  let document: unknown;
  try {
    document = loadAuthoringYaml(text);
  } catch (error) {
    throw new MaterializeError(
      `invalid connector authoring file ${path}: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  if (!isRecord(document)) {
    throw new MaterializeError(
      `connector authoring file ${path} must contain a mapping`,
    );
  }
  return document;
}

function requireString(
  document: Record<string, unknown>,
  key: string,
  path: string,
): string {
  const value = document[key];
  if (typeof value !== "string" || value.length === 0) {
    throw new MaterializeError(
      `connector authoring file ${path} requires a non-empty string '${key}'`,
    );
  }
  return value;
}

function isInside(root: string, candidate: string): boolean {
  const delta = relative(root, candidate);
  return (
    delta !== "" &&
    delta !== ".." &&
    !delta.startsWith("../") &&
    !isAbsolute(delta)
  );
}

/** Resolve a reference against `base` and require it to stay inside `root`. */
function resolveContained(
  root: string,
  base: string,
  reference: string,
  field: string,
): string {
  if (isAbsolute(reference) || reference.includes("://")) {
    throw new MaterializeError(
      `${field} must be a local catalog path relative to this file; HTTPS catalog references are not supported in the preview`,
    );
  }
  let resolved: string;
  try {
    resolved = realpathSync(resolve(base, reference));
  } catch {
    throw new MaterializeError(
      `${field} '${reference}' does not exist inside the agent workspace root`,
    );
  }
  if (!isInside(root, resolved)) {
    throw new MaterializeError(
      `${field} '${reference}' resolves outside the agent workspace root`,
    );
  }
  if (!statSync(resolved).isFile()) {
    throw new MaterializeError(`${field} '${reference}' is not a file`);
  }
  return resolved;
}

function operationNames(
  name: string,
  field: string,
  values: unknown[],
): string[] {
  for (const value of values) {
    if (typeof value !== "string" || value.length === 0) {
      throw new MaterializeError(
        `connector '${name}': expose.${field} entries must be non-empty operation names`,
      );
    }
  }
  return values as string[];
}

/** Apply the consumer's fail-closed exposure policy to the catalog. */
function selectOperations(
  name: string,
  expose: Record<string, unknown>,
  definitions: ToolDefinitions,
): string[] {
  const available = definitions.tools.map((tool) => tool.name);
  const allow = expose["allow"];
  const allowAll = expose["allow_all"] === true;
  const disallowRaw = expose["disallow"] ?? [];
  if (!Array.isArray(disallowRaw)) {
    throw new MaterializeError(
      `connector '${name}': expose.disallow must be a list of operation names`,
    );
  }
  const disallow = operationNames(name, "disallow", disallowRaw);

  let selected: Set<string>;
  if (allowAll) {
    if (allow !== undefined && allow !== null) {
      throw new MaterializeError(
        `connector '${name}': expose.allow and expose.allow_all are mutually exclusive`,
      );
    }
    selected = new Set(available);
  } else {
    if (!Array.isArray(allow) || allow.length === 0) {
      throw new MaterializeError(
        `connector '${name}': expose is fail-closed — declare at least one allow rule or set allow_all: true`,
      );
    }
    const allowNames = operationNames(name, "allow", allow);
    const unmatched = [...new Set(allowNames)]
      .filter((item) => !available.includes(item))
      .sort();
    if (unmatched.length > 0) {
      throw new MaterializeError(
        `connector '${name}': allow rules match no catalog operation: ${JSON.stringify(unmatched)}; available operations: ${JSON.stringify([...available].sort())}`,
      );
    }
    selected = new Set(allowNames);
  }

  const unmatchedDisallow = [...new Set(disallow)]
    .filter((item) => !available.includes(item))
    .sort();
  if (unmatchedDisallow.length > 0) {
    throw new MaterializeError(
      `connector '${name}': disallow rules match no catalog operation: ${JSON.stringify(unmatchedDisallow)}`,
    );
  }
  for (const item of disallow) selected.delete(item);

  if (selected.size === 0) {
    throw new MaterializeError(
      `connector '${name}': exposure filters select no operations; an agent must expose at least one connector tool`,
    );
  }
  return available.filter((toolName) => selected.has(toolName));
}

/**
 * Resolve the credential binding and endpoint every selected tool uses.
 *
 * The endpoint is consumer-owned with no catalog fallback: a catalog-chosen
 * destination combined with a consumer-chosen credential would send the
 * tenant secret to whoever published the catalog.
 */
function applyConnectorBindings(
  name: string,
  baseUrl: string,
  auth: Auth | null,
  definitions: ToolDefinitions,
  selected: string[],
): ToolDefinitions {
  const selectedNames = new Set(selected);
  const document: Record<string, unknown> = {
    ...structuredClone(definitions),
    source: {
      ...structuredClone(definitions.source),
      name,
      base_url: baseUrl,
    },
    auth,
    tools: definitions.tools
      .filter((tool) => selectedNames.has(tool.name))
      .map((tool) => structuredClone(tool)),
  };
  const parsed = ToolDefinitionsSchema.safeParse(document);
  if (!parsed.success) {
    throw new MaterializeError(
      `connector '${name}': invalid effective definition: ${formatZodIssues(parsed.error)}`,
    );
  }
  return parsed.data;
}

/** Cross-reference one `tool.yaml` with its catalog and apply its bindings. */
function materializeConnector(
  toolYamlPath: string,
  root: string,
): ToolDefinitions {
  const document = loadToolYaml(toolYamlPath);
  const name = requireString(document, "name", toolYamlPath);
  try {
    validateConnectorName(name);
  } catch (error) {
    throw new MaterializeError(
      `connector '${name}': ${error instanceof Error ? error.message : String(error)}`,
    );
  }

  const reference = requireString(document, "tool_defs", toolYamlPath);
  const catalogPath = resolveContained(
    root,
    dirname(toolYamlPath),
    reference,
    "tool_defs",
  );
  let definitions: ToolDefinitions;
  try {
    definitions = loadToolDefs(catalogPath);
  } catch (error) {
    throw new MaterializeError(
      `connector '${name}': invalid catalog ${basename(catalogPath)}: ${error instanceof Error ? error.message : String(error)}`,
    );
  }

  const sourceValue = document["source"] ?? {};
  if (!isRecord(sourceValue)) {
    throw new MaterializeError(`connector '${name}': source must be a mapping`);
  }
  const baseUrl = sourceValue["base_url"];
  if (typeof baseUrl !== "string" || baseUrl.length === 0) {
    throw new MaterializeError(
      `connector '${name}': source.base_url is required in tool.yaml (the catalog's base URL is never used as a fallback)`,
    );
  }

  let auth: Auth | null = null;
  const authValue = document["auth"];
  if (authValue !== undefined && authValue !== null) {
    if (!isRecord(authValue)) {
      throw new MaterializeError(`connector '${name}': auth must be a mapping`);
    }
    const parsed = AuthSchema.safeParse(authValue);
    if (!parsed.success) {
      throw new MaterializeError(
        `connector '${name}': invalid auth: ${formatZodIssues(parsed.error)}`,
      );
    }
    auth = parsed.data.type === "none" ? null : parsed.data;
  }

  const exposeValue = document["expose"] ?? {};
  if (!isRecord(exposeValue)) {
    throw new MaterializeError(`connector '${name}': expose must be a mapping`);
  }
  const selected = selectOperations(name, exposeValue, definitions);
  return applyConnectorBindings(name, baseUrl, auth, definitions, selected);
}

/**
 * Materialize connector authoring files into an in-memory runtime bundle.
 *
 * `root` is the containment boundary (the `agent.yaml` directory): every local
 * path inside each `tool.yaml` must resolve inside it. Pure function of the
 * input files — no filesystem writes, no network.
 */
export function materializeConnectors(
  connectorPaths: readonly string[],
  options: { root: string },
): ConnectorBundle {
  if (connectorPaths.length === 0) {
    throw new MaterializeError("no connector authoring files configured");
  }
  let root: string;
  try {
    root = realpathSync(options.root);
  } catch {
    throw new MaterializeError(`agent root ${options.root} does not exist`);
  }
  if (!statSync(root).isDirectory()) {
    throw new MaterializeError(`agent root ${options.root} does not exist`);
  }

  const toolYamlPaths = connectorPaths.map((provided) => {
    if (isAbsolute(provided)) {
      try {
        const resolved = realpathSync(provided);
        if (!isInside(root, resolved)) throw new Error("outside");
        return resolved;
      } catch {
        throw new MaterializeError(
          `connector authoring file ${provided} does not exist inside ${root}`,
        );
      }
    }
    return resolveContained(root, root, provided, "connectors source");
  });

  const connectors: BundledConnector[] = [];
  const tools: BundledTool[] = [];
  const seenConnectors = new Set<string>();
  const seenTools = new Map<string, string>();
  for (const toolYamlPath of toolYamlPaths) {
    const definitions = materializeConnector(toolYamlPath, root);
    const name = definitions.source.name;
    if (seenConnectors.has(name)) {
      throw new MaterializeError(`duplicate connector name '${name}'`);
    }
    seenConnectors.add(name);

    const connector: BundledConnector = { name, definitions };
    for (const operation of definitions.tools) {
      const previous = seenTools.get(operation.name);
      if (previous !== undefined) {
        throw new MaterializeError(
          `connector tool name collision between '${previous}' and '${name}': ${operation.name}`,
        );
      }
      seenTools.set(operation.name, name);
      tools.push(new BundledTool(connector, operation));
    }
    connectors.push(connector);
  }

  return new ConnectorBundle(connectors, tools);
}

/**
 * Resolve the connector bundle for this process.
 *
 * The `agent.yaml` `connectors` entries are materialized in memory from local
 * files. With none configured this is an immediate no-op without file or
 * network I/O, matching the no-connector contract.
 */
export function resolveRuntimeBundle(
  agentConfig: RuntimeAgentConfig,
): ConnectorBundle {
  const references = agentConfig.connectors;
  if (!Array.isArray(references) || references.length === 0) {
    return new ConnectorBundle();
  }
  if (agentConfig.path === null) {
    throw new MaterializeError(
      "connector configuration requires a located agent.yaml to resolve against",
    );
  }
  let root: string;
  try {
    root = realpathSync(dirname(agentConfig.path));
  } catch {
    throw new MaterializeError(
      "connectors source must be a path inside the agent.yaml directory",
    );
  }
  for (const entry of references) {
    try {
      const resolved = realpathSync(resolve(root, entry.source));
      if (!isInside(root, resolved)) throw new Error("outside");
    } catch {
      throw new MaterializeError(
        "connectors source must be a path inside the agent.yaml directory",
      );
    }
  }
  return materializeConnectors(
    references.map((entry) => entry.source),
    { root },
  );
}
