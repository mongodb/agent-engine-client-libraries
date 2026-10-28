/**
 * Validated runtime views shared by AER and Tool Pod connector registration.
 *
 * The authoring materializer builds these from `agent.yaml`, `tool.yaml`, and
 * the local compiled catalog. AER consumes each tool's schema and registration
 * metadata; Tool Pod consumes the same views to bind executor callables.
 * Nothing here reads files or performs I/O.
 *
 * Mirrors `agent_engine_runner_shared/connectors/bundle.py`.
 */

import type { Operation, ToolDefinitions } from "./definitions.js";

/** Connector execution budget exposed as ordinary tool metadata. */
export const DEFAULT_CONNECTOR_TIMEOUT_SECONDS = 30;

export class ConnectorBundleError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ConnectorBundleError";
  }
}

export interface BundledConnector {
  readonly name: string;
  readonly definitions: ToolDefinitions;
}

export interface ConnectorToolSchema {
  readonly name: string;
  readonly description: string;
  readonly inputSchema: Record<string, unknown>;
}

export interface ConnectorRegistrationMetadata {
  readonly name: string;
  readonly description: string;
  readonly is_local: false;
  readonly provider_type: string | null;
  readonly scopes: string[];
  readonly network: string[];
  readonly timeout_seconds: number;
  readonly redact_fields: string[];
  readonly connector: string;
}

export class BundledTool {
  constructor(
    readonly connector: BundledConnector,
    readonly operation: Operation,
  ) {}

  /** Return a detached SDK-visible schema. */
  get schema(): ConnectorToolSchema {
    return {
      name: this.operation.name,
      description: this.operation.description,
      inputSchema: structuredClone(this.operation.inputSchema),
    };
  }

  /** Return the single outbound host declared by this connector. */
  get network(): string[] {
    const hostname = new URL(this.connector.definitions.source.base_url)
      .hostname;
    return [hostname.replace(/^\[|\]$/g, "")];
  }

  /** Return detached AER/Tool Pod policy, with delegated binding outside the schema. */
  get registrationMetadata(): ConnectorRegistrationMetadata {
    const auth = this.connector.definitions.auth;
    const delegated = auth && auth.type === "oauth" ? auth : null;
    return {
      name: this.operation.name,
      description:
        this.operation.description.trim() ||
        `REST connector tool ${this.operation.name}`,
      is_local: false,
      provider_type: delegated?.provider_type ?? null,
      scopes: delegated ? [...delegated.scopes] : [],
      network: this.network,
      timeout_seconds: DEFAULT_CONNECTOR_TIMEOUT_SECONDS,
      redact_fields: [],
      connector: this.connector.name,
    };
  }
}

export class ConnectorBundle {
  constructor(
    readonly connectors: readonly BundledConnector[] = [],
    readonly tools: readonly BundledTool[] = [],
  ) {}

  /** Reject all name collisions before a caller mutates its registry. */
  requireAvailableNames(registeredNames: ReadonlySet<string>): void {
    const collisions = [
      ...new Set(
        this.tools
          .map((tool) => tool.operation.name)
          .filter((name) => registeredNames.has(name)),
      ),
    ].sort();
    if (collisions.length > 0) {
      throw new ConnectorBundleError(
        `connector tool names collide with registered or reserved tools: ${collisions.join(", ")}`,
      );
    }
  }
}
