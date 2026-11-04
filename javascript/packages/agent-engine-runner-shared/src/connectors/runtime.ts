/**
 * Tool Pod callables backed by one connector bundle.
 *
 * A runtime owns one executor per connector for the life of the Tool Pod
 * process. Registration is derived from the same bundle AER uses, so the two
 * processes cannot disagree about schemas, metadata, or names.
 *
 * Mirrors `agent_engine_runner_shared/connectors/runtime.py`.
 */

import { AsyncLocalStorage } from "node:async_hooks";

import type { RuntimeAgentConfig } from "../agent_config.js";
import { getCurrentAuthorization } from "../context.js";
import { ConnectorExecutor } from "./executor.js";
import type {
  BundledTool,
  ConnectorBundle,
  ConnectorRegistrationMetadata,
} from "./bundle.js";
import { resolveRuntimeBundle } from "./materialize.js";

const currentConnectorEnvironment = new AsyncLocalStorage<
  Readonly<Record<string, string>>
>();

/**
 * Bind the tenant environment captured for one connector invocation.
 *
 * The Tool Pod may run concurrent calls whose metadata secrets differ. Keep
 * the captured environment request-local instead of reading the live process
 * environment when the executor eventually builds the request.
 */
export function withConnectorEnvironment<T>(
  env: Readonly<Record<string, string>>,
  fn: () => T,
): T {
  return currentConnectorEnvironment.run(env, fn);
}

/** One connector tool and the Tool Pod callable bound to its executor. */
export class ConnectorToolRegistration {
  constructor(
    readonly tool: BundledTool,
    readonly callable: (arguments_: Record<string, unknown>) => Promise<string>,
  ) {}

  get metadata(): ConnectorRegistrationMetadata {
    return this.tool.registrationMetadata;
  }
}

export class ConnectorToolRuntime {
  private readonly executors = new Map<string, ConnectorExecutor>();
  private readonly _registrations: ConnectorToolRegistration[];
  private closed = false;

  constructor(bundle: ConnectorBundle) {
    try {
      for (const connector of bundle.connectors) {
        this.executors.set(
          connector.name,
          new ConnectorExecutor(connector.definitions),
        );
      }
    } catch (error) {
      for (const executor of this.executors.values()) {
        void executor.close();
      }
      throw error;
    }
    this._registrations = bundle.tools.map(
      (tool) =>
        new ConnectorToolRegistration(
          tool,
          this._makeCallable(
            this.executors.get(tool.connector.name),
            tool.operation.name,
          ),
        ),
    );
  }

  get registrations(): readonly ConnectorToolRegistration[] {
    return this._registrations;
  }

  private _makeCallable(
    executor: ConnectorExecutor | undefined,
    toolName: string,
  ): (arguments_: Record<string, unknown>) => Promise<string> {
    if (executor === undefined) {
      throw new Error(
        `connector executor is missing for tool ${JSON.stringify(toolName)}`,
      );
    }
    return async (arguments_) => {
      const env = currentConnectorEnvironment.getStore();
      if (env === undefined) {
        throw new Error(
          `connector invocation environment is unavailable for tool ${JSON.stringify(toolName)}`,
        );
      }
      return executor.execute(toolName, arguments_, {
        env,
        authorization: getCurrentAuthorization(),
      });
    };
  }

  /** Release every connector's pooled transport once. */
  async close(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    await Promise.all(
      [...this.executors.values()].map((executor) => executor.close()),
    );
  }
}

/** The runtime surface the connector registry reads from ToolServer. */
export interface ConnectorRuntimeTarget {
  readonly tools: Record<
    string,
    (arguments_: Record<string, unknown>) => unknown
  >;
  readonly toolDefinitions: Record<string, Record<string, unknown>>;
  getAgentConfig(): RuntimeAgentConfig;
}

/**
 * Materialize, collision-check, and install connector tools on a runtime.
 *
 * Callers supply the names already reserved by their mode (built-ins, MCP
 * tools, author tools). Returns `null` when the agent configures no
 * connectors; otherwise the caller owns the returned runtime and must close it
 * after the invocation lifecycle.
 */
export function registerConnectorTools(
  target: ConnectorRuntimeTarget,
  reservedNames: Iterable<string> = [],
): ConnectorToolRuntime | null {
  const bundle = resolveRuntimeBundle(target.getAgentConfig());
  if (bundle.tools.length === 0) return null;
  bundle.requireAvailableNames(
    new Set([...Object.keys(target.tools), ...reservedNames]),
  );
  const runtime = new ConnectorToolRuntime(bundle);
  for (const registration of runtime.registrations) {
    const name = registration.tool.operation.name;
    target.tools[name] = registration.callable;
    target.toolDefinitions[name] = { ...registration.metadata };
  }
  return runtime;
}
