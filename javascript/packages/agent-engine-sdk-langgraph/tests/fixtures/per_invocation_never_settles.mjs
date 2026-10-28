/**
 * Child process for the never-settling per-execution build case.
 *
 * Runs the real /execute route against a real opted-in App whose builder never
 * settles. Nothing about the fatal path is mocked here, so the process is
 * expected to exit 1 through fatalWithTermination and to have written the
 * termination message the parent asserts on. Running it out of process is what
 * makes the exit code and the message observable without replacing
 * process.exit, which in turn is what kept the in-process version from
 * producing a trustworthy signal.
 *
 * The parent supplies RUNNER_ABANDONED_BUILD_GRACE, RUNNER_EXECUTION_TIMEOUT,
 * OE_URL, RUNNER_MODE and TERMINATION_LOG_PATH.
 */

import Fastify from "fastify";

import {
  AERServer,
  setTerminationLogPathForTest,
} from "../../../agent-engine-runner-shared/dist/index.js";
import { App } from "../../dist/index.js";

setTerminationLogPathForTest(process.env["TERMINATION_LOG_PATH"]);

const app = new App({ appName: "test-agent" });
app.entrypoint(async () => new Promise(() => {}), {
  loadStrategy: "per-execution",
});

const server = Object.create(AERServer.prototype);
server.runtime = {
  appName: "test-agent",
  appVersion: "1.0.0",
  memoryEnabled: false,
  graphBuilder: app,
  tools: {},
  toolDefinitions: {},
  getAgentConfig: () => ({ featureEnabled: () => false, mcp: undefined }),
  getMongodbUri: () => null,
  getCurrentUserId: () => null,
  getAgent: (opts) => app.getAgent(opts),
  buildGraph: (input, ctx) => app.buildGraph(input, ctx),
  usesPerInvocationBuild: () => app.usesPerInvocationBuild(),
};
// Class fields, which Object.create does not install.
server.chunkSeq = new Map();
server.ownerCallbackUrl = new Map();
server.abandonedBuilds = new Map();
server.sendStreamChunk = async () => {};
server.reportCallback = async () => {};
server.handleExecute = (request) => server.doHandleExecute(request);

const fastify = Fastify();
server.registerRoutes(fastify);
await fastify.ready();

// The grace timer is unref'd in production, so hold the loop open until the
// fatal path exits the process.
setTimeout(() => {}, 60_000);

fastify
  .inject({
    method: "POST",
    url: "/execute",
    payload: {
      execution_id: "exec-never",
      message: "hi",
      platform_api_url: "http://oe.test:8000",
    },
  })
  .catch(() => {});
