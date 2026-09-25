export {
  ContentPolicyOTLPSpanExporter,
  JSONLSpanExporter,
  MongoDBSpanExporter,
  getContentCaptureMode,
  getTracePath,
} from "./exporters.js";
export {
  setupTracing,
  shutdownTracing,
  attachMongoTracing,
  tracingStatus,
  getCurrentTraceContext,
  getTracer,
  runInstrumentor,
  scrubCredentials,
  type SetupTracingArgs,
} from "./setup.js";
