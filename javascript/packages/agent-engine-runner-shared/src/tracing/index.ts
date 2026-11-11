export {
  ContentPolicyOTLPSpanExporter,
  JSONLSpanExporter,
  MongoDBSpanExporter,
  PlatformSpanExporter,
  getContentCaptureMode,
  getTracePath,
} from "./exporters.js";
export {
  platformTraceExporterEnabled,
  setupTracing,
  shutdownTracing,
  forceFlushTracing,
  attachMongoTracing,
  tracingStatus,
  getCurrentTraceContext,
  getTracer,
  runInstrumentor,
  scrubCredentials,
  type SetupTracingArgs,
} from "./setup.js";
