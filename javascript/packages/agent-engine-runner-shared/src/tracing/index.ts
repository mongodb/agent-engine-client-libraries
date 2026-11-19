export {
  ContentPolicyOTLPSpanExporter,
  JSONLSpanExporter,
  MongoDBSpanExporter,
  PlatformSpanExporter,
  getContentCaptureMode,
  getTracePath,
} from "./exporters.js";
export {
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
