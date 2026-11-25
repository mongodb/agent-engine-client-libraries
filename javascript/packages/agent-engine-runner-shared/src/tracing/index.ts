export {
  ContentPolicyOTLPSpanExporter,
  PlatformSpanExporter,
  getContentCaptureMode,
} from "./exporters.js";
export {
  setupTracing,
  shutdownTracing,
  forceFlushTracing,
  tracingStatus,
  getCurrentTraceContext,
  getTracer,
  runInstrumentor,
  scrubCredentials,
  type SetupTracingArgs,
} from "./setup.js";
