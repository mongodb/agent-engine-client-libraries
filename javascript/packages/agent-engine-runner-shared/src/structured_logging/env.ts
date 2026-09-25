/**
 * Install-time snapshot of the pod/runner environment that decorates every
 * structured-logging record (service, component, tenant/project, pod name).
 */

import { SERVICE_BY_MODE } from "./constants.js";

export function envOrNull(name: string): string | null {
  const v = process.env[name];
  return v && v.length > 0 ? v : null;
}

export function serviceFor(mode: string | null): string {
  if (!mode) return "agent-execution-runtime";
  return SERVICE_BY_MODE[mode.toLowerCase()] ?? mode.toLowerCase();
}

export function resolveLevel(level?: string | null): string {
  return (level ?? envOrNull("LOG_LEVEL") ?? "info").toLowerCase();
}

export interface FormatterEnv {
  service: string;
  component: string;
  tenantId: string | null;
  projectId: string | null;
  workspaceIdEnv: string | null;
  bootId: string | null;
  podName: string | null;
}

export function snapshotEnv(mode: string | null): FormatterEnv {
  // Env vars (RUNNER_MODE, ORG_ID, PROJECT_ID, WORKSPACE_ID, AGENTIC_BOOT_ID,
  // POD_NAME) are snapshotted here at install time, not re-read per record —
  // every log event goes through the layout and re-resolving env on each call
  // would noticeably load the hot path. Callers must ensure the pod env is
  // set before `installStructuredLogging()` runs; if the install runs before
  // the entrypoint stamps env, the layout caches stale (or missing) values for
  // the process lifetime. The `mode` arg lets callers override
  // RUNNER_MODE explicitly — useful when the pod env happens to be unset
  // but the caller knows its mode (e.g. `setupLogging` passes its `mode=`
  // arg through).
  const resolvedMode = mode ?? envOrNull("RUNNER_MODE");
  return {
    service: serviceFor(resolvedMode),
    component: (resolvedMode ?? "agent").toLowerCase(),
    tenantId: envOrNull("ORG_ID"),
    projectId: envOrNull("PROJECT_ID"),
    workspaceIdEnv: envOrNull("WORKSPACE_ID"),
    bootId: envOrNull("AGENTIC_BOOT_ID"),
    podName: envOrNull("POD_NAME") ?? envOrNull("HOSTNAME"),
  };
}
