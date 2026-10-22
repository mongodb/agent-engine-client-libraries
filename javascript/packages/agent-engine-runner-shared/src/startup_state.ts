const startupCompleteCallbacks = new Set<() => void>();

export function onStartupComplete(callback: () => void): () => void {
  startupCompleteCallbacks.add(callback);
  return () => startupCompleteCallbacks.delete(callback);
}

export function markStartupComplete(): void {
  for (const callback of startupCompleteCallbacks) callback();
  startupCompleteCallbacks.clear();
}
