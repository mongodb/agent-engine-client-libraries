/**
 * Global test setup — mirrors runner-shared/tests/conftest.py.
 *
 * Defaults RUNNER_MODE to 'aer' for all tests unless already set in the
 * environment. Individual tests that need a different mode can override
 * with vi.stubEnv('RUNNER_MODE', '...') which auto-restores after the test.
 */
if (!process.env["RUNNER_MODE"]) {
  process.env["RUNNER_MODE"] = "aer";
}
