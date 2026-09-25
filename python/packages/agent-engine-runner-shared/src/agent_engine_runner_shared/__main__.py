"""Generic platform entrypoint for the runner-shared image.

Dispatches via RUNNER_MODE (aer | tool | tool_function) through
TenantRuntime.register_and_run(). Tool and function modes do not need a graph
builder; aer mode does, so user agents provide their own __main__.py for that
case (App.run() enforces the builder check there).
"""

from agent_engine_runner_shared.runtime import TenantRuntime

if __name__ == "__main__":
    TenantRuntime().register_and_run()
