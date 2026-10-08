#!/usr/bin/env bash
# Run the publishable SDK package tests from the packages tree root.
#
# Works from this packages tree (the snapshot destination, or
# client-libraries/packages in the source repo). Does not run pact
# verification or the OpenAPI corpus — those need files that do not ship here.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_ROOT="$ROOT/python"
JAVASCRIPT_ROOT="$ROOT/javascript"

log() { printf '%s\n' "$*"; }

run_python_tests() {
  log "Python: ruff"
  (
    cd "$PYTHON_ROOT"
    uv run --with ruff==0.12.0 ruff check .
    uv run --with ruff==0.12.0 ruff format --check .
  )

  log "Python: agent-engine-sdk"
  (
    cd "$PYTHON_ROOT"
    uv sync --package agent-engine-sdk --extra dev
  )
  (
    cd "$PYTHON_ROOT/packages/agent-engine-sdk"
    uv run pytest
  )

  log "Python: agent-engine-sdk-memory"
  (
    cd "$PYTHON_ROOT"
    uv sync --package agent-engine-sdk-memory --extra dev
  )
  (
    cd "$PYTHON_ROOT/packages/agent-engine-sdk-memory"
    uv run pyright
    uv run pytest
  )

  log "Python: agent-engine-runner-shared"
  (
    cd "$PYTHON_ROOT"
    uv sync --package agent-engine-runner-shared --extra dev --extra tracing
  )
  (
    cd "$PYTHON_ROOT/packages/agent-engine-runner-shared"
    uv run pyright
    uv run pytest
  )

  log "Python: agent-engine-sdk-langgraph"
  (
    cd "$PYTHON_ROOT"
    uv sync --package agent-engine-sdk-langgraph --extra dev
  )
  (
    cd "$PYTHON_ROOT/packages/agent-engine-sdk-langgraph"
    uv run pyright
    uv run pytest
  )

  log "Python: agent-engine-sdk-adk"
  (
    cd "$PYTHON_ROOT"
    uv sync --package agent-engine-sdk-adk --extra dev
  )
  (
    cd "$PYTHON_ROOT/packages/agent-engine-sdk-adk"
    uv run pyright
    uv run pytest
  )

  log "Python: agent-engine-sdk-openai-agents"
  (
    cd "$PYTHON_ROOT"
    uv sync --package agent-engine-sdk-openai-agents --extra dev
  )
  (
    cd "$PYTHON_ROOT/packages/agent-engine-sdk-openai-agents"
    uv run pyright
    uv run pytest
  )
}

run_javascript_tests() {
  log "JavaScript: install workspace"
  (
    cd "$JAVASCRIPT_ROOT"
    npm ci --ignore-scripts --no-audit --no-fund --prefer-offline
  )

  local workspace
  for workspace in \
    @mongodb-js/agent-engine-sdk \
    @mongodb-js/agent-engine-sdk-memory \
    @mongodb-js/agent-engine-runner-shared
  do
    log "JavaScript: $workspace"
    (
      cd "$JAVASCRIPT_ROOT"
      npm run build --workspace="$workspace"
      npm run lint --workspace="$workspace"
      npm test --workspace="$workspace"
    )
  done

  log "JavaScript: @mongodb-js/agent-engine-sdk-langgraph"
  (
    cd "$JAVASCRIPT_ROOT"
    npm run typecheck --workspace=@mongodb-js/agent-engine-sdk-langgraph
    npm run lint --workspace=@mongodb-js/agent-engine-sdk-langgraph
    npm test --workspace=@mongodb-js/agent-engine-sdk-langgraph
  )
}

usage() {
  cat <<'EOF'
Usage: run-sdk-tests.sh [--python] [--javascript]
  Default (no flags): run both language suites.
EOF
}

RUN_PYTHON=false
RUN_JAVASCRIPT=false
if [ "$#" -eq 0 ]; then
  RUN_PYTHON=true
  RUN_JAVASCRIPT=true
else
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --python) RUN_PYTHON=true; shift ;;
      --javascript) RUN_JAVASCRIPT=true; shift ;;
      -h|--help) usage; exit 0 ;;
      *) usage >&2; exit 2 ;;
    esac
  done
fi

if [ "$RUN_PYTHON" = "true" ]; then
  run_python_tests
fi
if [ "$RUN_JAVASCRIPT" = "true" ]; then
  run_javascript_tests
fi
log "SDK tests passed"
