#!/usr/bin/env bash
# Start Verdaccio and/or pypiserver on loopback for SDK release rehearsal.
#
#   start-mock-registries.sh           # both
#   start-mock-registries.sh --npm     # Verdaccio only (needs docker)
#   start-mock-registries.sh --pypi    # pypiserver only (needs uv)
#
# Prints NPM_REGISTRY and NPM_NPMRC (npmrc with the publish token) for npm,
# and PYPI_REPOSITORY_URL for PyPI. Does not pull images implicitly.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERDACCIO_IMAGE="${VERDACCIO_IMAGE:-verdaccio/verdaccio:6.2.9@sha256:e3fc342181fd581de012659e55c14e8df4103aac6183698bbfd7cfc3c55d78d2}"
VERDACCIO_NAME="${VERDACCIO_NAME:-sdk-release-verdaccio}"
VERDACCIO_PORT="${VERDACCIO_PORT:-4873}"
VERDACCIO_USER="${VERDACCIO_USER:-sdk-release}"
VERDACCIO_PASSWORD="${VERDACCIO_PASSWORD:-sdk-release}"
NPM_NPMRC="${NPM_NPMRC:-${TMPDIR:-/tmp}/sdk-release-npmrc}"
PYPI_PORT="${PYPI_PORT:-8080}"
PYPI_DATA="${PYPI_DATA:-${TMPDIR:-/tmp}/sdk-release-pypi}"
PYPISERVER_VERSION="${PYPISERVER_VERSION:-2.3.2}"
PYPI_LOG="${PYPI_LOG:-${TMPDIR:-/tmp}/sdk-release-pypiserver.log}"
# uvx may resolve and download pypiserver on a cold cache before the server
# binds, so the budget covers install time, not just startup.
PYPI_READY_TIMEOUT="${PYPI_READY_TIMEOUT:-90}"
VERDACCIO_READY_TIMEOUT="${VERDACCIO_READY_TIMEOUT:-30}"
CONFIG="$ROOT/.github/verdaccio/config.yaml"

START_NPM=false
START_PYPI=false
if [ "$#" -eq 0 ]; then
  START_NPM=true
  START_PYPI=true
else
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --npm) START_NPM=true; shift ;;
      --pypi) START_PYPI=true; shift ;;
      *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
    esac
  done
fi

die() { echo "ERROR: $*" >&2; exit 1; }

command -v curl >/dev/null || die "curl is required"
command -v python3 >/dev/null || die "python3 is required"

if [ "$START_NPM" = "true" ]; then
  command -v docker >/dev/null || die "docker is required for --npm"
  [[ -f "$CONFIG" ]] || die "missing Verdaccio config: $CONFIG"
  if ! docker image inspect "$VERDACCIO_IMAGE" >/dev/null 2>&1; then
    die "$VERDACCIO_IMAGE is not available locally. Preload it with: docker pull $VERDACCIO_IMAGE"
  fi
  if docker container inspect "$VERDACCIO_NAME" >/dev/null 2>&1; then
    docker rm --force "$VERDACCIO_NAME" >/dev/null
  fi
  docker run --detach \
    --name "$VERDACCIO_NAME" \
    --publish "127.0.0.1:${VERDACCIO_PORT}:4873" \
    --pull never \
    --volume "$CONFIG:/verdaccio/conf/config.yaml:ro" \
    "$VERDACCIO_IMAGE" >/dev/null
fi

if [ "$START_PYPI" = "true" ]; then
  command -v uv >/dev/null || die "uv is required for --pypi"
  mkdir -p "$PYPI_DATA"
  # --passwords/--authenticate set to "." disable authentication so CI can
  # upload without a password file; --disable-fallback makes missing packages
  # 404 instead of proxying to real PyPI, which the skip-if-exists check
  # relies on; --interface keeps the registry off the network. The launcher
  # raises pypiserver's vendored-Bottle upload limit, which SDK distributions
  # exceed.
  uvx --from "pypiserver==${PYPISERVER_VERSION}" python "$ROOT/scripts/run-mock-pypi.py" run \
    --interface 127.0.0.1 \
    --port "$PYPI_PORT" \
    --disable-fallback \
    --passwords . \
    --authenticate . \
    "$PYPI_DATA" >"$PYPI_LOG" 2>&1 &
  PYPI_PID=$!
fi

dump_log() {
  local log=${1:-}
  [ -n "$log" ] && [ -f "$log" ] || return 0
  echo "--- $log ---" >&2
  cat "$log" >&2
  echo "--- end $log ---" >&2
}

# Without the launcher's PID and log, a crashed server and a slow one are
# indistinguishable: both surface as a bare readiness timeout with no output.
wait_http() {
  local url=$1
  local label=$2
  local timeout=$3
  local pid=${4:-}
  local log=${5:-}
  local i
  for ((i = 0; i < timeout; i++)); do
    if curl -sf "$url" >/dev/null; then
      return 0
    fi
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
      dump_log "$log"
      die "$label exited before becoming ready at $url"
    fi
    sleep 1
  done
  dump_log "$log"
  die "$label did not become ready at $url within ${timeout}s"
}

# Verdaccio rejects anonymous publish even with `publish: $all`, and the npm
# client aborts with ENEEDAUTH before sending a request. Register the CI
# publisher and hand its token to npm as a scoped userconfig.
register_verdaccio_user() {
  local url="http://127.0.0.1:${VERDACCIO_PORT}/-/user/org.couchdb.user:${VERDACCIO_USER}"
  local payload response token
  payload="$(printf '{"name":"%s","password":"%s","email":"%s@example.invalid"}' \
    "$VERDACCIO_USER" "$VERDACCIO_PASSWORD" "$VERDACCIO_USER")"
  response="$(curl -sS -X PUT "$url" -H 'Content-Type: application/json' --data "$payload")"
  token="$(printf '%s' "$response" | python3 -c 'import json, sys; print(json.load(sys.stdin)["token"])' 2>/dev/null)" \
    || die "Verdaccio user registration failed: $response"
  printf '%s' "$token"
}

if [ "$START_NPM" = "true" ]; then
  wait_http "http://127.0.0.1:${VERDACCIO_PORT}/-/ping" "Verdaccio" "$VERDACCIO_READY_TIMEOUT"
  token="$(register_verdaccio_user)"
  printf '//127.0.0.1:%s/:_authToken=%s\n' "$VERDACCIO_PORT" "$token" > "$NPM_NPMRC"
  echo "NPM_REGISTRY=http://127.0.0.1:${VERDACCIO_PORT}"
  echo "NPM_NPMRC=$NPM_NPMRC"
fi
if [ "$START_PYPI" = "true" ]; then
  wait_http "http://127.0.0.1:${PYPI_PORT}/simple/" "pypiserver" \
    "$PYPI_READY_TIMEOUT" "$PYPI_PID" "$PYPI_LOG"
  echo "PYPI_REPOSITORY_URL=http://127.0.0.1:${PYPI_PORT}"
fi
