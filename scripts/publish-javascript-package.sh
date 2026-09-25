#!/usr/bin/env bash
# Publish one JavaScript SDK package. Skips if that version is already on the
# registry. The caller must have already run npm ci and built this package
# (and its workspace dependencies). Sequencing across packages is the caller's
# job (GHA needs).
#
# Required:
#   --name NAME            npm package name (e.g. @mongodb-js/agent-engine-sdk)
#   --directory DIR        directory under javascript/packages/
#   NPM_REGISTRY           registry URL, e.g. http://127.0.0.1:4873
#
# Optional:
#   --packages-root PATH   packages tree (default: cwd)
#   NPM_NPMRC              npmrc holding the registry auth token (mock registry)
#   DRY_RUN                "true" packs and validates without uploading
#   PROBE_ATTEMPTS         registry probe retries (default: 5)
#   PROBE_CONNECT_TIMEOUT  connection timeout per probe in seconds (default: 10)
#   PROBE_MAX_TIME         total timeout per probe in seconds (default: 30)
set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }

PROBE_ATTEMPTS="${PROBE_ATTEMPTS:-5}"
PROBE_CONNECT_TIMEOUT="${PROBE_CONNECT_TIMEOUT:-10}"
PROBE_MAX_TIME="${PROBE_MAX_TIME:-30}"

# This probe gates a publish chain: an unretried 429 or 5xx from a real
# registry fails this package and skips every dependent, which leaves a
# partially published release that cannot be replayed cleanly.
probe_status() {
  local url=$1
  local output=$2
  local attempt code="000" delay=1
  for ((attempt = 1; attempt <= PROBE_ATTEMPTS; attempt++)); do
    code="$(curl -sS --connect-timeout "$PROBE_CONNECT_TIMEOUT" --max-time "$PROBE_MAX_TIME" \
      -o "$output" -w '%{http_code}' "$url" || true)"
    case "$code" in
      200|404)
        printf '%s' "$code"
        return 0
        ;;
    esac
    [ "$attempt" -lt "$PROBE_ATTEMPTS" ] || break
    echo "registry probe got HTTP ${code} for ${url}; retrying in ${delay}s" >&2
    sleep "$delay"
    delay=$((delay * 2))
  done
  printf '%s' "$code"
}

NAME=""
DIRECTORY=""
PACKAGES_ROOT="."
while [ "$#" -gt 0 ]; do
  case "$1" in
    --name) NAME=${2:?}; shift 2 ;;
    --directory) DIRECTORY=${2:?}; shift 2 ;;
    --packages-root) PACKAGES_ROOT=${2:?}; shift 2 ;;
    -h|--help)
      sed -n '2,16p' "$0"
      exit 0
      ;;
    *) die "unknown argument: $1" ;;
  esac
done

[ -n "$NAME" ] || die "--name is required"
[ -n "$DIRECTORY" ] || die "--directory is required"
: "${NPM_REGISTRY:?NPM_REGISTRY is not set}"

PACKAGES_ROOT="$(cd "$PACKAGES_ROOT" && pwd)"
manifest="$PACKAGES_ROOT/javascript/packages/$DIRECTORY/package.json"
[ -f "$manifest" ] || die "missing $manifest"

version="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["version"])' "$manifest")"
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([0-9A-Za-z.-]*)?$ ]] \
  || die "$manifest has an invalid version: $version"

encoded="$(python3 -c 'import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1], safe="@"))' "$NAME")"
url="${NPM_REGISTRY%/}/${encoded}/${version}"
code="$(probe_status "$url" /dev/null)"
case "$code" in
  404) ;;
  200)
    echo "skip ${NAME}@${version} (already on npm registry)"
    exit 0
    ;;
  *)
    die "npm registry returned HTTP ${code} for ${url}"
    ;;
esac

DRY_RUN="${DRY_RUN:-false}"
if [ "$DRY_RUN" = "true" ]; then
  echo "dry-run ${NAME}@${version}"
else
  echo "publish ${NAME}@${version}"
fi
original="$(mktemp)"
trap 'cp "$original" "$manifest"; rm -f "$original"' EXIT
cp "$manifest" "$original"
NPM_REGISTRY="$NPM_REGISTRY" python3 - "$manifest" <<'PY'
import json, os, sys

path = sys.argv[1]
data = json.load(open(path, encoding="utf-8"))
data.pop("private", None)
scripts = data.get("scripts")
if isinstance(scripts, dict):
    scripts.pop("prepare", None)
    data["scripts"] = scripts
data["publishConfig"] = {"registry": os.environ["NPM_REGISTRY"]}
with open(path, "w", encoding="utf-8") as fh:
    json.dump(data, fh, indent=2)
    fh.write("\n")
PY

(
  cd "$PACKAGES_ROOT/javascript/packages/$DIRECTORY"
  [ -n "${NPM_NPMRC:-}" ] && export NPM_CONFIG_USERCONFIG="$NPM_NPMRC"
  publish_args=(--ignore-scripts --access public --registry "$NPM_REGISTRY")
  if [ "$DRY_RUN" = "true" ]; then
    # npm publish --dry-run never contacts the registry with credentials, so
    # it proves the tarball is well formed but not that release-day auth
    # works. Exercise the credential path explicitly.
    npm whoami --registry "$NPM_REGISTRY" \
      || die "npm authentication failed against ${NPM_REGISTRY}"
    publish_args+=(--dry-run)
  fi
  npm publish "${publish_args[@]}"
)
