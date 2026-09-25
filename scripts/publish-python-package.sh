#!/usr/bin/env bash
# Build and publish one Python SDK package. Skips if that version is already
# on the registry. Sequencing across packages is the caller's job (GHA needs).
#
# Required:
#   --name DIST            distribution name (e.g. agent-engine-sdk)
#   --directory DIR        directory under python/packages/
#   PYPI_REPOSITORY_URL    upload endpoint, e.g. http://127.0.0.1:8080
#
# Optional:
#   --packages-root PATH   packages tree (default: cwd)
#   PYPI_INDEX_URL         PEP 503 index for the skip check. Real PyPI serves
#                          uploads and the index from different hosts; the mock
#                          serves both, so this defaults to the upload URL.
#   UV_PUBLISH_USERNAME / UV_PUBLISH_PASSWORD  (default: ci/ci for mock indexes)
#   DRY_RUN                "true" builds and validates without uploading
#   PROBE_ATTEMPTS         registry probe retries (default: 5)
#   PROBE_CONNECT_TIMEOUT  connection timeout per probe in seconds (default: 10)
#   PROBE_MAX_TIME         total timeout per probe in seconds (default: 30)
set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }

PROBE_ATTEMPTS="${PROBE_ATTEMPTS:-5}"
PROBE_CONNECT_TIMEOUT="${PROBE_CONNECT_TIMEOUT:-10}"
PROBE_MAX_TIME="${PROBE_MAX_TIME:-30}"

# This probe gates a publish chain: an unretried 429 or 5xx from a real index
# fails this package and skips every dependent, which leaves a partially
# published release that cannot be replayed cleanly.
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
    echo "index probe got HTTP ${code} for ${url}; retrying in ${delay}s" >&2
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
      sed -n '2,17p' "$0"
      exit 0
      ;;
    *) die "unknown argument: $1" ;;
  esac
done

[ -n "$NAME" ] || die "--name is required"
[ -n "$DIRECTORY" ] || die "--directory is required"
: "${PYPI_REPOSITORY_URL:?PYPI_REPOSITORY_URL is not set}"

PACKAGES_ROOT="$(cd "$PACKAGES_ROOT" && pwd)"
python_root="$PACKAGES_ROOT/python"
pyproject="$python_root/packages/$DIRECTORY/pyproject.toml"
[ -f "$pyproject" ] || die "missing $pyproject"

version="$(sed -n 's/^version = "\(.*\)"/\1/p' "$pyproject" | head -n1)"
[ -n "$version" ] || die "no version in $pyproject"

normalized="$(printf '%s' "$NAME" | tr '[:upper:]' '[:lower:]' | sed -E 's/[-_.]+/-/g')"
check_url="${PYPI_INDEX_URL:-$PYPI_REPOSITORY_URL}"
check_url="${check_url%/}/simple/"
url="${check_url}${normalized}/"

code="$(probe_status "$url" /dev/null)"
case "$code" in
  200|404) ;;
  *)
    die "PyPI index returned HTTP ${code} for ${url}"
    ;;
esac

DRY_RUN="${DRY_RUN:-false}"
if [ "$DRY_RUN" = "true" ]; then
  echo "dry-run ${NAME}==${version} (validates artifacts and registry state; does not validate upload authorization)"
else
  echo "publish ${NAME}==${version}"
fi
dist="$(mktemp -d)"
trap 'rm -rf "$dist"' EXIT
(
  cd "$python_root"
  uv build --package "$NAME" --out-dir "$dist"
)
shopt -s nullglob
artifacts=( "$dist"/* )
[ "${#artifacts[@]}" -gt 0 ] || die "uv build produced no artifacts for $NAME"

publish_args=(
  --publish-url "$PYPI_REPOSITORY_URL"
  --check-url "$check_url"
  --username "${UV_PUBLISH_USERNAME:-ci}"
  --password "${UV_PUBLISH_PASSWORD:-ci}"
)
if [ "$DRY_RUN" = "true" ]; then
  publish_args+=(--dry-run)
fi

uv publish "${publish_args[@]}" "${artifacts[@]}"
