#!/usr/bin/env bash
# Tests for publish-python-package.sh registry probes and uv publish arguments.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$SCRIPT_DIR/publish-python-package.sh"
TEMP_ROOT="$(mktemp -d)"
trap 'rm -rf "$TEMP_ROOT"' EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }

root="$TEMP_ROOT/packages"
mkdir -p "$root/python/packages/agent-engine-sdk"
cat > "$root/python/packages/agent-engine-sdk/pyproject.toml" <<'TOML'
[project]
name = "agent-engine-sdk"
version = "0.0.86"
TOML

bin="$TEMP_ROOT/bin"
mkdir -p "$bin"
cat > "$bin/uv" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
printf 'uv %s\n' "$*" >> "${UV_LOG:?}"
if [ "${1:-}" = "build" ]; then
  while [ "$#" -gt 0 ]; do
    if [ "$1" = "--out-dir" ]; then
      mkdir -p "$2"
      printf 'wheel\n' > "$2/agent_engine_sdk-0.0.86-py3-none-any.whl"
      printf 'sdist\n' > "$2/agent_engine_sdk-0.0.86.tar.gz"
      exit 0
    fi
    shift
  done
  exit 1
fi
STUB
chmod +x "$bin/uv"

cat > "$bin/curl" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
printf 'curl %s\n' "$*" >> "${CURL_LOG:?}"
count=0
[ ! -f "${CURL_COUNT:?}" ] || count="$(cat "$CURL_COUNT")"
count=$((count + 1))
printf '%s' "$count" > "$CURL_COUNT"
code="$(printf '%s\n' "${CURL_CODES:?}" | sed -n "${count}p")"
[ -n "$code" ] || code="$(printf '%s\n' "$CURL_CODES" | tail -n 1)"
body=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    -o) body=$2; shift 2 ;;
    -w|--connect-timeout|--max-time) shift 2 ;;
    -sS) shift ;;
    *) shift ;;
  esac
done
if [ -n "$body" ]; then
  : > "$body"
  if [ "$code" = "200" ]; then
    printf '<a href="agent_engine_sdk-0.0.86-py3-none-any.whl">wheel</a>\n' > "$body"
  fi
fi
printf '%s' "$code"
STUB
chmod +x "$bin/curl"

cat > "$bin/sleep" <<'STUB'
#!/usr/bin/env bash
exit 0
STUB
chmod +x "$bin/sleep"

run_publish() {
  local codes=$1
  shift
  rm -f "$CURL_COUNT"
  PATH="$bin:$PATH" \
    PYPI_REPOSITORY_URL="http://upload.test" \
    PYPI_INDEX_URL="http://index.test/" \
    UV_LOG="$UV_LOG" CURL_LOG="$CURL_LOG" CURL_COUNT="$CURL_COUNT" CURL_CODES="$codes" \
    bash "$SCRIPT" --name agent-engine-sdk --directory agent-engine-sdk --packages-root "$root" "$@"
}

# An existing wheel must not skip a missing sdist; uv handles files individually.
UV_LOG="$TEMP_ROOT/uv-partial.log"
CURL_LOG="$TEMP_ROOT/curl-partial.log"
CURL_COUNT="$TEMP_ROOT/curl-partial.count"
out="$(run_publish '200')"
printf '%s\n' "$out" | grep -Fq 'publish agent-engine-sdk==0.0.86' || fail "expected publish"
grep -Fq 'publish --publish-url http://upload.test --check-url http://index.test/simple/' "$UV_LOG" \
  || fail "missing per-artifact check URL"
grep -Fq -- '--connect-timeout 10 --max-time 30' "$CURL_LOG" || fail "missing probe timeouts"

# Dry run validates artifacts and registry state without claiming token validation.
UV_LOG="$TEMP_ROOT/uv-dry-run.log"
CURL_LOG="$TEMP_ROOT/curl-dry-run.log"
CURL_COUNT="$TEMP_ROOT/curl-dry-run.count"
out="$(DRY_RUN=true UV_PUBLISH_USERNAME=release-user UV_PUBLISH_PASSWORD=release-token run_publish '404')"
printf '%s\n' "$out" | grep -Fq 'does not validate upload authorization' \
  || fail "dry-run output must describe its authorization limitation"
grep -Fq -- '--username release-user --password release-token --dry-run' "$UV_LOG" \
  || fail "missing dry-run credentials or flag"

# Transient failures retry, while exhaustion fails before invoking uv.
UV_LOG="$TEMP_ROOT/uv-retry.log"
CURL_LOG="$TEMP_ROOT/curl-retry.log"
CURL_COUNT="$TEMP_ROOT/curl-retry.count"
run_publish $'503\n429\n404' >/dev/null
[ "$(cat "$CURL_COUNT")" = "3" ] || fail "expected three probe attempts"

UV_LOG="$TEMP_ROOT/uv-exhausted.log"
CURL_LOG="$TEMP_ROOT/curl-exhausted.log"
CURL_COUNT="$TEMP_ROOT/curl-exhausted.count"
if PROBE_ATTEMPTS=2 run_publish $'503\n503' >"$TEMP_ROOT/exhausted.out" 2>"$TEMP_ROOT/exhausted.err"; then
  fail "expected exhausted probes to fail"
fi
grep -Fq 'PyPI index returned HTTP 503' "$TEMP_ROOT/exhausted.err" || fail "missing exhaustion error"
[ ! -f "$UV_LOG" ] || fail "uv should not run after probe exhaustion"

echo "publish-python-package tests passed"
