#!/usr/bin/env bash
# Tests for publish-javascript-package.sh registry probes, auth, and manifest staging.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$SCRIPT_DIR/publish-javascript-package.sh"
TEMP_ROOT="$(mktemp -d)"
trap 'rm -rf "$TEMP_ROOT"' EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }

root="$TEMP_ROOT/packages"
pkg="$root/javascript/packages/agent-engine-sdk"
mkdir -p "$pkg"
cat > "$pkg/package.json" <<'JSON'
{
  "name": "@mongodb-js/agent-engine-sdk",
  "version": "0.0.86",
  "private": true,
  "scripts": { "prepare": "npm run build", "build": "tsc" }
}
JSON

bin="$TEMP_ROOT/bin"
mkdir -p "$bin"
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
printf '%s' "$code"
STUB
chmod +x "$bin/curl"

cat > "$bin/npm" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
printf 'npm %s userconfig=%s\n' "$*" "${NPM_CONFIG_USERCONFIG:-}" >> "${NPM_LOG:?}"
if [ "${1:-}" = "whoami" ] && [ "${NPM_AUTH_FAIL:-false}" = "true" ]; then
  exit 1
fi
if [ "${1:-}" = "publish" ] && [ -n "${PUBLISHED_MANIFEST:-}" ]; then
  python3 -c 'import json,sys; json.dump(json.load(open("package.json")), open(sys.argv[1],"w"), indent=2)' \
    "$PUBLISHED_MANIFEST"
fi
STUB
chmod +x "$bin/npm"

cat > "$bin/sleep" <<'STUB'
#!/usr/bin/env bash
exit 0
STUB
chmod +x "$bin/sleep"

run_publish() {
  local codes=$1
  shift
  rm -f "$CURL_COUNT"
  PATH="$bin:$PATH" NPM_REGISTRY="http://npm.test" \
    NPM_LOG="$NPM_LOG" CURL_LOG="$CURL_LOG" CURL_COUNT="$CURL_COUNT" CURL_CODES="$codes" \
    PUBLISHED_MANIFEST="${PUBLISHED_MANIFEST:-}" \
    bash "$SCRIPT" --name "@mongodb-js/agent-engine-sdk" --directory agent-engine-sdk \
      --packages-root "$root" "$@"
}

# Existing versions skip before npm or manifest mutation.
NPM_LOG="$TEMP_ROOT/npm-skip.log"
CURL_LOG="$TEMP_ROOT/curl-skip.log"
CURL_COUNT="$TEMP_ROOT/curl-skip.count"
out="$(run_publish '200')"
printf '%s\n' "$out" | grep -Fq 'skip @mongodb-js/agent-engine-sdk@0.0.86' || fail "expected skip"
[ ! -f "$NPM_LOG" ] || fail "npm should not run when the version exists"
grep -Fq -- '--connect-timeout 10 --max-time 30' "$CURL_LOG" || fail "missing probe timeouts"

# A normal publish strips private/prepare while publishing and restores the manifest.
NPM_LOG="$TEMP_ROOT/npm-publish.log"
CURL_LOG="$TEMP_ROOT/curl-publish.log"
CURL_COUNT="$TEMP_ROOT/curl-publish.count"
PUBLISHED_MANIFEST="$TEMP_ROOT/published.json"
run_publish '404' >/dev/null
grep -Fq 'npm publish --ignore-scripts --access public --registry http://npm.test' "$NPM_LOG" \
  || fail "missing npm publish"
python3 - "$PUBLISHED_MANIFEST" "$pkg/package.json" <<'PY' || fail "manifest staging mismatch"
import json, sys
published = json.load(open(sys.argv[1]))
restored = json.load(open(sys.argv[2]))
assert "private" not in published
assert "prepare" not in published.get("scripts", {})
assert restored["private"] is True
assert restored["scripts"]["prepare"] == "npm run build"
PY

# Dry run authenticates before packing and uses the configured npmrc.
NPM_LOG="$TEMP_ROOT/npm-dry-run.log"
CURL_LOG="$TEMP_ROOT/curl-dry-run.log"
CURL_COUNT="$TEMP_ROOT/curl-dry-run.count"
mock_npmrc="$TEMP_ROOT/mock.npmrc"
touch "$mock_npmrc"
DRY_RUN=true NPM_NPMRC="$mock_npmrc" run_publish '404' >/dev/null
grep -Fq 'npm whoami --registry http://npm.test' "$NPM_LOG" || fail "missing npm auth check"
grep -Fq 'npm publish --ignore-scripts --access public --registry http://npm.test --dry-run' "$NPM_LOG" \
  || fail "missing npm dry run"
grep -Fq "userconfig=$mock_npmrc" "$NPM_LOG" || fail "missing npm userconfig"

# Authentication failure must stop before npm publish.
NPM_LOG="$TEMP_ROOT/npm-auth-fail.log"
CURL_LOG="$TEMP_ROOT/curl-auth-fail.log"
CURL_COUNT="$TEMP_ROOT/curl-auth-fail.count"
if DRY_RUN=true NPM_AUTH_FAIL=true run_publish '404' >"$TEMP_ROOT/auth.out" 2>"$TEMP_ROOT/auth.err"; then
  fail "expected authentication failure"
fi
grep -Fq 'npm authentication failed' "$TEMP_ROOT/auth.err" || fail "missing auth error"
if grep -Fq 'npm publish' "$NPM_LOG"; then fail "publish ran after auth failure"; fi

# Transient failures retry, while exhaustion fails before invoking npm.
NPM_LOG="$TEMP_ROOT/npm-retry.log"
CURL_LOG="$TEMP_ROOT/curl-retry.log"
CURL_COUNT="$TEMP_ROOT/curl-retry.count"
run_publish $'503\n429\n404' >/dev/null
[ "$(cat "$CURL_COUNT")" = "3" ] || fail "expected three probe attempts"

NPM_LOG="$TEMP_ROOT/npm-exhausted.log"
CURL_LOG="$TEMP_ROOT/curl-exhausted.log"
CURL_COUNT="$TEMP_ROOT/curl-exhausted.count"
if PROBE_ATTEMPTS=2 run_publish $'503\n503' >"$TEMP_ROOT/exhausted.out" 2>"$TEMP_ROOT/exhausted.err"; then
  fail "expected exhausted probes to fail"
fi
grep -Fq 'npm registry returned HTTP 503' "$TEMP_ROOT/exhausted.err" || fail "missing exhaustion error"
[ ! -f "$NPM_LOG" ] || fail "npm should not run after probe exhaustion"

echo "publish-javascript-package tests passed"
