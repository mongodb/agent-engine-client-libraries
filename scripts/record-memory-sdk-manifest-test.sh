#!/usr/bin/env bash
# Tests for record-memory-sdk-manifest.sh. A stable build records the wheel
# hash. A pre-release leaves only the marker the record workflow understands.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$SCRIPT_DIR/record-memory-sdk-manifest.sh"
TEMP_ROOT="$(mktemp -d)"
trap 'rm -rf "$TEMP_ROOT"' EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }

root="$TEMP_ROOT/packages"
mkdir -p "$root/python/packages/agent-engine-sdk-memory" "$TEMP_ROOT/dist"
cat > "$root/python/packages/agent-engine-sdk-memory/pyproject.toml" <<'TOML'
[project]
name = "agent-engine-sdk-memory"
version = "1.2.3"
TOML
git -C "$root" init -q
git -C "$root" config user.email test@example.com
git -C "$root" config user.name test
printf 'fixture\n' > "$root/README"
git -C "$root" add README
git -C "$root" -c commit.gpgsign=false -c core.hooksPath=/dev/null commit -q -m fixture
expected_sha="$(git -C "$root" rev-parse HEAD)"
printf 'wheel\n' > "$TEMP_ROOT/dist/agent_engine_sdk_memory-1.2.3-py3-none-any.whl"
built_sha256="$(python3 -c 'import hashlib; print(hashlib.sha256(b"wheel\n").hexdigest())')"

# curl stands in for the PyPI JSON API. PYPI_SHA256 is the hash PyPI reports
# for the wheel; unset means PyPI has no file for the version.
bin="$TEMP_ROOT/bin"
mkdir -p "$bin"
cat > "$bin/curl" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
printf 'curl %s\n' "$*" >> "${CURL_LOG:?}"
[ -n "${PYPI_SHA256:-}" ] || exit 22
out=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    -o) out=$2; shift 2 ;;
    *) shift ;;
  esac
done
printf '{"urls": [{"filename": "agent_engine_sdk_memory-1.2.3-py3-none-any.whl", "digests": {"sha256": "%s"}}]}' \
  "$PYPI_SHA256" > "$out"
STUB
chmod +x "$bin/curl"
export PATH="$bin:$PATH" CURL_LOG="$TEMP_ROOT/curl.log" PYPI_LOOKUP_DELAY=0

workspace="$TEMP_ROOT/workspace"
mkdir -p "$workspace"
PYPI_SHA256="$built_sha256" GITHUB_WORKSPACE="$workspace" \
  bash "$SCRIPT" --dist-dir "$TEMP_ROOT/dist" --packages-root "$root" >"$TEMP_ROOT/stable.out"
manifest="$workspace/sdk-memory-manifest.json"
[ -f "$manifest" ] || fail "expected sdk-memory-manifest.json"
[ ! -e "$workspace/sdk-memory-manifest.json.skipped" ] || fail "stable publish left a skip marker"
grep -Fq 'pypi.org/pypi/agent-engine-sdk-memory/1.2.3/json' "$CURL_LOG" \
  || fail "record did not check the published wheel on PyPI"
python3 - "$manifest" "$expected_sha" <<'PY' || fail "manifest did not record the built wheel"
import hashlib, json, sys
manifest, expected_sha = sys.argv[1:3]
data = json.load(open(manifest, encoding="utf-8"))
assert data["package"] == "agent-engine-sdk-memory"
assert data["version"] == "1.2.3"
assert data["filename"] == "agent_engine_sdk_memory-1.2.3-py3-none-any.whl"
assert data["git_sha"] == expected_sha
assert data["sha256"] == hashlib.sha256(b"wheel\n").hexdigest()
assert "url" not in data
PY

sed -i.bak 's/^version = .*/version = "1.2.4-alpha.0"/' \
  "$root/python/packages/agent-engine-sdk-memory/pyproject.toml"
alpha_ws="$TEMP_ROOT/alpha-workspace"
mkdir -p "$alpha_ws"
GITHUB_WORKSPACE="$alpha_ws" bash "$SCRIPT" --dist-dir "$TEMP_ROOT/dist" --packages-root "$root" \
  >"$TEMP_ROOT/alpha.out" || fail "alpha memory record must succeed"
[ ! -e "$alpha_ws/sdk-memory-manifest.json" ] || fail "alpha publish recorded a memory manifest"
[ -e "$alpha_ws/sdk-memory-manifest.json.skipped" ] \
  || fail "alpha publish must leave the skip marker"
grep -Fq 'not recording pre-release' "$TEMP_ROOT/alpha.out" || fail "expected alpha skip message"

sed -i.bak 's/^version = .*/version = "1.2.3"/' \
  "$root/python/packages/agent-engine-sdk-memory/pyproject.toml"

# skip-existing left an earlier upload of this version on PyPI. The rebuilt
# wheel differs, so its hash must not be recorded: the canary could never
# download a file with that hash.
rebuild_ws="$TEMP_ROOT/rebuild-workspace"
mkdir -p "$rebuild_ws"
PYPI_SHA256="$(printf 'c%.0s' $(seq 64))" GITHUB_WORKSPACE="$rebuild_ws" \
  bash "$SCRIPT" --dist-dir "$TEMP_ROOT/dist" --packages-root "$root" >"$TEMP_ROOT/rebuild.out" \
  || fail "rebuild of a published version must exit cleanly"
[ ! -e "$rebuild_ws/sdk-memory-manifest.json" ] || fail "rebuild recorded a hash PyPI does not serve"
[ -e "$rebuild_ws/sdk-memory-manifest.json.skipped" ] || fail "rebuild must leave the skip marker"
grep -Fq 'published nothing new' "$TEMP_ROOT/rebuild.out" || fail "expected rebuild skip message"

# A wheel that never shows up on PyPI means the publish did not land. That is
# a failure, not a skip, and the lookup is retried before giving up.
absent_ws="$TEMP_ROOT/absent-workspace"
mkdir -p "$absent_ws"
: > "$CURL_LOG"
if PYPI_LOOKUP_ATTEMPTS=3 GITHUB_WORKSPACE="$absent_ws" \
  bash "$SCRIPT" --dist-dir "$TEMP_ROOT/dist" --packages-root "$root" \
  >"$TEMP_ROOT/absent.out" 2>"$TEMP_ROOT/absent.err"; then
  fail "wheel missing from PyPI must fail"
fi
grep -Fq 'is not on PyPI' "$TEMP_ROOT/absent.err" || fail "expected missing-on-PyPI error"
[ "$(wc -l < "$CURL_LOG")" -eq 3 ] || fail "PyPI lookup was not retried"
[ ! -e "$absent_ws/sdk-memory-manifest.json" ] && [ ! -e "$absent_ws/sdk-memory-manifest.json.skipped" ] \
  || fail "failed lookup must leave neither a manifest nor a skip marker"
rm -f "$TEMP_ROOT/dist/agent_engine_sdk_memory-1.2.3-py3-none-any.whl"
missing_ws="$TEMP_ROOT/missing-workspace"
mkdir -p "$missing_ws"
if GITHUB_WORKSPACE="$missing_ws" bash "$SCRIPT" --dist-dir "$TEMP_ROOT/dist" --packages-root "$root" \
  >"$TEMP_ROOT/missing.out" 2>"$TEMP_ROOT/missing.err"; then
  fail "missing wheel must fail"
fi
grep -Fq 'memory SDK wheel not found' "$TEMP_ROOT/missing.err" || fail "expected missing-wheel error"

echo "record-memory-sdk-manifest tests passed"
