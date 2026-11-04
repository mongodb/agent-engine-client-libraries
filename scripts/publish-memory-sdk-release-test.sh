#!/usr/bin/env bash
# Tests for publish-memory-sdk-release.sh release tags.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$SCRIPT_DIR/publish-memory-sdk-release.sh"
TEMP_ROOT="$(mktemp -d)"
trap 'rm -rf "$TEMP_ROOT"' EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }

bin="$TEMP_ROOT/bin"
mkdir -p "$bin"
cat > "$bin/gh" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
printf 'gh %s\n' "$*" >> "${GH_LOG:?}"
if [ "${1:-}" = "release" ] && [ "${2:-}" = "view" ]; then
  tag="${3:-}"
  case " ${GH_EXISTING_TAGS:-} " in
    *" ${tag} "*) ;;
    *) exit 1 ;;
  esac
  # --json assets lists the release's asset names (the script pipes it through
  # --jq). An existing release carries the manifest unless GH_EMPTY_TAGS names
  # it, which stands in for a release whose asset upload failed.
  case " $* " in
    *" --json assets "*)
      case " ${GH_EMPTY_TAGS:-} " in
        *" ${tag} "*) ;;
        *) echo "sdk-memory-manifest.json" ;;
      esac
      ;;
  esac
  exit 0
fi
if [ "${1:-}" = "release" ] && [ "${2:-}" = "download" ]; then
  [ "${GH_DOWNLOAD_FAIL:-}" != "1" ] || exit 1
  cat "${GH_CURRENT_MANIFEST:?}"
fi
exit 0
STUB
chmod +x "$bin/gh"

# A regression that shells out to curl fails here instead of using the real
# client. The recorder must ignore dispatch configuration, including a token
# that contains a newline or CR.
cat > "$bin/curl" <<'STUB'
#!/usr/bin/env bash
echo "curl must not be called" >&2
exit 99
STUB
chmod +x "$bin/curl"

write_manifest() {
  local dest=$1
  cat > "$dest" <<JSON
{
  "filename": "agent_engine_sdk_memory-${2:-1.2.3}-py3-none-any.whl",
  "git_sha": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "package": "agent-engine-sdk-memory",
  "sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "version": "${2:-1.2.3}"
}
JSON
}

manifest="$TEMP_ROOT/sdk-memory-manifest.json"
write_manifest "$manifest"

GH_LOG="$TEMP_ROOT/gh-plain.log"
rm -f "$GH_LOG"
PATH="$bin:$PATH" GH_LOG="$GH_LOG" \
  MEMORY_CANARY_DISPATCH_URL="https://attacker.example/steal" \
  MEMORY_CANARY_DISPATCH_TOKEN="$(printf 'secret\ntoken\r')" \
  bash "$SCRIPT" "$manifest" >"$TEMP_ROOT/plain.out" 2>"$TEMP_ROOT/plain.err"
grep -Fq 'sdk-memory-v1.2.3' "$GH_LOG" || fail "missing immutable release"
grep -Fq 'sdk-memory-current' "$GH_LOG" || fail "missing current release"
grep -Fq -- '--target aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' "$GH_LOG" || fail "missing target sha"
[ "$(grep -c -- '--latest=false' "$GH_LOG")" -eq 2 ] || fail "SDK releases must not become Latest"

# Publish 1.2.3 with an older, identical, or newer current record. An identical
# retry must not replace the asset. Numeric ordering matters: 1.2.10 is newer
# than 1.2.3.
current_manifest="$TEMP_ROOT/current.json"
for current_version in 1.2.2 1.2.3 1.2.10; do
  write_manifest "$current_manifest" "$current_version"
  GH_LOG="$TEMP_ROOT/gh-$current_version.log"
  PATH="$bin:$PATH" GH_LOG="$GH_LOG" \
    GH_EXISTING_TAGS="sdk-memory-v1.2.3 sdk-memory-current" \
    GH_CURRENT_MANIFEST="$current_manifest" \
    bash "$SCRIPT" "$manifest" >"$TEMP_ROOT/$current_version.out" \
    || fail "retry with current $current_version must succeed"
  grep -Eq 'release (create|delete)|release upload sdk-memory-v' "$GH_LOG" \
    && fail "retry changed an immutable release or recreated current"
  if [ "$current_version" = "1.2.2" ]; then
    grep -Fq 'release upload sdk-memory-current' "$GH_LOG" || fail "new version was not promoted"
  else
    grep -Fq 'release upload' "$GH_LOG" && fail "identical or older version rewrote current"
  fi
done

# Repair an empty immutable release whether current is also missing or has
# advanced since the failed run. Repairing history must not roll back current.
for current_state in missing newer; do
  write_manifest "$current_manifest" 1.2.10
  empty_tags="sdk-memory-v1.2.3"
  [ "$current_state" != "missing" ] || empty_tags+=" sdk-memory-current"
  GH_LOG="$TEMP_ROOT/gh-empty-$current_state.log"
  PATH="$bin:$PATH" GH_LOG="$GH_LOG" \
    GH_EXISTING_TAGS="sdk-memory-v1.2.3 sdk-memory-current" GH_EMPTY_TAGS="$empty_tags" \
    GH_CURRENT_MANIFEST="$current_manifest" \
    bash "$SCRIPT" "$manifest" >"$TEMP_ROOT/empty-$current_state.out" 2>&1 \
    || fail "publish onto an empty immutable release must succeed"
  grep -Fxq "gh release upload sdk-memory-v1.2.3 $manifest" "$GH_LOG" \
    || fail "missing immutable manifest was not repaired"
  grep -Fq 'release create' "$GH_LOG" && fail "empty release was recreated"
  if [ "$current_state" = "missing" ]; then
    grep -Fq 'release upload sdk-memory-current' "$GH_LOG" || fail "missing current manifest was not repaired"
  else
    grep -Fq 'release upload sdk-memory-current' "$GH_LOG" && fail "historical repair rolled back current"
  fi
done

# Unreadable or conflicting current records must not be replaced.
for invalid in malformed conflicting download; do
  write_manifest "$current_manifest"
  case "$invalid" in
    malformed) printf '{' > "$current_manifest" ;;
    conflicting) sed -i.bak 's/bbbbbbbb/cccccccc/g' "$current_manifest" ;;
  esac
  GH_LOG="$TEMP_ROOT/gh-$invalid.log"
  download_fail=0
  [ "$invalid" != "download" ] || download_fail=1
  if PATH="$bin:$PATH" GH_LOG="$GH_LOG" \
    GH_EXISTING_TAGS="sdk-memory-v1.2.3 sdk-memory-current" GH_CURRENT_MANIFEST="$current_manifest" \
    GH_DOWNLOAD_FAIL="$download_fail" \
    bash "$SCRIPT" "$manifest" >"$TEMP_ROOT/$invalid.out" 2>&1; then
    fail "$invalid current record must fail"
  fi
  grep -Eq 'release (upload|create|delete)' "$GH_LOG" && fail "$invalid current record mutated releases"
done

# A pre-release publish records no manifest and leaves the marker instead. The
# release step must exit cleanly without touching GitHub, so the canary keeps
# its recorded wheel.
mkdir -p "$TEMP_ROOT/alpha"
: >"$TEMP_ROOT/alpha/sdk-memory-manifest.json.skipped"
GH_LOG="$TEMP_ROOT/gh-alpha.log"
rm -f "$GH_LOG"
PATH="$bin:$PATH" GH_LOG="$GH_LOG" \
  bash "$SCRIPT" "$TEMP_ROOT/alpha/sdk-memory-manifest.json" \
  >"$TEMP_ROOT/alpha.out" 2>&1 || fail "pre-release with no manifest must exit cleanly"
grep -Fq 'was not recorded' "$TEMP_ROOT/alpha.out" || fail "expected pre-release skip message"
[ ! -e "$GH_LOG" ] || fail "pre-release touched GitHub releases"

# Without the marker, a missing manifest means the record step failed, and must
# not pass silently.
mkdir -p "$TEMP_ROOT/stable"
if PATH="$bin:$PATH" GH_LOG="$TEMP_ROOT/gh-stable.log" \
  bash "$SCRIPT" "$TEMP_ROOT/stable/sdk-memory-manifest.json" \
  >/dev/null 2>"$TEMP_ROOT/stable.err"; then
  fail "missing manifest without the skip marker must fail"
fi
grep -Fq 'manifest not found' "$TEMP_ROOT/stable.err" || fail "expected missing-manifest error"

# A manifest that exists is always published, even beside a stale marker: the
# skip only covers the no-manifest case.
present="$TEMP_ROOT/present"
mkdir -p "$present"
write_manifest "$present/sdk-memory-manifest.json"
: >"$present/sdk-memory-manifest.json.skipped"
GH_LOG="$TEMP_ROOT/gh-present.log"
rm -f "$GH_LOG"
PATH="$bin:$PATH" GH_LOG="$GH_LOG" \
  bash "$SCRIPT" "$present/sdk-memory-manifest.json" >/dev/null 2>&1 \
  || fail "present manifest must publish"
grep -Fq 'release create sdk-memory-v1.2.3' "$GH_LOG" || fail "present manifest was not published"

echo "publish-memory-sdk-release tests passed"
