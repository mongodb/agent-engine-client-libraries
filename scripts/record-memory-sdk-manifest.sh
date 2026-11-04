#!/usr/bin/env bash
# Hash the memory SDK wheel this build produced and write sdk-memory-manifest.json.
# The canary downloads that exact PyPI file and checks the hash. The manifest
# carries no download URL. Call this only after a real publish to public PyPI:
# a dry run must not get here. A pre-release is published but not recorded.
# The canary installs the recorded wheel as its latest, and pre-releases stay
# out of that channel. The publish step skips a version PyPI already has, so a
# rebuild of that version is not recorded either: its hash need not match the
# file on PyPI. The marker file tells the record workflow either skip was
# deliberate, so the decision lives only here.
set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }

DIST_DIR="dist"
PACKAGES_ROOT="."
while [ "$#" -gt 0 ]; do
  case "$1" in
    --dist-dir) DIST_DIR=${2:?}; shift 2 ;;
    --packages-root) PACKAGES_ROOT=${2:?}; shift 2 ;;
    -h|--help)
      sed -n '2,10p' "$0"
      exit 0
      ;;
    *) die "unknown argument: $1" ;;
  esac
done

PACKAGES_ROOT="$(cd "$PACKAGES_ROOT" && pwd)"
DIST_DIR="$(cd "$DIST_DIR" && pwd)"
pyproject="$PACKAGES_ROOT/python/packages/agent-engine-sdk-memory/pyproject.toml"
[ -f "$pyproject" ] || die "missing $pyproject"

version="$(sed -n 's/^version = "\(.*\)"/\1/p' "$pyproject" | head -n1)"
[ -n "$version" ] || die "no version in $pyproject"

# pyproject carries the semver release version; the wheel filename uses its
# PEP 440 form (0.11.7-alpha.0 -> 0.11.7a0). A stable version is x.y.z on
# both. Duplicated from the monorepo's scripts/release_versions.py because
# only this tree is synced.
if [[ "$version" =~ ^([0-9]+\.[0-9]+\.[0-9]+)(-alpha\.([0-9]+))?$ ]]; then
  pypi_version="${BASH_REMATCH[1]}${BASH_REMATCH[3]:+a${BASH_REMATCH[3]}}"
else
  die "$pyproject has an unsupported version: $version"
fi

workspace="${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is not set}"
manifest="$workspace/sdk-memory-manifest.json"
marker="$workspace/sdk-memory-manifest.json.skipped"

if [ "$pypi_version" != "$version" ]; then
  echo "not recording pre-release memory SDK ${pypi_version}; the canary keeps its recorded wheel"
  rm -f "$manifest"
  : > "$marker"
  exit 0
fi

wheel="$DIST_DIR/agent_engine_sdk_memory-${version}-py3-none-any.whl"
[ -f "$wheel" ] || die "memory SDK wheel not found at ${wheel}"
filename="$(basename "$wheel")"
built_sha256="$(python3 -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$wheel")"

# PyPI's JSON API can lag a fresh upload, so a missing file is retried before
# it counts as a failed publish.
pypi_json="$(mktemp)"
trap 'rm -f "$pypi_json"' EXIT
attempts="${PYPI_LOOKUP_ATTEMPTS:-6}"
pypi_sha256=""
for attempt in $(seq 1 "$attempts"); do
  if curl -fsSL --max-time 30 -o "$pypi_json" \
    "https://pypi.org/pypi/agent-engine-sdk-memory/${version}/json"; then
    pypi_sha256="$(python3 - "$pypi_json" "$filename" <<'PY'
import json
import sys

path, filename = sys.argv[1:3]
for entry in json.load(open(path, encoding="utf-8")).get("urls", []):
    if entry.get("filename") == filename:
        print(entry.get("digests", {}).get("sha256", ""))
        break
PY
)" || pypi_sha256=""
    [ -z "$pypi_sha256" ] || break
  fi
  [ "$attempt" -eq "$attempts" ] || sleep "${PYPI_LOOKUP_DELAY:-10}"
done
[ -n "$pypi_sha256" ] || die "${filename} is not on PyPI after publishing"

if [ "$pypi_sha256" != "$built_sha256" ]; then
  echo "PyPI already has a different ${filename}; this run published nothing new, so the canary keeps its recorded wheel"
  rm -f "$manifest"
  : > "$marker"
  exit 0
fi

git_sha="$(git -C "$PACKAGES_ROOT" rev-parse HEAD)"
python3 - "$wheel" "$version" "$git_sha" "$manifest" "$built_sha256" <<'PY'
import json
import os
import sys

wheel, version, git_sha, out, digest = sys.argv[1:6]
manifest = {
    "filename": os.path.basename(wheel),
    "git_sha": git_sha,
    "package": "agent-engine-sdk-memory",
    "sha256": digest,
    "version": version,
}
tmp = out + ".tmp"
with open(tmp, "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(tmp, out)
PY
rm -f "$marker"
echo "recorded memory SDK manifest ${version} at ${manifest}"
