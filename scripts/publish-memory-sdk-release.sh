#!/usr/bin/env bash
# Publish the recorded memory SDK manifest as GitHub Release assets.
# uv has verified that the recorded wheel is on PyPI. It carries no download
# URL. An existing immutable release is left unchanged so a later publish
# cannot replace the hash recorded for that version. This script does not
# notify the canary: the published tree must not name the private repository.
set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }

manifest=${1:?manifest path required}

# record-memory-sdk-manifest.sh decides which publishes are not recorded: a
# pre-release, or a rebuild of a version PyPI already had. It leaves this
# marker instead of a manifest, so there is nothing to do. Any other missing
# manifest is still an error.
if [ ! -f "$manifest" ] && [ -f "${manifest}.skipped" ]; then
  echo "memory SDK publish was not recorded; leaving the recorded wheel unchanged"
  exit 0
fi
[ -f "$manifest" ] || die "manifest not found: $manifest"
[ "$(basename "$manifest")" = "sdk-memory-manifest.json" ] || die "manifest file must be named sdk-memory-manifest.json"

export GH_PROMPT_DISABLED=1
current="sdk-memory-current"
asset="sdk-memory-manifest.json"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
current_manifest="$scratch/current.json"
current_exists=false
if current_assets="$(gh release view "$current" --json assets --jq '.assets[].name')"; then
  current_exists=true
  if printf '%s\n' "$current_assets" | grep -qxF "$asset"; then
    gh release download "$current" --pattern "$asset" --output - > "$current_manifest"
  fi
fi

meta="$(python3 - "$manifest" "$current_manifest" <<'PY'
import json
import re
import sys
from pathlib import Path

def read_manifest(path):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or data.get("package") != "agent-engine-sdk-memory":
        sys.exit("manifest package is invalid")
    patterns = {"version": r"[0-9]+\.[0-9]+\.[0-9]+", "git_sha": r"[0-9a-f]{40}", "sha256": r"[0-9a-f]{64}"}
    for field, pattern in patterns.items():
        value = data.get(field)
        if not isinstance(value, str) or len(value) > 64 or re.fullmatch(pattern, value) is None:
            sys.exit(f"manifest {field} is invalid")
    return data

incoming = read_manifest(sys.argv[1])
action = "publish"
if Path(sys.argv[2]).exists():
    current = read_manifest(sys.argv[2])
    incoming_version = tuple(map(int, incoming["version"].split(".")))
    current_version = tuple(map(int, current["version"].split(".")))
    if incoming_version < current_version:
        action = "older"
    elif incoming_version == current_version:
        if incoming["sha256"] != current["sha256"]:
            sys.exit("current version already records a different wheel hash")
        action = "unchanged"
print(incoming["version"], incoming["git_sha"], action, sep="\n")
PY
)" || die "refusing to publish an invalid memory SDK manifest"
version="$(printf '%s\n' "$meta" | sed -n '1p')"
git_sha="$(printf '%s\n' "$meta" | sed -n '2p')"
action="$(printf '%s\n' "$meta" | sed -n '3p')"

# Existing version records are never overwritten. These releases share the
# CLI repository, so neither SDK tag may become GitHub Latest.
immutable="sdk-memory-v${version}"
if gh release view "$immutable" >/dev/null 2>&1; then
  # gh release create makes the release before uploading its asset, so a failed
  # upload leaves an empty release that this existence check would otherwise
  # skip forever. Attach the manifest only when it is missing; never replace one
  # that is already recorded.
  if gh release view "$immutable" --json assets --jq '.assets[].name' | grep -qxF "$asset"; then
    echo "release ${immutable} already exists; leaving the recorded wheel unchanged"
  else
    echo "release ${immutable} has no manifest; attaching it"
    gh release upload "$immutable" "$manifest"
  fi
else
  gh release create "$immutable" "$manifest" \
    --title "agent-engine-sdk-memory ${version}" \
    --notes "Recorded wheel for agent-engine-sdk-memory ${version}." \
    --target "$git_sha" --latest=false
fi

if [ "$action" = "older" ]; then
  echo "memory SDK ${version} is older than current; leaving the recorded wheel unchanged"
  exit 0
fi

# Avoid the replacement's 404 window on identical retries.
if [ "$action" = "unchanged" ]; then
  echo "memory SDK ${version} is already current; leaving the manifest unchanged"
elif [ "$current_exists" = "true" ]; then
  gh release upload "$current" "$manifest" --clobber
else
  gh release create "$current" "$manifest" \
    --title "agent-engine-sdk-memory current" \
    --notes "Latest recorded agent-engine-sdk-memory wheel." \
    --target "$git_sha" --latest=false
fi
