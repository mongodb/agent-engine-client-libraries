#!/usr/bin/env bash
#
# Generate the TypeScript SDK Markdown API reference. Each package's reference is
# written to <out>/<package-dir>/docs/api.md (mirroring the Python SDK layout);
# <out> defaults to packages/, i.e. the committed location.
#
# Callers must build the workspace packages first (npm run build --workspaces)
# so cross-package @mongodb-js/* references resolve against the sibling dist/*.d.ts
# instead of degrading to unresolved types. The `docs` npm script does this; the
# scripts/test.sh drift gate reuses the build the sdk-typescript suite already ran.
#
# Output is normalized (trailing whitespace stripped, single trailing newline) so
# the committed tree is byte-identical across runs and machines. Determinism also
# depends on typedoc.json's disableSources (no per-commit source links) and
# hideGenerator (no version footer).
set -euo pipefail

cd "$(dirname "$0")/.."

out="${1:-packages}"

staging="$(mktemp -d)"
trap 'rm -rf "$staging"' EXIT

npx typedoc --out "$staging"

# Move TypeDoc's per-entry `src` pages to <package-dir>/docs/api.md, inline
# namespaces, add a table of contents, and fix up the relative links.
node scripts/restructure-docs.mjs "$staging"

# Mirrors the Python pydoc-markdown normalizer (agent-engine-sdk-langgraph/Makefile).
find "$staging" -type f -name '*.md' -exec \
  perl -0pi -e 's/[^\S\n]+$//mg; s/\n+\z/\n/' {} +

for page in "$staging"/*/docs/api.md; do
  rel="${page#"$staging"/}"
  mkdir -p "$out/$(dirname "$rel")"
  cp "$page" "$out/$rel"
done
