.PHONY: docs docs-check

# Fail the recipe if any stage of a pipe fails; otherwise a pydoc-markdown
# error is masked by the succeeding perl stage, truncating docs/api.md while
# `make docs` still reports success (and docs-check then passes empty-vs-empty).
SHELL := /usr/bin/env bash -o pipefail
.SHELLFLAGS := -ec

# pydoc-markdown transitively pins docstring-parser<0.12, which conflicts
# with deepagents' langchain-anthropic (docstring-parser>=0.15). Run it as
# an isolated `uv tool` so it never enters the project environment.
PYDOC_MARKDOWN := uv tool run --from 'pydoc-markdown>=4.8' pydoc-markdown
# pydoc-markdown emits trailing spaces on blank lines and an extra trailing
# blank line; strip both so docs-check only fails on content changes.
NORMALIZE_DOCS := perl -0pe 's/[^\S\n]+$$//mg; s/\n+\z/\n/'

# Output path for the generated reference. Defaults to the committed file;
# override (e.g. DOCS_OUT=/tmp/api.md) to generate elsewhere without clobbering
# it — used by the sdk-docs-internal-refs validation flow.
DOCS_OUT ?= docs/api.md

# Regenerate API reference from docstrings.
docs:
	$(PYDOC_MARKDOWN) pydoc-markdown.yml | $(NORMALIZE_DOCS) > $(DOCS_OUT)

# CI: fail if docs/api.md is stale.
docs-check:
	@$(PYDOC_MARKDOWN) pydoc-markdown.yml | $(NORMALIZE_DOCS) > /tmp/api-docs-check.md
	@diff -u docs/api.md /tmp/api-docs-check.md || { \
		echo ""; \
		echo "ERROR: docs/api.md is stale. Run 'make docs' to regenerate."; \
		exit 1; \
	}
	@echo "API docs are up to date."
