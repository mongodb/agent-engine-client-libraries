"""Argv entrypoint for the ``agentengine create-tool`` helper container.

The Go CLI runs this module inside the runner-base helper container and
parses one sentinel-prefixed JSON result line from stdout. It owns no
policy: every decision lives in :mod:`agent_engine_runner_shared.connectors.generate`;
this module only translates argv into library calls and results into the
container's output contract.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from pydantic import ValidationError

from agent_engine_runner_shared.connectors.generate import (
    GenerateError,
    compile_catalog,
    regenerate_catalog,
    scaffold_tool_yaml,
)
from agent_engine_runner_shared.connectors.generate.authoring import AUTH_CHOICES

__all__ = ["RESULT_PREFIX", "main"]

RESULT_PREFIX = "__AGENTIC_CREATE_TOOL_RESULT__ "

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_HELPER = 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="create-tool",
        description="Compile OpenAPI documents into connector authoring files.",
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--spec", help="OpenAPI 3.0 document to compile into a complete catalog")
    modes.add_argument(
        "--tool-defs", help="Existing local catalog to scaffold an authoring file for"
    )
    modes.add_argument(
        "--config",
        help="Existing tool.yaml that declares source.spec; regenerates its catalog",
    )
    parser.add_argument(
        "name",
        nargs="?",
        help="Connector name; required with --spec, defaults to the catalog's with --tool-defs",
    )
    parser.add_argument("--base-url", help="Tenant API base URL recorded in the authoring file")
    parser.add_argument(
        "--auth",
        choices=AUTH_CHOICES,
        help="Credential to present instead of the one the document proposes",
    )
    parser.add_argument("--provider", help="Project OAuth app whose token each call presents")
    parser.add_argument(
        "--scopes",
        help="Comma-separated OAuth scopes (with --provider) instead of the document's",
    )
    parser.add_argument("--allow", help="Comma-separated operation allow list (default: allow_all)")
    parser.add_argument(
        "--out-dir",
        help="Connector folder for the generated files (required with --spec/--tool-defs)",
    )
    parser.add_argument(
        "--skip-unsupported",
        action="store_true",
        help="Exclude operations the preview cannot represent instead of failing; "
        "every skip is printed and recorded in the catalog's provenance",
    )
    return parser


def _scopes(raw: str | None) -> list[str] | None:
    """Split ``--scopes``; omitted means the document's scopes. An empty entry
    is refused rather than dropped, so a typo cannot silently narrow the grant."""
    if raw is None:
        return None
    scopes = [scope.strip() for scope in raw.split(",")]
    if not all(scopes):
        raise GenerateError("--scopes must not contain empty entries")
    return scopes


def _allow(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    names = [name.strip() for name in raw.split(",") if name.strip()]
    if not names:
        raise GenerateError("--allow must name at least one operation")
    return names


def _result_line(paths: list[str]) -> str:
    """Render the result line with workspace-relative paths so the CLI can
    display real host files without accepting helper-owned spellings."""
    relative: list[str] = []
    for path in paths:
        spelled = os.path.relpath(path, os.getcwd())
        # Refuse true parent escapes only; a contained directory whose name
        # merely starts with ".." (e.g. "..generated") is legal.
        if spelled == ".." or spelled.startswith("../"):
            raise GenerateError(f"generated {path} is outside the workspace")
        relative.append(spelled)
    return RESULT_PREFIX + json.dumps({"written": relative})


def main(argv: list[str] | None = None) -> int:
    # argparse usage errors are input validation, not infrastructure: map the
    # SystemExit they raise to the generation-refusal exit code (a --help
    # request still exits 0). Without this, a typo'd flag value would be
    # reported as argparse's own status 2 — the helper-failure code.
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exc:
        return EXIT_OK if exc.code in (None, 0) else EXIT_FAILED
    skipped: tuple[str, ...] = ()
    try:
        scopes = _scopes(args.scopes)
        allow = _allow(args.allow)
        if args.config is not None:
            if (
                args.name is not None
                or args.auth
                or args.provider is not None
                or args.scopes is not None
                or args.allow is not None
                or args.base_url
                or args.out_dir
            ):
                raise GenerateError(
                    "--config regeneration keeps the name and settings written in "
                    "tool.yaml and writes beside it; a name, --auth, --provider, "
                    "--scopes, --allow, --base-url and --out-dir do not apply"
                )
            result = regenerate_catalog(args.config, skip_unsupported=args.skip_unsupported)
            written = [str(result.catalog)]
            skipped = result.skipped
        elif args.spec is not None:
            if not args.name:
                raise GenerateError("a connector name is required with --spec")
            if not args.out_dir:
                raise GenerateError("--out-dir is required with --spec")
            # --auth and --provider override the document's own security (e.g.
            # the official Jira document declares basic; a DC PAT is Bearer).
            result = compile_catalog(
                args.spec,
                name=args.name,
                base_url=args.base_url,
                auth=args.auth,
                provider=args.provider,
                scopes=scopes,
                allow=allow,
                skip_unsupported=args.skip_unsupported,
                out_dir=args.out_dir,
            )
            written = [str(result.catalog)]
            if result.tool_yaml is not None:
                written.append(str(result.tool_yaml))
            skipped = result.skipped
        else:
            if not args.base_url:
                raise GenerateError("--base-url is required with --tool-defs")
            if not args.out_dir:
                raise GenerateError("--out-dir is required with --tool-defs")
            if args.skip_unsupported:
                # Scaffolding references an existing catalog unchanged; there
                # is nothing to skip, so the requested behavior would be a
                # silent no-op. Refuse instead.
                raise GenerateError(
                    "--skip-unsupported applies to --spec compilation; "
                    "scaffolding references an existing catalog unchanged"
                )
            written = [
                str(
                    scaffold_tool_yaml(
                        args.tool_defs,
                        name=args.name,
                        base_url=args.base_url,
                        auth=args.auth,
                        provider=args.provider,
                        scopes=scopes,
                        allow=allow,
                        out_dir=args.out_dir,
                    )
                )
            ]
    except (GenerateError, ValidationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILED
    except Exception as exc:  # noqa: BLE001 - the container owns the contract
        # Operational failures (filesystem, environment) are helper-class:
        # retryable, distinct from input refusals. The message is sanitized
        # by the generator's quoting discipline; never include a traceback.
        print(f"error: {exc}".replace("\n", " "), file=sys.stderr)
        return EXIT_HELPER
    for skip in skipped:
        print(f"skipped: {skip}", file=sys.stderr)
    print(_result_line(written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
