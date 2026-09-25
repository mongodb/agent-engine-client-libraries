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

from agent_engine_runner_shared.connectors.definitions import Auth
from agent_engine_runner_shared.connectors.generate import (
    GenerateError,
    compile_catalog,
    regenerate_catalog,
    scaffold_tool_yaml,
)

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
    parser.add_argument("--name", help="Connector name (compile mode)")
    parser.add_argument("--base-url", help="Tenant API base URL recorded in the authoring file")
    parser.add_argument(
        "--auth",
        choices=("none", "bearer", "api_key"),
        help="Credential presentation; the authority over the spec's own security schemes",
    )
    parser.add_argument(
        "--auth-env", help="Workspace secret environment reference for the credential"
    )
    parser.add_argument("--auth-header", help="Header name for API key credentials")
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


def _auth(args: argparse.Namespace) -> Auth | None:
    if args.auth == "none":
        if args.auth_env or args.auth_header:
            raise GenerateError("--auth none cannot take --auth-env or --auth-header")
        return Auth(type="none")
    if args.auth is None:
        if args.auth_env or args.auth_header:
            raise GenerateError("--auth-env and --auth-header require --auth")
        return None
    if not args.auth_env:
        raise GenerateError(f"--auth {args.auth} requires --auth-env")
    return Auth(type=args.auth, env=args.auth_env, header=args.auth_header)


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
        auth = _auth(args)
        allow = _allow(args.allow)
        if args.config is not None:
            if (
                args.auth
                or args.auth_env
                or args.auth_header
                or args.allow is not None
                or args.base_url
                or args.name
                or args.out_dir
            ):
                raise GenerateError(
                    "--config regeneration preserves the authored settings from "
                    "tool.yaml and writes beside it; --auth, --auth-env, "
                    "--auth-header, --allow, --base-url, --name and --out-dir do not apply"
                )
            result = regenerate_catalog(args.config, skip_unsupported=args.skip_unsupported)
            written = [str(result.catalog)]
            skipped = result.skipped
        elif args.spec is not None:
            if not args.name:
                raise GenerateError("--name is required with --spec")
            if not args.out_dir:
                raise GenerateError("--out-dir is required with --spec")
            # The consumer's --auth is the authority over the spec's own
            # security-scheme definitions: the generator derives the catalog
            # override from the credential the developer chose (e.g. the
            # official Jira document declares basic; a DC PAT is Bearer).
            # An explicit Auth(type="none") is the anonymous override.
            result = compile_catalog(
                args.spec,
                name=args.name,
                base_url=args.base_url,
                auth=auth,
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
                        base_url=args.base_url,
                        auth=auth,
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
