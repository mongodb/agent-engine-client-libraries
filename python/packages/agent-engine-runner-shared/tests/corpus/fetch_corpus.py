"""Fetch the SHA-256-pinned OpenAPI acceptance corpus.

Documents come from a pinned commit of the APIs.guru GitHub repository rather
than api.apis.guru: the commit is immutable, and the fetch shares GitHub's
availability with CI itself instead of adding a third-party dependency.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

MANIFEST = Path(__file__).with_name("manifest.json")


def _git(cwd: Path, *args: str) -> None:
    env = dict(os.environ)
    # CI passes the job's GITHUB_TOKEN so the fetch is authenticated (public
    # repos are readable with any token); local runs fetch anonymously. The
    # header goes through env config, not argv, so a failed command's error
    # message cannot print the (base64, hence unmasked) credential.
    if token := env.get("GITHUB_TOKEN"):
        basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        env |= {
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
            "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {basic}",
        }
    subprocess.run(["git", *args], cwd=cwd, env=env, check=True)


def _checkout(repository: str, commit: str, paths: list[str], into: Path) -> None:
    _git(into, "init", "--quiet")
    _git(into, "remote", "add", "origin", repository)
    _git(into, "sparse-checkout", "set", "--no-cone", *paths)
    # Blob-less shallow fetch: checkout then downloads only the sparse paths.
    _git(into, "fetch", "--quiet", "--depth=1", "--filter=blob:none", "origin", commit)
    _git(into, "checkout", "--quiet", "FETCH_HEAD")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=MANIFEST.parent / "specs")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp)
        _checkout(
            manifest["source_repository"],
            manifest["source_commit"],
            [api["path"] for api in manifest["apis"]],
            source,
        )
        failures = 0
        for api in manifest["apis"]:
            try:
                body = (source / api["path"]).read_bytes()
                digest = hashlib.sha256(body).hexdigest()
                if digest != api["sha256"]:
                    raise ValueError(f"SHA-256 mismatch: expected {api['sha256']}, got {digest}")
                (args.out_dir / f"{api['id']}.json").write_bytes(body)
                print(f"{api['id']}: {len(body) // 1024} KiB")
            except Exception as exc:  # noqa: BLE001 - isolate and report every source
                failures += 1
                print(f"{api['id']}: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
