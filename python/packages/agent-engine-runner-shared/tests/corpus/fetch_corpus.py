"""Fetch the SHA-256-pinned OpenAPI acceptance corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

MAX_BYTES = 20 * 1024 * 1024
MANIFEST = Path(__file__).with_name("manifest.json")


def _fetch(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "MongoDB Agent Engine OpenAPI corpus"})
    with urlopen(request, timeout=120) as response:
        chunks: list[bytes] = []
        size = 0
        while chunk := response.read(64 * 1024):
            size += len(chunk)
            if size > MAX_BYTES:
                raise ValueError(f"document exceeds the {MAX_BYTES}-byte corpus cap")
            chunks.append(chunk)
    return b"".join(chunks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=MANIFEST.parent / "specs")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for api in manifest["apis"]:
        destination = args.out_dir / f"{api['id']}.json"
        try:
            body = _fetch(api["url"])
            digest = hashlib.sha256(body).hexdigest()
            if digest != api["sha256"]:
                raise ValueError(f"SHA-256 mismatch: expected {api['sha256']}, got {digest}")
            destination.write_bytes(body)
            print(f"{api['id']}: {len(body) // 1024} KiB")
        except Exception as exc:  # noqa: BLE001 - isolate and report every source
            failures += 1
            print(f"{api['id']}: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
