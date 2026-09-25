"""Read request-scoped secrets from the fctr metadata directory.

In RPC mode the OE delivers per-tool secrets via ``SetMetadata`` before each
tool call and clears them after (when restriction is enabled). fctr
materializes the keyset as files at ``/run/meta/<KEY>``. This module reads
those files so the tool server can inject them into ``os.environ`` for the
duration of the call — or permanently when restriction is disabled.
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Generator

logger = logging.getLogger(__name__)

_METADATA_DIR = "/run/meta"

# Overlap accounting for applied-env regions. os.environ is process-global:
# if a region overlaps one that has secrets applied, its snapshot captures
# those secrets as baseline and its restore re-installs them permanently
# while deleting the other call's live ones. The ToolServer's
# per-instance _execute_gate serializes its own routes, but that
# invariant lives in the caller — these counters make the module itself fail
# closed (raise before any env mutation) if any future call site overlaps a
# secret-bearing region. Secret-less overlap stays allowed: the snapshots
# are then identical to the live env and restore is idempotent. Mirrors the
# TypeScript agent_engine_runner_shared.server.metadata guard.
_active_regions = 0
_active_secret_regions = 0
# Guards the admission check, counters, and env mutation across threads —
# ToolServer instances are not confined to one thread.
_region_lock = threading.Lock()


def read_metadata_secrets(metadata_dir: str = _METADATA_DIR) -> Dict[str, str]:
    """Read all key-value pairs from the metadata directory.

    Returns a dict mapping filename (the secret name) to file content (the
    secret value). Skips files that cannot be read and logs a warning.
    Returns an empty dict when the directory does not exist.
    """
    meta_path = Path(metadata_dir)
    if not meta_path.is_dir():
        return {}

    secrets: Dict[str, str] = {}
    for entry in meta_path.iterdir():
        if not entry.is_file():
            continue
        try:
            secrets[entry.name] = entry.read_text()
        except OSError:
            logger.warning("failed to read metadata file %s", entry, exc_info=True)
    return secrets


def merge_metadata_env(metadata_dir: str = _METADATA_DIR) -> None:
    """Overwrite ``os.environ`` from ``/run/meta``; no restore or prune."""
    secrets = read_metadata_secrets(metadata_dir)
    if not secrets:
        return
    os.environ.update(secrets)
    logger.debug(
        "merged %d metadata secret(s) into os.environ (no restore)",
        len(secrets),
    )


@contextmanager
def applied_metadata_env(
    metadata_dir: str = _METADATA_DIR,
) -> Generator[None, None, None]:
    """Context manager that loads metadata secrets into ``os.environ``, then restores it.

    On entry: snapshots ``os.environ``, reads ``metadata_dir``, and merges
    the secrets in.

    On exit the restore depends on how the region was entered:

    * *clean* (no other region was active — the serialized production flow):
      full snapshot restore. Keys the region added are removed, overwritten
      keys are reverted, and keys the body deleted are reinstated.
    * *tainted* (entered while another, necessarily secret-less, region was
      active): delete-only restore. Keys added since the snapshot are removed,
      but no values are written back, since the snapshot holds the other
      region's request-scoped state and rewriting it could outlive that
      region's own restore. A tainted body's overwrite of a pre-existing key
      therefore survives the region.

    Raises:
        RuntimeError: if the region overlaps a concurrent secret-bearing
            region — either another region already has secrets applied, or
            this region carries secrets while any region is active. Raised
            before ``os.environ`` is touched (fail closed), so
            callers see a per-request error instead of a silent cross-request
            credential swap. Production call sites (``/execute``,
            ``/invoke_llm``, ``/invoke_llm/stream``) are serialized by the
            ToolServer's per-instance execute gate and do not hit this.
    """
    global _active_regions, _active_secret_regions

    secrets = read_metadata_secrets(metadata_dir)

    # The admission check, snapshot, env mutation, and counter updates form
    # one critical section: ToolServer instances may run on different
    # threads, and without the lock two regions could both pass the check
    # (or one could snapshot secrets the other is mid-way through applying
    # or restoring). Restore likewise runs under the lock, *before* the
    # counters are decremented, so no thread can be admitted while another
    # region's secrets are still live.
    with _region_lock:
        if _active_secret_regions > 0 or (secrets and _active_regions > 0):
            # Fail closed before mutating anything: proceeding would snapshot
            # or clobber another request's live secrets.
            raise RuntimeError(
                "metadata env region overlaps a concurrent secret-bearing region; "
                "refusing to apply request secrets to the shared os.environ"
            )

        snapshot = os.environ.copy()
        # A region that entered on a quiet environment owns the true
        # baseline: its restore is a full snapshot restore — deleting added
        # keys, reverting overwritten ones, and reinstating keys the callback
        # deleted — the documented contract for the serialized (production)
        # flow. A region that entered while another was active holds a
        # snapshot tainted with that region's request-scoped state: writing
        # any value from it could permanently reassert the other request's
        # data after its restore, so a tainted restore only deletes the keys
        # added since its snapshot and never writes values. (Tainted regions
        # are secret-less by the guard above, so at worst a tainted
        # callback's own overwrite of a pre-existing key outlives it.)
        entered_clean = _active_regions == 0

        def _restore_env() -> None:
            for key in set(os.environ.keys()) - set(snapshot.keys()):
                # pop(..., None) rather than del: the key set is computed
                # before the removals, and the region lock only serializes
                # other metadata regions — arbitrary tool/user code on another
                # thread may delete an os.environ key in between. A KeyError
                # here would escape before the counters are decremented and
                # leave the process permanently fail-closed.
                os.environ.pop(key, None)
            if entered_clean:
                os.environ.update(snapshot)

        try:
            os.environ.update(secrets)
        except Exception:
            # os.environ rejects some values (e.g. containing NUL) and
            # update() may have applied a prefix of the keys before raising:
            # roll the partial application back, and only count the region
            # once the apply succeeded — otherwise the counters stay
            # unbalanced and every later region rejects.
            _restore_env()
            raise
        _active_regions += 1
        if secrets:
            _active_secret_regions += 1
            logger.debug("applied %d metadata secret(s) to os.environ", len(secrets))

    try:
        yield
    finally:
        with _region_lock:
            # The decrements are unconditional: if _restore_env() ever raises,
            # skipping them would leave _active_secret_regions elevated for the
            # life of the process and every later /execute and /invoke_llm on
            # the pod would hit the fail-closed RuntimeError until restart.
            # (The TS side is structurally immune — it decrements first and
            # `delete` never throws.)
            try:
                _restore_env()
            finally:
                _active_regions -= 1
                if secrets:
                    _active_secret_regions -= 1
