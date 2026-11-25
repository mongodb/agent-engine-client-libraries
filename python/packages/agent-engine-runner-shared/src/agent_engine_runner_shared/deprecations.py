"""Warnings for SDK arguments the runner no longer honors."""

from __future__ import annotations

import warnings

_REMOVED_TRACE_OPTIONS = ("database_name", "traces_collection_name")


def warn_trace_options_removed(
    database_name: object | None = None,
    traces_collection_name: object | None = None,
) -> None:
    """Warn when a caller passes a trace-storage option the runner ignores.

    The adapters still accept these arguments so an agent built against an
    earlier release keeps starting; they have no effect because traces now go
    only to the Orchestration Engine.
    """
    for argument, value in zip(
        _REMOVED_TRACE_OPTIONS, (database_name, traces_collection_name), strict=True
    ):
        if value is not None:
            warnings.warn(
                f"App({argument}=...) is deprecated and ignored. Traces go only "
                "to the Orchestration Engine; this argument will be removed in "
                "a future release.",
                DeprecationWarning,
                stacklevel=3,
            )
