"""How the adapter reacts when an installed google-adk differs from tested ones.

Applications choose their own google-adk release, so the adapter cannot assume
the private ADK internals it patches keep their shape. Two policies apply:

- A durable mechanism a feature cannot run without refuses only applications
  that use that feature, before the Runner starts, naming the tested range.
- A workaround for an ADK bug steps aside, leaving stock ADK behavior, and
  logs one warning. The bugs it covers occur only in rare topologies.
"""

from __future__ import annotations

import logging
import re
from functools import cache
from importlib.metadata import PackageNotFoundError, version

from agent_engine_sdk_adk.errors import UnsupportedDurableADKError

__all__ = [
    "TESTED_ADK_MAX",
    "TESTED_ADK_MIN",
    "adk_release_before",
    "unsupported_adk_feature",
    "warn_workaround_inactive",
]

# The minimum-version suite and the package lock; a unit test keeps them equal.
TESTED_ADK_MIN = "2.9.0"
TESTED_ADK_MAX = "2.10.0"

_logger = logging.getLogger(__name__)
_inactive_workarounds: set[str] = set()


@cache
def _installed_adk_version() -> str:
    try:
        return version("google-adk")
    except PackageNotFoundError:
        return "unknown"


def adk_release_before(major: int, minor: int) -> bool | None:
    """Whether the installed google-adk is older than ``major.minor``.

    ``None`` when the installed version is unknown or unparsable, such as a
    vendored ADK without package metadata.
    """
    match = re.match(r"(\d+)\.(\d+)", _installed_adk_version())
    if match is None:
        return None
    return (int(match[1]), int(match[2])) < (major, minor)


def _tested_range() -> str:
    return (
        f"google-adk {_installed_adk_version()} is installed; this SDK is tested "
        f"with google-adk {TESTED_ADK_MIN} through {TESTED_ADK_MAX}"
    )


def unsupported_adk_feature(
    feature: str, *, alternative: str | None = None
) -> UnsupportedDurableADKError:
    """Build the refusal for a feature whose ADK internals changed shape.

    ``feature`` names what the application uses, such as
    ``"parallel_worker nodes"``; ``alternative`` optionally names a supported
    shape that avoids the feature.
    """
    remedy = f"pin google-adk<={TESTED_ADK_MAX}"
    if alternative:
        remedy += f", or {alternative}"
    return UnsupportedDurableADKError(
        f"Google ADK durable workflows do not support {feature} on this ADK "
        f"release: the ADK interfaces the adapter needs for it changed. "
        f"{_tested_range()}. To resolve, {remedy}."
    )


def warn_workaround_inactive(name: str, consequence: str) -> None:
    """Log once per process that an ADK bug workaround is not applied.

    ``consequence`` describes the stock ADK behavior applications now get.
    """
    if name in _inactive_workarounds:
        return
    _inactive_workarounds.add(name)
    _logger.warning(
        "Agent Engine's workaround for an ADK bug (%s) is inactive: %s. "
        "Durable runs continue with stock ADK behavior: %s. To restore it, "
        "pin google-adk<=%s.",
        name,
        _tested_range(),
        consequence,
        TESTED_ADK_MAX,
    )
