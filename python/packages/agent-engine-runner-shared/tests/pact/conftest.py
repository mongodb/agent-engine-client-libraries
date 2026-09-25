"""Skip the pact provider-verification suite unless pact-python is installed.

pact-python is intentionally NOT a committed dependency: its platform-specific
FFI wheel (pact-python-ffi) would churn the shared client-libraries workspace
lock. Run these tests with an ephemeral install instead:

    uv run --with pact-python python -m pytest tests/pact/

Under a plain ``uv run pytest`` (no pact-python) this conftest skips the whole
directory at collection time.
"""

import pytest

pytest.importorskip(
    "pact",
    reason="pact-python not installed; run: uv run --with pact-python python -m pytest tests/pact/",
)
