import pytest

# Configure anyio for async tests
pytest_plugins = ("anyio",)


@pytest.fixture
def anyio_backend():
    """Pin async tests to asyncio backend only (trio is not a dependency)."""
    return "asyncio"
