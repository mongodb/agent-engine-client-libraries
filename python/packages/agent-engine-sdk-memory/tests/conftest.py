from unittest.mock import patch

import pytest


@pytest.fixture
def mock_httpx_client():
    """Mock httpx.Client for testing."""
    with patch("agent_engine_sdk_memory._client.httpx.Client") as mock:
        yield mock.return_value
