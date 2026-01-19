"""
Pytest configuration and shared fixtures.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_directory():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_user_id():
    """Provide a sample user ID."""
    return "test_user_123"


@pytest.fixture
def sample_project_id():
    """Provide a sample project ID."""
    return "project_abc"
