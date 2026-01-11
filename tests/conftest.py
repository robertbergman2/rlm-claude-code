"""
Pytest configuration and fixtures for RLM-Claude-Code tests.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path so we can import src package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.types import Message, MessageRole, SessionContext, ToolOutput
from src.config import RLMConfig


@pytest.fixture
def mock_context():
    """Provide a basic mock context."""
    return SessionContext(
        messages=[
            Message(role=MessageRole.USER, content="Hello"),
            Message(role=MessageRole.ASSISTANT, content="Hi there!"),
        ],
        files={"README.md": "# Test"},
        tool_outputs=[],
        working_memory={},
    )


@pytest.fixture
def mock_config():
    """Provide a basic mock config."""
    return RLMConfig()


@pytest.fixture
def large_context():
    """Provide a context with ~100K tokens of synthetic data."""
    files = {f"file_{i}.py": f"# Content {i}\n" * 1000 for i in range(50)}
    return SessionContext(
        messages=[
            Message(role=MessageRole.USER, content="Large context test"),
        ],
        files=files,
        tool_outputs=[],
        working_memory={},
    )


@pytest.fixture
def debug_context():
    """Provide a context with debugging scenario."""
    return SessionContext(
        messages=[
            Message(role=MessageRole.USER, content="Fix the test"),
        ],
        files={
            "src/auth/controller.py": "def login(): raise Error()",
            "src/auth/service.py": "def validate(): return False",
            "tests/auth.test.py": "def test_login(): assert response == 401",
        },
        tool_outputs=[
            ToolOutput(
                tool_name="bash",
                content="FAIL: Expected 401, got 500\nStack trace...",
                exit_code=1,
            )
        ],
        working_memory={},
    )


# Markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "hypothesis: property-based tests"
    )
    config.addinivalue_line(
        "markers", "slow: tests that take >1s"
    )
    config.addinivalue_line(
        "markers", "security: security-related tests"
    )
