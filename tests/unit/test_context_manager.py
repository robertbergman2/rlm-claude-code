"""
Unit tests for context_manager module.

Implements: Spec §3 tests
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.types import Message, MessageRole, SessionContext, ToolOutput
from src.context_manager import (
    externalize_context,
    externalize_conversation,
    externalize_files,
    externalize_tool_outputs,
)


class TestExternalizeConversation:
    """Tests for externalize_conversation function."""

    def test_empty_conversation(self):
        """Empty message list returns empty list."""
        result = externalize_conversation([])

        assert result == []

    def test_single_message(self):
        """Single message is externalized correctly."""
        messages = [Message(role=MessageRole.USER, content="Hello")]

        result = externalize_conversation(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"
        assert result[0]["timestamp"] is None

    def test_multiple_messages(self):
        """Multiple messages maintain order."""
        messages = [
            Message(role=MessageRole.USER, content="Question"),
            Message(role=MessageRole.ASSISTANT, content="Answer"),
            Message(role=MessageRole.USER, content="Follow-up"),
        ]

        result = externalize_conversation(messages)

        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert result[2]["content"] == "Follow-up"

    def test_preserves_timestamp(self):
        """Timestamp is preserved in output."""
        messages = [Message(role=MessageRole.USER, content="Hi", timestamp=12345.0)]

        result = externalize_conversation(messages)

        assert result[0]["timestamp"] == 12345.0


class TestExternalizeFiles:
    """Tests for externalize_files function."""

    def test_empty_files(self):
        """Empty dict returns empty dict."""
        result = externalize_files({})

        assert result == {}

    def test_single_file(self):
        """Single file is copied correctly."""
        files = {"test.py": "print('hello')"}

        result = externalize_files(files)

        assert result == {"test.py": "print('hello')"}

    def test_files_are_copied(self):
        """Files dict is copied, not referenced."""
        files = {"test.py": "code"}

        result = externalize_files(files)
        result["new.py"] = "new code"

        assert "new.py" not in files


class TestExternalizeToolOutputs:
    """Tests for externalize_tool_outputs function."""

    def test_empty_outputs(self):
        """Empty list returns empty list."""
        result = externalize_tool_outputs([])

        assert result == []

    def test_single_output(self):
        """Single output is externalized correctly."""
        outputs = [ToolOutput(tool_name="bash", content="output", exit_code=0)]

        result = externalize_tool_outputs(outputs)

        assert len(result) == 1
        assert result[0]["tool"] == "bash"
        assert result[0]["content"] == "output"
        assert result[0]["exit_code"] == 0

    def test_multiple_outputs(self):
        """Multiple outputs maintain order."""
        outputs = [
            ToolOutput(tool_name="bash", content="first"),
            ToolOutput(tool_name="read", content="second"),
        ]

        result = externalize_tool_outputs(outputs)

        assert len(result) == 2
        assert result[0]["tool"] == "bash"
        assert result[1]["tool"] == "read"


class TestExternalizeContext:
    """Tests for externalize_context function."""

    def test_empty_context(self):
        """Empty context produces expected structure."""
        ctx = SessionContext()

        result = externalize_context(ctx)

        assert "conversation" in result
        assert "files" in result
        assert "tool_outputs" in result
        assert "working_memory" in result
        assert "context_stats" in result

    def test_context_stats(self):
        """Context stats are computed correctly."""
        ctx = SessionContext(
            messages=[Message(role=MessageRole.USER, content="test")],
            files={"a.py": "code", "b.py": "more"},
            tool_outputs=[ToolOutput(tool_name="bash", content="out")],
        )

        result = externalize_context(ctx)
        stats = result["context_stats"]

        assert stats["conversation_count"] == 1
        assert stats["file_count"] == 2
        assert stats["tool_output_count"] == 1
        assert stats["total_tokens"] > 0

    def test_working_memory_copied(self):
        """Working memory is copied, not referenced."""
        ctx = SessionContext(working_memory={"key": "value"})

        result = externalize_context(ctx)
        result["working_memory"]["new_key"] = "new_value"

        assert "new_key" not in ctx.working_memory

    def test_full_context_externalization(self):
        """Full context is externalized correctly."""
        ctx = SessionContext(
            messages=[
                Message(role=MessageRole.USER, content="Hello"),
                Message(role=MessageRole.ASSISTANT, content="Hi"),
            ],
            files={"main.py": "print('hello')"},
            tool_outputs=[ToolOutput(tool_name="bash", content="Hello")],
            working_memory={"state": "active"},
        )

        result = externalize_context(ctx)

        assert len(result["conversation"]) == 2
        assert "main.py" in result["files"]
        assert len(result["tool_outputs"]) == 1
        assert result["working_memory"]["state"] == "active"
