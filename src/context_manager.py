"""
Context externalization for RLM-Claude-Code.

Implements: Spec §3 Context Externalization
"""

from typing import Any

from .types import Message, SessionContext, ToolOutput


def externalize_conversation(messages: list[Message]) -> list[dict[str, Any]]:
    """
    Externalize conversation history as REPL-accessible data.

    Implements: Spec §3.1 Context Variable Schema
    """
    return [
        {
            "role": m.role.value,
            "content": m.content,
            "timestamp": m.timestamp,
        }
        for m in messages
    ]


def externalize_files(files: dict[str, str]) -> dict[str, str]:
    """
    Externalize cached files as REPL-accessible data.

    Implements: Spec §3.1 Context Variable Schema
    """
    # Direct passthrough for now; could add metadata
    return files.copy()


def externalize_tool_outputs(outputs: list[ToolOutput]) -> list[dict[str, Any]]:
    """
    Externalize tool outputs as REPL-accessible data.

    Implements: Spec §3.1 Context Variable Schema
    """
    return [
        {
            "tool": o.tool_name,
            "content": o.content,
            "exit_code": o.exit_code,
            "timestamp": o.timestamp,
        }
        for o in outputs
    ]


def externalize_context(context: SessionContext) -> dict[str, Any]:
    """
    Externalize complete session context for REPL.

    Implements: Spec §3 Context Externalization
    """
    return {
        "conversation": externalize_conversation(context.messages),
        "files": externalize_files(context.files),
        "tool_outputs": externalize_tool_outputs(context.tool_outputs),
        "working_memory": context.working_memory.copy(),
        "context_stats": {
            "total_tokens": context.total_tokens,
            "conversation_count": len(context.messages),
            "file_count": len(context.files),
            "tool_output_count": len(context.tool_outputs),
        },
    }


__all__ = [
    "externalize_context",
    "externalize_conversation",
    "externalize_files",
    "externalize_tool_outputs",
]
