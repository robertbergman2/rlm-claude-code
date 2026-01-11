"""
RLM-Claude-Code: Recursive Language Model integration for Claude Code.

Transform Claude Code into an RLM agent for unbounded context handling
and improved reasoning via REPL-based context decomposition.
"""

__version__ = "0.1.0"

from .complexity_classifier import should_activate_rlm
from .context_manager import externalize_context
from .orchestrator import RLMOrchestrator
from .repl_environment import RLMEnvironment
from .trajectory import TrajectoryEvent, TrajectoryRenderer
from .types import SessionContext

__all__ = [
    "RLMOrchestrator",
    "SessionContext",
    "externalize_context",
    "RLMEnvironment",
    "should_activate_rlm",
    "TrajectoryEvent",
    "TrajectoryRenderer",
]
