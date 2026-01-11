"""
Unit tests for complexity classifier.

Implements: Spec §6.3 tests
"""

import pytest
import sys
from pathlib import Path

# Add project root to path so we can import src package
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.complexity_classifier import (
    extract_complexity_signals,
    should_activate_rlm,
    is_definitely_simple,
)


class TestExtractComplexitySignals:
    """Tests for extract_complexity_signals function."""
    
    def test_detects_multi_file_reference(self, mock_context):
        """Prompt mentioning multiple files sets references_multiple_files."""
        prompt = "Fix the bug in auth.py and update tests.py"
        
        signals = extract_complexity_signals(prompt, mock_context)
        
        assert signals.references_multiple_files is True
    
    def test_detects_cross_context_reasoning(self, mock_context):
        """'Why X given Y' patterns trigger cross_context_reasoning."""
        prompt = "Why is the test failing given the fix we made?"
        
        signals = extract_complexity_signals(prompt, mock_context)
        
        assert signals.requires_cross_context_reasoning is True
    
    def test_detects_debugging_task(self, mock_context):
        """Error-related keywords trigger debugging_task."""
        prompt = "There's a bug causing the authentication to crash"
        
        signals = extract_complexity_signals(prompt, mock_context)
        
        assert signals.debugging_task is True
    
    def test_detects_pattern_search(self, mock_context):
        """'Find all' patterns trigger asks_about_patterns."""
        prompt = "Find all places where we call the API"
        
        signals = extract_complexity_signals(prompt, mock_context)
        
        assert signals.asks_about_patterns is True
    
    def test_simple_query_no_signals(self, mock_context):
        """Simple queries should not trigger complexity signals."""
        prompt = "Show me package.json"
        
        signals = extract_complexity_signals(prompt, mock_context)
        
        assert signals.references_multiple_files is False
        assert signals.requires_cross_context_reasoning is False
        assert signals.debugging_task is False


class TestShouldActivateRlm:
    """Tests for should_activate_rlm function."""
    
    def test_activates_on_cross_context_reasoning(self, mock_context):
        """Cross-context reasoning immediately activates RLM."""
        prompt = "Why does X happen when Y is configured?"
        
        should_activate, reason = should_activate_rlm(prompt, mock_context)
        
        assert should_activate is True
        assert "cross_context" in reason
    
    def test_activates_on_debugging_with_large_output(self, debug_context):
        """Debugging with large tool output activates RLM."""
        prompt = "Fix the failing test"
        debug_context.tool_outputs[0].content = "x" * 15000  # Large output
        
        should_activate, reason = should_activate_rlm(prompt, debug_context)
        
        assert should_activate is True
    
    def test_respects_manual_override_on(self, mock_context):
        """Manual RLM mode forces activation."""
        prompt = "Simple question"
        
        should_activate, reason = should_activate_rlm(
            prompt, mock_context, rlm_mode_forced=True
        )
        
        assert should_activate is True
        assert reason == "manual_override"
    
    def test_respects_manual_override_off(self, mock_context):
        """Manual simple mode prevents activation."""
        prompt = "Complex debugging task with errors"
        
        should_activate, reason = should_activate_rlm(
            prompt, mock_context, simple_mode_forced=True
        )
        
        assert should_activate is False
        assert reason == "simple_mode_forced"


class TestIsDefinitelySimple:
    """Tests for is_definitely_simple function."""
    
    def test_simple_file_read(self, mock_context):
        """Simple file reads are definitely simple."""
        prompt = "show package.json"
        
        assert is_definitely_simple(prompt, mock_context) is True
    
    def test_git_status(self, mock_context):
        """Git status is definitely simple."""
        prompt = "git status"
        
        assert is_definitely_simple(prompt, mock_context) is True
    
    def test_complex_not_simple(self, mock_context):
        """Complex queries are not definitely simple."""
        prompt = "Find all the bugs in the authentication module"
        
        assert is_definitely_simple(prompt, mock_context) is False
    
    def test_large_context_not_simple(self, large_context):
        """Even simple prompts with large context aren't definitely simple."""
        prompt = "ok"
        
        # Large context (100K tokens) should not be considered simple
        # even for acknowledgment prompts
        assert is_definitely_simple(prompt, large_context) is False
