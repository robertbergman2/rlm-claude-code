"""
Task complexity classification for RLM activation.

Implements: Spec §6.3 Task Complexity-Based Activation
"""

import re
from pathlib import Path

from .types import SessionContext, TaskComplexitySignals


def extract_complexity_signals(prompt: str, context: SessionContext) -> TaskComplexitySignals:
    """
    Extract complexity signals from prompt and context.

    Implements: Spec §6.3 Task Complexity-Based Activation

    Must be fast (<50ms) so uses heuristics, not LLM calls.
    """
    prompt_lower = prompt.lower()

    # File reference patterns - count total file references across all patterns
    file_extension_pattern = r"\b\w+\.(ts|js|py|go|rs|tsx|jsx)\b"
    file_matches = len(re.findall(file_extension_pattern, prompt_lower))

    # Also check for module pair mentions (auth and api, etc.)
    module_pair_pattern = r"\b(auth|api|db|ui|test|config)\b.*\b(auth|api|db|ui|test|config)\b"
    has_module_pair = bool(re.search(module_pair_pattern, prompt_lower))

    # Also check for conjunction patterns suggesting multiple targets
    conjunction_pattern = r"\b(and|also|plus)\s+(update|fix|change|modify)"
    has_conjunction = bool(re.search(conjunction_pattern, prompt_lower))

    # Multiple files if: 2+ explicit files, or module pair, or conjunction with file
    references_multiple = (
        file_matches >= 2 or has_module_pair or (has_conjunction and file_matches >= 1)
    )

    # Cross-context reasoning patterns
    cross_context_patterns = [
        r"\bwhy\b.*\b(when|if|given|since)\b",
        r"\bhow\b.*\b(relate|connect|affect|impact)\b",
        r"\bwhat\b.*\b(cause|led to|result)\b",
        r"\b(trace|follow|track)\b.*\b(through|across)\b",
    ]

    # Temporal patterns
    temporal_patterns = [
        r"\b(before|after|since|when|changed|used to|previously)\b",
        r"\b(history|log|commit|version|diff)\b",
        r"\blast\s+(time|session|attempt)\b",
    ]

    # Pattern search indicators
    pattern_patterns = [
        r"\b(find|search|locate|grep|all|every|each)\b.*\b(where|that|which)\b",
        r"\bhow many\b",
        r"\blist\s+(all|every)\b",
    ]

    # Debug indicators (using word stems to catch variations like failing/failed)
    debug_patterns = [
        r"\b(error|exception|fail\w*|crash\w*|bug|issue|broken)\b",
        r"\b(stack\s*trace|traceback|stderr)\b",
        r"\b(debug\w*|diagnos\w*|investigat\w*|troubleshoot\w*)\b",
    ]

    return TaskComplexitySignals(
        references_multiple_files=references_multiple,
        requires_cross_context_reasoning=any(
            re.search(p, prompt_lower) for p in cross_context_patterns
        ),
        involves_temporal_reasoning=any(re.search(p, prompt_lower) for p in temporal_patterns),
        asks_about_patterns=any(re.search(p, prompt_lower) for p in pattern_patterns),
        debugging_task=any(re.search(p, prompt_lower) for p in debug_patterns),
        context_has_multiple_domains=len(context.active_modules) > 2,
        recent_tool_outputs_large=sum(len(o.content) for o in context.tool_outputs[-5:]) > 10000,
        conversation_has_state_changes=False,  # TODO: Implement
        files_span_multiple_modules=len(
            {Path(f).parts[0] for f in context.files if Path(f).parts}
        )
        > 2,
        previous_turn_was_confused=False,  # TODO: Implement
        task_is_continuation="continue" in prompt_lower
        or "same" in prompt_lower
        or "also" in prompt_lower[:50],
    )


def should_activate_rlm(
    prompt: str,
    context: SessionContext,
    rlm_mode_forced: bool = False,
    simple_mode_forced: bool = False,
) -> tuple[bool, str]:
    """
    Determine if RLM mode should activate.

    Implements: Spec §6.3 Task Complexity-Based Activation

    Biased toward activation—when in doubt, use RLM.

    Returns:
        (should_activate, reason) tuple
    """
    # Manual overrides
    if rlm_mode_forced:
        return True, "manual_override"
    if simple_mode_forced:
        return False, "simple_mode_forced"

    signals = extract_complexity_signals(prompt, context)

    # High-signal indicators (each sufficient alone)
    if signals.requires_cross_context_reasoning:
        return True, "cross_context_reasoning"
    if signals.debugging_task and signals.recent_tool_outputs_large:
        return True, "debugging_with_large_output"
    if signals.references_multiple_files and signals.files_span_multiple_modules:
        return True, "multi_module_task"

    # Accumulative signals
    score = 0
    reasons = []

    if signals.references_multiple_files:
        score += 2
        reasons.append("multi_file")
    if signals.involves_temporal_reasoning:
        score += 2
        reasons.append("temporal")
    if signals.asks_about_patterns:
        score += 1
        reasons.append("pattern_search")
    if signals.context_has_multiple_domains:
        score += 1
        reasons.append("multi_domain")
    if signals.previous_turn_was_confused:
        score += 2
        reasons.append("prior_confusion")
    if signals.task_is_continuation:
        score += 1
        reasons.append("continuation")

    # Token count as tiebreaker
    if context.total_tokens > 80000:
        score += 1
        reasons.append("large_context")

    # Threshold: 2+ signals → activate
    if score >= 2:
        return True, f"complexity_score:{score}:{'+'.join(reasons)}"

    return False, "simple_task"


def is_definitely_simple(prompt: str, context: SessionContext) -> bool:
    """
    Returns True ONLY for queries that definitely don't need RLM.

    Implements: Spec §6.5 Hybrid Mode

    When in doubt, return False (use RLM).
    """
    prompt_lower = prompt.lower().strip()

    # Explicit simple patterns
    simple_patterns = [
        r"^(show|cat|read|view|open)\s+[\w./]+$",
        r"^(run|execute)\s+(npm|yarn|pnpm|cargo|go|python|pytest)\s+\w+$",
        r"^git\s+(status|log|diff|branch)$",
        r"^what\'?s?\s+in\s+[\w./]+\??$",
        r"^(ok|okay|thanks|got it|understood|yes|no|sure)\.?$",
    ]

    for pattern in simple_patterns:
        if re.match(pattern, prompt_lower) and context.total_tokens < 20000:
            return True

    # Short prompts with no complexity indicators
    if len(prompt) < 50 and context.total_tokens < 10000:
        complexity_words = [
            "why",
            "how",
            "find",
            "fix",
            "bug",
            "error",
            "all",
            "every",
            "change",
            "update",
            "refactor",
            "test",
        ]
        if not any(word in prompt_lower for word in complexity_words):
            return True

    return False


__all__ = [
    "extract_complexity_signals",
    "should_activate_rlm",
    "is_definitely_simple",
]
