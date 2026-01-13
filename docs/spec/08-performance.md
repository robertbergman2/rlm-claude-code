# SPEC-08: Performance Optimizations

## Overview

This specification defines capabilities for improving RLM performance through asynchronous execution, prompt caching, and context compression.

**Research Basis**:
- [RLM Paper](https://arxiv.org/abs/2512.24601) - notes "lack of asynchrony" as key limitation
- [Claude Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
- [KVzip (SNU, 2025)](https://techxplore.com/news/2025-11-ai-tech-compress-llm-chatbot.html)

## Dependencies

```
SPEC-08 (Performance) ───► SPEC-05 (Budget Tracking)
```

---

## 8.1 Asynchronous Recursive Calls

### Requirements

[SPEC-08.01] The system SHALL support fully asynchronous execution of recursive calls.

[SPEC-08.02] The system SHALL support true parallelism via asyncio.TaskGroup.

[SPEC-08.03] The system SHALL support configurable max_concurrency (default: 10).

[SPEC-08.04] The system SHALL support speculative execution:
- Execute primary operation with alternatives
- Cancel losers when winner completes
- Return first successful result

[SPEC-08.05] The system SHALL handle partial failures gracefully:
- Continue with successful results
- Report failed operations
- Synthesize best-effort result

[SPEC-08.06] The async executor SHALL respect budget constraints.

### Acceptance Criteria

- [ ] Parallel execution works correctly
- [ ] Speculative execution cancels losers
- [ ] Partial failures handled gracefully
- [ ] 3-5x latency reduction for multi-call queries
- [ ] Budget constraints respected during parallel execution

---

## 8.2 Prompt Caching Integration

### Requirements

[SPEC-08.10] The system SHALL structure prompts for optimal cache hits.

[SPEC-08.11] Shared context (files, conversation history) SHALL be placed first (cacheable).

[SPEC-08.12] Query-specific content SHALL be placed last (not cached).

[SPEC-08.13] The system SHALL maintain cache prefix registry for recursive calls.

[SPEC-08.14] The system SHALL track cache hit/miss metrics.

[SPEC-08.15] For recursive calls with shared context, the system SHALL reuse cached prefixes.

### Acceptance Criteria

- [ ] Prompt structure optimizes for caching
- [ ] Cache prefix registry maintained
- [ ] Cache hit rate >50% for recursive calls
- [ ] Cost reduction >50% for cache-eligible queries
- [ ] Latency reduction >30% for cache hits

---

## 8.3 Context Compression

### Requirements

[SPEC-08.20] The system SHALL compress intermediate results when they exceed target_tokens.

[SPEC-08.21] Compression SHALL use two-stage approach:
1. Extractive: Select key sentences using relevance scoring
2. Abstractive: LLM summarization if still over budget

[SPEC-08.22] Compression SHALL preserve:
- Key facts and findings
- Error messages and stack traces
- Code snippets and file references

[SPEC-08.23] The system SHALL support configurable target_tokens (default: 2000).

[SPEC-08.24] Compression SHALL be applied automatically to tool outputs >5000 tokens.

[SPEC-08.25] The system SHALL track compression ratio metrics.

### Acceptance Criteria

- [ ] Compression achieves 3-4x reduction
- [ ] Key information preserved after compression
- [ ] Automatic compression triggers appropriately
- [ ] No loss of task-critical information
- [ ] Latency impact <500ms for compression
