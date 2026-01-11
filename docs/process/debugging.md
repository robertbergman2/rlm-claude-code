# Debugging Workflow

This document describes how to debug RLM-Claude-Code issues.

## Core Principle: Trajectory First

**Before making any code changes, capture and analyze the trajectory.**

RLM behavior is determined by the sequence of REPL executions and recursive calls. Without the trajectory, debugging is guesswork.

## Debugging Workflow

### Step 1: Reproduce with Trajectory Capture

```bash
# Enable debug verbosity and JSON export
python -m src.orchestrator \
    --query "the failing query" \
    --context-file tests/fixtures/contexts/failing_context.json \
    --verbosity debug \
    --export-trajectory /tmp/debug_trajectory.json
```

### Step 2: Analyze Trajectory Structure

```python
# scripts/analyze_trajectory.py
import json
from pathlib import Path

def analyze_trajectory(path: str):
    """Analyze a trajectory JSON file."""
    trajectory = json.loads(Path(path).read_text())
    
    print(f"Total events: {len(trajectory)}")
    print(f"Max depth reached: {max(e['depth'] for e in trajectory)}")
    print(f"Recursive calls: {sum(1 for e in trajectory if e['type'] == 'recurse_start')}")
    print(f"REPL executions: {sum(1 for e in trajectory if e['type'] == 'repl_exec')}")
    
    # Find errors
    errors = [e for e in trajectory if e['type'] == 'error']
    if errors:
        print(f"\nErrors found: {len(errors)}")
        for err in errors:
            print(f"  - depth={err['depth']}: {err['content'][:100]}")
    
    # Find long-running events
    for i, event in enumerate(trajectory[1:], 1):
        duration = event['timestamp'] - trajectory[i-1]['timestamp']
        if duration > 5.0:  # >5 seconds
            print(f"\nSlow event: {event['type']} took {duration:.1f}s")
            print(f"  Content: {event['content'][:100]}")

if __name__ == "__main__":
    import sys
    analyze_trajectory(sys.argv[1])
```

### Step 3: Identify Issue Category

| Symptom | Category | See Section |
|---------|----------|-------------|
| Wrong answer | Reasoning | §3.1 |
| Missing context | Externalization | §3.2 |
| Infinite loop | Recursion | §3.3 |
| Timeout | Performance | §3.4 |
| Crash | Security/Error | §3.5 |
| No trajectory | Activation | §3.6 |

---

## 3.1 Reasoning Issues

**Symptom**: RLM produces incorrect answer despite having correct context.

**Diagnosis**:
```python
# Check what the root model actually saw
for event in trajectory:
    if event['type'] == 'analyze':
        print("Root saw:", event['metadata']['context_summary'])
```

**Common Causes**:

1. **Insufficient peeking**: Root didn't peek enough context
   - Fix: Adjust root prompt to encourage more exploration
   
2. **Bad search patterns**: grep/search missed relevant content
   - Fix: Log search queries and results, improve patterns

3. **Summarization loss**: Recursive summarization dropped key info
   - Fix: Increase max_tokens for summarization calls

4. **Depth limit hit**: Needed depth=2 but only had depth=1
   - Fix: Enable depth=2 for this query pattern

**Debugging Commands**:
```bash
# Replay with more aggressive exploration
python -m src.orchestrator \
    --query "the query" \
    --strategy aggressive_peek \
    --verbosity debug
```

---

## 3.2 Context Externalization Issues

**Symptom**: Context data missing or corrupted in REPL.

**Diagnosis**:
```python
# Compare original vs externalized
from src.context_manager import externalize_conversation

original = load_session_context()
externalized = externalize_conversation(original.messages)

# Check for data loss
assert len(externalized) == len(original.messages)
for orig, ext in zip(original.messages, externalized):
    assert orig.content == ext['content'], f"Content mismatch at {orig}"
```

**Common Causes**:

1. **Encoding issues**: Non-UTF8 content corrupted
   - Fix: Use `errors='replace'` in string handling

2. **Truncation**: Large files truncated without notice
   - Fix: Log truncation events, preserve truncation markers

3. **Serialization bugs**: Complex objects not JSON-serializable
   - Fix: Use pydantic models with proper serializers

**Debugging Commands**:
```bash
# Dump externalized context for inspection
python -m src.context_manager \
    --input tests/fixtures/contexts/problem.json \
    --output /tmp/externalized.json \
    --verbose
```

---

## 3.3 Recursion Issues

**Symptom**: RLM loops infinitely or exceeds depth limit unexpectedly.

**Diagnosis**:
```python
# Check recursion pattern
recurse_events = [e for e in trajectory if 'recurse' in e['type']]
for event in recurse_events:
    print(f"depth={event['depth']}: {event['content'][:50]}")

# Look for repeated queries
queries = [e['content'] for e in recurse_events if e['type'] == 'recurse_start']
from collections import Counter
repeated = Counter(queries).most_common(5)
print("Most repeated queries:", repeated)
```

**Common Causes**:

1. **Query echoing**: Sub-call returns query as answer, causing re-query
   - Fix: Detect and break echo patterns

2. **Incomplete termination**: FINAL signal not recognized
   - Fix: Check output parsing regex

3. **Spawn loop**: Each level spawns multiple children
   - Fix: Add spawn budget per depth level

**Debugging Commands**:
```bash
# Run with strict depth limit
python -m src.orchestrator \
    --query "the query" \
    --max-depth 1 \
    --max-recursive-calls 3 \
    --verbosity debug
```

---

## 3.4 Performance Issues

**Symptom**: RLM takes too long (>30s for typical queries).

**Diagnosis**:
```python
# Profile by event type
from collections import defaultdict

durations = defaultdict(list)
for i, event in enumerate(trajectory[1:], 1):
    duration = event['timestamp'] - trajectory[i-1]['timestamp']
    durations[event['type']].append(duration)

for event_type, times in sorted(durations.items(), key=lambda x: -sum(x[1])):
    print(f"{event_type}: total={sum(times):.1f}s, count={len(times)}, avg={sum(times)/len(times):.2f}s")
```

**Common Causes**:

1. **Too many recursive calls**: Partitioning too aggressively
   - Fix: Increase chunk size, reduce partition count

2. **Slow REPL execution**: Complex operations in sandbox
   - Fix: Profile sandbox, optimize hot paths

3. **Model latency**: Waiting on API calls
   - Fix: Enable parallel sub-calls, use caching

4. **Large context serialization**: JSON encoding bottleneck
   - Fix: Use streaming serialization, compress large values

**Debugging Commands**:
```bash
# Profile with cProfile
python -m cProfile -s cumulative -m src.orchestrator \
    --query "the query" \
    2>&1 | head -50

# Benchmark specific component
pytest tests/benchmarks/test_performance.py -v --benchmark-only
```

---

## 3.5 Security and Error Issues

**Symptom**: REPL crashes or produces security error.

**Diagnosis**:
```python
# Find error events
for event in trajectory:
    if event['type'] == 'error':
        print(f"Error at depth={event['depth']}:")
        print(f"  Message: {event['content']}")
        if 'metadata' in event:
            print(f"  Code: {event['metadata'].get('code', 'N/A')[:200]}")
```

**Common Causes**:

1. **Sandbox violation**: Code tried blocked operation
   - Expected: Security working correctly
   - Fix: If legitimate, add to allowlist

2. **Import error**: Required module not available
   - Fix: Add to REPL globals

3. **Syntax error in generated code**: Model produced invalid Python
   - Fix: Add syntax validation before execution

4. **Resource exhaustion**: Memory or CPU limit hit
   - Fix: Add resource limits to sandbox

**Debugging Commands**:
```bash
# Run REPL interactively to test execution
python -m src.repl_environment --interactive

# In REPL:
>>> execute("import os; os.system('ls')")  # Should fail
>>> execute("from pydantic import BaseModel")  # Should succeed
```

---

## 3.6 Activation Issues

**Symptom**: RLM doesn't activate when it should (or activates when it shouldn't).

**Diagnosis**:
```bash
# Check complexity classifier output
python -m src.complexity_classifier \
    --prompt "the user prompt" \
    --context-file tests/fixtures/contexts/context.json \
    --verbose
```

**Output**:
```
Complexity Signals:
  references_multiple_files: True
  requires_cross_context_reasoning: False
  involves_temporal_reasoning: False
  asks_about_patterns: True
  debugging_task: False
  ...
  
Score: 3
Threshold: 2
Decision: ACTIVATE (pattern_search + multi_file)
```

**Common Causes**:

1. **Pattern not matched**: Prompt phrasing not recognized
   - Fix: Add pattern to classifier

2. **Score below threshold**: Signals present but not enough
   - Fix: Lower threshold or weight signals higher

3. **Context signals wrong**: Context analysis incorrect
   - Fix: Debug context analysis functions

**Debugging Commands**:
```bash
# Test classifier on prompt corpus
python -m src.complexity_classifier \
    --test-corpus tests/fixtures/prompts/complex_prompts.txt \
    --expected-activation true \
    --verbose
```

---

## Debugging Tools

### Trajectory Viewer (CLI)

```bash
# Pretty-print trajectory
python -m src.tools.trajectory_viewer /tmp/debug_trajectory.json

# Filter by depth
python -m src.tools.trajectory_viewer /tmp/debug_trajectory.json --depth 1

# Filter by type
python -m src.tools.trajectory_viewer /tmp/debug_trajectory.json --type repl_exec
```

### REPL Inspector

```bash
# Dump REPL state at specific point
python -m src.tools.repl_inspector \
    --trajectory /tmp/debug_trajectory.json \
    --at-event 15
    
# Output:
# REPL State at event 15:
#   conversation: 45 messages
#   files: {'main.py': '...', 'utils.py': '...'}
#   working_memory: {'bug_location': 'line 42'}
```

### Diff Tool

```bash
# Compare two trajectories
python -m src.tools.trajectory_diff \
    /tmp/working_trajectory.json \
    /tmp/failing_trajectory.json
```

---

## Logging Configuration

```python
# src/logging_config.py
import logging

LOGGING_CONFIG = {
    'version': 1,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'detailed',
            'level': 'INFO',
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': '~/.claude/rlm-debug.log',
            'formatter': 'detailed',
            'level': 'DEBUG',
        },
    },
    'formatters': {
        'detailed': {
            'format': '%(asctime)s %(name)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s'
        },
    },
    'loggers': {
        'rlm.orchestrator': {'level': 'DEBUG'},
        'rlm.repl': {'level': 'DEBUG'},
        'rlm.recursive': {'level': 'DEBUG'},
        'rlm.classifier': {'level': 'INFO'},
    },
}
```

Enable verbose logging:
```bash
export RLM_LOG_LEVEL=DEBUG
export RLM_LOG_FILE=~/.claude/rlm-debug.log
```

---

## Common Fixes Checklist

- [ ] Captured trajectory before debugging
- [ ] Identified issue category
- [ ] Reproduced in isolation
- [ ] Added regression test
- [ ] Updated snapshot tests if behavior changed
- [ ] Documented fix in this file if novel
