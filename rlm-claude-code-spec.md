# Claude Code RLM Integration Specification

## TL;DR

Transform Claude Code into a Recursive Language Model (RLM) agent by externalizing conversation context as manipulable Python variables in a REPL environment, enabling the root model to programmatically decompose, analyze, and recursively query context—eliminating context rot while preserving Claude Code's core agentic capabilities.

---

## 1. Problem Statement

### 1.1 Context Rot in Long Coding Sessions

Claude Code sessions accumulate context through:
- Conversation history (multi-turn interactions)
- File reads (codebase exploration)
- Tool outputs (bash commands, test results, diffs)
- Agentic reasoning traces

**The problem**: As sessions grow beyond 60-100K tokens, model performance degrades—it "forgets" earlier context, misses details, and makes decisions without full information. This is especially acute in:
- Large refactoring tasks spanning multiple files
- Long debugging sessions with extensive logs
- Multi-step planning with accumulated state

### 1.2 Why RLM?

The RLM paper demonstrates that **context-centric decomposition** (treating the prompt as a variable to be programmatically manipulated) outperforms:
- Direct model calls with full context
- RAG/retrieval approaches (BM25, semantic search)
- ReAct-style agents with retrieval tools

Key insight: **The model decides how to decompose context, not a predefined scaffold.**

---

## 2. Architecture Overview

### 2.1 High-Level Design

```
┌─────────────────────────────────────────────────────────────────┐
│                        Claude Code CLI                          │
├─────────────────────────────────────────────────────────────────┤
│                     RLM Orchestration Layer                     │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────────┐   │
│  │   Context     │  │    REPL       │  │   Recursive       │   │
│  │   Manager     │  │  Environment  │  │   Call Handler    │   │
│  └───────────────┘  └───────────────┘  └───────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                   Claude Code Core (Preserved)                  │
│  ┌─────────┐  ┌──────────┐  ┌────────┐  ┌─────────────────┐   │
│  │  Tools  │  │  Agents  │  │  Hooks │  │  MCP Servers    │   │
│  └─────────┘  └──────────┘  └────────┘  └─────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                      Model Provider Layer                       │
│           (Anthropic API / Router / Local Models)               │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Core Components

| Component | Responsibility |
|-----------|----------------|
| **Context Manager** | Externalizes conversation context as Python variables |
| **REPL Environment** | Sandboxed Python execution with context access |
| **Recursive Call Handler** | Manages sub-model calls with isolated state |
| **Router Integration** | Directs root vs. recursive calls to appropriate models |

---

## 3. Context Externalization

### 3.1 Context Variable Schema

Instead of including full conversation history in the prompt, externalize as:

```python
# Available in REPL environment
conversation: List[Message] = [...]  # Full conversation history
files: Dict[str, str] = {...}         # Cached file contents
tool_outputs: List[ToolOutput] = [...] # Recent tool results
working_memory: Dict[str, Any] = {}   # Session state

# Metadata
context_stats = {
    "total_tokens": 156000,
    "conversation_tokens": 45000,
    "file_tokens": 89000,
    "tool_output_tokens": 22000
}

# Helper functions (pre-loaded)
def peek(var, start=0, end=1000): ...
def search(var, pattern, regex=False): ...
def summarize(var, max_tokens=500): ...  # Uses sub-call
def recursive_query(query, context_var): ...  # Sub-RLM call
```

### 3.2 Root Prompt Structure

The root model receives a **minimal prompt** without full context:

```markdown
You are Claude Code operating in RLM mode. Your conversation context 
is stored in variables in a Python REPL environment.

## Available Variables
- `conversation`: {n_messages} messages, {n_tokens} tokens
- `files`: {n_files} files cached ({file_tokens} tokens)
- `tool_outputs`: {n_outputs} recent outputs ({output_tokens} tokens)

## Actions
1. **Peek**: View portions of context: `peek(conversation, 0, 5)`
2. **Search**: Find patterns: `search(files['main.py'], 'def ')`
3. **Query**: Ask sub-questions: `recursive_query("What's the error?", tool_outputs[-1])`
4. **Execute**: Run code/tools normally when ready

## Current Query
{user_query}

## Rules
- Don't request full context dumps—use programmatic access
- Partition large contexts before analysis
- Use recursive_query for semantic understanding of large chunks
- When confident, output: FINAL(answer) or FINAL_VAR(var_name)
```

---

## 4. REPL Environment Design

### 4.1 Sandbox Architecture

```python
class RLMEnvironment:
    """Sandboxed REPL for context manipulation"""
    
    def __init__(self, context: SessionContext):
        self.globals = {
            # Context variables
            'conversation': context.messages,
            'files': context.cached_files,
            'tool_outputs': context.tool_outputs,
            'working_memory': {},
            
            # Helper functions
            'peek': self._peek,
            'search': self._search,
            'summarize': self._summarize,
            'recursive_query': self._recursive_query,
            'recursive_llm': self._recursive_query,  # alias
            
            # Standard library (restricted)
            're': re,
            'json': json,
            'len': len,
            'str': str,
            'list': list,
            'dict': dict,
            'sorted': sorted,
            'enumerate': enumerate,
            
            # === EXTENDED TOOLING ===
            # These are available for context analysis and validation
            'pydantic': pydantic,           # Data validation/parsing
            'hypothesis': hypothesis,       # Property-based testing
            'cp': cpmpy,                    # Constraint programming
            'cpmpy': cpmpy,                 # Alias
        }
        self.locals = {}
        
    def execute(self, code: str) -> ExecutionResult:
        """Execute code in sandbox, return result"""
        # Use RestrictedPython or similar for safety
        ...
```

### 4.1.1 Extended Python Tooling

The RLM REPL environment includes production-grade Python tooling for robust context manipulation:

**Available Tools**:

| Tool | Version | Purpose in RLM |
|------|---------|----------------|
| `uv` | latest | Fast package management, virtual env creation |
| `ty` | latest | Type checking extracted context/generated code |
| `ruff` | latest | Linting/formatting code in context |
| `pydantic` | v2.x | Schema validation for structured context data |
| `hypothesis` | latest | Property-based testing for verification queries |
| `cpmpy` | latest | Constraint programming for verification and analysis |

**CPMpy Integration**:

CPMpy enables constraint-driven reasoning in the REPL—aligning with the Ananke architecture of treating code generation as constrained search through valid programs.

```python
import cpmpy as cp

# Example: Verify a scheduling constraint from context
# Given: task dependencies extracted from code comments

tasks = cp.intvar(0, 100, shape=5, name="task_start")
durations = [10, 5, 8, 3, 12]  # Extracted from context

model = cp.Model()

# Constraint: task[1] must start after task[0] finishes
model += tasks[1] >= tasks[0] + durations[0]

# Constraint: tasks 2 and 3 cannot overlap (extracted from mutex comment)
model += (tasks[2] + durations[2] <= tasks[3]) | (tasks[3] + durations[3] <= tasks[2])

# Verify feasibility
if model.solve():
    print("Schedule is feasible:", tasks.value())
else:
    print("Constraint violation detected")
```

**Use Cases in RLM**:

| Scenario | CPMpy Application |
|----------|-------------------|
| Dependency analysis | Model import/call graphs as constraints |
| Type compatibility | Encode type relationships as logical constraints |
| Resource verification | Cumulative constraints for resource bounds |
| State machine validation | Transition constraints on state variables |
| Test case generation | Generate inputs satisfying path constraints |

**Depth=2 Verification Pattern**:

```python
# At depth=2, use CPMpy to verify proposed changes don't break invariants

async def verify_change_safety(
    change: ProposedEdit,
    invariants: List[str],  # Extracted from docstrings/comments
    context: str
) -> VerificationResult:
    """
    Use constraint solving to verify a proposed change.
    
    Implements: Spec §6.4 Depth=2 Verification
    """
    model = cp.Model()
    
    # Parse invariants into constraints
    for inv in invariants:
        constraint = parse_invariant_to_cpmpy(inv, context)
        if constraint:
            model += constraint
    
    # Check if change maintains satisfiability
    model += encode_change_effect(change)
    
    if model.solve():
        return VerificationResult(safe=True, witness=model.solution_hint())
    else:
        # Use unsat core to explain why
        return VerificationResult(
            safe=False, 
            reason=extract_conflict(model)
        )
```

**REPL Integration**:

```python
# Pydantic: Validate structured data from context
from pydantic import BaseModel, Field
from typing import List

class ErrorReport(BaseModel):
    file: str
    line: int
    message: str
    severity: str = Field(pattern=r'^(error|warning|info)$')

# Parse tool output into structured form
errors = [ErrorReport.model_validate_json(line) for line in tool_outputs[-1].split('\n')]

# Hypothesis: Property-based verification
from hypothesis import given, strategies as st

@given(st.sampled_from(files.keys()))
def test_all_files_have_extension(filename):
    assert '.' in filename

# Run verification (useful for depth=2 validation)
test_all_files_have_extension()

# CPMpy: Constraint verification
import cpmpy as cp

# Model a simple constraint problem from context
x = cp.intvar(1, 10, name="x")
y = cp.intvar(1, 10, name="y")
model = cp.Model([x + y == 15, x < y])
model.solve()
print(f"Solution: x={x.value()}, y={y.value()}")
```

**CLI Tools (via subprocess in sandbox)**:

```python
# These run in isolated subprocess, output captured

# Type check a code snippet from context
async def typecheck_snippet(code: str) -> TypeCheckResult:
    """Run ty on extracted code"""
    result = await sandbox_exec(['ty', 'check', '-'], stdin=code)
    return TypeCheckResult(
        success=result.returncode == 0,
        errors=parse_ty_output(result.stderr)
    )

# Lint code before suggesting edits
async def lint_snippet(code: str) -> LintResult:
    """Run ruff on extracted code"""
    result = await sandbox_exec(['ruff', 'check', '--stdin-filename=snippet.py', '-'], stdin=code)
    return LintResult(
        issues=parse_ruff_output(result.stdout),
        auto_fixable=count_fixable(result.stdout)
    )
```

**Environment Setup** (on plugin install):

```bash
#!/bin/bash
# scripts/setup_repl_env.sh

# Create isolated venv for REPL
uv venv ~/.claude/rlm-repl-env --python 3.12

# Install core tooling
uv pip install --python ~/.claude/rlm-repl-env \
    pydantic>=2.0 \
    hypothesis>=6.0 \
    cpmpy>=0.9.20 \
    ruff \
    ty

# Verify installation
~/.claude/rlm-repl-env/bin/python -c "import pydantic, hypothesis, cpmpy; print('OK')"
```

**Security Constraints**:

```python
# ALLOWED in REPL
- pydantic.BaseModel, Field, validators
- hypothesis.given, strategies, settings
- cpmpy.Model, intvar, boolvar, constraints, solve
- subprocess calls to: ty, ruff (sandboxed)
- Read-only file inspection via context variables

# BLOCKED in REPL  
- subprocess calls to: rm, mv, curl, wget, etc.
- Direct filesystem writes (use Claude Code tools)
- Network access (except via Claude Code tools)
- os.system, eval on user input
- Import of arbitrary modules
```
```

### 4.2 Recursive Call Implementation

```python
async def _recursive_query(
    self, 
    query: str, 
    context: Any,
    model: str = None,  # Defaults to configured recursive model
    max_tokens: int = 4000
) -> str:
    """
    Spawn a sub-RLM call over provided context.
    
    The sub-call receives:
    - The query
    - The context (as a string variable)
    - NO access to parent REPL state (isolated)
    
    Returns: The sub-model's response
    """
    context_str = self._serialize_context(context)
    
    # Sub-call prompt
    sub_prompt = f"""
    You are analyzing a portion of context for a larger task.
    
    ## Context
    The following is stored in variable `context`:
    {context_str[:500]}... [truncated, {len(context_str)} chars total]
    
    ## Query
    {query}
    
    Provide a focused, factual response. If you need to examine
    the context programmatically, you have access to a Python REPL
    with `context` as a variable.
    """
    
    # Route to recursive model (could be same or different)
    response = await self.router.complete(
        model=model or self.config.recursive_model,
        prompt=sub_prompt,
        max_tokens=max_tokens,
        depth=self.depth + 1
    )
    
    return response.content
```

### 4.3 Output Protocol

The root model signals completion via:

```python
# Direct answer
FINAL(Your answer here)

# Answer stored in variable
result = "Complex computed answer"
FINAL_VAR(result)

# Continue with tool use (normal Claude Code flow)
<tool_use>bash</tool_use>
<command>npm test</command>
```

---

## 5. Integration with Claude Code

### 5.1 Plugin Structure

```
rlm-agent/
├── .claude-plugin/
│   └── plugin.json
├── agents/
│   └── rlm-orchestrator.md      # Root agent definition
├── skills/
│   └── rlm-context-management/
│       └── SKILL.md             # Context management strategies
├── hooks/
│   └── hooks.json               # Session hooks
├── src/
│   ├── context_manager.py       # Context externalization
│   ├── repl_environment.py      # REPL sandbox
│   ├── recursive_handler.py     # Sub-call management
│   └── router_integration.py    # Model routing
└── README.md
```

### 5.2 Hook Integration

```json
{
  "hooks": {
    "SessionStart": [{
      "hooks": [{
        "type": "command",
        "command": "python ${CLAUDE_PLUGIN_ROOT}/src/init_rlm.py",
        "timeout": 5
      }]
    }],
    "UserPromptSubmit": [{
      "matcher": ".*",
      "hooks": [{
        "type": "command", 
        "command": "python ${CLAUDE_PLUGIN_ROOT}/src/check_context_threshold.py",
        "timeout": 3
      }]
    }],
    "PreCompact": [{
      "hooks": [{
        "type": "command",
        "command": "python ${CLAUDE_PLUGIN_ROOT}/src/externalize_context.py"
      }]
    }]
  }
}
```

### 5.3 Router Configuration

Integrate with `claude-code-router` for model selection:

```json
{
  "Router": {
    "default": "anthropic,claude-sonnet-4",
    "rlm_root": "anthropic,claude-opus-4-5-20251101",
    "rlm_recursive": "anthropic,claude-sonnet-4",
    "rlm_recursive_deep": "anthropic,claude-haiku-4-5-20251001",
    "rlm_max_depth": 2,
    "rlm_mode": "complexity",
    "rlm_simple_bypass": true
  },
  "rlm": {
    "activation": {
      "mode": "complexity",
      "fallback_token_threshold": 80000,
      "complexity_score_threshold": 2
    },
    "depth": {
      "default": 2,
      "max": 3,
      "spawn_repl_at_depth_1": true
    },
    "hybrid": {
      "enabled": true,
      "simple_query_bypass": true,
      "simple_confidence_threshold": 0.95
    },
    "trajectory": {
      "verbosity": "normal",
      "streaming": true,
      "colors": true,
      "export_enabled": true,
      "export_path": "~/.claude/rlm-trajectories/"
    },
    "models": {
      "root": "claude-opus-4-5-20251101",
      "recursive_depth_1": "claude-sonnet-4",
      "recursive_depth_2": "claude-haiku-4-5-20251001"
    },
    "cost_controls": {
      "max_recursive_calls_per_turn": 10,
      "max_tokens_per_recursive_call": 8000,
      "abort_on_cost_threshold": 50000
    }
  }
}
```

**Model Selection by Depth**:

| Depth | Model | Rationale |
|-------|-------|-----------|
| 0 (root) | Opus 4.5 | Complex orchestration, strategy selection |
| 1 | Sonnet 4 | Analysis, summarization, focused queries |
| 2 | Haiku 4.5 | Verification, simple extraction, cheap |

**Routing Logic**:
- `rlm_mode: "complexity"` → Activate based on task complexity analysis (default)
- `rlm_mode: "always"` → Always use RLM orchestration
- `rlm_mode: "token"` → Activate when context exceeds token threshold
- `rlm_mode: "manual"` → Only via `/rlm` command

---

## 6. Key Design Decisions

### 6.1 Why Not Just Fork Claude Code?

**Decision**: Build as a plugin, not a fork.

**Rationale**:
- Anthropic frequently updates Claude Code (>300 commits)
- Plugins get updates automatically
- Core tool execution, permissions, hooks all preserved
- Community can extend/improve independently

### 6.2 REPL Language Choice

**Decision**: Python REPL (not JavaScript)

**Rationale**:
- RLM paper uses Python; proven patterns exist
- Better string manipulation for context processing
- Claude is stronger at Python code generation
- RestrictedPython provides sandbox primitives
- Can shell out to bash for system operations

### 6.3 Task Complexity-Based Activation

**Decision**: Activate RLM based on **task complexity analysis**, not token count.

**Rationale**: Token count is a proxy metric. The real question is: "Does this task require reasoning over distributed context?" A 20K token session with a complex multi-file refactor needs RLM more than a 100K token session asking "what's in package.json?"

**Complexity Classifier**:

```python
@dataclass
class TaskComplexitySignals:
    """Signals extracted from user prompt + context"""
    
    # Prompt analysis
    references_multiple_files: bool      # "fix the bug in auth and update tests"
    requires_cross_context_reasoning: bool  # "why is X happening given Y?"
    involves_temporal_reasoning: bool    # "what changed since last working version?"
    asks_about_patterns: bool            # "find all places where..."
    debugging_task: bool                 # error traces, stack dumps in context
    
    # Context analysis  
    context_has_multiple_domains: bool   # frontend + backend + infra
    recent_tool_outputs_large: bool      # >10K tokens of test/build output
    conversation_has_state_changes: bool # user changed requirements mid-session
    files_span_multiple_modules: bool    # auth/, api/, db/, etc.
    
    # Historical signals
    previous_turn_was_confused: bool     # model asked clarifying questions
    task_is_continuation: bool           # "now do the same for..."

def should_activate_rlm(
    prompt: str, 
    context: SessionContext,
    config: RLMConfig
) -> tuple[bool, str]:
    """
    Determine if RLM mode should activate.
    Returns (should_activate, reason).
    
    Biased toward activation—when in doubt, use RLM.
    """
    signals = extract_complexity_signals(prompt, context)
    
    # Always activate (manual override)
    if context.rlm_mode_forced:
        return True, "manual_override"
    
    # Never activate (explicit simple mode)
    if context.simple_mode_forced:
        return False, "simple_mode_forced"
    
    # Complexity scoring (biased toward activation)
    score = 0
    reasons = []
    
    # High-signal indicators (each sufficient alone)
    if signals.requires_cross_context_reasoning:
        return True, "cross_context_reasoning"
    if signals.debugging_task and signals.recent_tool_outputs_large:
        return True, "debugging_with_large_output"
    if signals.references_multiple_files and signals.files_span_multiple_modules:
        return True, "multi_module_task"
    
    # Accumulative signals
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
    
    # Token count as tiebreaker (not primary signal)
    if context.total_tokens > 80000:
        score += 1
        reasons.append("large_context")
    
    # Threshold: 2+ signals → activate (conservative toward RLM)
    if score >= 2:
        return True, f"complexity_score:{score}:{'+'.join(reasons)}"
    
    return False, "simple_task"
```

**Prompt Analysis Implementation**:

```python
def extract_complexity_signals(prompt: str, context: SessionContext) -> TaskComplexitySignals:
    """
    Lightweight signal extraction—runs on every prompt.
    Must be fast (<50ms) so uses heuristics, not LLM calls.
    """
    prompt_lower = prompt.lower()
    
    # File reference patterns
    file_patterns = [
        r'\b\w+\.(ts|js|py|go|rs|tsx|jsx)\b',  # explicit files
        r'\b(and|also|plus)\s+(update|fix|change|modify)',  # conjunctions
        r'\b(auth|api|db|ui|test|config)\b.*\b(auth|api|db|ui|test|config)\b',  # multiple modules
    ]
    references_multiple = sum(
        len(re.findall(p, prompt_lower)) > 1 
        for p in file_patterns
    )
    
    # Cross-context reasoning patterns
    cross_context_patterns = [
        r'\bwhy\b.*\b(when|if|given|since)\b',
        r'\bhow\b.*\b(relate|connect|affect|impact)\b',
        r'\bwhat\b.*\b(cause|led to|result)\b',
        r'\b(trace|follow|track)\b.*\b(through|across)\b',
    ]
    
    # Temporal patterns
    temporal_patterns = [
        r'\b(before|after|since|when|changed|used to|previously)\b',
        r'\b(history|log|commit|version|diff)\b',
        r'\blast\s+(time|session|attempt)\b',
    ]
    
    # Pattern search indicators
    pattern_patterns = [
        r'\b(find|search|locate|grep|all|every|each)\b.*\b(where|that|which)\b',
        r'\bhow many\b',
        r'\blist\s+(all|every)\b',
    ]
    
    # Debug indicators
    debug_patterns = [
        r'\b(error|exception|fail|crash|bug|issue|broken)\b',
        r'\b(stack\s*trace|traceback|stderr)\b',
        r'\b(debug|diagnose|investigate|troubleshoot)\b',
    ]
    
    return TaskComplexitySignals(
        references_multiple_files=references_multiple >= 2,
        requires_cross_context_reasoning=any(
            re.search(p, prompt_lower) for p in cross_context_patterns
        ),
        involves_temporal_reasoning=any(
            re.search(p, prompt_lower) for p in temporal_patterns
        ),
        asks_about_patterns=any(
            re.search(p, prompt_lower) for p in pattern_patterns
        ),
        debugging_task=any(
            re.search(p, prompt_lower) for p in debug_patterns
        ),
        context_has_multiple_domains=len(context.active_modules) > 2,
        recent_tool_outputs_large=sum(
            len(o.content) for o in context.tool_outputs[-5:]
        ) > 10000,
        conversation_has_state_changes=context.has_requirement_changes,
        files_span_multiple_modules=len(set(
            Path(f).parts[0] for f in context.cached_files.keys()
        )) > 2,
        previous_turn_was_confused=context.last_response_had_clarification,
        task_is_continuation='continue' in prompt_lower or 
                            'same' in prompt_lower or
                            'also' in prompt_lower[:50],
    )
```

### 6.4 Recursive Depth = 2

**Decision**: Default max depth = 2, configurable up to 3.

**Rationale**: 
- Depth=1 handles most cases but fails on verification chains
- Your Ananke/Maze work involves constraint propagation that benefits from deeper recursion
- Depth=2 allows: Root → Analysis → Verification (three-layer reasoning)
- Cost is manageable: ~3-5x single call, still cheaper than naive 200K+ context

**Depth=2 Architecture**:

```python
class RecursiveREPL:
    """
    REPL environment that can spawn child REPLs for depth=2.
    """
    
    def __init__(self, context: Any, depth: int, max_depth: int = 2):
        self.context = context
        self.depth = depth
        self.max_depth = max_depth
        self.child_repls: List[RecursiveREPL] = []
        
    async def recursive_query(
        self, 
        query: str, 
        context: Any,
        spawn_repl: bool = False  # Allow sub-query to have its own REPL
    ) -> str:
        """
        Spawn a recursive call. If spawn_repl=True AND depth < max_depth,
        the child gets its own REPL environment for further decomposition.
        """
        if self.depth >= self.max_depth:
            # At max depth: simple completion, no REPL
            return await self._simple_completion(query, context)
        
        if spawn_repl:
            # Create child REPL with isolated state
            child = RecursiveREPL(
                context=context,
                depth=self.depth + 1,
                max_depth=self.max_depth
            )
            self.child_repls.append(child)
            return await child.run_rlm_loop(query)
        else:
            # Simple sub-call without REPL (faster, cheaper)
            return await self._simple_completion(query, context)
```

**When to use depth=2**:

| Pattern | Depth=1 | Depth=2 |
|---------|---------|---------|
| "Find the bug" | Root analyzes, sub-calls examine files | Same |
| "Find and verify fix" | Root analyzes, may miss verification | Root → Find → Verify chain |
| "Refactor safely" | Root plans, sub-calls check files | Root → Plan → Validate each change |
| "Compare approaches" | Root struggles with parallel analysis | Root → Analyze A → Analyze B → Compare |

### 6.5 Hybrid Mode with Conservative Simple Classification

**Decision**: Hybrid mode enabled, but **biased toward RLM activation**.

**Rationale**: False negatives (missing RLM when needed) are worse than false positives (using RLM when unnecessary). RLM overhead on simple queries is ~2-3 seconds—acceptable. Missing context on complex queries causes real failures.

**Simple Query Definition** (conservative):

```python
def is_definitely_simple(prompt: str, context: SessionContext) -> bool:
    """
    Returns True ONLY for queries that definitely don't need RLM.
    When in doubt, return False (use RLM).
    """
    prompt_lower = prompt.lower().strip()
    
    # Explicit simple patterns (exhaustive list)
    simple_patterns = [
        # Direct file operations
        r'^(show|cat|read|view|open)\s+[\w./]+$',
        # Single command execution
        r'^(run|execute)\s+(npm|yarn|pnpm|cargo|go|python|pytest)\s+\w+$',
        # Git status checks
        r'^git\s+(status|log|diff|branch)$',
        # Simple questions about single files
        r'^what\'?s?\s+in\s+[\w./]+\??$',
        # Acknowledgments
        r'^(ok|okay|thanks|got it|understood|yes|no|sure)\.?$',
    ]
    
    for pattern in simple_patterns:
        if re.match(pattern, prompt_lower):
            # Even if pattern matches, check context isn't complex
            if context.total_tokens < 20000 and not context.has_errors:
                return True
    
    # Short prompts with no context references
    if len(prompt) < 50 and context.total_tokens < 10000:
        # Check for ANY complexity indicators
        if not any(word in prompt_lower for word in [
            'why', 'how', 'find', 'fix', 'bug', 'error', 'all', 
            'every', 'change', 'update', 'refactor', 'test'
        ]):
            return True
    
    # Default: not simple (use RLM)
    return False
```

**Expected Distribution**:
- ~10-15% of queries classified as "definitely simple"
- ~85-90% routed through RLM
- Users can override with `/simple` command for explicit bypass

### 6.6 Streaming Trajectory Visibility

**Decision**: Full trajectory visibility with human-friendly streaming output.

**Rationale**: RLM orchestration is multi-step reasoning. Without visibility:
- Users can't debug why a query failed
- No way to learn what strategies work
- Feels like a black box (violates your constraint-driven principles)

**Trajectory Display Design**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ RLM Mode • depth=0/2 • task: multi_module_task                          │
├─────────────────────────────────────────────────────────────────────────┤
│ ▶ ANALYZE │ Examining context structure...                              │
│   ├─ conversation: 45 messages (23K tokens)                             │
│   ├─ files: 12 cached (67K tokens)                                      │
│   └─ tool_outputs: 8 recent (34K tokens)                                │
│                                                                         │
│ ▶ REPL    │ peek(tool_outputs[-1], 0, 1500)                            │
│   └─ [test output: FAIL auth/login.test.ts:45]                         │
│                                                                         │
│ ▶ REASON  │ Error is 500 vs expected 401. Need to trace auth handler.  │
│                                                                         │
│ ▶ REPL    │ search(files, 'AuthController')                            │
│   └─ Found in: src/auth/controllers/auth.controller.ts                  │
│                                                                         │
│ ▶ RECURSE │ depth=1 │ "How does login() handle validation failures?"   │
│   │       │ context: auth.controller.ts (2.3K tokens)                   │
│   │                                                                     │
│   │ ▶ ANALYZE │ Scanning controller implementation...                   │
│   │ ▶ REPL    │ search(context, 'validateCredentials')                 │
│   │ ▶ REASON  │ Found: throws generic Error, not HTTP error            │
│   │ ◀ RETURN  │ "Controller throws Error() not UnauthorizedError()"    │
│                                                                         │
│ ▶ REASON  │ Root cause identified. Preparing fix.                       │
│                                                                         │
│ ▶ FINAL   │ Bug in auth.controller.ts:23 - throws Error() instead of   │
│           │ UnauthorizedError(). Fix: change to throw new               │
│           │ UnauthorizedError('Invalid credentials')                    │
└─────────────────────────────────────────────────────────────────────────┘
```

**Streaming Implementation**:

```python
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator, Optional
import asyncio

class TrajectoryEventType(Enum):
    RLM_START = "rlm_start"
    ANALYZE = "analyze"
    REPL_EXEC = "repl_exec"
    REPL_RESULT = "repl_result"
    REASON = "reason"
    RECURSE_START = "recurse_start"
    RECURSE_END = "recurse_end"
    FINAL = "final"
    ERROR = "error"
    TOOL_USE = "tool_use"  # Normal Claude Code tool

@dataclass
class TrajectoryEvent:
    type: TrajectoryEventType
    depth: int
    content: str
    metadata: Optional[dict] = None
    timestamp: float = None
    
    def __post_init__(self):
        self.timestamp = self.timestamp or time.time()

class TrajectoryRenderer:
    """
    Renders trajectory events to terminal with streaming support.
    Designed for human readability during execution.
    """
    
    ICONS = {
        TrajectoryEventType.RLM_START: "◆",
        TrajectoryEventType.ANALYZE: "▶",
        TrajectoryEventType.REPL_EXEC: "▶",
        TrajectoryEventType.REPL_RESULT: "└─",
        TrajectoryEventType.REASON: "▶",
        TrajectoryEventType.RECURSE_START: "▶",
        TrajectoryEventType.RECURSE_END: "◀",
        TrajectoryEventType.FINAL: "▶",
        TrajectoryEventType.ERROR: "✗",
        TrajectoryEventType.TOOL_USE: "⚙",
    }
    
    LABELS = {
        TrajectoryEventType.RLM_START: "RLM",
        TrajectoryEventType.ANALYZE: "ANALYZE",
        TrajectoryEventType.REPL_EXEC: "REPL",
        TrajectoryEventType.REPL_RESULT: "",
        TrajectoryEventType.REASON: "REASON",
        TrajectoryEventType.RECURSE_START: "RECURSE",
        TrajectoryEventType.RECURSE_END: "RETURN",
        TrajectoryEventType.FINAL: "FINAL",
        TrajectoryEventType.ERROR: "ERROR",
        TrajectoryEventType.TOOL_USE: "TOOL",
    }
    
    def __init__(self, verbosity: str = "normal"):
        """
        verbosity: "minimal" | "normal" | "verbose" | "debug"
        """
        self.verbosity = verbosity
        self.current_depth = 0
        
    def render_event(self, event: TrajectoryEvent) -> str:
        """Render single event to terminal string"""
        indent = "│   " * event.depth
        icon = self.ICONS[event.type]
        label = self.LABELS[event.type]
        
        # Depth indicator for recursive calls
        depth_indicator = ""
        if event.type == TrajectoryEventType.RECURSE_START:
            depth_indicator = f" │ depth={event.depth + 1} │"
        
        # Truncate content based on verbosity
        content = self._truncate_content(event.content, event.type)
        
        # Color coding (ANSI)
        color = self._get_color(event.type)
        reset = "\033[0m"
        dim = "\033[2m"
        
        if event.type == TrajectoryEventType.REPL_RESULT:
            # Indented result, dimmed
            return f"{indent}  {dim}{icon} [{content}]{reset}"
        elif label:
            return f"{indent}{color}{icon} {label:7}{reset}{depth_indicator} │ {content}"
        else:
            return f"{indent}{content}"
    
    def _truncate_content(self, content: str, event_type: TrajectoryEventType) -> str:
        """Truncate based on verbosity and event type"""
        limits = {
            "minimal": {"default": 60, "repl_result": 40, "reason": 80},
            "normal": {"default": 120, "repl_result": 80, "reason": 200},
            "verbose": {"default": 300, "repl_result": 200, "reason": 500},
            "debug": {"default": 1000, "repl_result": 1000, "reason": 1000},
        }
        
        key = "repl_result" if event_type == TrajectoryEventType.REPL_RESULT else \
              "reason" if event_type == TrajectoryEventType.REASON else "default"
        limit = limits[self.verbosity][key]
        
        if len(content) <= limit:
            return content
        return content[:limit - 3] + "..."
    
    def _get_color(self, event_type: TrajectoryEventType) -> str:
        """ANSI color codes for event types"""
        colors = {
            TrajectoryEventType.RLM_START: "\033[1;36m",  # Bold cyan
            TrajectoryEventType.ANALYZE: "\033[34m",       # Blue
            TrajectoryEventType.REPL_EXEC: "\033[33m",     # Yellow
            TrajectoryEventType.REASON: "\033[32m",        # Green
            TrajectoryEventType.RECURSE_START: "\033[35m", # Magenta
            TrajectoryEventType.RECURSE_END: "\033[35m",   # Magenta
            TrajectoryEventType.FINAL: "\033[1;32m",       # Bold green
            TrajectoryEventType.ERROR: "\033[1;31m",       # Bold red
            TrajectoryEventType.TOOL_USE: "\033[36m",      # Cyan
        }
        return colors.get(event_type, "")


class StreamingTrajectory:
    """
    Manages streaming trajectory output during RLM execution.
    """
    
    def __init__(self, renderer: TrajectoryRenderer):
        self.renderer = renderer
        self.events: List[TrajectoryEvent] = []
        self.subscribers: List[asyncio.Queue] = []
        
    async def emit(self, event: TrajectoryEvent):
        """Emit event to all subscribers"""
        self.events.append(event)
        rendered = self.renderer.render_event(event)
        
        for queue in self.subscribers:
            await queue.put(rendered)
    
    def subscribe(self) -> asyncio.Queue:
        """Subscribe to trajectory stream"""
        queue = asyncio.Queue()
        self.subscribers.append(queue)
        return queue
    
    async def stream(self) -> AsyncIterator[str]:
        """Async iterator for streaming output"""
        queue = self.subscribe()
        try:
            while True:
                line = await queue.get()
                yield line
        except asyncio.CancelledError:
            self.subscribers.remove(queue)
            raise
    
    def get_full_trajectory(self) -> List[TrajectoryEvent]:
        """Get complete trajectory for logging/replay"""
        return self.events.copy()
    
    def export_json(self) -> str:
        """Export trajectory as JSON for analysis"""
        return json.dumps([
            {
                "type": e.type.value,
                "depth": e.depth,
                "content": e.content,
                "metadata": e.metadata,
                "timestamp": e.timestamp,
            }
            for e in self.events
        ], indent=2)
```

**Integration with Claude Code Output**:

```python
class RLMOrchestrator:
    """Main orchestration loop with streaming trajectory"""
    
    async def run(
        self, 
        query: str, 
        context: SessionContext
    ) -> AsyncIterator[Union[TrajectoryEvent, ToolUse, AssistantMessage]]:
        """
        Run RLM loop, yielding events as they occur.
        Consumers can render trajectory AND tool outputs together.
        """
        trajectory = StreamingTrajectory(
            TrajectoryRenderer(verbosity=self.config.trajectory_verbosity)
        )
        
        # Start event
        await trajectory.emit(TrajectoryEvent(
            type=TrajectoryEventType.RLM_START,
            depth=0,
            content=f"depth=0/{self.config.max_depth} • task: {self.activation_reason}",
            metadata={"query": query, "context_tokens": context.total_tokens}
        ))
        yield trajectory.events[-1]
        
        # Main loop
        repl = RecursiveREPL(context, depth=0, max_depth=self.config.max_depth)
        
        async for event in self._rlm_loop(query, repl, trajectory):
            yield event
            
            # Check for final answer
            if event.type == TrajectoryEventType.FINAL:
                break
            
            # Check for tool use (pass through to Claude Code)
            if event.type == TrajectoryEventType.TOOL_USE:
                # Let Claude Code handle the tool, get result
                tool_result = yield event  # Bidirectional generator
                # Feed result back into REPL
                repl.inject_tool_result(tool_result)
```

**Verbosity Levels**:

| Level | Shows | Use Case |
|-------|-------|----------|
| `minimal` | RECURSE, FINAL, ERROR only | Production, minimal noise |
| `normal` | All events, truncated content | Default, balanced visibility |
| `verbose` | All events, full content | Debugging specific issues |
| `debug` | Everything + internal state | Development, trajectory analysis |

**Configuration**:

```json
{
  "rlm": {
    "trajectory_verbosity": "normal",
    "trajectory_colors": true,
    "trajectory_timestamps": false,
    "trajectory_export_path": "~/.claude/rlm-trajectories/",
    "trajectory_export_format": "json"
  }
}
```

---

### Phase 1: Core Infrastructure (Week 1-2)

| Task | Description | Deliverable |
|------|-------------|-------------|
| Context Manager | Externalize conversation/files to Python vars | `context_manager.py` |
| REPL Sandbox | Secure Python execution environment | `repl_environment.py` |
| Basic Router | Route root vs. recursive calls | `router_integration.py` |
| CLI Integration | `/rlm` command to toggle mode | Plugin command |

**Validation**: NIAH benchmark on 500K token synthetic context

### Phase 2: Claude Code Integration (Week 3-4)

| Task | Description | Deliverable |
|------|-------------|-------------|
| Hook System | Automatic activation on threshold | `hooks.json` |
| Tool Preservation | Ensure bash/edit/read tools work in RLM mode | Integration tests |
| State Persistence | Save/restore RLM state across sessions | State manager |
| Error Handling | Graceful fallback to standard mode | Error handlers |

**Validation**: Real coding tasks (bug fix, feature implementation)

### Phase 3: Optimization (Week 5-6)

| Task | Description | Deliverable |
|------|-------------|-------------|
| Async Sub-calls | Parallel recursive queries | Async handler |
| Caching | Cache summarizations, frequently-accessed context | Cache layer |
| Cost Tracking | Monitor token usage in RLM mode | Cost dashboard |
| Prompt Tuning | Optimize root/recursive prompts | Prompt library |

**Validation**: OOLONG benchmark subset, cost comparison

### Phase 4: Advanced Features (Week 7-8)

| Task | Description | Deliverable |
|------|-------------|-------------|
| Depth=2 Support | Enable deeper recursion for massive contexts | Recursive REPL |
| Model Routing | Different models for different query types | Smart router |
| Learning | Adapt strategies based on success patterns | Feedback loop |
| Visualization | Trajectory viewer for debugging | Web UI |

**Validation**: Production dogfooding on real projects

---

## 8. Technical Considerations

### 8.1 Security

The REPL environment must be sandboxed:

```python
# ALLOWED
- String manipulation
- List/dict operations
- Regex (re module)
- JSON parsing
- len(), str(), sorted(), etc.

# BLOCKED
- File system access (use Claude Code's tools)
- Network access (use Claude Code's tools)  
- subprocess/os.system
- import of arbitrary modules
- eval/exec of user-provided strings
```

**Implementation**: Use `RestrictedPython` with custom guards, or containerized execution.

### 8.2 Performance

| Operation | Target Latency | Strategy |
|-----------|---------------|----------|
| Context externalization | <500ms | Lazy serialization |
| REPL execution | <100ms | Persistent interpreter |
| Recursive sub-call | <10s | Streaming, timeout |
| Full RLM turn | <30s | Parallel sub-calls |

### 8.3 Cost Model

```
RLM Cost = Root Prompt Tokens + Σ(Recursive Call Tokens)

Where:
- Root Prompt Tokens ≈ 2K (minimal prompt) + REPL outputs
- Recursive Call = Query + Context chunk (typically 4-16K)
- Expected: 2-5 recursive calls per complex query
- Breakeven: When context > 60K tokens (RLM becomes cheaper)
```

### 8.4 State Management

```python
class RLMSession:
    """Persistent state across turns"""
    
    conversation: List[Message]  # Full history
    files: Dict[str, str]        # Cached files
    working_memory: Dict         # User-defined state
    repl_state: Dict            # Python interpreter state
    trajectory: List[RLMStep]   # For debugging/replay
    
    def save(self, path: Path): ...
    def load(cls, path: Path) -> 'RLMSession': ...
```

---

## 9. Open Questions

### 9.1 Resolved ✓

| Question | Decision | Section |
|----------|----------|---------|
| Activation strategy | Complexity-based, not token count | §6.3 |
| Recursive depth | Default=2, max=3 | §6.4 |
| Hybrid mode | Enabled, conservative simple classification | §6.5 |
| Trajectory visibility | Streaming, human-friendly, configurable verbosity | §6.6 |

### 9.2 Architecture (Remaining)

1. **REPL Isolation at Depth=2**: When depth=1 spawns a child REPL, should it share any state with parent?
   - Proposal: Fully isolated, but can pass serialized context
   - Risk: Memory overhead of multiple interpreter instances

2. **Tool Interleaving**: When RLM decides to use a Claude Code tool (bash, edit), how does that flow?
   - Option A: Yield control to Claude Code, resume RLM after
   - Option B: RLM orchestrates tool use through its own handler
   - Proposal: Option A (preserves Claude Code's permission model)

3. **Checkpoint/Resume**: Can an RLM session be paused and resumed?
   - Use case: Long-running tasks, cost management
   - Complexity: REPL state serialization, trajectory replay

### 9.3 Complexity Classifier

1. **False Positive Cost**: What's the overhead when RLM activates unnecessarily?
   - Needs: Benchmark simple queries through RLM vs direct
   - Hypothesis: ~2-3 second overhead acceptable

2. **Classifier Calibration**: How to tune complexity signals for different codebases?
   - Proposal: Project-level config for signal weights
   - Alternative: Learning from user feedback (thumbs up/down)

3. **Multi-Modal Signals**: Should image/diagram content affect complexity scoring?
   - Current: Not considered
   - Future: If Claude Code adds vision, integrate

### 9.4 Trajectory UX

1. **Terminal Width**: How to handle narrow terminals?
   - Proposal: Adaptive truncation, collapsible depth levels

2. **Interruption**: Can user interrupt mid-RLM and redirect?
   - Use case: "Actually, focus on X instead"
   - Needs: REPL state checkpoint, trajectory branch

3. **Replay**: Should users be able to replay trajectories?
   - Use case: "Show me how you solved that yesterday"
   - Proposal: JSON export + viewer tool (phase 4)

### 9.5 Research (Empirical Testing Needed)

1. **Optimal Depth=2 Strategy**: When should depth=1 spawn a REPL child vs simple completion?
   - Hypothesis: Spawn REPL for verification tasks, simple for extraction
   - Needs: A/B testing on real coding tasks

2. **Model Selection at Depth**: Is Haiku sufficient for depth=2, or does it fail on verification?
   - Needs: Benchmark Haiku vs Sonnet at depth=2

3. **Complexity Classifier Accuracy**: What's the precision/recall of the heuristic classifier?
   - Needs: Labeled dataset of "simple" vs "complex" queries

4. **Cost-Performance Curve**: At what complexity score does RLM ROI become positive?
   - Needs: Token cost tracking + success rate by complexity

---

## 10. Success Criteria

### 10.1 Quantitative

| Metric | Target | Measurement |
|--------|--------|-------------|
| Context handling | 500K+ tokens | Synthetic benchmarks |
| Accuracy on long context | +30% vs baseline | OOLONG subset |
| Cost per query | ≤ baseline at 80K+ | Token tracking |
| Latency overhead | <50% vs direct | E2E timing |
| Complexity classifier precision | >85% | Labeled test set |
| Complexity classifier recall | >95% | Labeled test set (favor false positives) |
| Depth=2 utilization | 20-30% of RLM queries | Production telemetry |
| Trajectory render latency | <10ms per event | Performance profiling |

### 10.2 Qualitative

- [ ] Users understand what RLM is doing (trajectory is readable)
- [ ] Tool execution works identically to standard mode
- [ ] Session state persists correctly across RLM/standard mode switches
- [ ] Debugging trajectory is inspectable and exportable
- [ ] Community can extend via standard plugin mechanisms
- [ ] Complexity activation feels "smart" (users don't notice false positives)
- [ ] Depth=2 is used for verification, not unnecessarily
- [ ] `/simple` escape hatch works reliably

### 10.3 User Experience Targets

| Scenario | Target Experience |
|----------|-------------------|
| Simple query in complex session | Bypass RLM, <2s response |
| Complex multi-file task | RLM activates, trajectory visible, 30-60s |
| Interruption mid-RLM | Graceful stop, context preserved |
| Verbosity change | Immediate effect, no restart |
| Trajectory export | JSON saved, replayable in viewer |

---

## 11. References

1. [Recursive Language Models Paper](https://arxiv.org/abs/2512.24601v1) - Zhang, Kraska, Khattab
2. [RLM Official Implementation](https://github.com/alexzhang13/rlm)
3. [Claude Code Documentation](https://code.claude.com/docs)
4. [Claude Code Plugins](https://github.com/anthropics/claude-code/tree/main/plugins)
5. [Claude Code Router](https://github.com/musistudio/claude-code-router)
6. [OpenCode](https://github.com/anomalyco/opencode) - Alternative architecture reference
7. [Anthropic Context Engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)

---

## Appendix A: Example RLM Trace (Depth=2)

**Scenario**: User asks to refactor auth module and verify no regressions.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ ◆ RLM     │ depth=0/2 • task: multi_module_task+verification                │
├─────────────────────────────────────────────────────────────────────────────┤
│ ▶ ANALYZE │ Context: 156K tokens across auth/, api/, tests/                 │
│   ├─ conversation: 45 messages (23K tokens)                                 │
│   ├─ files: 18 cached (98K tokens) - auth/*, api/*, tests/*                 │
│   └─ tool_outputs: 3 recent (35K tokens) - test failures                    │
│                                                                             │
│ ▶ REPL    │ peek(tool_outputs[-1], 0, 2000)                                 │
│   └─ [npm test: FAIL auth/login.test.ts:45 - expected 401, got 500]        │
│                                                                             │
│ ▶ REASON  │ Test failure indicates error handling issue. Need to:           │
│           │ 1) Find the bug, 2) Fix it, 3) Verify no side effects          │
│                                                                             │
│ ▶ RECURSE │ depth=1 │ "Analyze auth error handling and find 500 source"    │
│   │       │ context: files['src/auth/**'] (34K tokens)                      │
│   │       │ spawn_repl: true                                                │
│   │                                                                         │
│   │ ▶ ANALYZE │ Scanning 8 auth module files...                             │
│   │ ▶ REPL    │ search(context, r'throw.*Error|catch.*Error')               │
│   │   └─ [Found 12 matches across 5 files]                                  │
│   │ ▶ REPL    │ recursive_query("Which throw statements lack HTTP codes?", │
│   │           │                  matches[:6], spawn_repl=False)             │
│   │   │                                                                     │
│   │   │ ▶ RECURSE │ depth=2 │ simple extraction query                       │
│   │   │   └─ [auth.controller.ts:23 throws generic Error]                   │
│   │   │   └─ [auth.service.ts:89 throws generic Error]                      │
│   │   │ ◀ RETURN  │ "2 locations throw generic Error without HTTP status"   │
│   │                                                                         │
│   │ ▶ REASON  │ auth.controller.ts:23 is in login path, likely culprit     │
│   │ ◀ RETURN  │ "Bug: auth.controller.ts:23 throws Error() not             │
│   │           │ UnauthorizedError(). Also check auth.service.ts:89"        │
│                                                                             │
│ ▶ REASON  │ Confirmed bug location. Before fixing, verify call sites.       │
│                                                                             │
│ ▶ RECURSE │ depth=1 │ "Find all callers of AuthController.login()"         │
│   │       │ context: files['src/**'] (67K tokens)                           │
│   │       │ spawn_repl: true                                                │
│   │                                                                         │
│   │ ▶ REPL    │ search(context, r'AuthController.*login|login.*route')      │
│   │   └─ [Found in: routes/auth.ts, middleware/session.ts, tests/*]        │
│   │ ▶ RECURSE │ depth=2 │ "Do any callers depend on catching Error?"       │
│   │   │ ▶ ANALYZE │ Checking error handling in 3 call sites...             │
│   │   │ ◀ RETURN  │ "No callers catch generic Error specifically.          │
│   │   │           │ All use HTTP status codes. Safe to change."            │
│   │ ◀ RETURN  │ "3 call sites verified. Change is safe."                   │
│                                                                             │
│ ▶ REASON  │ Verification complete. Safe to proceed with fix.               │
│                                                                             │
│ ⚙ TOOL    │ Edit src/auth/controllers/auth.controller.ts                   │
│   └─ Line 23: throw new Error(...) → throw new UnauthorizedError(...)      │
│                                                                             │
│ ⚙ TOOL    │ Edit src/auth/services/auth.service.ts                         │
│   └─ Line 89: throw new Error(...) → throw new UnauthorizedError(...)      │
│                                                                             │
│ ⚙ TOOL    │ bash: npm test -- --grep "auth"                                │
│   └─ [PASS: 12/12 tests passing]                                           │
│                                                                             │
│ ▶ FINAL   │ Fixed auth error handling:                                      │
│           │ • auth.controller.ts:23 - now throws UnauthorizedError (401)   │
│           │ • auth.service.ts:89 - now throws UnauthorizedError (401)      │
│           │ Verified: All 12 auth tests passing, no regression.             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Trajectory Statistics**:
```
Total turns: 1
RLM depth reached: 2
Recursive calls: 4 (2 at depth=1, 2 at depth=2)
REPL executions: 6
Tool uses: 3 (2 edits, 1 bash)
Total tokens: ~45K (vs ~156K for naive approach)
Estimated cost: $0.47 (vs $1.56 naive)
Time: 34 seconds
```

**What Depth=2 Enabled**:
- Depth=1 found the bug locations
- Depth=2 verified no callers depend on current behavior (safe to change)
- Without depth=2, would have either skipped verification or used expensive root-level analysis

---

## Appendix B: Comparison with Alternatives

| Approach | Pros | Cons |
|----------|------|------|
| **RLM Plugin** | Preserves Claude Code, community extensible | Complexity in state sync |
| **Claude Code Router** | Simple model switching | No context decomposition |
| **OpenCode Fork** | Full control | Loses Anthropic updates |
| **Custom Agent** | Maximum flexibility | Reimplements everything |

**Recommendation**: RLM Plugin is optimal for Rand's stated goals of better reasoning and context handling while retaining Claude Code's core capabilities.
