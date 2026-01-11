# Testing Strategy

This document defines the testing approach for RLM-Claude-Code.

## Test Categories

### 1. Unit Tests

Test individual functions in isolation.

**Location**: `tests/unit/`

**Naming**: `test_<module>_<function>.py`

**Example**:
```python
# tests/unit/test_complexity_classifier.py
import pytest
from src.complexity_classifier import extract_complexity_signals, TaskComplexitySignals

class TestExtractComplexitySignals:
    """Tests for extract_complexity_signals function."""
    
    def test_detects_multi_file_reference(self):
        """Prompt mentioning multiple files sets references_multiple_files."""
        prompt = "Fix the bug in auth.py and update tests.py"
        context = MockContext(total_tokens=5000)
        
        signals = extract_complexity_signals(prompt, context)
        
        assert signals.references_multiple_files is True
    
    def test_detects_cross_context_reasoning(self):
        """'Why X given Y' patterns trigger cross_context_reasoning."""
        prompt = "Why is the test failing given the fix we made?"
        context = MockContext(total_tokens=5000)
        
        signals = extract_complexity_signals(prompt, context)
        
        assert signals.requires_cross_context_reasoning is True
    
    def test_simple_query_no_signals(self):
        """Simple queries should not trigger complexity signals."""
        prompt = "Show me package.json"
        context = MockContext(total_tokens=1000)
        
        signals = extract_complexity_signals(prompt, context)
        
        assert signals.references_multiple_files is False
        assert signals.requires_cross_context_reasoning is False
        assert signals.debugging_task is False
```

### 2. Integration Tests

Test component interactions.

**Location**: `tests/integration/`

**Example**:
```python
# tests/integration/test_rlm_loop.py
import pytest
from src.orchestrator import RLMOrchestrator
from src.context_manager import SessionContext

class TestRLMLoop:
    """Tests for the complete RLM orchestration loop."""
    
    @pytest.fixture
    def orchestrator(self):
        return RLMOrchestrator(config=TestConfig())
    
    @pytest.fixture
    def large_context(self):
        """Create a context with 100K tokens of synthetic data."""
        return SessionContext(
            messages=[...],
            files={...},
            tool_outputs=[...]
        )
    
    @pytest.mark.asyncio
    async def test_rlm_finds_needle_in_haystack(self, orchestrator, large_context):
        """RLM should find specific information in large context."""
        # Inject known needle
        large_context.files["needle.txt"] = "SECRET_VALUE_12345"
        
        events = []
        async for event in orchestrator.run(
            query="Find the secret value in the files",
            context=large_context
        ):
            events.append(event)
        
        final_event = next(e for e in events if e.type == TrajectoryEventType.FINAL)
        assert "SECRET_VALUE_12345" in final_event.content
    
    @pytest.mark.asyncio
    async def test_respects_max_depth(self, orchestrator, large_context):
        """RLM should not exceed configured max depth."""
        orchestrator.config.max_depth = 2
        
        max_depth_seen = 0
        async for event in orchestrator.run(
            query="Analyze everything",
            context=large_context
        ):
            if hasattr(event, 'depth'):
                max_depth_seen = max(max_depth_seen, event.depth)
        
        assert max_depth_seen <= 2
```

### 3. Property-Based Tests (Hypothesis)

Test invariants that should hold for all inputs.

**Location**: `tests/property/`

**Example**:
```python
# tests/property/test_context_manager.py
from hypothesis import given, strategies as st, settings
from src.context_manager import externalize_conversation, Message

@given(st.lists(st.builds(
    Message,
    role=st.sampled_from(['user', 'assistant']),
    content=st.text(min_size=1, max_size=10000)
), min_size=0, max_size=100))
@settings(max_examples=200)
def test_externalize_preserves_message_count(messages):
    """Externalization should preserve all messages."""
    externalized = externalize_conversation(messages)
    
    assert len(externalized) == len(messages)

@given(st.dictionaries(
    keys=st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=['L', 'N', 'P'])),
    values=st.text(min_size=0, max_size=50000),
    min_size=0,
    max_size=50
))
def test_externalize_files_preserves_content(files):
    """File externalization should preserve exact content."""
    from src.context_manager import externalize_files
    
    externalized = externalize_files(files)
    
    for path, content in files.items():
        assert externalized[path] == content

@given(st.integers(min_value=0, max_value=3))
def test_recursive_depth_never_negative(depth):
    """Recursive handler should never create negative depths."""
    from src.recursive_handler import RecursiveREPL
    
    repl = RecursiveREPL(context="test", depth=depth, max_depth=3)
    
    assert repl.depth >= 0
```

### 4. Trajectory Snapshot Tests

Test that RLM behavior produces expected trajectories.

**Location**: `tests/snapshots/`

**Example**:
```python
# tests/snapshots/test_trajectory_snapshots.py
import pytest
from src.orchestrator import RLMOrchestrator

class TestTrajectorySnapshots:
    """Snapshot tests for trajectory output."""
    
    @pytest.mark.asyncio
    async def test_simple_query_trajectory(self, snapshot):
        """Simple query should produce minimal trajectory."""
        orchestrator = RLMOrchestrator(config=TestConfig())
        context = MockContext(total_tokens=5000)
        
        events = []
        async for event in orchestrator.run(
            query="What's in README.md?",
            context=context
        ):
            events.append(event.to_dict())
        
        snapshot.assert_match(events, "simple_query_trajectory.json")
    
    @pytest.mark.asyncio
    async def test_recursive_query_trajectory(self, snapshot):
        """Recursive query should show depth transitions."""
        orchestrator = RLMOrchestrator(config=TestConfig())
        context = MockContext(total_tokens=100000)
        
        events = []
        async for event in orchestrator.run(
            query="Find the bug and verify the fix won't break anything",
            context=context
        ):
            events.append(event.to_dict())
        
        snapshot.assert_match(events, "recursive_query_trajectory.json")
```

### 5. Security Tests

Verify sandbox restrictions.

**Location**: `tests/security/`

**Example**:
```python
# tests/security/test_repl_sandbox.py
import pytest
from src.repl_environment import RLMEnvironment

class TestREPLSandbox:
    """Security tests for REPL sandbox."""
    
    @pytest.fixture
    def repl(self):
        return RLMEnvironment(context=MockContext())
    
    def test_blocks_os_system(self, repl):
        """os.system should be blocked."""
        with pytest.raises(SecurityError):
            repl.execute("import os; os.system('ls')")
    
    def test_blocks_subprocess(self, repl):
        """Direct subprocess access should be blocked."""
        with pytest.raises(SecurityError):
            repl.execute("import subprocess; subprocess.run(['ls'])")
    
    def test_blocks_file_write(self, repl):
        """File writes should be blocked."""
        with pytest.raises(SecurityError):
            repl.execute("open('/tmp/test', 'w').write('pwned')")
    
    def test_blocks_network(self, repl):
        """Network access should be blocked."""
        with pytest.raises(SecurityError):
            repl.execute("import urllib.request; urllib.request.urlopen('http://evil.com')")
    
    def test_allows_pydantic(self, repl):
        """Pydantic should be accessible."""
        result = repl.execute("""
from pydantic import BaseModel
class Test(BaseModel):
    value: int
Test(value=42).value
""")
        assert result.output == 42
    
    def test_allows_hypothesis(self, repl):
        """Hypothesis should be accessible."""
        result = repl.execute("""
from hypothesis import strategies as st
st.integers(min_value=0, max_value=10).example()
""")
        assert 0 <= result.output <= 10
    
    def test_allows_cpmpy(self, repl):
        """CPMpy should be accessible for constraint solving."""
        result = repl.execute("""
import cpmpy as cp
x = cp.intvar(1, 10, name="x")
y = cp.intvar(1, 10, name="y")
model = cp.Model([x + y == 10, x < y])
model.solve()
(x.value(), y.value())
""")
        x_val, y_val = result.output
        assert x_val + y_val == 10
        assert x_val < y_val
    
    def test_allowed_subprocess_ty(self, repl):
        """ty subprocess should be allowed."""
        result = repl.execute("""
await sandbox_exec(['ty', '--version'])
""")
        assert result.returncode == 0
    
    def test_allowed_subprocess_ruff(self, repl):
        """ruff subprocess should be allowed."""
        result = repl.execute("""
await sandbox_exec(['ruff', '--version'])
""")
        assert result.returncode == 0
    
    def test_blocked_subprocess_curl(self, repl):
        """curl subprocess should be blocked."""
        with pytest.raises(SecurityError):
            repl.execute("await sandbox_exec(['curl', 'http://evil.com'])")
```

### 6. Benchmark Tests

Measure performance against targets.

**Location**: `tests/benchmarks/`

**Example**:
```python
# tests/benchmarks/test_performance.py
import pytest

class TestPerformance:
    """Performance benchmarks."""
    
    def test_complexity_classifier_latency(self, benchmark):
        """Complexity classifier must run under 50ms."""
        from src.complexity_classifier import extract_complexity_signals
        
        prompt = "Fix the authentication bug and update the tests"
        context = MockContext(total_tokens=50000)
        
        result = benchmark(extract_complexity_signals, prompt, context)
        
        assert benchmark.stats.stats.mean < 0.050  # 50ms
    
    def test_repl_execution_latency(self, benchmark):
        """REPL execution must complete under 100ms."""
        from src.repl_environment import RLMEnvironment
        
        repl = RLMEnvironment(context=MockContext())
        code = "len(files)"
        
        result = benchmark(repl.execute, code)
        
        assert benchmark.stats.stats.mean < 0.100  # 100ms
    
    def test_trajectory_render_latency(self, benchmark):
        """Trajectory render must complete under 10ms per event."""
        from src.trajectory import TrajectoryRenderer, TrajectoryEvent, TrajectoryEventType
        
        renderer = TrajectoryRenderer(verbosity="normal")
        event = TrajectoryEvent(
            type=TrajectoryEventType.REASON,
            depth=1,
            content="Analyzing the error patterns in the output"
        )
        
        result = benchmark(renderer.render_event, event)
        
        assert benchmark.stats.stats.mean < 0.010  # 10ms
```

## Running Tests

```bash
# All tests
pytest tests/ -v

# Specific category
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/property/ -v
pytest tests/security/ -v
pytest tests/benchmarks/ --benchmark-only

# With coverage
pytest tests/ -v --cov=src/ --cov-report=html

# Watch mode
pytest-watch tests/ -v

# Specific test file
pytest tests/unit/test_complexity_classifier.py -v

# Specific test
pytest tests/unit/test_complexity_classifier.py::TestExtractComplexitySignals::test_detects_multi_file_reference -v
```

## Test Configuration

```toml
# pyproject.toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
markers = [
    "hypothesis: property-based tests",
    "slow: tests that take >1s",
    "security: security-related tests",
]
addopts = "-v --tb=short"

[tool.coverage.run]
source = ["src"]
branch = true

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "raise NotImplementedError",
]
```

## Fixtures

Common fixtures in `tests/conftest.py`:

```python
# tests/conftest.py
import pytest
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class MockContext:
    """Mock SessionContext for testing."""
    total_tokens: int = 10000
    messages: List[dict] = None
    files: Dict[str, str] = None
    tool_outputs: List[str] = None
    
    def __post_init__(self):
        self.messages = self.messages or []
        self.files = self.files or {"README.md": "# Test"}
        self.tool_outputs = self.tool_outputs or []

@dataclass  
class TestConfig:
    """Test configuration."""
    max_depth: int = 2
    rlm_mode: str = "always"
    trajectory_verbosity: str = "normal"

@pytest.fixture
def mock_context():
    return MockContext()

@pytest.fixture
def test_config():
    return TestConfig()

@pytest.fixture
def large_context():
    """Context with 100K tokens of synthetic data."""
    files = {f"file_{i}.py": f"# Content {i}\n" * 1000 for i in range(50)}
    return MockContext(
        total_tokens=100000,
        files=files
    )
```

## Test Data

Store test fixtures in `tests/fixtures/`:

```
tests/fixtures/
├── contexts/
│   ├── small_context.json
│   ├── large_context.json
│   └── error_context.json
├── trajectories/
│   ├── expected_simple.json
│   └── expected_recursive.json
└── prompts/
    ├── simple_prompts.txt
    ├── complex_prompts.txt
    └── edge_cases.txt
```
