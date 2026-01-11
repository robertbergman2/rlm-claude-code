# Code Review Process

Review all changes following this checklist before committing. Provide specific, actionable feedback with `file:line` references.

## Review Categories

### 1. Correctness

**Does the code do what it's supposed to?**

- [ ] Implementation matches spec section (cite paragraph)
- [ ] Edge cases handled (empty context, max depth, timeout)
- [ ] Error states produce meaningful messages
- [ ] No regressions in existing functionality

**RLM-Specific Checks**:
- [ ] Context externalization preserves all data
- [ ] REPL execution is properly sandboxed
- [ ] Recursive calls respect depth limits
- [ ] Trajectory events emitted in correct order

### 2. Type Safety

**Are types correct and complete?**

```bash
# Must pass with no errors
ty check src/

# Check for Any leakage
ty check src/ --strict
```

- [ ] All public functions have return type annotations
- [ ] Pydantic models validate at runtime boundaries
- [ ] No `# type: ignore` without justification comment
- [ ] Generic types used where appropriate (not `Any`)

### 3. Testing

**Is the code adequately tested?**

- [ ] Unit tests for new functions
- [ ] Integration tests for component interactions
- [ ] Hypothesis property tests for data transformations
- [ ] Trajectory snapshot tests for RLM behavior

**Coverage Requirements**:
| Component | Minimum Coverage |
|-----------|-----------------|
| context_manager.py | 90% |
| repl_environment.py | 85% |
| recursive_handler.py | 90% |
| complexity_classifier.py | 95% |
| trajectory_renderer.py | 80% |

### 4. Performance

**Does the code meet performance targets?**

- [ ] Complexity classifier runs in <50ms
- [ ] REPL execution per cell <100ms
- [ ] Trajectory render per event <10ms
- [ ] No O(n²) algorithms on context size

**Profiling Command**:
```bash
pytest tests/benchmarks/ --benchmark-only
```

### 5. Security

**Is the sandbox secure?**

- [ ] No `eval()` on untrusted input
- [ ] subprocess calls use allowlist
- [ ] No filesystem writes outside approved paths
- [ ] API keys not logged or exposed in trajectories

### 6. Style

**Does the code follow project conventions?**

```bash
# Must pass with no errors
ruff check src/
ruff format src/ --check
```

- [ ] Docstrings on public APIs (Google style)
- [ ] Meaningful variable names (no single letters except loops)
- [ ] Functions under 50 lines (extract if longer)
- [ ] No commented-out code

## Issue Classification

### Blocking Issues

**Must fix before commit:**

- Type errors (`ty check` failures)
- Test failures
- Security vulnerabilities
- Spec violations
- Data loss scenarios

### Non-Blocking Issues

**File as GitHub issues for later:**

- Performance improvements below threshold
- Additional test coverage
- Documentation gaps
- Refactoring opportunities
- Nice-to-have features

## Review Output Format

```markdown
## Code Review: [component/feature]

### Blocking Issues

1. **[file:line]** [Category] Description
   - Current: `code snippet`
   - Should be: `suggested fix`

### Non-Blocking Issues

1. **[file:line]** [Category] Description (filed as #issue)

### Approved Changes

- [file] Brief description of what changed

### Notes

Any context for future sessions.
```

## Slash Command

Use `/code-review` to trigger this process:

```markdown
Review the current changes following docs/process/code-review.md.

Focus areas:
- Correctness: Implementation matches spec?
- Types: ty check passes?
- Tests: Coverage adequate?
- Security: Sandbox maintained?

Provide file:line references for all issues.
```
