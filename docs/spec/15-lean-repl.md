# SPEC-15: Lean REPL Integration

## Overview

This specification defines capabilities for integrating a Lean 4 REPL into RLM, enabling formal verification, theorem proving, and mathematical reasoning with machine-checked guarantees.

**Status**: DEFERRED - Specs and epics created for future implementation.

**Research Basis**:
- [Lean 4 REPL](https://github.com/leanprover-community/repl) - JSON-based Lean interface
- [lean-repl-py](https://pypi.org/project/lean-repl-py/) - Python wrapper
- [LeanCopilot](https://github.com/lean-dojo/LeanCopilot) - LLM tactic suggestion
- [LeanDojo](https://leandojo.org/) - AI-driven theorem proving
- [Harmonic/Aristotle](https://harmonic.fun/) - IMO-level automated proving

## Dependencies

```
SPEC-15 (Lean REPL) ───► SPEC-01 (REPL Functions) [integration pattern]
SPEC-15 (Lean REPL) ───► SPEC-13 (Rich Output) [proof state display]
SPEC-15 (Lean REPL) ───► SPEC-14 (Always-On) [verification checkpoints]
```

---

## 15.1 Lean REPL Functions

### Requirements

[SPEC-15.01] The system SHALL expose Lean operations in the REPL sandbox.

[SPEC-15.02] Lean REPL functions SHALL include:
```python
def lean_check(code: str) -> LeanResult:
    """Execute Lean code and return verification result."""

def lean_tactic(tactic: str, state: int) -> TacticResult:
    """Apply a tactic to a proof state."""

def lean_prove(theorem: str, tactics: list[str]) -> ProofResult:
    """Attempt to prove theorem with given tactics."""

def lean_verify(claim: str) -> VerificationResult:
    """Verify a natural language claim by formalization."""
```

[SPEC-15.03] LeanResult SHALL include:
- `success: bool` - Whether code type-checked
- `messages: list[LeanMessage]` - Compiler output
- `env: int | None` - Environment ID for chaining
- `sorries: list[Sorry]` - Incomplete proofs

[SPEC-15.04] TacticResult SHALL include:
- `proof_state: int` - New state ID
- `goals: list[str]` - Remaining proof goals
- `status: Literal["complete", "incomplete", "error"]`

[SPEC-15.05] ProofResult SHALL include:
- `success: bool` - Whether proof completed
- `proof_term: str | None` - Final proof term if successful
- `attempts: list[TacticResult]` - Tactic application history
- `error: str | None` - Error message if failed

### Acceptance Criteria

- [ ] All functions callable from REPL
- [ ] Results correctly typed
- [ ] Error handling comprehensive
- [ ] State management works

---

## 15.2 Lean Process Management

### Requirements

[SPEC-15.10] The system SHALL manage Lean REPL as a subprocess.

[SPEC-15.11] Lean subprocess SHALL be started lazily on first use.

[SPEC-15.12] The system SHALL support connection pooling for concurrent proofs.

[SPEC-15.13] Subprocess timeout SHALL be configurable (default: 30s per command).

[SPEC-15.14] The system SHALL handle Lean crashes gracefully with retry.

[SPEC-15.15] Environment states SHALL be serializable via pickle for persistence.

### Acceptance Criteria

- [ ] Lazy startup works correctly
- [ ] Pool manages concurrent access
- [ ] Timeout prevents hangs
- [ ] Crash recovery works
- [ ] State serialization works

---

## 15.3 JSON Protocol Interface

### Requirements

[SPEC-15.20] The system SHALL communicate with Lean via JSON protocol.

[SPEC-15.21] Command types SHALL include:
| Command | Purpose |
|---------|---------|
| `{"cmd": "..."}` | Execute Lean code |
| `{"tactic": "...", "proofState": N}` | Apply tactic |
| `{"file": "path"}` | Process .lean file |
| `{"pickleTo": "path", "env": N}` | Serialize state |
| `{"unpickleFrom": "path"}` | Deserialize state |

[SPEC-15.22] Response parsing SHALL handle all Lean message types.

[SPEC-15.23] The system SHALL support environment chaining via `env` field.

[SPEC-15.24] Proof state IDs SHALL be tracked for backtracking.

### Acceptance Criteria

- [ ] All command types work
- [ ] Response parsing handles edge cases
- [ ] Environment chaining works
- [ ] Backtracking to earlier states works

---

## 15.4 Proof State Visualization

### Requirements

[SPEC-15.30] The system SHALL render proof states using Rich output (SPEC-13).

[SPEC-15.31] Proof state format SHALL use turnstile notation:
```
⊢ Proof: example (x : Nat) : x = x
  ├─ Goal 1: x = x
  │  └─ Context: x : Nat
  └─ Status: 1 goal remaining
```

[SPEC-15.32] Tactic application SHALL show before/after states.

[SPEC-15.33] Completed proofs SHALL show `✓` with proof term summary.

[SPEC-15.34] Failed proofs SHALL show `✗` with error context.

### Acceptance Criteria

- [ ] Proof states render clearly
- [ ] Goal context visible
- [ ] Tactic effects shown
- [ ] Completion/failure clear

---

## 15.5 Auto-Formalization

### Requirements

[SPEC-15.40] The system SHALL support auto-formalization of claims.

[SPEC-15.41] `lean_verify(claim)` SHALL:
1. Translate natural language to Lean theorem statement
2. Attempt proof with standard tactics
3. Report success, failure, or "unable to formalize"

[SPEC-15.42] Auto-formalization SHALL use LLM for translation.

[SPEC-15.43] Translation confidence SHALL be reported.

[SPEC-15.44] The system SHALL cache successful formalizations in memory.

[SPEC-15.45] Common patterns SHALL have template-based formalization:
- "X terminates" → `theorem X_terminates : ...`
- "X is sorted" → `theorem X_sorted : Sorted X`
- "X implies Y" → `theorem X_implies_Y : X → Y`

### Acceptance Criteria

- [ ] Natural language claims translate
- [ ] Confidence scores meaningful
- [ ] Caching reduces redundant work
- [ ] Common patterns handled

---

## 15.6 Integration with RLM Reasoning

### Requirements

[SPEC-15.50] The system SHALL support verification checkpoints in RLM.

[SPEC-15.51] Verification checkpoints SHALL trigger on:
- Claims about algorithm correctness
- Security property assertions
- Mathematical statements
- Invariant declarations

[SPEC-15.52] Checkpoint results SHALL boost or reduce confidence.

[SPEC-15.53] Failed verification SHALL trigger re-reasoning path.

[SPEC-15.54] Successful verification SHALL be stored in memory with high confidence.

### Acceptance Criteria

- [ ] Checkpoints trigger appropriately
- [ ] Confidence adjustment works
- [ ] Re-reasoning on failure works
- [ ] Verified facts persist

---

## 15.7 Installation and Setup

### Requirements

[SPEC-15.60] Lean integration SHALL be an optional dependency.

[SPEC-15.61] Installation command SHALL be:
```bash
uv sync --extra lean
```

[SPEC-15.62] The system SHALL detect if Lean is available and degrade gracefully.

[SPEC-15.63] First-time setup SHALL download/configure Lean automatically.

[SPEC-15.64] Mathlib import SHALL be optional (significant download size).

[SPEC-15.65] Setup status SHALL be reported via `/rlm status`.

### Acceptance Criteria

- [ ] Optional install works
- [ ] Graceful degradation without Lean
- [ ] Auto-setup functional
- [ ] Mathlib opt-in works
- [ ] Status reporting accurate

---

## Implementation Notes

### Dependencies

Add to `pyproject.toml`:
```toml
[project.optional-dependencies]
lean = [
    "lean-repl-py>=0.1.12",
]
```

### Files to Create

- `src/lean_repl.py` - Lean process management and protocol
- `src/lean_functions.py` - REPL-exposed functions
- `src/lean_formalization.py` - Auto-formalization logic
- `tests/integration/test_lean_repl.py` - Integration tests

### Architecture

```
┌─────────────────────────────────────────────┐
│              RLM REPL Sandbox               │
│  lean_check() lean_tactic() lean_prove()   │
└─────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│            LeanREPLManager                  │
│  • Process pool                             │
│  • State tracking                           │
│  • Timeout management                       │
└─────────────────────────────────────────────┘
                     │
                     ▼ JSON over stdin/stdout
┌─────────────────────────────────────────────┐
│         Lean 4 REPL Subprocess              │
│  • Type checking                            │
│  • Tactic application                       │
│  • Proof verification                       │
└─────────────────────────────────────────────┘
```

### Research References

- [VeriBench](https://openreview.net/forum?id=rWkGFmnSNl) - Benchmark showing Claude 3.7 Sonnet at 12.5% compilation, 30% proof rate
- [Aristotle](https://arxiv.org/abs/2510.01346) - Harmonic's IMO-level prover architecture
- [APOLLO](https://arxiv.org/abs/2505.05758) - LLM-Lean collaboration patterns
- [Pantograph](https://github.com/lenianiva/Pantograph) - Programmatic Lean interface
