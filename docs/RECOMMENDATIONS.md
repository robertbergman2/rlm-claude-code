# RLM-Claude-Code: Strategic Recommendations

**Analysis Date**: 2026-01-13
**Based on**: Deep codebase analysis + research synthesis from academic literature and industry best practices

---

## Executive Summary

RLM-Claude-Code is a sophisticated implementation of Recursive Language Model principles, successfully integrating context externalization, recursive decomposition, persistent memory, and intelligent orchestration. The architecture is solid, with ~1000+ tests and comprehensive spec coverage.

This document presents **research-grounded recommendations** to advance the project across six dimensions:
1. **Intelligence** - Reasoning quality and decomposition strategies
2. **Performance** - Latency, cost efficiency, and scaling
3. **Capabilities** - New features and expanded functionality
4. **Reliability** - Robustness, error handling, and guarantees
5. **User Experience** - Observability, feedback, and control
6. **Maintenance** - Technical debt, testing, and sustainability

---

## Part I: Current State Assessment

### Strengths

1. **Solid Theoretical Foundation**: Direct implementation of the [RLM paper](https://arxiv.org/abs/2512.24601) principles with appropriate adaptations for Claude Code integration

2. **Comprehensive Architecture**:
   - RestrictedPython sandbox with thoughtful security model
   - Hypergraph memory store with SQLite+WAL
   - Tiered memory evolution (task→session→longterm→archive)
   - Reasoning traces with decision trees
   - Enhanced budget tracking with burn rate monitoring

3. **Testing Discipline**: 1000+ tests across unit, integration, property-based, and security categories

4. **Observability**: Streaming trajectory with configurable verbosity levels

### Gaps Identified

| Gap | Current State | Opportunity |
|-----|---------------|-------------|
| **Asynchronous execution** | Deferred operations processed serially | Paper notes "lack of asynchrony" as key limitation |
| **Semantic search** | Keyword-based relevance scoring | No embeddings for memory/context retrieval |
| **Adaptive depth** | Static depth budget | No per-query compute allocation |
| **Verification** | No formal guarantees | No integration with formal methods |
| **Learning loop** | Strategy cache with basic similarity | No feedback from outcomes to improve classifier |

---

## Part II: Intelligence Recommendations

### 2.1 Implement Tree of Thoughts Integration

**Research basis**: [Tree of Thoughts (NeurIPS 2023)](https://arxiv.org/abs/2305.10601) showed 4% → 74% success rate improvement on Game of 24 by enabling deliberate reasoning with backtracking.

**Current gap**: RLM uses linear recursive decomposition without explicit exploration of alternative paths or backtracking.

**Recommendation**: Hybrid ToT-RLM architecture

```python
# Proposed: src/reasoning/tot_integration.py

class ThoughtNode:
    """Node in the thought tree."""
    thought: str
    state: dict[str, Any]
    children: list["ThoughtNode"]
    value_estimate: float
    is_terminal: bool

class ToTREPL(RLMEnvironment):
    """Extended REPL with thought branching."""

    def branch(self, thoughts: list[str]) -> list[ThoughtNode]:
        """Generate multiple thought branches for exploration."""

    def evaluate_state(self, node: ThoughtNode) -> float:
        """Self-evaluate progress toward goal."""

    def backtrack(self, to_node: ThoughtNode) -> None:
        """Return to previous state for alternative exploration."""
```

**Implementation priority**: HIGH - This directly addresses the "parallel exploration" pattern the RLM paper notes as emergent but unstructured.

### 2.2 Add Compute-Optimal Allocation

**Research basis**: [Inference Scaling Laws (ICLR 2025)](https://arxiv.org/abs/2408.03314) demonstrated 4x improvement in test-time compute efficiency through adaptive per-prompt allocation.

**Current gap**: Static depth budget (default=2) regardless of query difficulty.

**Recommendation**: Implement adaptive depth based on estimated query difficulty

```python
# Proposed: src/compute_allocation.py

@dataclass
class ComputeAllocation:
    """Per-query compute budget allocation."""
    depth_budget: int
    model_tier: ModelTier
    parallel_calls: int
    timeout_ms: int
    estimated_cost: float

def allocate_compute(
    query: str,
    context: SessionContext,
    total_budget: float,
) -> ComputeAllocation:
    """
    Allocate compute optimally based on query difficulty.

    Uses lightweight difficulty estimation (like current complexity_classifier)
    but maps to continuous compute allocation rather than binary activation.
    """
```

**Key insight**: The research shows smaller models + more inference compute often beats larger models. Consider Haiku with depth=3 vs Opus with depth=1 for certain query types.

### 2.3 Integrate Formal Verification for Code Tasks

**Research basis**: [PREFACE (GLSVLSI 2025)](https://dl.acm.org/doi/10.1145/3716368.3735300) showed RL-guided prompt repair achieves formally verifiable code generation without fine-tuning.

**Current gap**: CPMpy is available in REPL but not systematically used for verification chains.

**Recommendation**: Add verification-aware decomposition for code generation/modification tasks

```python
# Enhancement to recursive_handler.py

class VerificationChain:
    """Chain of verification steps for code changes."""

    def generate_preconditions(self, change: CodeChange) -> list[Constraint]:
        """Generate constraints that must hold before change."""

    def generate_postconditions(self, change: CodeChange) -> list[Constraint]:
        """Generate constraints that must hold after change."""

    def verify(self, constraints: list[Constraint], code: str) -> VerificationResult:
        """Verify constraints hold using CPMpy or type checker."""
```

**Concrete example**: For a refactoring task, automatically generate: "All call sites still type-check" as a postcondition, then spawn recursive calls to verify each.

---

## Part III: Smarter RLM - Tool Orchestration, Routing, and Learning

This section addresses the core question: **How can the RLM become genuinely smarter over time?** The goal is an RLM that proactively uses the REPL for calculation, programmatically orchestrates tools, intelligently routes between models, enriches context before reasoning, and learns from every interaction.

### 3.1 Proactive REPL for Programmatic Reasoning

**Research basis**: [ARTIST (Microsoft, 2025)](https://www.microsoft.com/en-us/research/wp-content/uploads/2025/04/AgenticReasoning.pdf) demonstrates that LLMs augmented with Python interpreters systematically decompose complex problems, alternating between internal reasoning and external computation.

**Current gap**: The REPL is available but reactive—the model uses it when prompted rather than proactively offloading computation.

**Recommendation**: Implement a **proactive computation layer** that automatically identifies opportunities for programmatic reasoning.

```python
# Proposed: src/proactive_repl.py

class ProactiveComputationAdvisor:
    """Advise when REPL computation is more reliable than LLM reasoning."""

    # Patterns where REPL >> LLM reasoning
    COMPUTATION_TRIGGERS = {
        "arithmetic": r"\b(\d+\s*[\+\-\*/]\s*\d+|\bcalculate\b|\bcompute\b)",
        "counting": r"\b(how many|count|number of|total)\b",
        "sorting": r"\b(sort|order|rank|largest|smallest|top \d+)\b",
        "filtering": r"\b(filter|where|matching|containing)\b",
        "aggregation": r"\b(sum|average|mean|median|max|min)\b",
        "date_math": r"\b(days since|weeks ago|before|after)\b",
        "string_ops": r"\b(extract|parse|split|format|regex)\b",
    }

    def suggest_repl_approach(
        self,
        query: str,
        context: SessionContext,
    ) -> REPLSuggestion | None:
        """
        Analyze query and suggest REPL-based approach if beneficial.

        Returns None if LLM reasoning is preferable.
        Returns REPLSuggestion with code template if REPL is better.
        """
        for pattern_type, pattern in self.COMPUTATION_TRIGGERS.items():
            if re.search(pattern, query, re.IGNORECASE):
                return self._generate_suggestion(pattern_type, query, context)
        return None

    def _generate_suggestion(
        self,
        pattern_type: str,
        query: str,
        context: SessionContext,
    ) -> REPLSuggestion:
        """Generate REPL code suggestion for the pattern type."""
        templates = {
            "counting": "len([x for x in {data} if {condition}])",
            "filtering": "[x for x in {data} if {condition}]",
            "aggregation": "sum({data}) / len({data})",  # etc.
        }
        return REPLSuggestion(
            pattern_type=pattern_type,
            template=templates.get(pattern_type, ""),
            explanation=f"Use REPL for {pattern_type} (more reliable than LLM arithmetic)",
        )
```

**Key insight from research**: The difference between an LLM and an agentic LLM is "the difference between a calculator and a pilot." The RLM should recognize when it's more effective to *be* the pilot that *uses* the calculator.

#### 3.1.1 Auto-Injected Computation Helpers

Extend the REPL with computation-specific functions that the model learns to use:

```python
# Enhancement to repl_environment.py

class ComputationalREPL(RLMEnvironment):
    """REPL with enhanced computational capabilities."""

    def __init__(self, ...):
        super().__init__(...)
        # Add computational helpers
        self.globals.update({
            # Safe math operations
            "calc": self._safe_calculate,
            "stats": self._compute_statistics,

            # Data manipulation
            "group_by": self._group_by,
            "pivot": self._pivot_table,
            "dedupe": self._deduplicate,

            # Code analysis (programmatic, not LLM)
            "count_lines": self._count_lines,
            "find_imports": self._find_imports,
            "call_graph": self._build_call_graph,

            # Context compilation
            "compile_context": self._compile_context,
            "enrich_with_deps": self._enrich_with_dependencies,
        })

    def _safe_calculate(self, expression: str) -> float:
        """
        Safely evaluate mathematical expressions.

        Uses ast.literal_eval + operator whitelist for safety.
        """
        # Parse and evaluate safely
        allowed_operators = {'+', '-', '*', '/', '**', '%', '//'}
        # ... safe evaluation logic

    def _compile_context(
        self,
        files: list[str],
        include_deps: bool = True,
        max_tokens: int = 50000,
    ) -> dict[str, Any]:
        """
        Programmatically compile relevant context for a task.

        Instead of dumping all files, analyze imports/dependencies
        and build minimal sufficient context.
        """
        compiled = {}
        for file in files:
            content = self.globals["files"].get(file, "")
            if include_deps:
                deps = self._find_imports(content)
                for dep in deps:
                    if dep not in compiled and self._is_local(dep):
                        compiled[dep] = self._load_file(dep)
            compiled[file] = content

        return self._truncate_to_budget(compiled, max_tokens)
```

### 3.2 Intelligent Tool Orchestration

**Research basis**: [LATS (ICML 2024)](https://arxiv.org/abs/2310.04406) unifies reasoning, acting, and planning using Monte Carlo Tree Search, doubling ReAct performance on HotPotQA. [Hybrid orchestration patterns](https://aws.amazon.com/blogs/machine-learning/customize-agent-workflows-with-advanced-orchestration-techniques-using-strands-agents/) show that combining ReWOO's planning discipline with ReAct's agility yields best results.

**Current gap**: Tool calls are reactive and unplanned. No explicit planning phase before tool execution.

**Recommendation**: Implement **LATS-inspired tool orchestration** with planning, acting, and reflection phases.

```python
# Proposed: src/tool_orchestration.py

from dataclasses import dataclass
from enum import Enum
from typing import Any

class OrchestrationPhase(Enum):
    PLAN = "plan"      # Generate tool use plan
    ACT = "act"        # Execute tools
    OBSERVE = "observe" # Analyze results
    REFLECT = "reflect" # Self-critique and adjust

@dataclass
class ToolPlan:
    """Structured plan for tool execution."""
    goal: str
    steps: list["ToolStep"]
    dependencies: dict[int, list[int]]  # step -> depends_on_steps
    estimated_cost: float
    confidence: float

@dataclass
class ToolStep:
    """Single step in a tool plan."""
    step_id: int
    tool: str
    args: dict[str, Any]
    expected_output: str
    fallback: "ToolStep | None" = None

class LATSOrchestrator:
    """
    LATS-inspired tool orchestration with MCTS.

    Key innovation: Use LLM as both action generator AND value function,
    enabling lookahead and backtracking.
    """

    def __init__(
        self,
        exploration_weight: float = 1.414,  # UCB1 constant
        max_rollouts: int = 10,
        max_depth: int = 5,
    ):
        self.exploration_weight = exploration_weight
        self.max_rollouts = max_rollouts
        self.max_depth = max_depth
        self._tree: dict[str, ToolTreeNode] = {}

    async def orchestrate(
        self,
        query: str,
        context: SessionContext,
        available_tools: list[Tool],
    ) -> OrchestrationResult:
        """
        Orchestrate tool use with MCTS-based planning.

        1. PLAN: Generate initial tool plan
        2. EXPAND: Use UCB1 to select promising nodes
        3. SIMULATE: Roll out tool execution
        4. BACKPROPAGATE: Update value estimates
        5. REFLECT: Self-critique failed paths
        """
        root = self._create_root_node(query, context)

        for _ in range(self.max_rollouts):
            # Selection: UCB1 to balance exploration/exploitation
            node = self._select_node(root)

            # Expansion: Generate candidate tool actions
            children = await self._expand_node(node, available_tools)

            # Simulation: Execute best candidate
            result = await self._simulate(children[0])

            # Backpropagation: Update value estimates
            self._backpropagate(node, result)

            # Early termination if goal achieved
            if result.success and result.confidence > 0.9:
                return result

        # Reflect on failures and return best effort
        return await self._reflect_and_synthesize(root)

    def _compute_ucb1(self, node: ToolTreeNode) -> float:
        """Upper Confidence Bound for node selection."""
        if node.visits == 0:
            return float('inf')

        exploitation = node.value / node.visits
        exploration = self.exploration_weight * (
            (2 * math.log(node.parent.visits) / node.visits) ** 0.5
        )
        return exploitation + exploration

    async def _reflect_on_failure(
        self,
        failed_path: list[ToolTreeNode],
    ) -> str:
        """
        Generate self-critique for failed execution path.

        This is the key LATS innovation: reflection enables learning
        from failures within the same query.
        """
        return await self._llm_reflect(
            f"This tool sequence failed: {failed_path}. "
            f"What went wrong and what should be tried instead?"
        )
```

#### 3.2.1 Tool Selection Policy

Add intelligent tool selection based on task analysis:

```python
# Enhancement to tool_bridge.py

class IntelligentToolSelector:
    """Select optimal tools based on task analysis."""

    # Tool capability matrix
    TOOL_CAPABILITIES = {
        "Bash": {"execution", "system_info", "file_ops", "git"},
        "Read": {"file_content", "inspection"},
        "Grep": {"search", "pattern_matching"},
        "Glob": {"file_discovery", "pattern_matching"},
        "Edit": {"modification", "refactoring"},
        "Write": {"creation", "generation"},
    }

    # Task -> preferred tools mapping
    TASK_TOOL_PREFERENCES = {
        "find_definition": ["Grep", "Read"],
        "find_usages": ["Grep", "Glob"],
        "understand_structure": ["Glob", "Read"],
        "modify_code": ["Read", "Edit"],  # Read first!
        "run_tests": ["Bash"],
        "check_status": ["Bash"],
    }

    def select_tools(
        self,
        task: str,
        context: SessionContext,
    ) -> list[tuple[str, float]]:
        """
        Select tools for task with confidence scores.

        Returns list of (tool_name, confidence) tuples.
        """
        task_type = self._classify_task(task)
        preferred = self.TASK_TOOL_PREFERENCES.get(task_type, [])

        # Score tools by relevance
        scored = []
        for tool, capabilities in self.TOOL_CAPABILITIES.items():
            relevance = self._compute_relevance(task_type, capabilities)
            preference_boost = 0.2 if tool in preferred else 0
            scored.append((tool, relevance + preference_boost))

        return sorted(scored, key=lambda x: -x[1])

    def suggest_tool_sequence(
        self,
        goal: str,
        available_info: dict[str, Any],
    ) -> list[ToolStep]:
        """
        Suggest optimal tool sequence for achieving goal.

        Uses task decomposition + tool selection.
        """
        subtasks = self._decompose_goal(goal)
        sequence = []

        for subtask in subtasks:
            best_tool = self.select_tools(subtask, available_info)[0][0]
            sequence.append(ToolStep(
                tool=best_tool,
                purpose=subtask,
                depends_on=[s.step_id for s in sequence if self._has_dependency(subtask, s)],
            ))

        return sequence
```

### 3.3 Intelligent Model Routing

**Research basis**: [RouteLLM (ICLR 2025)](https://github.com/lm-sys/RouteLLM) achieves 85% cost reduction while maintaining 95% of GPT-4 performance. [Cost-Aware Contrastive Routing (2025)](https://arxiv.org/html/2508.12491) improves accuracy-cost tradeoff by 25%.

**Current gap**: `smart_router.py` exists but uses simple heuristics. No learned routing based on query characteristics.

**Recommendation**: Implement **learned routing** with query difficulty estimation and model capability matching.

```python
# Proposed: src/learned_router.py

from dataclasses import dataclass
import numpy as np

@dataclass
class RoutingDecision:
    """Decision from the router."""
    model: str
    confidence: float
    estimated_quality: float
    estimated_cost: float
    reasoning: str

class LearnedRouter:
    """
    Route queries to optimal model based on learned preferences.

    Inspired by RouteLLM but adapted for RLM's recursive structure.
    """

    # Model capability profiles
    MODEL_PROFILES = {
        "opus": {
            "strengths": ["complex_reasoning", "multi_step", "creative", "nuanced"],
            "cost_per_1k": 0.015,
            "quality_baseline": 0.95,
        },
        "sonnet": {
            "strengths": ["analysis", "code", "structured", "speed"],
            "cost_per_1k": 0.003,
            "quality_baseline": 0.85,
        },
        "haiku": {
            "strengths": ["extraction", "classification", "simple_qa", "speed"],
            "cost_per_1k": 0.00025,
            "quality_baseline": 0.70,
        },
    }

    def __init__(self, router_model: str = "learned"):
        self.router_model = router_model
        self._routing_history: list[RoutingOutcome] = []
        self._query_embedder = self._load_embedder()

    def route(
        self,
        query: str,
        context: SessionContext,
        cost_sensitivity: float = 0.5,  # 0=quality only, 1=cost only
    ) -> RoutingDecision:
        """
        Route query to optimal model.

        Uses combination of:
        1. Query difficulty estimation
        2. Task type classification
        3. Context complexity analysis
        4. Historical performance on similar queries
        """
        # Extract features
        features = self._extract_features(query, context)

        # Estimate difficulty (0-1 scale)
        difficulty = self._estimate_difficulty(features)

        # Get historical performance for similar queries
        similar_outcomes = self._find_similar_outcomes(query)

        # Score each model
        scores = {}
        for model, profile in self.MODEL_PROFILES.items():
            quality_score = self._estimate_quality(model, features, similar_outcomes)
            cost_score = 1.0 - (profile["cost_per_1k"] / 0.015)  # Normalized

            # Weighted combination
            scores[model] = (
                (1 - cost_sensitivity) * quality_score +
                cost_sensitivity * cost_score
            )

        best_model = max(scores, key=scores.get)

        return RoutingDecision(
            model=best_model,
            confidence=scores[best_model],
            estimated_quality=self._estimate_quality(best_model, features, similar_outcomes),
            estimated_cost=self.MODEL_PROFILES[best_model]["cost_per_1k"],
            reasoning=self._explain_routing(best_model, features, scores),
        )

    def _estimate_difficulty(self, features: QueryFeatures) -> float:
        """
        Estimate query difficulty for routing.

        Factors:
        - Reasoning depth required
        - Domain specificity
        - Ambiguity level
        - Context size
        """
        difficulty = 0.0

        # Multi-step reasoning increases difficulty
        if features.requires_multi_step:
            difficulty += 0.3

        # Cross-domain queries are harder
        if features.crosses_domains:
            difficulty += 0.2

        # Large context increases difficulty
        difficulty += min(0.2, features.context_tokens / 100000)

        # Ambiguous queries need stronger models
        if features.ambiguity_score > 0.5:
            difficulty += 0.2

        return min(1.0, difficulty)

    def record_outcome(
        self,
        query: str,
        model: str,
        success: bool,
        quality_score: float,
        cost: float,
    ) -> None:
        """
        Record routing outcome for learning.

        This feedback loop enables the router to improve over time.
        """
        self._routing_history.append(RoutingOutcome(
            query_embedding=self._embed_query(query),
            model=model,
            success=success,
            quality_score=quality_score,
            cost=cost,
            timestamp=time.time(),
        ))

        # Periodic model update
        if len(self._routing_history) % 100 == 0:
            self._update_routing_model()
```

#### 3.3.1 Cascading Router for Cost Optimization

Implement cascading (try cheaper first, escalate if needed):

```python
# Enhancement to learned_router.py

class CascadingRouter(LearnedRouter):
    """
    Try cheaper models first, escalate on low confidence.

    Research shows cascading with reliable judges achieves
    5x+ cost savings at zero performance degradation.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.8,
        max_escalations: int = 2,
    ):
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.max_escalations = max_escalations
        self.cascade_order = ["haiku", "sonnet", "opus"]

    async def route_with_cascade(
        self,
        query: str,
        context: SessionContext,
    ) -> CascadingResult:
        """
        Execute query with cascading escalation.

        1. Start with cheapest viable model
        2. If confidence < threshold, escalate
        3. Use self-consistency check for confidence estimation
        """
        for i, model in enumerate(self.cascade_order):
            result = await self._execute_with_model(query, context, model)

            # Estimate confidence via self-consistency
            confidence = await self._estimate_confidence(result, query)

            if confidence >= self.confidence_threshold:
                return CascadingResult(
                    answer=result,
                    model_used=model,
                    escalations=i,
                    total_cost=self._compute_cascade_cost(i),
                )

            if i >= self.max_escalations:
                break

        # Return best effort from strongest model
        return CascadingResult(
            answer=result,
            model_used=self.cascade_order[-1],
            escalations=len(self.cascade_order) - 1,
            total_cost=self._compute_cascade_cost(len(self.cascade_order) - 1),
        )
```

### 3.4 Programmatic Context Enrichment

**Research basis**: [RAG-Reasoning (EMNLP 2025)](https://github.com/DavidZWZ/Awesome-RAG-Reasoning) shows that bridging retrieval with reasoning enables more powerful agentic AI. [Knowledge graph integration](https://dl.acm.org/doi/10.1145/3701716.3715473) provides continuously updated factual grounding.

**Current gap**: Context is externalized but not actively enriched before reasoning.

**Recommendation**: Implement **proactive context enrichment** that automatically gathers relevant information before the LLM reasons.

```python
# Proposed: src/context_enrichment.py

class ContextEnricher:
    """
    Proactively enrich context before LLM reasoning.

    Instead of reactive retrieval, analyze query intent and
    pre-fetch likely-needed information.
    """

    def __init__(self, memory_store: MemoryStore):
        self.memory_store = memory_store
        self._enrichment_strategies = {
            "code_task": self._enrich_for_code,
            "debug_task": self._enrich_for_debugging,
            "analysis_task": self._enrich_for_analysis,
            "question": self._enrich_for_question,
        }

    async def enrich(
        self,
        query: str,
        context: SessionContext,
    ) -> EnrichedContext:
        """
        Enrich context based on query intent.

        Enrichment happens BEFORE LLM reasoning, not during.
        """
        intent = self._classify_intent(query)
        strategy = self._enrichment_strategies.get(intent, self._default_enrich)

        enriched = await strategy(query, context)

        return EnrichedContext(
            original=context,
            additions=enriched,
            enrichment_reasoning=self._explain_enrichment(intent, enriched),
        )

    async def _enrich_for_code(
        self,
        query: str,
        context: SessionContext,
    ) -> dict[str, Any]:
        """
        Enrich context for code-related tasks.

        Automatically gather:
        - Related file dependencies
        - Type definitions
        - Test files
        - Recent git changes
        """
        enriched = {}

        # Extract mentioned files
        mentioned_files = self._extract_file_references(query)

        for file in mentioned_files:
            # Add the file itself
            if file not in context.files:
                enriched[f"file:{file}"] = await self._load_file(file)

            # Add imports/dependencies
            deps = await self._analyze_dependencies(file)
            for dep in deps[:5]:  # Limit to top 5
                if dep not in context.files:
                    enriched[f"dep:{dep}"] = await self._load_file(dep)

            # Add type definitions if TypeScript/Python
            types = await self._find_type_definitions(file)
            if types:
                enriched[f"types:{file}"] = types

            # Add related tests
            test_file = self._find_test_file(file)
            if test_file and test_file not in context.files:
                enriched[f"test:{file}"] = await self._load_file(test_file)

        # Add relevant memories
        memories = self.memory_store.search(query, limit=5)
        for mem in memories:
            enriched[f"memory:{mem.id}"] = mem.content

        return enriched

    async def _enrich_for_debugging(
        self,
        query: str,
        context: SessionContext,
    ) -> dict[str, Any]:
        """
        Enrich context for debugging tasks.

        Automatically gather:
        - Error stack traces (parsed)
        - Related log entries
        - Recent changes to affected files
        - Similar past debugging experiences
        """
        enriched = {}

        # Parse any error messages in context
        errors = self._extract_errors(context.tool_outputs)
        for error in errors:
            # Get source location
            source_file, line = self._parse_error_location(error)
            if source_file:
                # Add surrounding context (±20 lines)
                enriched[f"error_context:{source_file}"] = await self._get_lines(
                    source_file, line - 20, line + 20
                )

            # Find git blame for the line
            blame = await self._git_blame(source_file, line)
            if blame:
                enriched[f"blame:{source_file}:{line}"] = blame

        # Find similar past debugging experiences
        experiences = self.memory_store.query_nodes(
            node_type="experience",
            min_confidence=0.6,
            limit=3,
        )
        for exp in experiences:
            if self._is_relevant_experience(exp, query):
                enriched[f"experience:{exp.id}"] = exp.content

        return enriched
```

### 3.5 Continuous Learning and Self-Improvement

**Research basis**: [Agent Lightning (Microsoft, 2025)](https://arxiv.org/abs/2508.03680) enables RL-based training of LLM agents with zero code modifications. [RLVR (Reinforcement Learning from Verifiable Rewards)](https://karpathy.bearblog.dev/year-in-review-2025/) emerged as the de facto new training stage for 2025.

**Current gap**: Strategy cache provides basic pattern matching but no true learning from outcomes.

**Recommendation**: Implement **outcome-based learning** that improves routing, tool selection, and decomposition over time.

```python
# Proposed: src/continuous_learning.py

from dataclasses import dataclass, field
from typing import Any
import json

@dataclass
class ExecutionOutcome:
    """Outcome of an RLM execution for learning."""
    query: str
    query_features: QueryFeatures
    strategy_used: StrategyType
    model_used: str
    depth_reached: int
    tools_used: list[str]
    success: bool
    user_satisfaction: float | None  # 0-1 if feedback provided
    cost: float
    latency_ms: float
    error_type: str | None = None

@dataclass
class LearningSignal:
    """Signal extracted from outcome for learning."""
    signal_type: str  # "routing", "strategy", "tool_selection"
    positive: bool
    features: dict[str, Any]
    weight: float = 1.0

class ContinuousLearner:
    """
    Learn from execution outcomes to improve future performance.

    Implements lightweight online learning without model fine-tuning.
    """

    def __init__(
        self,
        persistence_path: Path,
        learning_rate: float = 0.1,
    ):
        self.persistence_path = persistence_path
        self.learning_rate = learning_rate

        # Learned adjustments
        self._routing_adjustments: dict[str, float] = {}
        self._strategy_preferences: dict[str, dict[str, float]] = {}
        self._tool_effectiveness: dict[str, dict[str, float]] = {}

        self._load_learned_state()

    def record_outcome(self, outcome: ExecutionOutcome) -> list[LearningSignal]:
        """
        Record execution outcome and extract learning signals.

        Returns list of signals that were extracted.
        """
        signals = []

        # Extract routing signal
        routing_signal = self._extract_routing_signal(outcome)
        if routing_signal:
            signals.append(routing_signal)
            self._update_routing(routing_signal)

        # Extract strategy signal
        strategy_signal = self._extract_strategy_signal(outcome)
        if strategy_signal:
            signals.append(strategy_signal)
            self._update_strategy_preferences(strategy_signal)

        # Extract tool effectiveness signal
        for tool in outcome.tools_used:
            tool_signal = self._extract_tool_signal(outcome, tool)
            if tool_signal:
                signals.append(tool_signal)
                self._update_tool_effectiveness(tool_signal)

        self._save_learned_state()
        return signals

    def _extract_routing_signal(
        self,
        outcome: ExecutionOutcome,
    ) -> LearningSignal | None:
        """
        Extract signal about model routing effectiveness.

        Positive if: correct model was used (success + reasonable cost)
        Negative if: wrong model (failure or excessive cost for simple task)
        """
        # Compute expected cost for difficulty
        expected_cost = self._expected_cost_for_difficulty(outcome.query_features)

        # Was routing appropriate?
        if outcome.success:
            # Success: check if cost was reasonable
            cost_ratio = outcome.cost / expected_cost
            if cost_ratio < 0.5:
                # Way under budget - could have used cheaper model
                return LearningSignal(
                    signal_type="routing",
                    positive=False,
                    features={
                        "query_type": outcome.query_features.primary_type,
                        "model_used": outcome.model_used,
                        "suggestion": "downgrade",
                    },
                    weight=0.3,  # Weak signal
                )
            elif cost_ratio > 2.0:
                # Way over budget - routing was necessary
                return LearningSignal(
                    signal_type="routing",
                    positive=True,
                    features={
                        "query_type": outcome.query_features.primary_type,
                        "model_used": outcome.model_used,
                    },
                    weight=1.0,
                )
        else:
            # Failure: model wasn't strong enough
            return LearningSignal(
                signal_type="routing",
                positive=False,
                features={
                    "query_type": outcome.query_features.primary_type,
                    "model_used": outcome.model_used,
                    "suggestion": "upgrade",
                    "error_type": outcome.error_type,
                },
                weight=1.0,
            )

        return None

    def _update_routing(self, signal: LearningSignal) -> None:
        """Update routing preferences based on signal."""
        query_type = signal.features["query_type"]
        model = signal.features["model_used"]

        key = f"{query_type}:{model}"
        current = self._routing_adjustments.get(key, 0.0)

        if signal.positive:
            # Reinforce this routing
            self._routing_adjustments[key] = current + self.learning_rate * signal.weight
        else:
            # Discourage this routing
            self._routing_adjustments[key] = current - self.learning_rate * signal.weight

    def get_routing_adjustment(
        self,
        query_type: str,
        model: str,
    ) -> float:
        """Get learned adjustment for routing decision."""
        key = f"{query_type}:{model}"
        return self._routing_adjustments.get(key, 0.0)

    def get_strategy_preference(
        self,
        query_type: str,
    ) -> dict[str, float]:
        """Get learned strategy preferences for query type."""
        return self._strategy_preferences.get(query_type, {})

    def get_tool_effectiveness(
        self,
        task_type: str,
    ) -> dict[str, float]:
        """Get learned tool effectiveness for task type."""
        return self._tool_effectiveness.get(task_type, {})
```

#### 3.5.1 Meta-Learning: Learning to Learn

```python
# Enhancement to continuous_learning.py

class MetaLearner(ContinuousLearner):
    """
    Meta-level learning: learn how to learn better.

    Tracks which learning signals are most predictive of success
    and adjusts learning rates accordingly.
    """

    def __init__(self, ...):
        super().__init__(...)
        self._signal_effectiveness: dict[str, float] = {}
        self._prediction_history: list[PredictionOutcome] = []

    def predict_success(
        self,
        query: str,
        proposed_strategy: StrategyType,
        proposed_model: str,
    ) -> tuple[float, str]:
        """
        Predict success probability based on learned patterns.

        Returns (probability, reasoning).
        """
        features = self._extract_features(query)

        # Get relevant adjustments
        routing_adj = self.get_routing_adjustment(
            features.primary_type, proposed_model
        )
        strategy_pref = self.get_strategy_preference(features.primary_type)

        # Compute base probability
        base_prob = 0.7  # Prior

        # Adjust based on learned preferences
        prob = base_prob + routing_adj * 0.1
        if proposed_strategy.value in strategy_pref:
            prob += strategy_pref[proposed_strategy.value] * 0.1

        # Clamp to [0.1, 0.99]
        prob = max(0.1, min(0.99, prob))

        reasoning = self._explain_prediction(features, routing_adj, strategy_pref)
        return prob, reasoning

    def update_meta_learning(
        self,
        prediction: float,
        actual_success: bool,
    ) -> None:
        """
        Update meta-learning based on prediction accuracy.

        If predictions are consistently wrong, adjust learning rate.
        """
        self._prediction_history.append(PredictionOutcome(
            predicted=prediction,
            actual=actual_success,
            timestamp=time.time(),
        ))

        # Analyze recent prediction accuracy
        recent = self._prediction_history[-100:]
        if len(recent) >= 50:
            accuracy = sum(
                1 for p in recent
                if (p.predicted > 0.5) == p.actual
            ) / len(recent)

            if accuracy < 0.6:
                # Predictions are poor - increase learning rate
                self.learning_rate = min(0.3, self.learning_rate * 1.2)
            elif accuracy > 0.8:
                # Predictions are good - decrease learning rate (stable)
                self.learning_rate = max(0.01, self.learning_rate * 0.9)
```

### 3.6 Integration: The Smart RLM Pipeline

Bring all components together into an integrated smart pipeline:

```python
# Proposed: src/smart_rlm.py

class SmartRLMPipeline:
    """
    Integrated smart RLM with all enhancements.

    Pipeline:
    1. Analyze query → extract features, estimate difficulty
    2. Enrich context → proactively gather relevant information
    3. Route → select optimal model based on learned preferences
    4. Plan → generate tool orchestration plan with LATS
    5. Execute → run with proactive REPL computation
    6. Learn → record outcome and update preferences
    """

    def __init__(self):
        self.enricher = ContextEnricher()
        self.router = CascadingRouter()
        self.orchestrator = LATSOrchestrator()
        self.proactive_repl = ProactiveComputationAdvisor()
        self.learner = MetaLearner()

    async def process(
        self,
        query: str,
        context: SessionContext,
    ) -> SmartRLMResult:
        """Process query through the smart pipeline."""

        # 1. Analyze
        features = self._analyze_query(query)
        difficulty = self._estimate_difficulty(features)

        # 2. Check for proactive computation opportunities
        repl_suggestion = self.proactive_repl.suggest_repl_approach(query, context)
        if repl_suggestion and repl_suggestion.confidence > 0.8:
            # Use REPL directly for computation
            result = await self._execute_repl_approach(repl_suggestion, context)
            self.learner.record_outcome(self._create_outcome(result, "repl"))
            return result

        # 3. Enrich context
        enriched = await self.enricher.enrich(query, context)

        # 4. Route to optimal model
        routing = self.router.route(
            query, enriched,
            cost_sensitivity=self._compute_cost_sensitivity(difficulty),
        )

        # 5. Plan tool orchestration
        plan = await self.orchestrator.plan(
            query, enriched,
            available_tools=self._get_available_tools(),
        )

        # 6. Execute with cascading
        result = await self.router.route_with_cascade(
            query, enriched,
        )

        # 7. Learn from outcome
        outcome = self._create_outcome(result, routing.model)
        signals = self.learner.record_outcome(outcome)

        return SmartRLMResult(
            answer=result.answer,
            model_used=result.model_used,
            enrichment_used=list(enriched.additions.keys()),
            learning_signals=signals,
            cost=result.total_cost,
        )
```

### 3.7 Summary: Making the RLM Smarter

| Component | Purpose | Key Technique | Research Basis |
|-----------|---------|---------------|----------------|
| **Proactive REPL** | Offload computation | Pattern detection + code templates | [ARTIST](https://www.microsoft.com/en-us/research/wp-content/uploads/2025/04/AgenticReasoning.pdf) |
| **Tool Orchestration** | Structured planning | LATS with MCTS | [LATS (ICML 2024)](https://arxiv.org/abs/2310.04406) |
| **Intelligent Routing** | Cost-quality optimization | Learned cascading | [RouteLLM (ICLR 2025)](https://github.com/lm-sys/RouteLLM) |
| **Context Enrichment** | Proactive preparation | Intent-based pre-fetch | [RAG-Reasoning](https://github.com/DavidZWZ/Awesome-RAG-Reasoning) |
| **Continuous Learning** | Self-improvement | Outcome-based updates | [Agent Lightning](https://arxiv.org/abs/2508.03680) |

**Key insight**: The difference between a reactive tool-user and a smart agent is *anticipation*. A smart RLM:
- **Anticipates** when computation beats reasoning (proactive REPL)
- **Plans** tool sequences before executing (LATS orchestration)
- **Selects** the right model for the task (learned routing)
- **Prepares** context before reasoning (proactive enrichment)
- **Learns** from every interaction (continuous improvement)

---

## Part IV: Performance Recommendations

### 3.1 Implement Asynchronous Recursive Calls

**Research basis**: The RLM paper explicitly notes "lack of asynchrony can cause each query to range from a few seconds to several minutes" as a key limitation.

**Current gap**: `pending_operations` are collected during sync REPL execution, then processed serially with `Semaphore(5)`.

**Recommendation**: Full async pipeline with speculative execution

```python
# Proposed: src/async_orchestrator.py

class AsyncRLMOrchestrator:
    """Fully asynchronous RLM execution engine."""

    async def execute_parallel(
        self,
        operations: list[DeferredOperation],
        max_concurrency: int = 10,
    ) -> list[Any]:
        """Execute operations with true parallelism."""
        async with asyncio.TaskGroup() as tg:
            tasks = [
                tg.create_task(self._execute_one(op))
                for op in operations
            ]
        return [t.result() for t in tasks]

    async def speculative_execute(
        self,
        primary: DeferredOperation,
        alternatives: list[DeferredOperation],
    ) -> Any:
        """Execute primary with speculative alternatives, cancel losers."""
```

**Expected impact**: 3-5x latency reduction for multi-call queries based on paper's benchmarks.

### 3.2 Add Prompt Caching Integration

**Research basis**: Claude's prompt caching can reduce costs by 90% and latency by 85% for repeated context.

**Current gap**: No explicit prompt caching strategy for recursive calls with shared context.

**Recommendation**: Structure recursive calls to maximize cache hits

```python
# Enhancement to api_client.py

class CacheAwareClient:
    """API client with prompt caching optimization."""

    def __init__(self):
        self.cache_prefix_registry: dict[str, str] = {}

    def prepare_cacheable_context(
        self,
        shared_context: str,
        query_specific: str,
    ) -> tuple[str, str]:
        """
        Structure prompt for optimal caching.

        Shared context (files, conversation) goes first (cacheable).
        Query-specific content goes last (not cached).
        """
```

### 3.3 Implement Context Compression

**Research basis**: [KVzip (SNU, 2025)](https://techxplore.com/news/2025-11-ai-tech-compress-llm-chatbot.html) achieves 3-4x memory compression with 2x latency reduction.

**Current gap**: Context externalization works, but no compression of intermediate results.

**Recommendation**: Add map-reduce compression for large intermediate results

```python
# Enhancement to repl_environment.py

def _compress_intermediate(
    self,
    content: str,
    target_tokens: int = 2000,
) -> str:
    """
    Compress intermediate results while preserving key information.

    Uses extractive then abstractive compression:
    1. Extract key sentences using attention/relevance scoring
    2. Abstractively summarize if still over budget
    """
```

---

## Part IV: Capability Recommendations

### 4.1 Add Embedding-Based Memory Retrieval

**Research basis**: [A-MEM (NeurIPS 2025)](https://arxiv.org/abs/2502.12110) and [Zep](https://blog.getzep.com/content/files/2025/01/ZEP__USING_KNOWLEDGE_GRAPHS_TO_POWER_LLM_AGENT_MEMORY_2025011700.pdf) demonstrate significant improvements with semantic retrieval.

**Current gap**: `memory_query()` uses keyword matching only; no embeddings.

**Recommendation**: Hybrid keyword + semantic retrieval

```python
# Enhancement to memory_store.py

class SemanticMemoryStore(MemoryStore):
    """Memory store with embedding-based retrieval."""

    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        super().__init__()
        self.embedding_model = embedding_model

    def create_node(self, ..., compute_embedding: bool = True) -> str:
        """Create node with optional embedding."""
        node_id = super().create_node(...)
        if compute_embedding:
            embedding = self._compute_embedding(content)
            self._store_embedding(node_id, embedding)
        return node_id

    def semantic_search(
        self,
        query: str,
        top_k: int = 10,
        hybrid_alpha: float = 0.5,  # Balance keyword vs semantic
    ) -> list[Node]:
        """Hybrid retrieval combining FTS5 + embedding similarity."""
```

**Schema addition**:
```sql
-- Add to SCHEMA_SQL
CREATE VIRTUAL TABLE IF NOT EXISTS node_embeddings USING vec0(
    node_id TEXT PRIMARY KEY,
    embedding FLOAT[1536]  -- text-embedding-3-small dimension
);
```

### 4.2 Implement Cross-Session Memory Promotion

**Research basis**: [G-Memory (2025)](https://arxiv.org/html/2511.07800v1) shows hierarchical memory enables progressive learning across sessions.

**Current gap**: Memory evolution exists but lacks automatic promotion based on cross-session patterns.

**Recommendation**: Add session-crossing pattern detection

```python
# Enhancement to memory_evolution.py

class CrossSessionPromoter:
    """Promote memories that prove valuable across sessions."""

    def analyze_session_patterns(
        self,
        session_ids: list[str],
    ) -> list[PromotionCandidate]:
        """
        Identify facts/experiences accessed across multiple sessions.

        Promotion criteria:
        1. Accessed in 3+ distinct sessions
        2. Associated with successful outcomes
        3. High confidence maintained over time
        """

    def promote_if_warranted(
        self,
        node_id: str,
        access_pattern: AccessPattern,
    ) -> bool:
        """Promote node if cross-session value is demonstrated."""
```

### 4.3 Add Multi-Turn Planning with Checkpoints

**Research basis**: [Anthropic's long-running agent harnesses](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) show checkpoint-based recovery enables complex multi-session tasks.

**Current gap**: Trajectory logging exists but not checkpointing for recovery.

**Recommendation**: Add checkpoint/restore for long-running RLM tasks

```python
# Proposed: src/checkpointing.py

@dataclass
class RLMCheckpoint:
    """Checkpoint for RLM session recovery."""

    session_id: str
    depth: int
    repl_state: dict[str, Any]
    working_memory: dict[str, Any]
    pending_operations: list[DeferredOperation]
    trajectory_events: list[TrajectoryEvent]
    cost_so_far: BudgetMetrics

    def save(self, path: Path) -> None:
        """Serialize checkpoint to disk."""

    @classmethod
    def load(cls, path: Path) -> "RLMCheckpoint":
        """Restore checkpoint from disk."""

class CheckpointingOrchestrator(RLMOrchestrator):
    """Orchestrator with automatic checkpointing."""

    def __init__(self, checkpoint_interval: int = 5):
        self.checkpoint_interval = checkpoint_interval
        self._turn_count = 0

    async def process_turn(self, ...) -> TrajectoryEvent:
        result = await super().process_turn(...)
        self._turn_count += 1

        if self._turn_count % self.checkpoint_interval == 0:
            await self._save_checkpoint()

        return result
```

---

## Part V: Reliability Recommendations

### 5.1 Add Confidence-Weighted Synthesis

**Research basis**: The RLM paper notes quality variance in recursive call results. Weighting by confidence improves aggregation.

**Current gap**: Recursive results are aggregated without confidence scoring.

**Recommendation**: Track and use confidence in synthesis

```python
# Enhancement to recursive_handler.py

@dataclass
class RecursiveResult:
    """Result from recursive call with confidence."""

    content: str
    confidence: float  # 0.0-1.0
    reasoning_trace: list[str]
    cost: BudgetMetrics

def synthesize_results(
    results: list[RecursiveResult],
    synthesis_strategy: str = "weighted",
) -> SynthesisResult:
    """
    Synthesize multiple recursive results.

    Strategies:
    - "weighted": Weight by confidence
    - "consensus": Only include high-confidence agreement
    - "diverse": Include disagreements for user decision
    """
```

### 5.2 Implement Execution Guarantees

**Research basis**: The paper acknowledges "lack of strong guarantees about controlling total API cost or total runtime."

**Current gap**: Soft limits exist but can be exceeded.

**Recommendation**: Hard execution boundaries with graceful degradation

```python
# Enhancement to enhanced_budget.py

class ExecutionGuarantees:
    """Hard guarantees for RLM execution."""

    def __init__(
        self,
        max_cost_usd: float = 1.0,
        max_duration_seconds: float = 300.0,
        max_recursive_calls: int = 20,
    ):
        self._budget_remaining = max_cost_usd
        self._deadline = time.time() + max_duration_seconds
        self._calls_remaining = max_recursive_calls

    def check_can_proceed(self, estimated_cost: float) -> bool:
        """Check if operation can proceed within guarantees."""

    def on_budget_exhausted(self) -> GracefulDegradationPlan:
        """Return plan for graceful degradation when budget exhausted."""
```

### 5.3 Add Circuit Breaker for Recursive Calls

**Research basis**: Standard reliability pattern for distributed systems; prevents cascade failures.

**Current gap**: Failed recursive calls retry but no circuit breaker.

**Recommendation**: Circuit breaker per model tier

```python
# Proposed: src/resilience.py

class RecursiveCallCircuitBreaker:
    """Circuit breaker for recursive LLM calls."""

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 60.0,
    ):
        self._failure_count = 0
        self._state = CircuitState.CLOSED
        self._last_failure_time: float | None = None

    def call_with_breaker(
        self,
        func: Callable[..., T],
        *args,
        **kwargs,
    ) -> T | FallbackResult:
        """Execute with circuit breaker protection."""
```

---

## Part VI: User Experience Recommendations

### 6.1 Add Progressive Disclosure in Trajectory

**Research basis**: Anthropic's [context engineering guide](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) emphasizes giving users visibility and control.

**Current gap**: Trajectory has verbosity levels but no progressive disclosure.

**Recommendation**: Expandable trajectory with drill-down

```python
# Enhancement to trajectory.py

class ProgressiveTrajectory:
    """Trajectory renderer with progressive disclosure."""

    def render_summary(self) -> str:
        """One-line summary of RLM progress."""
        # "RLM: 3 recursive calls, found bug in auth.ts:45"

    def render_overview(self) -> str:
        """Key events without details."""
        # Show RECURSE boundaries, FINAL, ERROR only

    def render_detail(self, event_id: str) -> str:
        """Full details for specific event."""

    def render_cost_breakdown(self) -> str:
        """Detailed cost attribution by component."""
```

### 6.2 Add Interactive Steering

**Research basis**: Research on human-AI collaboration shows steering improves outcomes more than full automation.

**Current gap**: Users can only force RLM on/off, not steer mid-execution.

**Recommendation**: Add steering points during execution

```python
# Proposed: src/steering.py

class SteeringPoint:
    """Point where user can steer RLM execution."""

    type: Literal["branch", "depth", "abort", "refine"]
    options: list[str]
    default: str

class InteractiveOrchestrator(RLMOrchestrator):
    """Orchestrator with interactive steering."""

    async def request_steering(
        self,
        point: SteeringPoint,
        timeout: float = 30.0,
    ) -> str:
        """Request user steering decision."""

    def set_auto_steering_policy(
        self,
        policy: Callable[[SteeringPoint], str],
    ) -> None:
        """Set policy for automatic steering (testing, CI)."""
```

### 6.3 Add Learning from User Corrections

**Research basis**: RLHF and [ToTRL (2025)](https://arxiv.org/abs/2505.12717) show learning from feedback significantly improves reasoning.

**Current gap**: Strategy cache exists but doesn't learn from explicit user corrections.

**Recommendation**: Capture and learn from user feedback

```python
# Enhancement to strategy_cache.py

class FeedbackLearner:
    """Learn from user corrections to RLM outputs."""

    def record_correction(
        self,
        query: str,
        rlm_output: str,
        user_correction: str,
        correction_type: CorrectionType,
    ) -> None:
        """Record a user correction for learning."""

    def adjust_classifier(
        self,
        corrections: list[Correction],
    ) -> ClassifierAdjustments:
        """
        Suggest adjustments to complexity classifier based on corrections.

        If users frequently correct RLM on query type X,
        adjust activation threshold for X.
        """
```

---

## Part VII: Architecture Recommendations

### 7.1 Modularize Orchestrator

**Current state**: `orchestrator.py` and `intelligent_orchestrator.py` have overlapping responsibilities.

**Recommendation**: Clean separation of concerns

```
orchestrator/
├── __init__.py
├── core.py              # Base orchestration loop
├── intelligent.py       # Claude-powered decisions
├── async_executor.py    # Async execution engine
├── checkpointing.py     # Session persistence
└── steering.py          # User interaction
```

### 7.2 Add Plugin Architecture for REPL Functions

**Research basis**: The RLM paper shows emergent strategies vary by domain. Extensibility enables domain-specific functions.

**Recommendation**: Plugin system for domain-specific REPL functions

```python
# Proposed: src/repl_plugins.py

class REPLPlugin(Protocol):
    """Protocol for REPL function plugins."""

    @property
    def name(self) -> str: ...

    @property
    def functions(self) -> dict[str, Callable]: ...

    def on_load(self, env: RLMEnvironment) -> None: ...

# Example: Code analysis plugin
class CodeAnalysisPlugin:
    name = "code_analysis"

    functions = {
        "ast_parse": ast_parse,
        "find_callers": find_callers,
        "find_callees": find_callees,
        "dependency_graph": dependency_graph,
    }
```

### 7.3 Separate Memory Store Interface

**Current state**: `MemoryStore` directly implements SQLite operations.

**Recommendation**: Abstract interface for storage backend flexibility

```python
# Proposed: src/memory/interface.py

class MemoryBackend(Protocol):
    """Abstract memory storage backend."""

    def create_node(self, ...) -> str: ...
    def get_node(self, node_id: str) -> Node | None: ...
    def search(self, query: str, ...) -> list[SearchResult]: ...
    # etc.

# Implementations
class SQLiteBackend(MemoryBackend): ...
class PostgresBackend(MemoryBackend): ...  # For team/cloud scenarios
class InMemoryBackend(MemoryBackend): ...  # For testing
```

---

## Part VIII: Prioritized Roadmap

### Phase A: Quick Wins (1-2 weeks of work)

| Item | Impact | Effort | Dependencies |
|------|--------|--------|--------------|
| 3.1 Async recursive calls | HIGH | Medium | None |
| 3.2 Prompt caching | HIGH | Low | None |
| 5.2 Hard execution guarantees | Medium | Low | None |
| 6.1 Progressive trajectory | Medium | Low | None |

### Phase B: Core Improvements (3-4 weeks)

| Item | Impact | Effort | Dependencies |
|------|--------|--------|--------------|
| 2.2 Compute-optimal allocation | HIGH | Medium | 5.2 |
| 4.1 Embedding-based retrieval | HIGH | Medium | None |
| 5.1 Confidence-weighted synthesis | Medium | Medium | None |
| 7.1 Modularize orchestrator | Medium | Medium | None |

### Phase C: Advanced Capabilities (5-8 weeks)

| Item | Impact | Effort | Dependencies |
|------|--------|--------|--------------|
| 2.1 ToT integration | HIGH | High | 7.1 |
| 4.3 Checkpointing | HIGH | High | 7.1 |
| 2.3 Formal verification | Medium | High | 7.2 |
| 6.2 Interactive steering | Medium | Medium | 6.1 |

### Phase D: Learning & Evolution (Ongoing)

| Item | Impact | Effort | Dependencies |
|------|--------|--------|--------------|
| 4.2 Cross-session promotion | Medium | Medium | 4.1 |
| 6.3 Learning from corrections | Medium | Medium | Strategy cache |
| 7.2 REPL plugin system | Medium | Medium | 7.1 |

---

## References

### Primary Sources
- [Recursive Language Models](https://arxiv.org/abs/2512.24601) - Zhang, Kraska, Khattab (MIT)
- [Tree of Thoughts](https://arxiv.org/abs/2305.10601) - Yao et al. (NeurIPS 2023)
- [Scaling LLM Test-Time Compute](https://arxiv.org/abs/2408.03314) - (ICLR 2025)

### Memory Systems
- [A-MEM: Agentic Memory](https://arxiv.org/abs/2502.12110) - NeurIPS 2025
- [Zep: Temporal Knowledge Graph](https://blog.getzep.com/content/files/2025/01/ZEP__USING_KNOWLEDGE_GRAPHS_TO_POWER_LLM_AGENT_MEMORY_2025011700.pdf)
- [G-Memory: Hierarchical Graph Memory](https://arxiv.org/html/2511.07800v1)

### Context Engineering
- [Anthropic: Effective Context Engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- [Anthropic: Long-Running Agent Harnesses](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- [KVzip Memory Compression](https://techxplore.com/news/2025-11-ai-tech-compress-llm-chatbot.html)

### Formal Verification
- [PREFACE: RL for Code Verification](https://dl.acm.org/doi/10.1145/3716368.3735300)
- [Formal Verification of LLM Code](https://arxiv.org/abs/2507.13290)

---

## Appendix: Measurement Framework

To track improvement, instrument these metrics:

### Intelligence Metrics
- **Decomposition quality**: % of queries where decomposition matches expert-labeled strategy
- **Synthesis accuracy**: % of synthesized answers rated correct by user/evaluator
- **Backtracking rate**: % of queries requiring backtracking (lower = better initial decomposition)

### Performance Metrics
- **P50/P95 latency**: End-to-end query completion time
- **Cost per query**: Total API cost per query, segmented by complexity
- **Cache hit rate**: % of recursive calls benefiting from prompt caching

### Reliability Metrics
- **Guarantee adherence**: % of queries completing within budget/time guarantees
- **Circuit breaker triggers**: Rate of circuit breaker activations
- **Recovery success**: % of checkpointed sessions successfully resumed

### UX Metrics
- **User override rate**: % of queries where user forces RLM on/off
- **Correction rate**: % of RLM outputs requiring user correction
- **Steering adoption**: % of steering opportunities where user provides input
