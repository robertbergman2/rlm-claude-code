"""
Consistency checker for epistemic verification.

Implements: SPEC-16.06 Consistency checker (evidence scrubbing)

Detects procedural hallucinations by comparing model responses with
and without evidence. When answers are too similar regardless of evidence,
the model may be "knowing but not using" the information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from src.epistemic.similarity import SemanticSimilarity, SimilarityConfig, SimilarityResult

if TYPE_CHECKING:
    from src.api_client import APIResponse
    from src.embedding_retrieval import EmbeddingProvider


class LLMClient(Protocol):
    """Protocol for LLM client to enable dependency injection."""

    async def complete(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> APIResponse: ...


@dataclass
class ConsistencyResult:
    """
    Result of consistency check between with/without evidence responses.

    Attributes:
        evidence_dependence: How much the answer depends on evidence (0.0-1.0)
            0.0 = answer is identical regardless of evidence (hallucination risk)
            1.0 = answer completely changes without evidence (high dependence)
        consistency_score: Semantic similarity between responses (0.0-1.0)
            1.0 = responses are identical (hallucination risk)
            0.0 = responses are completely different (good evidence use)
        response_with_evidence: The answer generated WITH evidence
        response_without_evidence: The answer generated WITHOUT evidence
        similarity_result: Full similarity comparison result
        is_suspicious: Whether the result suggests possible hallucination
        suspicion_reason: Why the result is suspicious (if applicable)
    """

    evidence_dependence: float
    consistency_score: float
    response_with_evidence: str
    response_without_evidence: str
    similarity_result: SimilarityResult | None = None
    is_suspicious: bool = False
    suspicion_reason: str | None = None

    def __post_init__(self) -> None:
        """Validate score ranges."""
        if not 0.0 <= self.evidence_dependence <= 1.0:
            raise ValueError(
                f"evidence_dependence must be between 0.0 and 1.0, got {self.evidence_dependence}"
            )
        if not 0.0 <= self.consistency_score <= 1.0:
            raise ValueError(
                f"consistency_score must be between 0.0 and 1.0, got {self.consistency_score}"
            )


@dataclass
class ConsistencyConfig:
    """
    Configuration for consistency checking.

    Attributes:
        suspicion_threshold: Consistency above this triggers suspicion (default 0.85)
        high_dependence_threshold: Dependence below this triggers suspicion (default 0.2)
        similarity_config: Configuration for semantic similarity
        model: Model to use for generating responses
        scrub_model: Model to use for scrubbed (without evidence) responses
    """

    suspicion_threshold: float = 0.85
    high_dependence_threshold: float = 0.2
    similarity_config: SimilarityConfig = field(default_factory=SimilarityConfig)
    model: str = "haiku"
    scrub_model: str = "haiku"

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0.0 <= self.suspicion_threshold <= 1.0:
            raise ValueError(
                f"suspicion_threshold must be between 0.0 and 1.0, got {self.suspicion_threshold}"
            )
        if not 0.0 <= self.high_dependence_threshold <= 1.0:
            raise ValueError(
                f"high_dependence_threshold must be between 0.0 and 1.0, "
                f"got {self.high_dependence_threshold}"
            )


# Prompts for evidence scrubbing
SCRUB_SYSTEM = """You are a helpful assistant. Answer the user's question based on your knowledge.
Do NOT make up specific details you don't know - say "I don't know" for specifics you're uncertain about."""

SCRUB_PROMPT_TEMPLATE = """Question: {question}

Please answer this question."""

WITH_EVIDENCE_SYSTEM = """You are a helpful assistant. Answer the user's question using ONLY the provided evidence.
Base your answer solely on the evidence given - do not use outside knowledge."""

WITH_EVIDENCE_PROMPT_TEMPLATE = """Evidence:
{evidence}

Question: {question}

Please answer this question using ONLY the evidence provided above."""


class ConsistencyChecker:
    """
    Checks consistency between responses with and without evidence.

    Implements the evidence scrubbing technique from hallucination detection:
    1. Generate response WITH evidence
    2. Generate response WITHOUT evidence (scrubbed)
    3. Compare semantic similarity
    4. High similarity = potential hallucination (not using evidence)

    Example:
        >>> checker = ConsistencyChecker(llm_client)
        >>> result = await checker.check_consistency(
        ...     question="What color is the widget?",
        ...     evidence="The widget is blue.",
        ... )
        >>> result.evidence_dependence
        0.95  # High dependence = good (answer uses evidence)
    """

    def __init__(
        self,
        llm_client: LLMClient,
        embedding_provider: EmbeddingProvider | None = None,
        config: ConsistencyConfig | None = None,
    ):
        """
        Initialize the consistency checker.

        Args:
            llm_client: LLM client for generating responses
            embedding_provider: Optional embedding provider for similarity
            config: Consistency checking configuration
        """
        self.client = llm_client
        self.config = config or ConsistencyConfig()
        self.similarity = SemanticSimilarity(
            embedding_provider=embedding_provider,
            llm_client=llm_client,
            config=self.config.similarity_config,
        )

    async def check_consistency(
        self,
        question: str,
        evidence: str,
        existing_answer: str | None = None,
    ) -> ConsistencyResult:
        """
        Check consistency between responses with and without evidence.

        Args:
            question: The question being answered
            evidence: The evidence that should inform the answer
            existing_answer: Optional pre-generated answer (skips with-evidence generation)

        Returns:
            ConsistencyResult with dependence score and analysis
        """
        # Get response WITH evidence (or use provided)
        if existing_answer is not None:
            response_with = existing_answer
        else:
            response_with = await self._generate_with_evidence(question, evidence)

        # Get response WITHOUT evidence (scrubbed)
        response_without = await self._generate_without_evidence(question)

        # Compare responses semantically
        similarity_result = await self.similarity.compare(response_with, response_without)

        # Compute evidence dependence (inverse of consistency)
        # High similarity → low dependence (answer doesn't change → not using evidence)
        # Low similarity → high dependence (answer changes significantly → using evidence)
        consistency_score = similarity_result.score
        evidence_dependence = 1.0 - consistency_score

        # Determine if suspicious
        is_suspicious, suspicion_reason = self._check_suspicion(
            evidence_dependence, consistency_score
        )

        return ConsistencyResult(
            evidence_dependence=evidence_dependence,
            consistency_score=consistency_score,
            response_with_evidence=response_with,
            response_without_evidence=response_without,
            similarity_result=similarity_result,
            is_suspicious=is_suspicious,
            suspicion_reason=suspicion_reason,
        )

    async def compute_evidence_dependence(
        self,
        question: str,
        answer: str,
        evidence: str,
    ) -> float:
        """
        Compute how much an answer depends on the given evidence.

        This is a convenience method that returns just the dependence score.

        Args:
            question: The question that was answered
            answer: The answer to evaluate
            evidence: The evidence the answer should depend on

        Returns:
            Evidence dependence score (0.0 = independent, 1.0 = fully dependent)
        """
        result = await self.check_consistency(
            question=question,
            evidence=evidence,
            existing_answer=answer,
        )
        return result.evidence_dependence

    async def batch_check_consistency(
        self,
        items: list[tuple[str, str, str | None]],
    ) -> list[ConsistencyResult]:
        """
        Check consistency for multiple question/evidence pairs.

        Args:
            items: List of (question, evidence, optional_answer) tuples

        Returns:
            List of ConsistencyResult objects
        """
        results: list[ConsistencyResult] = []
        for question, evidence, answer in items:
            result = await self.check_consistency(question, evidence, answer)
            results.append(result)
        return results

    async def _generate_with_evidence(self, question: str, evidence: str) -> str:
        """Generate response using the provided evidence."""
        prompt = WITH_EVIDENCE_PROMPT_TEMPLATE.format(
            question=question,
            evidence=evidence,
        )

        response = await self.client.complete(
            messages=[{"role": "user", "content": prompt}],
            system=WITH_EVIDENCE_SYSTEM,
            model=self.config.model,
            max_tokens=1024,
            temperature=0.0,
        )

        return response.content

    async def _generate_without_evidence(self, question: str) -> str:
        """Generate response WITHOUT evidence (scrubbed)."""
        prompt = SCRUB_PROMPT_TEMPLATE.format(question=question)

        response = await self.client.complete(
            messages=[{"role": "user", "content": prompt}],
            system=SCRUB_SYSTEM,
            model=self.config.scrub_model,
            max_tokens=1024,
            temperature=0.0,
        )

        return response.content

    def _check_suspicion(
        self,
        evidence_dependence: float,
        consistency_score: float,
    ) -> tuple[bool, str | None]:
        """
        Determine if the consistency result is suspicious.

        Returns:
            Tuple of (is_suspicious, reason)
        """
        # High consistency = answers are too similar = not using evidence
        if consistency_score >= self.config.suspicion_threshold:
            return True, (
                f"Answer is {consistency_score:.0%} similar with/without evidence. "
                "Model may not be using the provided evidence."
            )

        # Low evidence dependence = answer doesn't change much = suspicious
        if evidence_dependence <= self.config.high_dependence_threshold:
            return True, (
                f"Evidence dependence is only {evidence_dependence:.0%}. "
                "Answer may be based on prior knowledge rather than evidence."
            )

        return False, None


def compute_consistency_score(
    response_with: str,
    response_without: str,
    similarity: SemanticSimilarity,
) -> float:
    """
    Compute consistency score between two responses (sync version).

    Uses embedding-only comparison for speed.

    Args:
        response_with: Response generated with evidence
        response_without: Response generated without evidence
        similarity: SemanticSimilarity instance

    Returns:
        Consistency score (0.0-1.0)
    """
    result = similarity.compare_sync(response_with, response_without)
    return result.score


def evidence_dependence_from_similarity(similarity_score: float) -> float:
    """
    Convert similarity score to evidence dependence.

    Evidence dependence is the inverse of similarity:
    - High similarity (0.9) → Low dependence (0.1) → Not using evidence
    - Low similarity (0.2) → High dependence (0.8) → Using evidence

    Args:
        similarity_score: Semantic similarity between with/without responses

    Returns:
        Evidence dependence score (0.0-1.0)
    """
    return 1.0 - similarity_score
