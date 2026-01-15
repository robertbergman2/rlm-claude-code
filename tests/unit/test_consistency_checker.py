"""
Unit tests for consistency checker.

Implements: SPEC-16.06 Unit tests for consistency checker
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from src.epistemic import (
    ConsistencyChecker,
    ConsistencyConfig,
    ConsistencyResult,
    compute_consistency_score,
    evidence_dependence_from_similarity,
)
from src.epistemic.similarity import SemanticSimilarity


@dataclass
class MockAPIResponse:
    """Mock API response for testing."""

    content: str
    input_tokens: int = 100
    output_tokens: int = 50
    model: str = "haiku"


class MockLLMClient:
    """Mock LLM client that returns predefined responses."""

    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or []
        self.call_count = 0
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> MockAPIResponse:
        self.calls.append(
            {
                "messages": messages,
                "system": system,
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )

        response_content = ""
        if self.call_count < len(self.responses):
            response_content = self.responses[self.call_count]
        self.call_count += 1

        return MockAPIResponse(content=response_content, model=model or "haiku")


class MockEmbeddingProvider:
    """Mock embedding provider for testing."""

    def __init__(self, embeddings: dict[str, list[float]] | None = None):
        self.embeddings = embeddings or {}
        self.call_count = 0

    def embed(self, text: str) -> list[float]:
        self.call_count += 1
        if text in self.embeddings:
            return self.embeddings[text]
        # Return a simple hash-based embedding for testing
        return [float(hash(text) % 100) / 100 for _ in range(8)]


class TestConsistencyResult:
    """Tests for ConsistencyResult dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic result creation."""
        result = ConsistencyResult(
            evidence_dependence=0.8,
            consistency_score=0.2,
            response_with_evidence="Answer with evidence",
            response_without_evidence="Answer without evidence",
        )
        assert result.evidence_dependence == 0.8
        assert result.consistency_score == 0.2
        assert not result.is_suspicious

    def test_suspicious_result(self) -> None:
        """Test suspicious result creation."""
        result = ConsistencyResult(
            evidence_dependence=0.1,
            consistency_score=0.9,
            response_with_evidence="Same answer",
            response_without_evidence="Same answer",
            is_suspicious=True,
            suspicion_reason="High consistency",
        )
        assert result.is_suspicious
        assert result.suspicion_reason == "High consistency"

    def test_invalid_evidence_dependence_raises(self) -> None:
        """Test invalid evidence_dependence raises ValueError."""
        with pytest.raises(ValueError, match="evidence_dependence must be between"):
            ConsistencyResult(
                evidence_dependence=1.5,
                consistency_score=0.5,
                response_with_evidence="a",
                response_without_evidence="b",
            )

    def test_invalid_consistency_score_raises(self) -> None:
        """Test invalid consistency_score raises ValueError."""
        with pytest.raises(ValueError, match="consistency_score must be between"):
            ConsistencyResult(
                evidence_dependence=0.5,
                consistency_score=-0.1,
                response_with_evidence="a",
                response_without_evidence="b",
            )

    def test_boundary_values(self) -> None:
        """Test boundary values are accepted."""
        result = ConsistencyResult(
            evidence_dependence=0.0,
            consistency_score=1.0,
            response_with_evidence="a",
            response_without_evidence="b",
        )
        assert result.evidence_dependence == 0.0
        assert result.consistency_score == 1.0


class TestConsistencyConfig:
    """Tests for ConsistencyConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = ConsistencyConfig()
        assert config.suspicion_threshold == 0.85
        assert config.high_dependence_threshold == 0.2
        assert config.model == "haiku"

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = ConsistencyConfig(
            suspicion_threshold=0.9,
            high_dependence_threshold=0.1,
            model="sonnet",
        )
        assert config.suspicion_threshold == 0.9
        assert config.model == "sonnet"

    def test_invalid_suspicion_threshold(self) -> None:
        """Test invalid suspicion_threshold raises ValueError."""
        with pytest.raises(ValueError, match="suspicion_threshold must be between"):
            ConsistencyConfig(suspicion_threshold=1.5)

    def test_invalid_high_dependence_threshold(self) -> None:
        """Test invalid high_dependence_threshold raises ValueError."""
        with pytest.raises(ValueError, match="high_dependence_threshold must be between"):
            ConsistencyConfig(high_dependence_threshold=-0.1)


class TestConsistencyChecker:
    """Tests for ConsistencyChecker class."""

    @pytest.mark.asyncio
    async def test_high_dependence_different_answers(self) -> None:
        """Test high dependence when answers differ significantly."""
        # Response with evidence: specific answer
        # Response without evidence: different/uncertain answer
        responses = [
            "The widget is blue based on the evidence.",
            "I don't know what color the widget is.",
        ]
        client = MockLLMClient(responses=responses)

        # Set up embeddings to make responses very different
        embeddings = {
            "The widget is blue based on the evidence.": [1.0, 0.0, 0.0, 0.0],
            "I don't know what color the widget is.": [0.0, 1.0, 0.0, 0.0],
        }
        provider = MockEmbeddingProvider(embeddings=embeddings)

        checker = ConsistencyChecker(client, embedding_provider=provider)
        result = await checker.check_consistency(
            question="What color is the widget?",
            evidence="The widget is blue.",
        )

        # High evidence dependence (answer changes significantly)
        assert result.evidence_dependence >= 0.4
        assert result.consistency_score <= 0.6
        assert not result.is_suspicious

    @pytest.mark.asyncio
    async def test_low_dependence_same_answers(self) -> None:
        """Test low dependence when answers are too similar."""
        # Both responses are nearly identical (hallucination risk)
        responses = [
            "The answer is 42.",
            "The answer is 42.",
        ]
        client = MockLLMClient(responses=responses)

        checker = ConsistencyChecker(client)
        result = await checker.check_consistency(
            question="What is the answer?",
            evidence="According to the document, the value is 42.",
        )

        # Identical responses = perfect consistency = low dependence
        assert result.evidence_dependence == 0.0
        assert result.consistency_score == 1.0
        assert result.is_suspicious
        assert "similar" in result.suspicion_reason.lower()

    @pytest.mark.asyncio
    async def test_existing_answer_skips_generation(self) -> None:
        """Test that existing_answer skips with-evidence generation."""
        # Only one response needed (for without evidence)
        responses = ["I don't know the answer."]
        client = MockLLMClient(responses=responses)

        checker = ConsistencyChecker(client)
        result = await checker.check_consistency(
            question="What is X?",
            evidence="X is Y.",
            existing_answer="X is Y according to the evidence.",
        )

        # Only one LLM call (for without evidence)
        assert client.call_count == 1
        assert result.response_with_evidence == "X is Y according to the evidence."

    @pytest.mark.asyncio
    async def test_compute_evidence_dependence(self) -> None:
        """Test convenience method for computing dependence."""
        responses = ["Uncertain answer without evidence."]
        client = MockLLMClient(responses=responses)

        embeddings = {
            "The specific answer from evidence.": [1.0, 0.0, 0.0, 0.0],
            "Uncertain answer without evidence.": [0.0, 0.8, 0.2, 0.0],
        }
        provider = MockEmbeddingProvider(embeddings=embeddings)

        checker = ConsistencyChecker(client, embedding_provider=provider)
        dependence = await checker.compute_evidence_dependence(
            question="What is X?",
            answer="The specific answer from evidence.",
            evidence="X is the specific answer.",
        )

        assert 0.0 <= dependence <= 1.0

    @pytest.mark.asyncio
    async def test_batch_check_consistency(self) -> None:
        """Test batch consistency checking."""
        responses = [
            "Answer 1 with evidence",
            "Answer 1 without",
            "Answer 2 with evidence",
            "Answer 2 without",
        ]
        client = MockLLMClient(responses=responses)
        checker = ConsistencyChecker(client)

        items = [
            ("Question 1?", "Evidence 1", None),
            ("Question 2?", "Evidence 2", None),
        ]
        results = await checker.batch_check_consistency(items)

        assert len(results) == 2
        # Each item needs 2 LLM calls
        assert client.call_count == 4

    @pytest.mark.asyncio
    async def test_suspicion_from_high_consistency(self) -> None:
        """Test suspicion is triggered by high consistency."""
        responses = ["Same answer", "Same answer"]
        client = MockLLMClient(responses=responses)

        config = ConsistencyConfig(suspicion_threshold=0.8)
        checker = ConsistencyChecker(client, config=config)

        result = await checker.check_consistency(
            question="Test?",
            evidence="Test evidence",
        )

        assert result.is_suspicious
        assert "similar" in result.suspicion_reason.lower()

    @pytest.mark.asyncio
    async def test_suspicion_from_low_dependence(self) -> None:
        """Test suspicion is triggered by low dependence threshold."""
        responses = ["Nearly same", "Nearly same answer"]
        client = MockLLMClient(responses=responses)

        embeddings = {
            "Nearly same": [1.0, 0.0, 0.0, 0.0],
            "Nearly same answer": [0.95, 0.05, 0.0, 0.0],
        }
        provider = MockEmbeddingProvider(embeddings=embeddings)

        config = ConsistencyConfig(
            suspicion_threshold=0.99,  # Won't trigger this
            high_dependence_threshold=0.3,  # Will trigger this
        )
        checker = ConsistencyChecker(client, embedding_provider=provider, config=config)

        result = await checker.check_consistency(
            question="Test?",
            evidence="Evidence",
        )

        # Very similar embeddings → high consistency → low dependence → suspicious
        if result.evidence_dependence <= 0.3:
            assert result.is_suspicious

    @pytest.mark.asyncio
    async def test_uses_correct_models(self) -> None:
        """Test that correct models are used for generation."""
        responses = ["With evidence", "Without evidence"]
        client = MockLLMClient(responses=responses)

        config = ConsistencyConfig(model="sonnet", scrub_model="haiku")
        checker = ConsistencyChecker(client, config=config)

        await checker.check_consistency(
            question="Test?",
            evidence="Evidence",
        )

        # First call (with evidence) uses model
        assert client.calls[0]["model"] == "sonnet"
        # Second call (without evidence) uses scrub_model
        assert client.calls[1]["model"] == "haiku"

    @pytest.mark.asyncio
    async def test_prompts_contain_question_and_evidence(self) -> None:
        """Test that prompts are constructed correctly."""
        responses = ["Answer with", "Answer without"]
        client = MockLLMClient(responses=responses)
        checker = ConsistencyChecker(client)

        await checker.check_consistency(
            question="What is the capital of France?",
            evidence="France is a country in Europe. Its capital is Paris.",
        )

        # First call should have evidence in the prompt
        with_evidence_prompt = client.calls[0]["messages"][0]["content"]
        assert "France" in with_evidence_prompt
        assert "Paris" in with_evidence_prompt
        assert "capital" in with_evidence_prompt.lower()

        # Second call should NOT have evidence
        without_evidence_prompt = client.calls[1]["messages"][0]["content"]
        assert "capital" in without_evidence_prompt.lower()
        # Evidence-specific content shouldn't be in scrubbed prompt
        assert "Paris" not in without_evidence_prompt


class TestComputeConsistencyScore:
    """Tests for compute_consistency_score helper."""

    def test_identical_responses(self) -> None:
        """Test identical responses give perfect consistency."""
        embeddings = {"hello world": [1.0, 0.0, 0.0, 0.0]}
        provider = MockEmbeddingProvider(embeddings=embeddings)
        similarity = SemanticSimilarity(embedding_provider=provider)

        score = compute_consistency_score("hello world", "hello world", similarity)
        assert score == 1.0

    def test_different_responses(self) -> None:
        """Test different responses give lower consistency."""
        embeddings = {
            "answer A": [1.0, 0.0, 0.0, 0.0],
            "answer B": [0.0, 1.0, 0.0, 0.0],
        }
        provider = MockEmbeddingProvider(embeddings=embeddings)
        similarity = SemanticSimilarity(embedding_provider=provider)

        score = compute_consistency_score("answer A", "answer B", similarity)
        assert score == 0.5  # Orthogonal vectors


class TestEvidenceDependenceFromSimilarity:
    """Tests for evidence_dependence_from_similarity helper."""

    def test_high_similarity_low_dependence(self) -> None:
        """Test high similarity yields low dependence."""
        dependence = evidence_dependence_from_similarity(0.9)
        assert dependence == pytest.approx(0.1)

    def test_low_similarity_high_dependence(self) -> None:
        """Test low similarity yields high dependence."""
        dependence = evidence_dependence_from_similarity(0.2)
        assert dependence == pytest.approx(0.8)

    def test_perfect_similarity_zero_dependence(self) -> None:
        """Test perfect similarity yields zero dependence."""
        dependence = evidence_dependence_from_similarity(1.0)
        assert dependence == 0.0

    def test_zero_similarity_full_dependence(self) -> None:
        """Test zero similarity yields full dependence."""
        dependence = evidence_dependence_from_similarity(0.0)
        assert dependence == 1.0


class TestConsistencyCheckerEdgeCases:
    """Edge case tests for ConsistencyChecker."""

    @pytest.mark.asyncio
    async def test_empty_evidence(self) -> None:
        """Test handling of empty evidence."""
        responses = ["Answer with empty evidence", "Answer without evidence"]
        client = MockLLMClient(responses=responses)
        checker = ConsistencyChecker(client)

        result = await checker.check_consistency(
            question="What is X?",
            evidence="",
        )

        # Should still produce a result
        assert 0.0 <= result.evidence_dependence <= 1.0
        assert 0.0 <= result.consistency_score <= 1.0

    @pytest.mark.asyncio
    async def test_very_long_evidence(self) -> None:
        """Test handling of very long evidence."""
        long_evidence = "This is evidence. " * 100
        responses = ["Summary answer", "No evidence answer"]
        client = MockLLMClient(responses=responses)
        checker = ConsistencyChecker(client)

        result = await checker.check_consistency(
            question="Summarize this?",
            evidence=long_evidence,
        )

        assert result is not None
        assert result.response_with_evidence == "Summary answer"

    @pytest.mark.asyncio
    async def test_special_characters_in_question(self) -> None:
        """Test handling of special characters."""
        responses = ["Answer 1", "Answer 2"]
        client = MockLLMClient(responses=responses)
        checker = ConsistencyChecker(client)

        result = await checker.check_consistency(
            question="What about 'quotes' and \"double quotes\"?",
            evidence="Evidence with {braces} and [brackets].",
        )

        assert result is not None
