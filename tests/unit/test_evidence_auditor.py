"""
Unit tests for evidence auditor.

Implements: SPEC-16.09 Unit tests for evidence auditor
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from src.epistemic import (
    AuditResult,
    BatchAuditResult,
    EvidenceAuditor,
    ExtractedClaim,
    compute_evidence_support,
)


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


class TestAuditResult:
    """Tests for AuditResult dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic result creation."""
        from src.epistemic import ClaimVerification

        verification = ClaimVerification(claim_id="c1", claim_text="Test")
        result = AuditResult(verification=verification, audit_model="haiku")

        assert result.verification.claim_id == "c1"
        assert result.gaps == []
        assert result.audit_model == "haiku"


class TestBatchAuditResult:
    """Tests for BatchAuditResult dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic batch result creation."""
        result = BatchAuditResult(
            results=[],
            total_claims=5,
            flagged_count=2,
            phantom_count=1,
        )
        assert result.total_claims == 5
        assert result.flagged_count == 2
        assert result.phantom_count == 1


class TestEvidenceAuditor:
    """Tests for EvidenceAuditor class."""

    @pytest.mark.asyncio
    async def test_audit_claim_supported(self) -> None:
        """Test auditing a well-supported claim."""
        response = json.dumps(
            {
                "support_score": 0.95,
                "issues": [],
                "reasoning": "The evidence explicitly supports this claim",
            }
        )
        client = MockLLMClient(responses=[response])
        auditor = EvidenceAuditor(client)

        claim = ExtractedClaim(
            claim_id="c1",
            claim_text="The function returns 42",
            evidence_ids=["e1"],
        )
        evidence = {"e1": "def func(): return 42"}

        result = await auditor.audit_claim(claim, evidence)

        assert result.verification.evidence_support == 0.95
        assert not result.verification.is_flagged
        assert len(result.gaps) == 0

    @pytest.mark.asyncio
    async def test_audit_claim_unsupported(self) -> None:
        """Test auditing an unsupported claim."""
        response = json.dumps(
            {
                "support_score": 0.2,
                "issues": ["unsupported"],
                "reasoning": "Evidence does not support this claim",
            }
        )
        client = MockLLMClient(responses=[response])
        auditor = EvidenceAuditor(client, support_threshold=0.7)

        claim = ExtractedClaim(
            claim_id="c1",
            claim_text="The function returns 100",
            evidence_ids=["e1"],
        )
        evidence = {"e1": "def func(): return 42"}

        result = await auditor.audit_claim(claim, evidence)

        assert result.verification.evidence_support == 0.2
        assert result.verification.is_flagged
        assert result.verification.flag_reason == "unsupported"
        assert len(result.gaps) >= 1

    @pytest.mark.asyncio
    async def test_audit_claim_phantom_citation(self) -> None:
        """Test detection of phantom citations."""
        response = json.dumps(
            {
                "support_score": 0.8,
                "issues": [],
                "reasoning": "Supported",
            }
        )
        client = MockLLMClient(responses=[response])
        auditor = EvidenceAuditor(client)

        claim = ExtractedClaim(
            claim_id="c1",
            claim_text="According to file.py, X is Y",
            evidence_ids=["nonexistent.py"],  # This doesn't exist
        )
        evidence = {"e1": "Some actual evidence"}

        result = await auditor.audit_claim(claim, evidence)

        # Should have phantom citation gap
        phantom_gaps = [g for g in result.gaps if g.gap_type == "phantom_citation"]
        assert len(phantom_gaps) == 1
        assert result.verification.is_flagged
        assert result.verification.flag_reason == "phantom_citation"

    @pytest.mark.asyncio
    async def test_audit_claim_no_evidence(self) -> None:
        """Test auditing claim with no matching evidence."""
        client = MockLLMClient(responses=[])
        auditor = EvidenceAuditor(client)

        claim = ExtractedClaim(
            claim_id="c1",
            claim_text="Some claim",
            evidence_ids=["missing"],
        )
        evidence = {}  # No evidence at all

        result = await auditor.audit_claim(claim, evidence)

        assert result.verification.evidence_support == 0.0
        assert result.verification.is_flagged
        assert result.verification.flag_reason == "unsupported"

    @pytest.mark.asyncio
    async def test_audit_claim_contradiction(self) -> None:
        """Test detection of contradictions."""
        response = json.dumps(
            {
                "support_score": 0.1,
                "issues": ["contradiction"],
                "reasoning": "Evidence contradicts the claim",
            }
        )
        client = MockLLMClient(responses=[response])
        auditor = EvidenceAuditor(client)

        claim = ExtractedClaim(
            claim_id="c1",
            claim_text="X equals 5",
            evidence_ids=["e1"],
        )
        evidence = {"e1": "X equals 10, not 5"}

        result = await auditor.audit_claim(claim, evidence)

        assert result.verification.is_flagged
        assert result.verification.flag_reason == "contradiction"
        contradiction_gaps = [g for g in result.gaps if g.gap_type == "contradicted"]
        assert len(contradiction_gaps) == 1

    @pytest.mark.asyncio
    async def test_audit_claim_extrapolation(self) -> None:
        """Test detection of over-extrapolation."""
        response = json.dumps(
            {
                "support_score": 0.5,
                "issues": ["extrapolation"],
                "reasoning": "Claim goes beyond what evidence states",
            }
        )
        client = MockLLMClient(responses=[response])
        auditor = EvidenceAuditor(client)

        claim = ExtractedClaim(
            claim_id="c1",
            claim_text="X is always Y in all cases",
            evidence_ids=["e1"],
        )
        evidence = {"e1": "X is Y in this example"}

        result = await auditor.audit_claim(claim, evidence)

        assert result.verification.is_flagged
        extrapolation_gaps = [g for g in result.gaps if g.gap_type == "over_extrapolation"]
        assert len(extrapolation_gaps) == 1

    @pytest.mark.asyncio
    async def test_audit_claim_uses_critical_model(self) -> None:
        """Test that critical claims use critical model."""
        response = json.dumps({"support_score": 0.9, "issues": [], "reasoning": "OK"})
        client = MockLLMClient(responses=[response])
        auditor = EvidenceAuditor(client, default_model="haiku", critical_model="sonnet")

        claim = ExtractedClaim(claim_id="c1", claim_text="Test", evidence_ids=["e1"])
        evidence = {"e1": "Evidence"}

        await auditor.audit_claim(claim, evidence, is_critical=True)

        assert client.calls[0]["model"] == "sonnet"

    @pytest.mark.asyncio
    async def test_audit_claim_uses_default_model(self) -> None:
        """Test that non-critical claims use default model."""
        response = json.dumps({"support_score": 0.9, "issues": [], "reasoning": "OK"})
        client = MockLLMClient(responses=[response])
        auditor = EvidenceAuditor(client, default_model="haiku", critical_model="sonnet")

        claim = ExtractedClaim(claim_id="c1", claim_text="Test", evidence_ids=["e1"])
        evidence = {"e1": "Evidence"}

        await auditor.audit_claim(claim, evidence, is_critical=False)

        assert client.calls[0]["model"] == "haiku"

    @pytest.mark.asyncio
    async def test_audit_claims_batch(self) -> None:
        """Test batch auditing of multiple claims."""
        responses = [
            json.dumps({"support_score": 0.9, "issues": [], "reasoning": "Good"}),
            json.dumps({"support_score": 0.3, "issues": ["unsupported"], "reasoning": "Bad"}),
        ]
        client = MockLLMClient(responses=responses)
        auditor = EvidenceAuditor(client, support_threshold=0.7)

        claims = [
            ExtractedClaim(claim_id="c1", claim_text="Claim 1", evidence_ids=["e1"]),
            ExtractedClaim(claim_id="c2", claim_text="Claim 2", evidence_ids=["e1"]),
        ]
        evidence = {"e1": "Some evidence"}

        result = await auditor.audit_claims(claims, evidence)

        assert result.total_claims == 2
        assert result.flagged_count == 1
        assert len(result.results) == 2

    @pytest.mark.asyncio
    async def test_audit_claims_respects_critical_ids(self) -> None:
        """Test that specified critical IDs use critical model."""
        responses = [
            json.dumps({"support_score": 0.9, "issues": [], "reasoning": "OK"}),
            json.dumps({"support_score": 0.9, "issues": [], "reasoning": "OK"}),
        ]
        client = MockLLMClient(responses=responses)
        auditor = EvidenceAuditor(client, default_model="haiku", critical_model="sonnet")

        claims = [
            ExtractedClaim(claim_id="c1", claim_text="Claim 1", evidence_ids=["e1"]),
            ExtractedClaim(claim_id="c2", claim_text="Claim 2", evidence_ids=["e1"]),
        ]
        evidence = {"e1": "Evidence"}

        await auditor.audit_claims(claims, evidence, critical_claim_ids={"c2"})

        # First claim should use default model
        assert client.calls[0]["model"] == "haiku"
        # Second claim should use critical model
        assert client.calls[1]["model"] == "sonnet"

    @pytest.mark.asyncio
    async def test_audit_handles_invalid_json(self) -> None:
        """Test that invalid JSON responses are handled gracefully."""
        client = MockLLMClient(responses=["This is not JSON at all"])
        auditor = EvidenceAuditor(client)

        claim = ExtractedClaim(claim_id="c1", claim_text="Test", evidence_ids=["e1"])
        evidence = {"e1": "Evidence"}

        result = await auditor.audit_claim(claim, evidence)

        # Should return default score, not crash
        assert result.verification.evidence_support == 0.5

    @pytest.mark.asyncio
    async def test_audit_clamps_score_to_valid_range(self) -> None:
        """Test that out-of-range scores are clamped."""
        response = json.dumps(
            {
                "support_score": 1.5,  # Invalid, should be clamped to 1.0
                "issues": [],
                "reasoning": "OK",
            }
        )
        client = MockLLMClient(responses=[response])
        auditor = EvidenceAuditor(client)

        claim = ExtractedClaim(claim_id="c1", claim_text="Test", evidence_ids=["e1"])
        evidence = {"e1": "Evidence"}

        result = await auditor.audit_claim(claim, evidence)

        assert result.verification.evidence_support == 1.0

    @pytest.mark.asyncio
    async def test_phantom_detection_with_content_match(self) -> None:
        """Test phantom detection considers evidence content."""
        response = json.dumps({"support_score": 0.9, "issues": [], "reasoning": "OK"})
        client = MockLLMClient(responses=[response])
        auditor = EvidenceAuditor(client)

        claim = ExtractedClaim(
            claim_id="c1",
            claim_text="Test",
            evidence_ids=["src/foo.py:42"],  # Reference within evidence
        )
        evidence = {"e1": "Content from src/foo.py:42 shows..."}

        result = await auditor.audit_claim(claim, evidence)

        # Should NOT be a phantom because the reference exists in content
        phantom_gaps = [g for g in result.gaps if g.gap_type == "phantom_citation"]
        assert len(phantom_gaps) == 0


class TestComputeEvidenceSupport:
    """Tests for compute_evidence_support function."""

    def test_basic_computation(self) -> None:
        """Test basic support computation."""
        result = {"support_score": 0.9, "issues": []}
        score = compute_evidence_support("claim", "evidence", result)
        assert score == 0.9

    def test_penalty_for_issues(self) -> None:
        """Test that issues reduce the score."""
        result = {"support_score": 0.9, "issues": ["partial", "extrapolation"]}
        score = compute_evidence_support("claim", "evidence", result)
        # 0.9 - 0.2 penalty = 0.7
        assert score == 0.7

    def test_score_not_negative(self) -> None:
        """Test that score doesn't go below 0."""
        result = {"support_score": 0.2, "issues": ["a", "b", "c", "d", "e"]}
        score = compute_evidence_support("claim", "evidence", result)
        assert score == 0.0

    def test_default_score(self) -> None:
        """Test default score when not provided."""
        result = {"issues": []}
        score = compute_evidence_support("claim", "evidence", result)
        assert score == 0.5


class TestGapBitsComputation:
    """Tests for gap bits computation."""

    @pytest.mark.asyncio
    async def test_high_support_low_bits(self) -> None:
        """Test that high support results in low gap bits."""
        response = json.dumps({"support_score": 0.95, "issues": [], "reasoning": "OK"})
        client = MockLLMClient(responses=[response])
        auditor = EvidenceAuditor(client)

        claim = ExtractedClaim(claim_id="c1", claim_text="Test", evidence_ids=["e1"])
        evidence = {"e1": "Evidence"}

        result = await auditor.audit_claim(claim, evidence)

        # High support = low gap bits
        assert result.verification.evidence_gap_bits < 0.1

    @pytest.mark.asyncio
    async def test_low_support_high_bits(self) -> None:
        """Test that low support results in high gap bits."""
        response = json.dumps({"support_score": 0.1, "issues": [], "reasoning": "Bad"})
        client = MockLLMClient(responses=[response])
        auditor = EvidenceAuditor(client)

        claim = ExtractedClaim(claim_id="c1", claim_text="Test", evidence_ids=["e1"])
        evidence = {"e1": "Evidence"}

        result = await auditor.audit_claim(claim, evidence)

        # Low support = high gap bits (~3.3 bits for 0.1 support)
        assert result.verification.evidence_gap_bits > 3.0
