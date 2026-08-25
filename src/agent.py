"""The evidence loop: search -> assess each result's stance and reliability
-> score the claim. This is the "multi-step" core of the agent -- one claim
can pull evidence from several sources before a verdict is reached, and the
verdict is a function of reliability-weighted stance, not just a majority
vote of raw hits.
"""
from dataclasses import dataclass, field

from src.claims import Claim, extract_claims
from src.search import search
from src.stance import classify
from src.reliability import score_url, score_corpus_doc

SUPPORT_MARGIN = 0.15  # how much supporting must beat contradicting to call a verdict


@dataclass
class EvidenceItem:
    source: str
    snippet: str
    stance: str
    stance_confidence: float
    reliability_tier: str
    reliability_score: float
    reliability_reason: str


@dataclass
class ClaimVerdict:
    claim: str
    verdict: str  # "Supported" | "Contradicted" | "Unverifiable"
    confidence: float
    evidence: list[EvidenceItem] = field(default_factory=list)
    search_mode: str = "offline"


@dataclass
class VerificationReport:
    document_credibility: float
    claim_verdicts: list[ClaimVerdict] = field(default_factory=list)


def _weighted_stance(evidence: list[EvidenceItem]) -> tuple[float, float]:
    supporting = sum(e.reliability_score * e.stance_confidence for e in evidence if e.stance == "supports")
    contradicting = sum(e.reliability_score * e.stance_confidence for e in evidence if e.stance == "contradicts")
    return supporting, contradicting


def _verdict_from_weights(supporting: float, contradicting: float) -> tuple[str, float]:
    total = supporting + contradicting
    if total < 1e-6:
        return "Unverifiable", 0.0
    if supporting - contradicting > SUPPORT_MARGIN:
        return "Supported", round(supporting / total, 4)
    if contradicting - supporting > SUPPORT_MARGIN:
        return "Contradicted", round(contradicting / total, 4)
    return "Unverifiable", round(1 - abs(supporting - contradicting) / total, 4)  # disputed evidence


def check_claim(claim_text: str, k: int = 3) -> ClaimVerdict:
    results, mode = search(claim_text, k)

    evidence = []
    for r in results:
        stance = classify(r.snippet, claim_text)
        reliability = score_corpus_doc(r.tier) if mode == "offline" else score_url(r.source)
        evidence.append(EvidenceItem(
            source=r.source, snippet=r.snippet,
            stance=stance["stance"], stance_confidence=stance["confidence"],
            reliability_tier=reliability["tier"], reliability_score=reliability["score"],
            reliability_reason=reliability["reason"],
        ))

    supporting, contradicting = _weighted_stance(evidence)
    verdict, confidence = _verdict_from_weights(supporting, contradicting)
    return ClaimVerdict(claim=claim_text, verdict=verdict, confidence=confidence, evidence=evidence, search_mode=mode)


def verify_document(text: str, k: int = 3) -> VerificationReport:
    claims = extract_claims(text)
    verdicts = [check_claim(c.text, k=k) for c in claims]

    # (supported - contradicted) / total, rescaled from [-1, 1] to [0, 1]:
    # all supported -> 1.0, all contradicted -> 0.0, all unverifiable -> 0.5 (neutral).
    scored = [v.confidence for v in verdicts if v.verdict == "Supported"]
    contradicted = [v for v in verdicts if v.verdict == "Contradicted"]
    credibility = (
        0.0 if not verdicts else
        round((len(scored) - len(contradicted)) / len(verdicts) * 0.5 + 0.5, 4)
    )
    return VerificationReport(document_credibility=credibility, claim_verdicts=verdicts)
