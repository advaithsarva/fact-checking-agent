"""End-to-end evidence-loop tests against the offline demo corpus. Needs the
real NLI model, so slower on first run (model download, cached after).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import check_claim, verify_document


def test_true_claim_with_direct_evidence_is_supported():
    v = check_claim("The Eiffel Tower is located in Paris, France.")
    assert v.verdict == "Supported"
    assert v.search_mode == "offline"


def test_well_known_myth_is_contradicted():
    v = check_claim("The Great Wall of China is visible from space with the naked eye.")
    assert v.verdict == "Contradicted"


def test_claim_with_no_relevant_evidence_is_unverifiable_not_guessed():
    v = check_claim("Bananas are the most popular fruit on Mars.")
    assert v.verdict == "Unverifiable"
    assert v.evidence == []


def test_verify_document_produces_one_verdict_per_claim_and_a_credibility_score():
    text = "The Eiffel Tower is located in Paris. Bananas are the most popular fruit on Mars."
    report = verify_document(text)
    assert len(report.claim_verdicts) == 2
    assert 0.0 <= report.document_credibility <= 1.0


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
