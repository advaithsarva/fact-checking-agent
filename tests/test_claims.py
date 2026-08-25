"""Claim extraction/filtering tests. Needs spaCy's local model (no network
after the initial `spacy download`), no NLI/search involved -- fast.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.claims import extract_claims


def test_declarative_sentence_is_a_claim():
    claims = extract_claims("The Eiffel Tower is located in Paris.")
    assert len(claims) == 1
    assert claims[0].text == "The Eiffel Tower is located in Paris."


def test_question_is_not_a_claim():
    claims = extract_claims("Is the sky blue?")
    assert claims == []


def test_first_person_opinion_is_not_a_claim():
    claims = extract_claims("I think this policy is a mistake.")
    assert claims == []


def test_mixed_document_keeps_only_checkable_sentences():
    text = (
        "The Eiffel Tower is located in Paris. I believe that's fascinating. "
        "Is that surprising? Scientists discovered a new exoplanet last year."
    )
    claims = extract_claims(text)
    texts = [c.text for c in claims]
    assert "The Eiffel Tower is located in Paris." in texts
    assert "Scientists discovered a new exoplanet last year." in texts
    assert len(claims) == 2


def test_compound_sentence_splits_when_both_halves_are_full_claims():
    claims = extract_claims("The Great Wall of China is visible from space and Napoleon was short.")
    texts = [c.text for c in claims]
    assert len(texts) == 2
    assert any("Great Wall" in t for t in texts)
    assert any("Napoleon" in t for t in texts)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
