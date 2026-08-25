"""Regression test for the multi-sentence-premise bug: an early version of
classify() fed the whole (often multi-sentence) evidence passage to the NLI
model in one shot, which returned high-confidence "neutral" even when one
sentence in the passage was a near-exact match for the claim. Splitting into
sentences fixed it -- but the naive fix (take the single highest-confidence
sentence result) had its own bug: a confident "neutral" on an unrelated
sentence would outrank a correct but slightly-less-confident "supports" on
the one sentence that mattered. See stance.py's docstring and comments.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stance import classify


def test_matching_sentence_wins_over_more_confident_neutral_ones():
    # sentence 1 is a near-exact match for the claim; sentences 2-3 are
    # unrelated to it but score a very confident "neutral" on their own --
    # this is exactly the shape that broke the naive max-confidence version.
    passage = (
        "Mount Everest is the highest mountain above sea level on Earth. "
        "Its peak elevation is commonly cited as 8,849 meters. "
        "It is part of the Himalayan mountain range."
    )
    claim = "Mount Everest is the highest mountain on Earth above sea level."
    result = classify(passage, claim)
    assert result["stance"] == "supports"


def test_empty_evidence_is_neutral_not_a_crash():
    assert classify("", "Some claim.") == {"stance": "neutral", "confidence": 0.0}


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
