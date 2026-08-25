"""Pure-logic reliability scoring tests -- no models, no network, instant."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reliability import score_url, score_corpus_doc


def test_known_domain_scores_high():
    result = score_url("https://www.nasa.gov/mission/artemis")
    assert result["tier"] == "high"


def test_gov_suffix_scores_high_even_when_not_listed():
    result = score_url("https://www.irs.gov/forms")
    assert result["tier"] == "high"


def test_unknown_domain_is_not_penalized_as_unreliable():
    result = score_url("https://some-random-blog.example")
    assert result["tier"] == "unknown"
    assert "not a claim it's unreliable" in result["reason"]


def test_corpus_tiers_map_to_distinct_scores():
    primary = score_corpus_doc("primary")
    secondary = score_corpus_doc("secondary")
    assert primary["score"] > secondary["score"]


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
