"""Source reliability: primary vs. secondary, known-reliable vs. unknown.

A real reliability model would need a maintained database of outlet
track records (something like NewsGuard or Media Bias/Fact Check). This is
the honest cheap version: a domain-suffix heuristic (.gov/.edu and a short
list of well-established reference/news sites score higher) plus whatever
tier the evidence source declares itself (used for the offline demo corpus,
where each document is hand-labelled primary/secondary -- see evidence/).

This is explicitly a heuristic, not a bias/accuracy judgment about any
outlet not on the list -- "unknown" means exactly that, not "unreliable".
"""
from urllib.parse import urlparse

_HIGH_RELIABILITY_DOMAINS = {
    "wikipedia.org", "reuters.com", "apnews.com", "bbc.com", "nature.com",
    "nasa.gov", "who.int", "un.org",
}
_HIGH_RELIABILITY_SUFFIXES = (".gov", ".edu")

_TIER_SCORE = {"primary": 0.9, "secondary": 0.6, "unknown": 0.4}


def score_url(url: str) -> dict:
    domain = urlparse(url).netloc.replace("www.", "")
    if domain in _HIGH_RELIABILITY_DOMAINS or any(domain.endswith(s) for s in _HIGH_RELIABILITY_SUFFIXES):
        return {"tier": "high", "score": 0.9, "reason": f"{domain} is a known reference/institutional source"}
    return {"tier": "unknown", "score": 0.4, "reason": f"{domain} is not on the known-reliable list (not a claim it's unreliable)"}


def score_corpus_doc(declared_tier: str) -> dict:
    """For the offline evidence corpus, where each doc declares its own tier
    in its front-matter (see evidence/*.txt) instead of having a real URL."""
    score = _TIER_SCORE.get(declared_tier, _TIER_SCORE["unknown"])
    return {"tier": declared_tier, "score": score, "reason": f"offline demo corpus, self-declared tier: {declared_tier}"}
