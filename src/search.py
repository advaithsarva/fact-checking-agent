"""Search providers: a live web search (Tavily API) or an offline demo
corpus, picked automatically by whether TAVILY_API_KEY is set.

This is the "it degrades" rule in practice (workflow.md Phase 3.5): no key,
no network, rate limited -- the agent still runs, end to end, against
evidence/*.txt, and every report says which mode produced it. A missing key
is never a crash and never a silent empty result.
"""
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

EVIDENCE_DIR = Path(__file__).resolve().parents[1] / "evidence"


@dataclass
class SearchResult:
    title: str
    source: str  # a URL for live search, a source name for the offline corpus
    snippet: str
    tier: str | None = None  # only set for offline corpus results


def _parse_evidence_doc(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    header, _, body = text.partition("---\n")
    meta = dict(re.findall(r"^(\w+):\s*(.+)$", header, re.MULTILINE))
    return {"tier": meta.get("tier", "unknown"), "source": meta.get("source", path.stem), "body": body.strip()}


@lru_cache(maxsize=1)
def _corpus_index():
    from rank_bm25 import BM25Okapi
    docs = [_parse_evidence_doc(p) for p in sorted(EVIDENCE_DIR.glob("*.txt"))]
    tokenized = [d["body"].lower().split() for d in docs]
    return docs, BM25Okapi(tokenized)


# BM25 scores from shared stopwords ("is", "the", "located") alone land
# around 0.6-1.4 on this corpus even for topically unrelated documents; a
# genuine keyword match on the claim's subject scores several times higher
# (6.8 for "Eiffel Tower" against the Eiffel Tower doc). Below this floor,
# feeding the "match" to the NLI stance model produces confident-looking but
# spurious contradictions on unrelated text -- see CLAUDE.md. Tuned to this
# six-document demo corpus's scale, not a universal constant.
_MIN_RELEVANCE_SCORE = 2.0


def search_offline(query: str, k: int = 3) -> list[SearchResult]:
    docs, bm25 = _corpus_index()
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(range(len(docs)), key=lambda i: -scores[i])[:k]
    return [
        SearchResult(title=docs[i]["source"], source=docs[i]["source"],
                     snippet=docs[i]["body"], tier=docs[i]["tier"])
        for i in ranked if scores[i] >= _MIN_RELEVANCE_SCORE
    ]


def search_live(query: str, k: int = 3) -> list[SearchResult]:
    import requests
    api_key = os.environ["TAVILY_API_KEY"]  # no fallback default -- see workflow.md Phase 3.5 rule 4
    resp = requests.post(
        "https://api.tavily.com/search",
        json={"api_key": api_key, "query": query, "max_results": k},
        timeout=10,
    )
    resp.raise_for_status()
    return [
        SearchResult(title=r.get("title", ""), source=r["url"], snippet=r.get("content", ""))
        for r in resp.json().get("results", [])
    ]


def search(query: str, k: int = 3) -> tuple[list[SearchResult], str]:
    """Returns (results, mode) where mode is 'live' or 'offline' -- callers
    surface this in the report so a demo run is never mistaken for a live
    fact-check."""
    if os.environ.get("TAVILY_API_KEY"):
        try:
            return search_live(query, k), "live"
        except Exception:
            pass  # falls through to offline rather than failing the whole claim
    return search_offline(query, k), "offline"
