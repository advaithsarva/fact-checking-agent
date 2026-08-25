"""Does a piece of evidence support or contradict a claim?

Runs a small NLI (natural language inference) cross-encoder:
premise = one evidence sentence, hypothesis = the claim. Entailment ->
supports, contradiction -> contradicts, neutral -> that sentence doesn't
bear on the claim.

Evidence snippets are multi-sentence passages, but this model (like most
NLI models, trained on single-sentence SNLI/MNLI pairs) is unreliable when
given a whole multi-sentence passage as the premise directly -- an early
version of this function did that and returned "neutral" at 93% confidence
for a passage containing the claim's exact wording, because the unrelated
surrounding sentences diluted the signal. Splitting into sentences and
classifying each one separately, then taking the single highest-confidence
result across all of them, fixes it: the one sentence that's actually about
the claim produces the strongest signal, in either direction.

No API key: this is a local model (~280MB, downloads once via
sentence-transformers), same "credential-free" choice made throughout this
project's siblings.
"""
import re
from functools import lru_cache

_MODEL_NAME = "cross-encoder/nli-deberta-v3-xsmall"
_LABELS = {0: "contradicts", 1: "supports", 2: "neutral"}
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(_MODEL_NAME)


def _classify_pair(premise: str, hypothesis: str) -> dict:
    scores = _model().predict([(premise, hypothesis)])[0]
    probs = _softmax(scores)
    label_idx = int(probs.argmax())
    return {"stance": _LABELS[label_idx], "confidence": float(probs[label_idx])}


def classify(evidence_text: str, claim_text: str) -> dict:
    if not evidence_text.strip():
        return {"stance": "neutral", "confidence": 0.0}
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(evidence_text) if s.strip()]
    results = [_classify_pair(s, claim_text) for s in sentences] or [{"stance": "neutral", "confidence": 0.0}]

    # A confident "supports"/"contradicts" on the one sentence that's actually
    # about the claim always outranks a confident "neutral" on the sentences
    # that aren't -- otherwise the (usually more numerous) unrelated sentences
    # win on raw confidence and mask the sentence that matters. See the
    # Everest/Napoleon regression this fixed, noted above.
    on_topic = [r for r in results if r["stance"] != "neutral"]
    best = max(on_topic or results, key=lambda r: r["confidence"])
    return {"stance": best["stance"], "confidence": round(best["confidence"], 4)}


def _softmax(x):
    import numpy as np
    e = np.exp(x - x.max())
    return e / e.sum()
