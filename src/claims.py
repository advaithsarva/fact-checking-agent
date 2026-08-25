"""Decomposes text into atomic, checkable factual claims.

"Atomic" here means: one sentence, further split on coordinating
conjunctions ("and", "but") when both halves have their own subject and
verb, so "Water boils at 100C and the Earth has one moon" becomes two
claims instead of one compound sentence a search can't target cleanly.

Claim decomposition is a hard problem in general (it needs to resolve
"it"/"they" back to what they refer to, split nested clauses, etc). This is
the honest, cheap version: it filters out sentences that are structurally
not factual claims (questions, commands, first-person opinions) and splits
on the one conjunction pattern that's common and easy to get right. It will
not decompose "The company, which was founded in 2010 and later acquired by
Globex, grew quickly" into three claims -- that needs real clause-level
parsing this project doesn't attempt.

It also inherits spaCy's sentence-boundary detection as-is, including its
edge cases: "Water boils at 100C. I believe that's fascinating." gets
merged into one sentence by en_core_web_sm (it doesn't treat "100C." as a
sentence end), which would let an opinion clause ride along inside an
otherwise-checkable sentence undetected. Writing "100 degrees Celsius"
instead of "100C" avoids it; this project doesn't patch around spaCy's
sentencizer for input like that.
"""
from dataclasses import dataclass
from functools import lru_cache

_OPINION_MARKERS = (
    "i think", "i believe", "in my opinion", "i feel", "we think",
    "arguably", "i'd argue", "personally",
)


@dataclass
class Claim:
    text: str
    source_sentence: str


@lru_cache(maxsize=1)
def _nlp():
    import spacy
    return spacy.load("en_core_web_sm")


def _is_checkable(sent) -> bool:
    text = sent.text.strip()
    if not text or text.endswith("?"):
        return False
    lowered = text.lower()
    if any(lowered.startswith(m) for m in _OPINION_MARKERS):
        return False
    # needs at least one verb and one subject-ish dependency to be a claim,
    # not a fragment or a heading
    has_verb = any(t.pos_ in ("VERB", "AUX") for t in sent)
    has_subject = any(t.dep_ in ("nsubj", "nsubjpass") for t in sent)
    return has_verb and has_subject


def _split_compound(sent) -> list[str]:
    """Split 'A and B' into ['A', 'B'] only when each half has its own
    subject+verb -- otherwise leave it whole rather than risk cutting a
    single claim in half mid-clause."""
    conjuncts = [t for t in sent if t.dep_ == "cc" and t.text.lower() in ("and", "but")]
    if not conjuncts:
        return [sent.text.strip()]

    split_point = conjuncts[0].i
    left = sent.doc[sent.start:split_point]
    right = sent.doc[split_point + 1:sent.end]
    left_ok = any(t.dep_ in ("nsubj", "nsubjpass") for t in left) and any(t.pos_ in ("VERB", "AUX") for t in left)
    right_ok = any(t.dep_ in ("nsubj", "nsubjpass") for t in right) and any(t.pos_ in ("VERB", "AUX") for t in right)
    if left_ok and right_ok:
        return [left.text.strip().rstrip(","), right.text.strip()]
    return [sent.text.strip()]


def extract_claims(text: str) -> list[Claim]:
    if not text.strip():
        return []
    doc = _nlp()(text)
    claims = []
    for sent in doc.sents:
        if not _is_checkable(sent):
            continue
        for piece in _split_compound(sent):
            if piece:
                claims.append(Claim(text=piece, source_sentence=sent.text.strip()))
    return claims
