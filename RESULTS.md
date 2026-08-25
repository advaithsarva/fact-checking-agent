# Results

## Verdict accuracy

Command:

```bash
python scripts/measure_agent.py
```

10 hand-written claims against the offline demo corpus (`evidence/*.txt`,
6 documents): 4 true claims that should be Supported, 3 false/mythical
claims that should be Contradicted, 3 off-topic claims with no matching
evidence that should be Unverifiable.

| Claim | Expected | Got | Confidence |
|---|---|---|---|
| The Eiffel Tower is located in Paris, France. | Supported | Supported | 1.0 |
| The Eiffel Tower is located in London, England. | Contradicted | Contradicted | 1.0 |
| The Great Wall of China is visible from space with the naked eye. | Contradicted | Contradicted | 1.0 |
| Napoleon Bonaparte was extremely short compared to other men of his time. | Contradicted | Contradicted | 1.0 |
| Water boils at 100 degrees Celsius at sea level. | Supported | Supported | 1.0 |
| The Moon is Earth's only natural satellite. | Supported | Supported | 1.0 |
| Mount Everest is the highest mountain on Earth above sea level. | Supported | Supported | 0.594 |
| Bananas are the most popular fruit on Mars. | Unverifiable | Unverifiable | 0.0 |
| The stock market rose sharply yesterday. | Unverifiable | Unverifiable | 0.0 |
| A new species of beetle was discovered in Peru last week. | Unverifiable | **Contradicted** | 1.0 |

**Accuracy: 9/10 = 0.90.**

### The one miss, explained

"A new species of beetle was discovered in Peru last week" has no real
match in the 6-document corpus. It should return no evidence and land on
Unverifiable. Instead the BM25 relevance filter (`_MIN_RELEVANCE_SCORE =
2.0` in `src/search.py`) lets it through: the query happens to share enough
common words ("was", "in", "of") with the Napoleon document to score 2.03 —
just over the threshold, purely from stopword overlap, not topical
relevance. The NLI model then confidently (1.0) calls one sentence of the
Napoleon passage a "contradiction" of a claim about beetles, which is
itself a second, smaller failure (an NLI model asked to judge two unrelated
sentences shouldn't return high-confidence contradiction at all — the same
class of issue documented in `stance.py`, just not fully eliminated by the
sentence-splitting fix).

Not fixed by raising the threshold for this specific example — that would
be tuning one number against the one query I happened to write, not fixing
the underlying issue (BM25 alone can't perfectly separate "shares stray
stopwords" from "is topically relevant" in a six-document corpus). A larger
corpus, a TF-IDF-weighted rather than raw-BM25 threshold, or a cheap
topic/entity-overlap check before trusting the NLI stance call would be the
actual fixes — left as a known limitation rather than a special-cased patch.

## Latency

10 claims (search + stance classification per claim, offline corpus) in
21.1s = ~2.1s/claim on CPU, dominated by the NLI model running once per
evidence sentence per claim. `verify_document()` on the 6-claim sample
article (`samples/article.txt`) takes proportionally longer since claim
extraction (spaCy) runs first.

## Test suite

```
tests/test_claims.py       5 passed  (offline, claim filtering/splitting)
tests/test_reliability.py  4 passed  (offline, pure logic)
tests/test_stance.py       2 passed  (real NLI model, incl. the sentence-selection regression fix)
tests/test_agent.py        4 passed  (real models, end-to-end evidence loop)
```
