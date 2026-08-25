# fact-checking-agent

Feed it an article or any text; it decomposes the factual claims, runs a
multi-step evidence loop per claim (search, read, assess source reliability,
judge stance), and produces a per-claim verdict with citations plus an
overall document credibility score.

## Architecture

```
text --> claim extraction (spaCy: filters questions/opinions, splits "A and B")
              |
              v
        for each claim:
          search(claim)                 --> Tavily API if TAVILY_API_KEY set, else the offline demo corpus
          for each result:
            classify(evidence, claim)   --> supports / contradicts / neutral   (NLI cross-encoder)
            score_reliability(source)   --> tier + score
          weighted-vote the evidence    --> Supported / Contradicted / Unverifiable + confidence
              |
              v
        verification report: per-claim verdict, evidence + citations, document credibility score
```

## Credential-free by default

No API key required to run it. Search falls back to a small **offline demo
corpus** (`evidence/*.txt`, six hand-written reference documents on general-
knowledge topics) when `TAVILY_API_KEY` isn't set, and every report says
which mode produced it (`search_mode: "offline"` in the JSON, `"offline
search"` in the HTML). Live web search is a drop-in upgrade, not a
requirement — this is the "it degrades" rule from `workflow.md` Phase 3.5:
no key, no network, rate limited, and the agent still runs end to end.

## Run it

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

python -m src.cli samples/article.txt -o report.html --json report.json
```

First run downloads the ~280MB NLI model; cached locally after. To use live
search instead of the demo corpus: `export TAVILY_API_KEY=...` first.

## Honest limitations

- **Claim decomposition is sentence-level plus one conjunction-split rule**,
  not real clause parsing. It correctly rejects questions and first-person
  opinions but won't untangle a sentence with three nested clauses. See
  `src/claims.py`.
- **Relation between evidence and claim is judged sentence-by-sentence** by
  a small (~280MB) NLI model, not a large reasoning model — accurate on
  direct factual statements, unreliable on claims needing multi-step
  inference the model wasn't trained for. See RESULTS.md for a documented
  miss.
- **The offline corpus is six documents on unrelated general-knowledge
  topics** — it demonstrates the mechanism (search, stance, reliability,
  scoring), not broad real-world fact-checking coverage. That needs live
  search, which needs a `TAVILY_API_KEY`.
- **No full-page fetching.** Search results carry snippets/document bodies
  only; `fetch_page(url)` from the original spec isn't implemented, so
  reliability scoring works on the source domain, not page content beyond
  the snippet.

## Tests

```bash
python tests/test_claims.py       # claim extraction/filtering, offline
python tests/test_reliability.py  # pure logic, instant
python tests/test_stance.py       # NLI stance, incl. a documented regression fix
python tests/test_agent.py        # end-to-end evidence loop, real models
```

## Results

See [RESULTS.md](RESULTS.md): verdict accuracy on a 10-claim gold set
against the offline corpus, including the one claim it gets wrong and why.

## Project origin

One of three portfolio projects specced in `domains/nlp/`, built the same
day as [multilingual-sentiment-pipeline](https://github.com/advaithsarva/multilingual-sentiment-pipeline)
and [graph-rag-knowledge-system](https://github.com/advaithsarva/graph-rag-knowledge-system).
