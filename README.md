# TalentLens

A config-driven, embeddings-based candidate ranker for the **Intelligent Candidate
Discovery & Ranking Challenge** (India Runs on Data & AI).

The dataset is built around a trap: keyword-matching fails on purpose. A Marketing
Manager who pasted "Pinecone, RAG, FAISS" into their skills is *not* a fit; a strong
engineer who never wrote "RAG" might be. So TalentLens does not score keywords. It
scores **meaning, then discounts by credibility and reachability**.

## The model

For every candidate we build one document (headline + summary + each role's title,
company and description + skills with proficiency), embed it with `all-MiniLM-L6-v2`,
and compare it to the embedded **intent** of the JD. That semantic match is shaped by
three readable factors and two multipliers:

```
fit = (0.55·semantic + 0.30·skill_coverage + 0.15·seniority)
      × credibility      # 0.15–1.0  : kill keyword-stuffers & impossible claims
      × availability     # 0.55–1.0  : down-weight the dormant / unreachable
```

- **semantic** — cosine(JD, candidate). Catches fits who don't use the buzzwords.
- **skill_coverage** — how many of the 4 must-have families (retrieval/embeddings,
  vector·hybrid search, ranking/recsys, evaluation) the profile actually evidences.
- **seniority** — triangular fit peaking at 6–8y, in-band 5–9y.
- **credibility** *(multiplier, not a filter)* — off-domain title with no engineering
  history → ×0.18; "expert" in a skill used 0 months → ×0.6; services-only career,
  fabricated tenure, CV/speech focus, LangChain-only → smaller penalties.
- **availability** — last-active recency, recruiter-response rate, open-to-work.

Because credibility and availability are **multiplicative**, a perfect-on-paper
keyword-stuffer can't climb: a high semantic score × 0.18 still lands at the bottom.

## The role is config, not code

Nothing is hardcoded to one job. A role is a folder:

```
jobs/senior_ai_engineer/
  description.md   # the full JD in Markdown — embedded verbatim as "intent"
  spec.json        # skill families, weights, seniority band, credibility rules
```

Drop in a new folder and point `--job` at it to rank for a different role — the engine
never changes.

## Run it

```bash
pip install -r requirements.txt

python rank.py --sample                                       # instant 50-candidate demo
python rank.py --job jobs/senior_ai_engineer \
               --candidates candidates.jsonl --out submission.csv   # full pool
```

The first run downloads MiniLM (~90 MB) and encodes the pool; embeddings are cached to
`.cache/`, so later runs are instant. Encoding the full 100K pool takes ~2–3 min on a
GPU (auto-detected) or ~10–15 min on CPU.

Candidate embeddings are **job-independent**, so you can precompute them once and reuse them
for any role — no GPU required at rank time:

```bash
python rank.py --candidates candidates.jsonl --out submission.csv \
               --save-emb embeddings.npz          # encode once, save (id-aligned)
python rank.py --candidates candidates.jsonl --out submission.csv \
               --emb embeddings.npz                # later runs skip encoding (~40 s on CPU)
```

Output is the required `candidate_id, rank, score, reasoning` CSV, where `reasoning` is one
clear sentence:

```
Strong fit. AI Engineer @ upGrad (7.6y). Covers 4/4 must-haves: retrieval & embeddings
(embedding); vector / hybrid search (Qdrant, Haystack); ranking & recsys (Recommendation
Systems); evaluation (a/b). Plus LLM fine-tuning. Active 2026-04, 62% recruiter response.
```

### Docker

```bash
docker build -t talentlens .
docker run --rm talentlens                                    # ranks the bundled sample
docker run --rm -v $PWD/data:/data talentlens \
  python rank.py --candidates /data/candidates.jsonl --out /data/submission.csv
```

### Colab

`demo_colab.ipynb` clones this repo, pulls the candidate file **and precomputed embeddings**
from Drive links, and runs `rank.py --emb`. The full 100K pool ranks in ~2–3 min on a plain
CPU runtime — no GPU needed. (Toggle `USE_PRECOMPUTED` off to encode from scratch.)

## Files

| path | what |
|---|---|
| `rank.py` | the engine: load → embed (cached) → score → CSV |
| `jobs/<role>/description.md`, `spec.json` | the role, as config |
| `sample_candidates.json` | bundled 50-candidate sample |
| `demo_colab.ipynb` | one-click Colab run on the full dataset |
| `Dockerfile` | reproducible CPU image |
| `docs/` | slide deck (PDF/PPTX) + submission PDF |

## Why these choices

- **Embed intent, not a checklist** — the JD is prose, so the embedding captures
  "shipped retrieval at a product company, not a researcher / title-chaser".
- **Multiplicative penalties over hard filters** — nothing is silently dropped; every
  candidate gets a score and a reason you can read.
- **Config-driven** — the same engine ranks any role; you tune `spec.json`, not Python.
