# Start here

A map of this project, written for someone returning to it after a gap — or reading it for the
first time. `README.md` covers the original environment setup; this covers what the work became.

## What the project actually found

It began as "can a local 9B language model predict TG2 inhibitors better than cheminformatics?"
The answer is no (a tuned random forest wins by 0.049 AUROC, p = 0.007). The durable result is
methodological: **on small, literature-derived bioactivity datasets, the evaluation protocol
decided the conclusion more often than the model did.** Four distinct failures were documented on
TG2 and three were reproduced on two unrelated targets.

Two of our own positive findings were withdrawn after auditing. That is the point of the project,
not an embarrassment to it.

## Read in this order

| | document | what it is |
|---|---|---|
| 1 | [`docs/summary.md`](docs/summary.md) | the index and the short version — **start here** |
| 2 | [`docs/paper.md`](docs/paper.md) | the full write-up, ~9,000 words, eight sections |
| 3 | [`docs/data.md`](docs/data.md) | provenance and validation. **Read before trusting any number** |

Then, by theme:

**Evaluation methodology (the main contribution)**
- [`docs/second_target.md`](docs/second_target.md) — do the failures reproduce on CDK2 and MAO-B?
- [`docs/reproducibility.md`](docs/reproducibility.md) — why two identical runs disagreed
- [`docs/forgetting.md`](docs/forgetting.md) — KL anchors, and a withdrawn hypothesis

**The chemistry**
- [`docs/selectivity.md`](docs/selectivity.md) — TG2 vs FXIII-A, first attempt
- [`docs/covalent.md`](docs/covalent.md) — kinact/Ki and covalent mechanism
- [`docs/screening_prep.md`](docs/screening_prep.md) — unit recovery, +45 new molecules
- [`docs/zed1227_analysis.md`](docs/zed1227_analysis.md) — the clinical compound as a reference

**Where it was heading**
- [`docs/experimental_plan.md`](docs/experimental_plan.md) — the preregistered 16-compound study
- [`docs/collaboration/`](docs/collaboration/) — summary, proposal, and **unsent** outreach drafts

## The state of play, in one paragraph

A random forest beats the fine-tuned LLM. Below ~0.4 Tanimoto every model ranks at chance on TG2
(but not on CDK2 or MAO-B — that floor is dataset-specific). Selectivity against FXIII-A is
predictable inside *one* 69-molecule series and nowhere else. The clinical compound ZED1227 is
already in our training data, and our own compounds beat it on both potency and selectivity — what
it has that they lack is gut-restricted delivery, which appears in no dataset we hold. **No
candidate shortlist has been produced and none is justified.**

## Running it

```bash
cd ~/projects/celiac-ai-research
bash scripts/setup_env.sh && source .venv/bin/activate
pytest                                  # 161 tests
python scripts/check_paper.py           # 61 paper values recomputed from results/
```

`scripts/check_paper.py` is worth knowing about: it recomputes every headline number in the paper
straight from `results/` and fails if the paper has drifted. It has already caught one stale table
and two rounding errors.

## What is NOT in this copy

- **`.venv/`** (6.5 GB) — rebuild with `scripts/setup_env.sh` from `requirements.txt`
- **`checkpoints/`** (8.8 GB) — 30+ LoRA adapters. Regenerable, and since the determinism fix
  (commit `4b4de52`) they rebuild **bit-exactly**, so they are reproducible rather than precious.
  Roughly 6 GPU-hours for the full set.

Everything else is here, including the complete `.git` history (21 commits) and the raw ChEMBL
downloads in `data/raw/` (214 MB), so nothing needs re-fetching.

## Standing constraints, carried throughout

Local compute only — no hosted model APIs, no cloud GPUs. Nothing pushed to any remote. No
compounds ordered, no laboratory contacted. Computational prediction, biochemical inhibition and a
celiac treatment are three different things, and the documents keep them apart.
