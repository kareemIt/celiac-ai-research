# Draft supervisor update — **NOT SENT** (2026-10-30)

Nothing here has been sent. Fill in the salutation, and decide how much of the appendix to keep,
before sending. Every figure in the body is reproducible from `results/` by the scripts named;
`scripts/check_paper.py` recomputes the paper's 61 headline values and fails if the write-up has
drifted. Written for a reader who knows machine learning but not medicinal chemistry, so the
domain terms are glossed inline.

---

**Subject:** Project overview and three questions — where the TG2 work has landed

Dear Professor ——,

This is a full overview rather than an increment, since the project has changed shape since we last
spoke. I have kept it to one read and put the three questions I need your judgement on at the end.

## 1. The original question, and the answer

Can a locally fine-tuned ~9B language model (Qwen, used as a Jev-style decision model) predict
which small molecules inhibit **TG2** — the enzyme that chemically modifies gluten peptides into
the forms that trigger celiac disease — better than standard cheminformatics?

**No.** On 631 molecules with measured TG2 potency, evaluated over 10 identical scaffold splits:

| model | AUROC |
|---|---|
| random forest on molecular fingerprints, tuned per split | **0.898 ± 0.057** |
| Qwen, QLoRA fine-tuned | 0.850 ± 0.058 |
| Qwen, zero-shot | 0.651 ± 0.144 |

The gap between the fine-tune and the random forest is −0.049 AUROC, 95% CI [−0.076, −0.021],
p = 0.007, with the LLM ahead on only 2 of 10 splits. Fine-tuning works emphatically — it adds
0.199 AUROC over the zero-shot model on every split — it simply does not reach a well-tuned
classical baseline at this data scale. A molecular fingerprint is a fixed binary feature vector
over substructures, so the baseline here is ordinary supervised learning on 2,048 features and a
few hundred rows.

## 2. What the project actually contributes

The model comparison is the least interesting result. The durable one is methodological:

> **On small, literature-derived datasets, the choice of evaluation protocol decided the published
> conclusion more often than the choice of model did.**

Three ways of splitting the data, in increasing order of honesty:

- **Random split** — invalid here, and measurably so. **38.6%** of random-split test molecules have
  a training molecule with a byte-identical fingerprint. The model is being asked to recall, not to
  generalise.
- **Scaffold split** — group by core structure, the community's usual answer. Better, and still
  optimistic.
- **Publication-held-out split** — hold out entire source papers. Each paper contributes its own
  chemistry measured in its own laboratory on its own assay, so this is the only split that
  separates *generalising to new chemistry* from *memorising a laboratory*. It is the split almost
  nobody reports.

The ordering of the penalties is the project's best-supported claim, because it replicates on two
unrelated targets chosen in advance — **CDK2** (a kinase, 2,646 molecules, 351 papers) and
**MAO-B** (an oxidase, 5,840 molecules, 414 papers):

| | TG2 | CDK2 | MAO-B |
|---|---|---|---|
| identical fingerprints under a random split | 38.6% | 7.3% | 8.9% |
| random → scaffold penalty | ≈ −0.08 | −0.030 | −0.059 |
| **scaffold → publication penalty** | large | **−0.068** | **−0.076** |

On both independent targets the publication penalty **exceeds** the scaffold penalty — 2.3× on
CDK2, 1.3× on MAO-B — across different enzyme classes, dataset sizes and class balances. On MAO-B
the full sequence is AUROC 0.923 → 0.865 → 0.789, with no confidence interval overlapping the next.

Three further findings sit alongside it: post-hoc calibration fails under scaffold shift for every
fingerprint model (the fitted temperature swings from 0.4 to 3.7 across folds) but works for the
fine-tuned LLM, which is overconfident in a consistent direction; conformal prediction keeps its
coverage under shift by widening its prediction sets instead; and abstaining on low confidence cuts
error from 15.5% to 6.5%, while abstaining on low chemical similarity does not help at all. One
failure — the chance-level floor below 0.4 fingerprint similarity — turned out to be specific to
the TG2 dataset and did *not* reproduce on CDK2 or MAO-B, which I report as a negative.

## 3. What we withdrew, and why I think that is the result

Four claims were retracted after auditing, two of them our own positive findings:

1. An apparent **LLM advantage on novel chemistry** — an artifact of pooling predictions across
   overlapping folds; it vanished when scored one row per molecule.
2. **KL anchors preventing catastrophic forgetting** — +0.035 AUROC, CI [−0.107, +0.225]. The
   decline that motivated the hypothesis disappeared once training was made bit-exact (the
   non-determinism was in the backward pass, not the optimizer; three identical runs now agree
   exactly, including saved weights).
3. A **false claim of mine** that TG2 inhibition had no clinical proof of concept. It does.
4. A **false claim of mine** that our data contained no drug-delivery information. It does — §5.

I would argue the self-correction record is the thesis's strongest evidence of method, not a
weakness in it, but that is your call rather than mine.

## 4. The chemistry thread: selectivity, and the clinical compound

Any celiac drug would be taken for life by otherwise-healthy people, so it must **not** inhibit
**Factor XIII-A**, a closely related enzyme that clots blood. That makes selectivity a safety gate
rather than an optimisation target. I assembled 147 molecules measured against both enzymes *within
a single publication*, so every ratio is a within-study comparison — necessary, because I measure a
0.50-log systematic offset between laboratories, larger than most of the effects being studied.

The result is narrow and holds up: selectivity is predictable **within one 69-molecule series**
(Spearman ρ = 0.42 under a cluster split) and **not across series** — a second series looked just
as good under a scaffold split and collapsed to ρ ≈ 0 under a cluster split. Generalisation between
the two series fails entirely.

**ZED1227** is the only TG2 inhibitor with human efficacy data (a 163-patient randomised trial), so
it was the obvious external reference point. Two things emerged:

- It **cannot validate our models**, for two reasons simultaneously: it is already inside our
  training data, so any prediction would be memorisation; and it sits outside the selectivity
  model's domain of applicability. The analysis script refuses to output a number and records why.
- **Potency and selectivity are not what makes it a drug.** It is 53 nM against TG2 and >900-fold
  selective over Factor XIII-A. Our own measured data already contains better — 14 nM and
  2,500-fold, plus a second compound at 1,081-fold. If our compounds already beat the clinical
  benchmark on both axes we can model, then neither axis is the bottleneck, and more modelling of
  either cannot help.

## 5. What is the bottleneck: getting the drug to the right place

ZED1227 is designed to stay in the gut and not enter the bloodstream, which is why it is safe to
take daily. I had reported that our datasets contained nothing about this property. That was wrong:
our files were downloaded *by target*, so they held only enzyme measurements, while the public
database files drug-delivery data under the *molecule*. Re-querying returned **380 measured values
on 113 molecules from 27 publications** — permeability, metabolic stability, clearance, protein
binding.

What it shows is uncomfortable. Our two most selective compounds have measured permeability **more
than 14× and 30× above** ZED1227's upper bound — i.e. they would be absorbed into the bloodstream,
the opposite of what is wanted. Both come from a **Huntington's disease** programme, where getting
into the brain was the design goal. Of all 147 selectivity-paired molecules, exactly one has
permeability low enough to resemble ZED1227, and that one is 1,300 nM rather than tens of
nanomolar and entirely non-selective.

**I did not train a model on this data, and I want to flag that as a decision.** It is 37 molecules
across 9 publications in four non-interchangeable assay systems; the largest single-protocol block
is 19 molecules. By the project's own finding, nothing below a cluster split is interpretable, and a
cluster split cannot be constructed inside 19 points. A model fitted across those 9 papers would be
predicting which laboratory a measurement came from. Solubility — the most basic delivery property
— has **zero** measurements for any molecule we hold.

## 6. The proposed experiment

`docs/experimental_plan.md` preregisters a prospective study: **16 previously untested compounds**
from the one series where the selectivity model survives a cluster split, selected to **span** the
predicted range rather than its top (selecting the top 16 would confound a good model with an easy
series), with predictions committed in writing before any measurement, and success defined in
advance as Spearman ≥ 0.5 and failure below 0.3. The new recommendation is one addition: measure
solubility and permeability on the same 16 compounds, in one laboratory under one protocol.

No laboratory has been contacted and no compound ordered; the outreach drafts in
`docs/collaboration/` are unsent and will not go anywhere without your review.

## The three questions I need your view on

1. **Is the completed computational work sufficient for a paper or the thesis as it stands?** My
   own view is that the publication-held-out finding, replicated on three targets, is the
   publishable contribution and the LLM-vs-random-forest comparison is the hook rather than the
   result. The risk is that a reviewer reads it as a negative-results paper.
2. **Is the proposed 16-compound validation scientifically worth running?** It is the only step I
   can see that tests a prediction against reality rather than re-measuring our own accuracy — but
   it needs an external laboratory and it produces no new model.
3. **Should we reconsider the chemical series, given §5?** The series where our model works comes
   from a Huntington's programme, and the delivery data suggests its compounds may be systemically
   absorbed as a class — which would make the series **wrong for celiac disease however well we
   rank selectivity within it**. The alternative series is chemically appropriate but has only 8
   paired measurements, far too few to test anything. So the choice is between testing the model
   where it can be tested and working on chemistry that could matter. I lean towards running the
   16-compound study as designed, precisely because solubility and permeability on those 16 would
   settle the question empirically for about the same effort — but I would rather raise the problem
   now than discover it after the measurements.

Everything is in the repository and reproducible: 23 commits, 171 tests passing, all 61 of the
paper's headline numbers recomputed from source on demand. No compounds ordered, no laboratory
contacted, nothing pushed to any remote, and no model training run since the paper.

If you read one document, `docs/summary.md` is the index; `docs/adme_analysis.md` is the new work.

With thanks,
Kareem

---

## Appendix — what to point her at

| document | what it is |
|---|---|
| `docs/summary.md` | the index and the short version |
| `docs/paper.md` | the full write-up, ~9,000 words |
| `docs/second_target.md` | the CDK2 and MAO-B replication |
| `docs/adme_analysis.md` | **new** — the delivery-property audit and the no-model decision |
| `docs/zed1227_analysis.md` | the clinical compound as a reference point, withdrawn claim marked |
| `docs/experimental_plan.md` | the preregistered 16-compound study |
| `docs/collaboration/` | one-page summary, proposal, and the **unsent** outreach drafts |

Reproduce the new work with `python scripts/fetch_adme.py` then `python scripts/analyse_adme.py`
(CPU, seconds). Tables land in `results/adme_measurements.csv` and `results/adme_summary.csv`.

## Caveats to have ready if she asks

- Every biochemical comparison is **cross-assay**: ZED1227's permeability bound comes from a
  different cell line than our compounds' values. Indicative, not equivalent.
- ZED1227's delivery profile is **hand-transcribed** from one paper's table into
  `data/curated/literature_adme.csv`, which records the source per value but has not been checked
  by a second reader.
- The within-series selectivity result (ρ = 0.42) is **retrospective, on one publication's
  chemistry**. That is the entire reason for proposing a prospective test.
