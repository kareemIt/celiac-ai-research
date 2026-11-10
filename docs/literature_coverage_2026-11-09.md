# Literature coverage: the six review topics against the repository's verified references

Kareem Itani · 2026-11-09 · no reference was added to `paper.md` in this pass. Nothing below is
cited from memory: every claim about what a reference says is taken from the repository's own
verified entries in [`paper.md`](paper.md) §11 and
[`state_of_the_art_comparison.md`](state_of_the_art_comparison.md) §Sources, whose verification
status those documents record.

**The single most important finding is a citation defect, not a coverage gap.** `paper.md`
benchmarks ten models and names Chemprop, CheMeleon, ChemBERTa-2, MoLFormer-XL and a tabular
foundation model more than fifty times across §5.10, §5.11, §5.13 and §5.18 — and cites **none of
them**. Its 29 verified references contain no primary citation for any model it evaluates. A
reviewer would raise this on the first pass.

The citations exist and are already verified, in `state_of_the_art_comparison.md` §Sources: Yang et
al. 2019 and Heid et al. 2024 for Chemprop, Burns et al. for CheMeleon, Chithrananda et al. and
Ahmad et al. for ChemBERTa/-2, Ross et al. for MoLFormer-XL, Ben Hicham et al. for the tabular
model, plus van Tilborg et al. (now `paper.md` [29]). That document records eleven of its eighteen
sources as checked against the publisher or arXiv page, and four (its 6, 12, 16, 18) as secondary
confirmation only, explicitly flagged as needing a primary check "before they appear in a
manuscript". So the work needed is **promotion of already-verified entries**, not a new search —
except for Chemprop v2 (its 6), Uni-Mol2 (12), ADMET-AI (16) and CLAMP (18), which need primary
verification first and are not used in any claim.

---

## Coverage by topic

| topic | references | verdict |
|---|---|---|
| Uncertainty estimation | [5] [6] [21] [22] [23] [24] [26] [28] | **Adequate — the strongest area** |
| Data leakage and splitting | [7] [8] [9] [10] [27] | **Adequate** |
| Measurement uncertainty in public data | [11] [12] [25] | **Adequate** |
| TG2 inhibitors and celiac treatment | [1] [16] [17] [20] | **Adequate for motivation, thin for mechanism** |
| Molecular machine learning | — | **Absent from `paper.md`** (see above) |
| Activity cliffs | [29] | **One paper for a §5.18-sized claim** |
| Negative and inactive examples | [11] [12] [18] [19] | **Genuine gap** |
| Selectivity and intestinal delivery | [17] [20] | **Genuine gap** |
| Overfitting diagnostics | — | **Not needed — we make no overfitting claim** |

### Adequately covered, and why

**Uncertainty estimation.** Eight references spanning temperature scaling [5], the conformal
foundations [6], hierarchical conformal prediction [21] [22], exchangeability failure [23],
conformal prediction in cheminformatics [24], class-conditional conformal [26] and the closest prior
art on drug–target data [28]. §5.13 and §5.16 are supported at the depth their claims require, and
§2.3's novelty claim was already narrowed once because [28] was found. No addition needed.

**Data leakage and splitting.** Bemis–Murcko [7], Butina [8], time-split validation [9], the
memorisation critique and AVE bias [10], and the finding that scaffold splits themselves overstate
virtual-screening performance [27]. This is the paper's central subject and the coverage matches it.

**Measurement uncertainty.** [11] and [12] establish the heterogeneous-IC50 error floor that §3
leans on, and [25] the large-scale ChEMBL comparison. Adequate.

**TG2 and celiac disease.** [1] for the prior computational study, [17] for TG2 as a target,
[20] for the ZED1227 trial and [16] for its registration. Enough to motivate the application, which
is all §1.1 needs. It is thin on TG2 mechanism and structure, but no claim in the paper rests on
that, so this is a presentation gap rather than an evidential one.

### The three real gaps

**1. Negative and inactive examples — the professor's own concern, and the least supported topic.**
§3.3 admits 46 censored `>` records as confirmed negatives and excludes 45 `Inconclusive` rows, and
§5.17 now reports sensitivity and specificity separately. None of that cites any methodological
literature. The repository has nothing on: how inactives should be defined from qualified
bioactivity records; whether censored measurements should be modelled as bounds rather than
thresholded; class-imbalance handling in QSAR; or decoy and presumed-inactive construction and its
biases. [18] and [19] are structural-alert filters, which is a different question. **This is the
topic where additional literature would most change what we can say**, because §3.3's censored-data
decision is currently defended only by internal argument and §5.17's class asymmetry has no
methodological context to sit in.

**2. Activity cliffs — one reference supporting a full results section.** §5.18 adopts
MoleculeACE's definition, replicates its central claim on three targets, and qualifies its framing
about rank correlation. Doing that against a single citation is thin. Missing: the origin of the
cliff concept and its matched-molecular-pair formalisation, and any review that would let us say
whether the generic-scaffold test dominating our pair counts (§5.18, §8) is a known property of the
definition or a feature of our data. We cannot settle that from one paper, and §5.18 currently says
so.

**3. Selectivity and intestinal delivery.** §5.7 models TG2 against FXIII-A and
`adme_analysis.md` reports the withdrawn permeability claim, with no citation for FXIII-A biology,
for the selectivity requirement in this indication, or for gut-restricted drug design. [20] carries
the entire therapeutic-requirements argument on its own. Since §7.2's conclusions about what a
celiac candidate would need are among the paper's most consequential statements for a reader, they
should rest on more than one trial report.

### Not a gap: overfitting

The brief lists overfitting as a review topic. No additional literature is needed, because **the
paper makes no overfitting claim** — §8 now states explicitly that every result measures
generalisation failure under distribution shift and that the train-versus-held-out gap was never
measured, so memorisation is not established. [10] already covers the benchmark-memorisation
argument for the one place it is relevant (§5.1's leakage result). Citing an overfitting literature
for findings that are not about overfitting would misdescribe our own work.

## What this implies for the manuscript

Ordered by how much it changes what the paper may claim:

1. **Promote the eleven verified model citations** from `state_of_the_art_comparison.md` into
   `paper.md` §11 and cite them at §5.10, §5.13 and §5.18. Mechanical, no new verification, and it
   closes a defect that currently affects every model-comparison claim in the paper.
2. **Search and verify the inactive-and-censored-data literature.** Gap 1 is the one that touches a
   dataset decision (§3.3) and a results section (§5.17) rather than only a sentence.
3. **Add two or three activity-cliff references** so §5.18 is not a one-citation section.
4. **Add selectivity and intestinal-delivery references** for §5.7 and §7.2.
5. **Primary-verify or drop** Chemprop v2, Uni-Mol2, ADMET-AI and CLAMP, which
   `state_of_the_art_comparison.md` flags as secondary-confirmation only. None backs a claim, so
   dropping them is acceptable; what is not acceptable is promoting them unverified.

**On the "20 papers".** The brief refers to twenty papers from an earlier review. That list is not
in the repository and was not attached, and the 29 references here are not it. The topic-level
analysis above is what can be done without it; reconciling the actual list against these 29 needs
the list itself. Until then no reference has been added on the strength of a remembered title.
