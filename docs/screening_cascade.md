# Could the five-gate screening framework be built? (audited 2026-11-04)

**The framework is already four-fifths built, and the blocking problem is not a missing model — it
is that no molecule carries all five gate inputs.** Measured selectivity exists only for molecules
*inside* the training data; applicability domain and conformal uncertainty are quantities about
molecules *outside* it. The intersection is **0 molecules**, and that is structural rather than a
matter of sample size.

Reproduce: `python scripts/audit_screening_cascade.py`. CPU, seconds, fits nothing and produces no
shortlist. Table: `results/screening_cascade_audit.csv`.

## 1. What already exists

[`screen_candidates.py`](../scripts/screen_candidates.py) implements four of the five proposed gates
on the 2,414-molecule approved-drug library, and [`candidates.md`](candidates.md) reports the result:

| proposed gate | implemented? | where |
|---|---|---|
| predicted TG2 inhibitory activity | **yes** | ensemble `p_potent`, 3 models |
| experimental or predicted selectivity | **no** | — |
| molecular similarity / applicability domain | **yes** | `max_tanimoto_train` against a 0.40 floor |
| prediction uncertainty | **yes** | conformal `{potent}`-only sets, plus `models_agree` and `spread` |
| ADME and drug-likeness | **yes** | QED, MW, logP, TPSA, PAINS/Brenk `alerts`, oral route |

Its output was **0 Tier-1 compounds and 1 Tier-2 compound**, because 0.8% of approved drugs (19 of
2,414) reach the applicability floor at all. The gate that does the work is the applicability domain,
and it does it before any other gate is consulted.

## 2. The five gates cannot be evaluated on the same molecule

| gate input | molecules where it exists |
|---|---|
| predicted / measured TG2 activity | 631 (the training set) |
| measured FXIII-A selectivity | 147 (within-publication pairs; **79** format-matched) |
| conformal uncertainty under publication shift | 444 (the LOPO arm) |
| applicability domain | 2,414 (the library) |
| ADME / drug-likeness | 2,414 (the library) |

| intersection | molecules |
|---|---|
| selectivity **and** conformal uncertainty | 142 |
| selectivity **and** a library molecule | **0** |
| **all five gate inputs** | **0** |

The reason is definitional, not accidental. A molecule with measured FXIII-A selectivity is a
molecule somebody assayed on both enzymes and published, which is why it is in our training set —
and a molecule in the training set has max Tanimoto 1.0 to the training set and no honest
out-of-sample uncertainty estimate. Conversely, every molecule for which the applicability-domain
and uncertainty gates are meaningful is one nobody has assayed against FXIII-A.

**So the five-gate cascade cannot be assembled from existing public data at any sample size.** It
needs either measured selectivity on molecules outside the TG2 training set, or a *predictive*
selectivity model — and [`selectivity_feasibility.md`](selectivity_feasibility.md) establishes that
the second is not available: two informative publications, pointing in opposite directions, probably
in different assay formats, with 13 selective molecules in the format-matched subset.

## 3. Do the activity and selectivity gates agree? Yes — and this corrects us

[`selectivity_feasibility.md`](selectivity_feasibility.md) measured r(TG2, FXIII-A) = +0.41 and we
initially read that as "potency-first ranking enriches for cross-reactivity". **That reading was
wrong.** Tested directly on the 147 paired molecules, against a 10-fold selectivity gate:

| TG2 potency stratum | n | passes the 10× gate | vs base rate |
|---|---|---|---|
| all paired | 147 | 13.6% | — |
| top half by pIC50 | 74 | 18.9% | +5.3% |
| top quartile | 37 | 27.0% | +13.4% |
| **top decile** | 15 | **46.7%** | **+33.1%** |

Top decile against the rest: **odds ratio 8.0, Fisher exact p = 0.001.**

The correlation between the two potencies is well below 1, so potency gains on TG2 outrun potency
gains on FXIII-A and net selectivity improves. **The activity and selectivity gates agree rather
than conflict**, which is the opposite of what we expected and makes the cascade *more* coherent,
not less.

Two qualifications keep this from being a general rule:

- **It is concentrated in one series.** Within the TG2-directed publication the top quartile's pass
  rate rises 11.6% → 33.3% (+21.7%); within the FXIII-A-directed publication it rises 12.3% → 13.3%
  (+1.1%). One of two publications is a single observation.
- **The ratio improves while the absolute off-target potency rises.** A molecule selected for being
  very potent on TG2 is also, in absolute terms, a more potent FXIII-A inhibitor than the molecules
  the gate rejected. For a drug taken chronically, absolute off-target potency is the quantity that
  matters, and a ratio gate does not see it. **A cascade should gate on absolute FXIII-A IC50 as
  well as on the ratio.** That is the one concrete design change this analysis supports.

It also sits alongside a limitation rather than removing it: measured potency enriches for measured
selectivity, while *predicted* potency carries no selectivity information (AUROC 0.454 in
[`selectivity.md`](selectivity.md)). The enrichment is real in the chemistry and not recoverable
from the model, which is a statement about our model's resolution.

## 4. What the existing results actually support about prioritisation

Stated as what is and is not licensed by measurement:

**Supported.**
- Ranking analogues of known TG2 chemotypes, within 0.4–0.7 Tanimoto of the training set, where the
  measured AUROC of ≈0.86 applies. This is the only regime with demonstrated skill.
- Reporting a prediction as a conformal set rather than a probability, **calibrated on held-out
  publications** ([`lopo_uncertainty.md`](lopo_uncertainty.md)), and treating a `{potent, not
  potent}` answer as a refusal rather than a weak yes.
- Using measured, within-publication, format-matched selectivity as evidence about those specific
  molecules.
- Capping absolute off-target potency rather than only the selectivity ratio (section 3).

**Not supported.**
- Any prospective shortlist from the approved-drug library: 99.2% of it is below the applicability
  floor where the models rank at chance.
- Any selectivity column on a predicted candidate list. No selectivity model transfers across
  series, and predicted potency carries no selectivity signal.
- Treating a decisive conformal answer on an unseen publication as reliable: those carry 25–39%
  error at α = 0.1.
- Any claim about clinical safety, from predicted or measured selectivity. FXIII-A inhibition in a
  cuvette is a flag, not a prediction about bleeding in a person.

## Verdict: no shortlist, and why

**No candidate shortlist is produced.** The condition the brief set — "unless the evidence is
sufficiently strong" — is not met, and the audit says precisely where it fails:

1. The selectivity gate cannot be applied to any prospective molecule, by measurement (0 library
   molecules have FXIII-A data) or by model (no selectivity model transfers).
2. The applicability gate removes 99.2% of the only prospective library scored.
3. The uncertainty gate is honest but not decisive on unfamiliar chemistry: 15–32% of molecules get
   a committed answer and a quarter to a third of those are wrong.

A cascade whose gates cannot be jointly evaluated would produce a ranking that *looks* like it
combined five lines of evidence while actually resting on one. That is a worse failure than
producing nothing.

**The strongest defensible next step is experimental, not computational:** measure FXIII-A IC50, in
one assay format, on molecules that are *not* already in the TG2 training set. Roughly 30–50 would
convert the selectivity gate from unbuildable to testable, give leave-one-publication-out a third
and fourth informative group, and let the 46.7%-against-13.6% enrichment in section 3 be checked on
chemistry it was not derived from. No amount of further modelling on the existing 147 pairs
substitutes for that.

## Limitations

- Section 3's strata are measured potency, not predicted. A real cascade ranks on predictions, and
  predicted potency is the weaker signal (AUROC 0.454 against a selectivity label).
- 15 molecules in the top decile. The Fisher test is exact, so the p-value is valid, but the 46.7%
  point estimate is held up by 7 molecules.
- The 147 pairs inherit the format-compatibility problems of
  [`selectivity_feasibility.md`](selectivity_feasibility.md) §2; the format-matched subset is 79 and
  section 3 was not recomputed on it alone, because the top decile would hold 8 molecules.
- `candidates.md`'s library screen is from 2026-10-27 and was not re-run; the gate counts are read
  from its stored output.
- The 10-fold gate and the 0.40 applicability floor are both inherited thresholds. No sweep was run
  over either.
