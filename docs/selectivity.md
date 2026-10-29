# Can we find a *selective* TG2 inhibitor? (audited 2026-10-29)

Reproduce: `python scripts/fetch_chembl_tg2.py --target all`, then
`python scripts/audit_tg_family.py` and `python scripts/run_selectivity.py`.
Tables: `results/tg_family_audit.csv`, `results/selectivity.csv`. CPU, about two minutes.

A TG2 inhibitor for celiac disease would be taken chronically by otherwise healthy people. The
transglutaminase family shares a catalytic cysteine, and most published TG2 inhibitors are covalent,
so cross-reactivity is the expected case rather than the exception. Factor XIII-A crosslinks fibrin;
inhibiting it causes bleeding. **Selectivity against FXIII-A is a safety gate, not a refinement**,
which is why it was the first thing checked in this phase.

## Phase 1 — what ChEMBL actually has

Four human enzymes plus one ortholog, each fetched to its own file and never merged. A measurement
against TGM1, TGM3, FXIII-A or guinea-pig TG2 is evidence about *that* enzyme and is never used as a
human TG2 label.

| | ChEMBL | records | molecules | papers | IC50 in nM | exact | censored |
|---|---|---|---|---|---|---|---|
| **TG2** (target, UniProt P21980) | CHEMBL2730 | 1,583 | 781 | 33 | 556 | 425 | 45 |
| TGM1 (P22735) | CHEMBL2810 | 167 | 164 | 11 | 92 | 83 | 9 |
| TGM3 (Q08188) | CHEMBL3363 | 103 | 95 | 10 | 63 | 8 | 55 |
| **FXIII-A** (antitarget, P00488) | CHEMBL4530 | 442 | 251 | 15 | 210 | 177 | 23 |
| TG2, guinea pig (ortholog) | CHEMBL3988613 | 55 | 51 | — | 55 | 22 | 21 |

All records are binding assays in the stated species; no non-human rows contaminate the human
targets. Target IDs were resolved from UniProt accessions through the ChEMBL target endpoint rather
than guessed, and all five are `SINGLE PROTEIN`.

Labels at the project's existing 1 µM cut:

| | molecules | potent | balance |
|---|---|---|---|
| TG2 (ChEMBL only) | 460 | 245 | 53% positive |
| **project TG2 set** (PubChem ∪ ChEMBL) | **631** | **302** | 48% positive |
| TGM1 | 92 | **5** | 5% positive |
| TGM3 | 63 | **1** | 2% positive |
| FXIII-A | 196 | 127 | 65% positive |
| TG2 guinea pig | 41 | **0** | 0% positive |

**TGM1 and TGM3 are not modellable.** Five and one potent molecules cannot train or test anything.
They are retained as descriptive evidence only.

### The expansion hypothesis fails

The previous phase recommended related transglutaminases as a source of chemically distant actives.
**The data does not support that, and the recommendation is withdrawn.**

| source | molecules not already in the TG2 set | median max-Tanimoto to it | below 0.5 |
|---|---|---|---|
| TGM1 | 5 | 0.341 | 5 |
| TGM3 | 1 | 0.232 | 1 |
| FXIII-A | 31 | 0.315 | 24 |
| TG2 guinea pig | 41 | 0.379 | **32** |

The off-target files are overwhelmingly the *same molecules* run on a second enzyme — selectivity
panels, not new chemistry. TGM1 contributes five molecules and TGM3 one. Scaffold overlap confirms
it: of TGM1's 53 Murcko scaffolds, 51 are already in the TG2 set; of TGM3's 42, 41 are; of
FXIII-A's 113, 99 are.

Guinea-pig TG2 is the one genuine source of unfamiliar chemistry — 41 molecules, 32 below 0.5
Tanimoto, and only 2 of its 18 scaffolds appear in the human set. **But every one of them is
inactive**: exact IC50s run 2.1 to 74 µM, median 26 µM, none below 1 µM. It can enlarge the
*negative* class in unfamiliar chemistry and cannot address the real shortage, which is distant
*actives*. It is also a different species and must stay a separate label.

**Net: the 42-molecule, 6-positive novel stratum that limits every generalization claim in this
project cannot be meaningfully grown from these sources.**

### One signal we are discarding

The IC50-only filter drops 990 TG2 records, and some of them are mechanistically the most
informative data in the file:

| | kinact/Ki | kinact | assays describing irreversible inhibition |
|---|---|---|---|
| TG2 | **286 molecules** | 148 | 238 molecules |
| TGM1 | 51 | 0 | 7 |
| FXIII-A | 11 | 0 | 6 |

kinact/Ki is the standard efficiency measure for a covalent inhibitor, and 286 molecules carry one —
comparable to the 302 potent molecules the whole project trains on. It is not interchangeable with
IC50 (it has different units and measures a rate over an affinity), so it cannot be merged into the
existing label. It is, however, the obvious basis for a *mechanism* model, which is noted as the
most promising unexploited asset in this dataset.

### Replicate quality

| | exact pIC50 | replicated | spanning > 1 log |
|---|---|---|---|
| TG2 | 425 | 82 | 17 |
| TGM1 | 83 | 0 | 0 |
| FXIII-A | 177 | 8 | 0 |
| TG2 guinea pig | 22 | 2 | 0 |

The off-target sets have almost no internal replication, so their values rest on single
measurements from single papers. That matters for the next section.

## Phase 2 — the selectivity model, and why its first answer was wrong

Only FXIII-A has enough paired data to attempt: **165 molecules measured on both TG2 and FXIII-A**,
of which **29 are TG2-potent and FXIII-A-clean**, 103 hit both, and 33 are not TG2-potent. The
103 is itself a finding — most known TG2 inhibitors do cross-react.

Tuned random forest on Morgan bits, the project's standard protocol, pooled one row per molecule
with a bootstrap CI over molecules:

| task | molecules | positives | AUROC | 95% CI |
|---|---|---|---|---|
| TG2 potency | 432 | 234 | 0.866 | [0.828, 0.900] |
| FXIII-A potency | 118 | 79 | 0.948 | [0.892, 0.989] |
| **selectivity, scaffold split** | 71 | 17 | **0.974** | [0.917, 1.000] |
| ranking selectivity by predicted TG2 potency | 134 | 20 | 0.454 | [0.329, 0.581] |

**That 0.974 is not a result, it is an artefact**, and it was caught before being reported as one.
Per-seed AUROC of 0.998 ± 0.006 on a task with 17 positives is not a plausible measurement. Two
checks explain it:

- Of the 98 scaffolds in the paired set, only **3 appear in both classes**. The selective and
  non-selective molecules are nearly scaffold-disjoint, so a scaffold split does not actually
  separate the question from the chemotype.
- The selective class is dominated by a handful of papers: **11 of 29 come from one document**, and
  the two classes are largely different series from different labs (Cbz-glycine peptidics against
  phenylsulfonyl-piperazines).

So the model can be learning "which series is this", and "which series" correlates with "which lab
measured it", which correlates with the label. The test that separates the two explanations is to
hold out whole papers instead of whole scaffolds:

| | AUROC | 95% CI | molecules scored |
|---|---|---|---|
| selectivity, scaffold split | 0.974 | [0.917, 1.000] | 71 |
| **selectivity, paper held out** | **0.574** | **[0.250, 0.864]** | 15 |

**A 0.40 collapse.** Held out across papers the model is indistinguishable from chance, on an
interval so wide it spans "much worse than guessing" to "good". Murcko scaffolds do not separate
papers when one lab's series spans several scaffolds, which is why the scaffold split — correct
everywhere else in this project — is the wrong control for this particular question.

Multi-task made no difference. Adding the FXIII-A model's prediction as a feature moved selectivity
AUROC from 0.901 to 0.911 on the subset where both are defined, and that comparison is an upper
bound rather than a fair estimate, because the auxiliary feature was fitted on the same molecules.

## Verdict

**Can we distinguish TG2 inhibition from broader transglutaminase inhibition?** **No.** Not with
this data, and not in a way that would transfer to a molecule from a new series. The only honest
number is the paper-held-out 0.574 with a 95% interval of [0.250, 0.864].

**Can the training data be expanded to cover unfamiliar chemistry?** **No.** The related
transglutaminases are selectivity panels on molecules we already have, and the one genuinely novel
source — guinea-pig TG2 — contains no actives.

**Does a selectivity-aware shortlist follow?** **No, and none is produced.** Phase 4 was explicitly
conditional on the models supporting it. Ranking by predicted TG2 potency carries *no* information
about selectivity (AUROC 0.454, an interval straddling chance), so any shortlist from the existing
potency model would be silent on the safety question that motivated this phase. Publishing one with
a selectivity column would misrepresent what was measured.

**What did we learn that is positive?**

1. Cross-reactivity is the norm: 103 of the 132 TG2-potent molecules with paired data also inhibit
   FXIII-A at 1 µM. That is a real, measured property of the known chemical space, and it is a
   reason for caution about every TG2 inhibitor series in the literature, not a modelling artefact.
2. FXIII-A potency itself is predictable (0.948) when the question is potency rather than
   selectivity — so the problem is not fingerprints, it is that selectivity is a small difference
   between two large numbers, measured on too few molecules spread across too few papers.
3. The 286-molecule kinact/Ki set is the largest piece of unused, mechanistically meaningful data
   in this project.

## Limitations

- 29 selective molecules across 6 papers. Every selectivity number here is small-sample.
- The paper-held-out split scores only 15 molecules per fold on average; its interval is wide in
  both directions and does not prove the absence of a signal, only that we cannot demonstrate one.
- The 1 µM cut is inherited from the TG2 work. A different threshold would move the selective count,
  and no threshold sweep was run.
- Document IDs come from the FXIII-A side of each pair; a molecule's TG2 measurement may originate
  elsewhere, so the grouping is approximate.
- TGM1, TGM3 and guinea-pig TG2 were audited, not modelled.
- Nothing here is a statement about safety in an organism. FXIII-A inhibition in a cuvette is a
  flag, not a clinical prediction.
