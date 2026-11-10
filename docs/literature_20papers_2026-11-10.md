# The twenty-paper reading list: verification and bearing on our findings

Date: 2026-11-10. The list was supplied by the supervisor on this date; it is the list brief 1
referred to but did not attach, so deliverable 1 starts here.

Every citation below was checked against an indexed primary record (Europe PMC `resultType=core`,
PubMed, arXiv or the publisher) on 2026-11-10. Nothing in this file is taken from a secondary
summary, a blog, or a search-result snippet. Where the supplied list's year disagrees with the
record, the record's year is used and the disagreement is noted.

**Reading depth.** Full methods were read for #1, #15, #16 and #17 (open access), and the complete
abstract with numeric results for #20 (paywalled; the deposited code and derived results at
doi:10.5281/zenodo.21522395 were also inspected). #19 is paywalled and only its abstract was
obtainable, so it stays marked *(abstract only)* and must not be cited as though we had assessed
its methodology. For the remaining fourteen, only the citation was verified; this file makes no
methodological claim about them.

**Four of my own first-pass readings were wrong.** They were taken from abstracts, and the methods
contradict them. Each is corrected below with the correction stated, not silently replaced:
#20 reports average precision and not AUROC; #17 does not demonstrate the memorisation mechanism
its abstract asserts; #15 does not contradict our splitting choices because it never tested them;
and #1's 93% is not a held-out test score.

## Group 1 — Celiac disease and TG2 inhibitor discovery

| # | citation | status |
|---|---|---|
| 1 | Wichka I, Lai P-K. Rapid discovery of Transglutaminase 2 inhibitors for celiac disease with boosting ensemble machine learning. *Comput Struct Biotechnol J* 2024;23:3669–3679. doi:10.1016/j.csbj.2024.10.019 | already ref 1 |
| 2 | Endrizzi F, Grunst M, Rudloff S, De Laffolie J, Zimmer K-P, Stricker S. Targeting transglutaminase 2: pathways to celiac disease therapies. *Gastroenterol Rep* 2025;13:goaf086. doi:10.1093/gastro/goaf086 | already ref 17 |
| 3 | Büchold C, Hils M, Gerlach U, Weber J, Pelzer C, Heil A, Aeschlimann D, Pasternack R. Features of ZED1227: The First-In-Class Tissue Transglutaminase Inhibitor Undergoing Clinical Evaluation for the Treatment of Celiac Disease. *Cells* 2022;11(10):1667. doi:10.3390/cells11101667. PMID 35626704, PMCID PMC9139979 | **new** |
| 4 | Schuppan D, Mäki M, Lundin KEA, et al. A randomized trial of a transglutaminase 2 inhibitor for celiac disease. *N Engl J Med* 2021;385(1):35–45. doi:10.1056/NEJMoa2032441. PMID 34192430 | already ref 20 |
| 5 | Dotsenko V, Tewes B, Hils M, et al. (CEC-3 Investigators). Transcriptomic analysis of intestine following administration of a transglutaminase 2 inhibitor to prevent gluten-induced intestinal damage in celiac disease. *Nat Immunol* 2024;25(7):1218–1230. doi:10.1038/s41590-024-01867-0. PMID 38914866, PMCID PMC11224021 | **new** |

## Group 2 — Existing treatments and other directions

| # | citation | status |
|---|---|---|
| 6 | Taavela J, Elli L, Bouma G, Tye-Din JA, Schuppan D, Lundin KEA, Schumann M. Upcoming Treatments in Celiac Disease: From Luminal Enzymes to Oral Immune Tolerance. *United European Gastroenterol J* 2026;14(3):e70222. doi:10.1002/ueg2.70222. PMID 42033586, PMCID PMC13109976 | **new** |
| 7 | Javed A, Saeed A, Akhtar A, Khalid A, Khan AS, Rabba W. Role of Regulatory T Cells and Transglutaminase 2 Inhibitors in Celiac Disease: A Systematic Review. *Cureus* 2025;17(5):e84901. doi:10.7759/cureus.84901. PMID 40575215, PMCID PMC12199851 | **new** |
| 8 | Kounatidis D, Pavlou A, Evangelopoulos A, et al. Rethinking Celiac Disease Management: Treatment Approaches Beyond the Gluten-Free Diet. *Biomedicines* 2025;14(1):29. doi:10.3390/biomedicines14010029. PMID 41595566, PMCID PMC12837603 | **new**; list says 2026, record says 2025 (published 22 Dec 2025 in the 2026 volume) |
| 9 | Tye-Din JA. Evolution in coeliac disease diagnosis and management. *JGH Open* 2024;8(7):e13107. doi:10.1002/jgh3.13107. PMCID PMC11217771 | **new** |

## Group 3 — Improving the models

| # | citation | status |
|---|---|---|
| 10 | Wu Z, Ramsundar B, Feinberg EN, Gomes J, Geniesse C, Pappu AS, Leswing K, Pande V. MoleculeNet: a benchmark for molecular machine learning. *Chem Sci* 2018;9(2):513–530. doi:10.1039/c7sc02664a. PMID 29629118, PMCID PMC5868307 | **new** |
| 11 | Yang K, Swanson K, Jin W, et al. Analyzing Learned Molecular Representations for Property Prediction. *J Chem Inf Model* 2019;59(8):3370–3388. doi:10.1021/acs.jcim.9b00237. PMID 31361484, PMCID PMC6727618 | **new**; note the published correction, PMID 31814400 |
| 12 | Heid E, Greenman KP, Chung Y, Li SC, Graff DE, Vermeire FH, Wu H, Green WH, McGill CJ. Chemprop: A Machine Learning Package for Chemical Property Prediction. *J Chem Inf Model* 2024;64(1):9–17. doi:10.1021/acs.jcim.3c01250. PMID 38147829, PMCID PMC10777403 | **new** |
| 13 | Graff DE, Morgan NK, Burns JW, et al. Chemprop v2: An Efficient, Modular Machine Learning Package for Chemical Property Prediction. *J Chem Inf Model* 2026;66(1):28–33. doi:10.1021/acs.jcim.5c02332. PMID 41453060 | **new**; clears a previously unverified citation |
| 14 | Qiao J, Jin J, Wang D, et al. A self-conformation-aware pre-training framework for molecular property prediction with substructure interpretability. *Nat Commun* 2025;16(1):4382. doi:10.1038/s41467-025-59634-0. PMID 40355450, PMCID PMC12069555 | **new** (SCAGE) |

Items 12 and 13 matter for honesty as much as for method: the paper benchmarks Chemprop and
CheMeleon and currently cites neither. Until these are in §11, ten of our benchmarked models are
uncited.

## Group 4 — Negative examples, overfitting, unreliable predictions

| # | citation | status |
|---|---|---|
| 15 | López-Pérez K, Miranda-Quintana RA. Extended Activity Cliffs-Driven Approaches on Data Splitting for the Study of Bioactivity Machine Learning Predictions. *Mol Inform* 2025;44(1):e202400054. doi:10.1002/minf.202400054. PMID 39558572 | **new** |
| 16 | Cui C, Su X, Zhang Z, Velez-Arce A, Wang J, Shi X, Zhang Y, Wu J, Chen YZ, Zitnik M, Shen W. Activity-cliff awareness enables robust graph learning for molecular property prediction. *Nat Commun* 2026;17(1):9376. doi:10.1038/s41467-026-75713-2. PMID 42680733, PMCID PMC13534492 | **new** (ACANet) |
| 17 | Tamura S, Miyao T, Bajorath J. Large-scale prediction of activity cliffs using machine and deep learning methods of increasing complexity. *J Cheminform* 2023;15:4. doi:10.1186/s13321-022-00676-7. PMCID PMC9825040 | **new** |
| 18 | Parrondo-Pizarro R, Lanini J, Rodríguez-Pérez R. Uncertainty Quantification in Molecular Machine Learning for Property Predictions under Data Shifts. *J Chem Inf Model* 2026;66(2):923–935. doi:10.1021/acs.jcim.5c02381. PMID 41533656 | **new** |
| 19 | Kötter A, Singh K, Matter H, Hessler G, Grebner C. Upgrading Reliability in Molecular Property Prediction by Robust Quantification of Uncertainty from Machine Learning Models. *J Chem Inf Model* 2025;65(20):10819–10831. doi:10.1021/acs.jcim.5c00464. PMID 41060699 | **new** |
| 20 | Agboola SS, Agboola OE, Olasehinde OR, et al. Activity cliffs resist prediction within and across protein kinases: a leakage-controlled machine-learning analysis of structure-potency discontinuities. *Bioorg Chem* 2026;181:110406. doi:10.1016/j.bioorg.2026.110406. PMID 42612264 | **new** |

## What bears directly on our own findings, after reading the methods

### #1 Wichka & Lai: the 93% is not a number our results can be compared against

This is the paper the supervisor recommended and the one we are asked to reproduce. Its protocol,
read in full, does not support using 93% as a benchmark for our AUROCs.

- **Dataset.** PubChem bioactivity for TG2 as of August 2023. 1,130 raw rows → 672 after
  deduplication by CID → **336 molecules** used for classification.
- **Labels.** Actives are IC50 < 0.6 µM (stated in Results, not Methods). The inactive class is
  **45 documented inactives pooled with 75 "unspecified" compounds** sliced at < 300 µM. So about
  five eighths of the negative class are compounds with no confirmed inactivity. This is precisely
  the practice brief 1 told us not to adopt, and it inflates any metric that rewards calling
  non-actives inactive.
- **Internal contradiction.** Methods 2.1 says inconclusive molecules were sliced in near the
  inactives; Results 3.1 says the inconclusive class was unfit for use because it lacked potency
  values. The two cannot both be true, and the paper does not resolve it.
- **Class balance.** 216 actives to 120 inactives, **64.3% active**. The majority class is the
  active class.
- **The 64% baseline is a majority-class predictor.** Reported as a DummyClassifier at 0.64
  accuracy with AUC 0.50. 216/336 = 0.643. So "93% versus a 64% baseline" is 93% against
  always-say-active on a 64%-active set — a 29-point gain over a trivial rule, with AUC 0.50 as
  the thing actually being beaten.
- **The 93% is a cross-validation mean, not a held-out test score.** 0.93 ± 0.02, stratified 5-fold
  CV repeated over 50 trials on all 336 molecules. There is no external test set. The separate 96%
  in their Table 1 is a different model (stump AdaBoost, default hyperparameters) on a 75:25 split
  and must not be quoted as the final model's performance.
- **Splitting.** Stratified random 75:25. **No scaffold split, no publication split, no temporal
  split.** By our own §5 results, a random split on this kind of data overstates performance on
  unfamiliar chemistry substantially.
- **Tuning.** Optuna, selected on accuracy; the paper never states that tuning was nested inside
  the CV folds, so the reported figure may be optimistic by an unknown amount.
- **Winning model.** AdaBoost over an MLP weak learner (one hidden layer, 50 units, ReLU, lr 0.001,
  max 1,000 iterations), 279 estimators, learning rate ≈ 0.0230. Features: 457 RDKit fingerprints
  plus 210 descriptors, variance ≤ 0.1 removed, Pearson r ≥ 0.9 groups collapsed, standard-scaled.

**Consequence for us.** A faithful reproduction is feasible — the model is fully specified — but the
comparison brief 1 asked for cannot be "their 93% versus our AUROC". The honest reproduction is to
rebuild their pipeline, then re-score it under our scaffold- and publication-held-out protocols with
our label policy, and report what their model does under evaluation it was not designed for. We
should also report their number on their own protocol so the two are not conflated. This vindicates
brief 1's instruction concretely: we now know *why* 93% does not mean their model is better.

### #20 Agboola et al.: independent corroboration, with my misreport corrected

**Correction: the figures are average precision, not AUROC.** I reported them as AUC earlier; that
was wrong, and average precision on an 11%-prevalence task is a different quantity.

- 16,897 Ki/Kd measurements, 20 human kinases, 8,661 unique standardised structures, 186,663
  single-cut matched molecular pairs, 116,845 labelled.
- Cliff is ΔpActivity ≥ 2.0; smooth is ≤ 0.5. Pairs between are unlabelled.
- Leave-one-kinase-out random forest: **average precision 0.238**, against a prevalence baseline of
  0.111 and a transformation-frequency baseline of 0.149. A real but modest gain.
- 97.1% of cliff-forming transformations occur on a single kinase; agreement among recurring
  transformations is median r = −0.058 (95% CI −0.13 to 0.41), indistinguishable from zero.
- Within-kinase average precision falls from **0.821 to 0.309** (Murcko scaffold grouping) and
  **0.211** (connected molecular component grouping), at which point within- and cross-kinase
  performance are not distinguishable: **p = 0.31** and **p = 0.17**. The test is not named in the
  abstract.
- Five within-kinase partitioning schemes of increasing stringency, from the deposited code: random
  pair-level, grouped by transformation, grouped by constant context, grouped by Murcko scaffold,
  grouped by connected molecular component.

This remains the strongest external support for our grouped-evaluation argument, and on kinases —
CDK2 is one. Two cautions on how we cite it: their task is pair classification, ours is
single-compound classification, so the collapse is analogous rather than identical; and their
non-significant p values are evidence of indistinguishability under their sample size, not proof of
equality.

### #17 Tamura, Miyao & Bajorath: the leakage collapse is real, the memorisation mechanism is not

**Correction: I repeated the abstract's claim that accuracy was driven by memorisation. The body
does not demonstrate it.**

- Task is **pair classification** (AC versus non-AC), not single-compound prediction. 100 ChEMBL
  v29 activity classes, Ki or Kd.
- Cliff threshold is **class-specific** — class mean potency plus two standard deviations — not a
  fixed 100-fold cut. Non-ACs are ΔpKi < 1. Pairs in between are not described.
- Two splits. Leakage-possible: MMPs randomly split 80/20. Leakage-excluded: 20% of **compounds**
  held out per class; pairs with both compounds held out go to test, pairs with neither to training,
  **mixed pairs discarded**. Run on **42 of the 100 classes** (those with ≥ 20 ACs).
- Median balanced accuracy ≈ 0.7 and MCC ≈ 0.5 with leakage possible, falling to ≈ 0.6 and < 0.25
  with leakage excluded.
- Models: SVM with MMP kernel, RF, XGBoost, two fully connected and two message-passing variants,
  plus 1NN and 5NN controls. Optuna, 100 iterations, three-fold inner CV, three trials.
- **No control experiment isolates memorisation.** The evidence is the leakage-possible versus
  leakage-excluded comparison. The authors do not measure how often correct predictions involve
  shared compounds, nor compare overlap against non-overlap pairs, and report no statistical test
  for the word "significantly". The abstract's causal claim is stronger than the body supports.

**Consequence for us.** The useful, supported finding is that complexity did not scale with accuracy
on *their* pair task, and that structurally distinct train/test sets reduce accuracy sharply. That
is a caution about expecting a better representation to fix cliff performance, but it is weaker than
I first stated and it is on a different task. It does not license a prediction that our proposal B
will fail. If we cite the memorisation claim at all, we must attribute it as the authors'
interpretation.

### #15 López-Pérez & Miranda-Quintana: does not contradict us, because it never tested our splits

**Correction: I reported this as being in tension with our scaffold- and publication-grouped
protocols. It is not. It never evaluates either one.**

- The eight strategies compared are **Random, Medoid, Uniform, Diverse, Kennard-Stone, eSALI,
  Anti-eSALI and bSALI** — diversity- and activity-landscape-based *selection* methods. **None is a
  scaffold split, and none is a publication or group split.**
- The design is a grid: method A picks the 20% test set, method B picks the training set from the
  remainder, all combinations including A = B. **Because method A chooses the test set, different
  strategies are scored on different test sets.** Only the test-set *size* is held constant. A
  comparison of RMSE across strategies therefore mixes generalisation with test-set difficulty, and
  cannot by itself establish that one strategy generalises better.
- 30 ChEMBL targets, curated from **van Tilborg et al.** — our ref 29, the same MoleculeACE data we
  use. Regression on Ki/EC50 scored by **RMSE**, five classic models (RF, kNN, SVM, GBM, MLP) on
  MACCS and ECFP4.
- The conclusion is bounded in the paper itself: random was "only surpassed in the low percentage
  cases by the anti-eSALI", and the authors caution that cliff effects are case-by-case and that no
  general best splitting method could be identified.

**Consequence for us.** Nothing to concede and nothing to claim. We should cite #15 for eSIM/eSALI
as landscape-roughness descriptors, and not cite it on the splitting question in either direction.
My earlier draft statement that "a peer-reviewed analysis prefers random splits" over grouped
splitting was unsupported and must not go into the paper.

### #16 Cui et al. (ACANet): the supplementary stress tests are the part that bears on us

- **ACA loss:** L = L_MAE + α·L_TSM, where the triplet term is a hinge over mined triplets using L2
  latent distances, with a **soft margin set by the label gap itself**, m = (y_A − y_N) − (y_A − y_P),
  so the loss vanishes when latent distances match label distances. Mining is online per batch
  (batch 128); positives differ from the anchor by ≤ cl, negatives by > cu; only triplets with
  strictly positive loss count.
- **52 datasets** across four task types: 9 low-sample narrow-scaffold regressions, 30 high-sample
  mixed-scaffold regressions (48,707 molecules), 3 MMP cliff classifications (ΔpKi ≥ 2 cliff, ≤ 1
  non-cliff), 10 ADMET delta tasks.
- **The headline split is not a scaffold split.** Molecules are spectrally clustered on ECFP, then
  split 80/20 *within each cluster*, stratified by cliff tag — an easier protocol than ours.
- **But two supplementary stress tests are in our regime, and they are the relevant evidence:** a
  Bemis–Murcko scaffold-held-out split (20% of scaffolds in test) where ACA improved **all 30**
  targets, and a cliff-rich split removing test molecules with > 0.90 similarity to training where
  ACA won on **28 of 30**.
- **Reported failure modes:** with no mined triplets the term is inert and performance drops; too
  large an α overfits hard triplets and hurt the easier 5-HT1A set; **strict structural gating by
  fingerprint, scaffold or SMILES threshold discarded useful triplets and did worse than latent-only
  mining**; cl and cu need per-dataset tuning; on some ADMET absorption sets it was only comparable
  to baselines. Leave-one-series-out is not evaluated.

**Consequence for us, and it resolves the disagreement I flagged.** #16 and #17 are not in conflict:
#17 found complexity not helping on *pair classification under compound-level holdout*, while #16
reports gains from a *loss function* on *single-compound regression under scaffold holdout and
similarity-capped splits*. The second is our setting. That makes #16 a better-motivated candidate
than our proposal B (count-based ECFP), which we chose for cheapness. Their own finding that strict
structural gating of triplets underperforms is also a direct warning about how we would adapt it to
publication-grouped data.

### #18 and #19: uncertainty, and the limit of what we can say

#19 (*abstract only — paywalled, full text not obtainable*) reports that popular UQ methods fail to
flag badly predicted compounds in regions of steep structure–activity relationship, and that the
train/test splitting scenario substantially changes measured UQ performance. Both are consistent
with our conformal results, where group calibration fixed coverage without making intervals usable.
The specific methods, datasets and splits are not stated in the record, so we cannot compare designs
and must cite it for its conclusion only. #18 reports that combining data-based and model-based
uncertainty through an error model stayed robust across degrees and types of shift — a concrete
alternative to the hierarchical conformal route, also *abstract only*.

### #3, #4 and #5 set the evidence bar

ZED1227's covalent, active-state-selective mechanism (#3), the six-week randomised trial (#4), and
the transcriptomic confirmation that oral TG2 inhibition prevented gluten-induced damage across 116
biopsies from 58 patients (#5) are what clinical evidence looks like for this target. They are the
reason the paper does not claim a treatment, and they belong in the discussion as the standard our
predictions do not meet.

**#3, #4 and #5 together set the evidence bar.** ZED1227's covalent, active-state-selective
mechanism (#3), the six-week randomised trial (#4), and the transcriptomic confirmation that oral
TG2 inhibition prevented gluten-induced damage across 116 biopsies from 58 patients (#5) are what
clinical evidence looks like for this target. They are the reason the paper does not claim a
treatment, and they belong in the discussion as the standard our predictions do not meet.

## Corrections to the existing reference list

Reference 3 in `docs/paper.md` reads "Li Y, Wagle J. Fine-tuning LLMs as Jev models.
arXiv:2610.02076." The arXiv identifier resolves to a real paper by those authors, submitted
2026-10-01, but its actual title is *LLM-as-Jev: LLMs Are Already Jev-Style Decision Models — When
and How to Fine-Tune Them*. The citation is genuine and the ✓ is warranted; the title string is
wrong and should be corrected.
