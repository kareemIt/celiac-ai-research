"""Experiment 2: do our models invert activity cliffs? Scored on saved predictions, no retraining.

Cliff definition is MoleculeACE's (van Tilborg, Alenicheva & Grisoni, JCIM 2022;62(23):5938-5951,
github.com/molML/MoleculeACE), with its published parameters rather than any of our own:

  structurally similar  -- at least one of  Tanimoto >= 0.9 on Morgan r=2, 1024 bits
                                            Tanimoto >= 0.9 on the generic Murcko scaffold's Morgan
  AND potency differs   -- fold change > 10, i.e. |delta pIC50| > 1

Deviation, recorded: MoleculeACE ORs in a third test, normalised SMILES Levenshtein similarity
>= 0.9. No edit-distance library is installed and adding one would touch the pinned environment
every committed result was produced under. Because the three tests are OR-ed, dropping one can only
*shrink* the cliff set -- every pair called a cliff here is a cliff by MoleculeACE's rule, but some
cliffs are missed. Counts are reported so the shortfall is visible.

Scoring is pairwise: for a discordant pair (one potent, one not) whose two molecules land in the
same test fold, the model is right if it gives the potent one the higher probability. Ties score
0.5. That needs no threshold, which is the point -- it sidesteps the operating-point question
docs/per_class_2026-11-09.md had to handle.

Usage: python scripts/analyse_cliffs.py [--target tg2|cdk2|maob] ...
Outputs: results/cliff_pairs_<target>.csv, results/cliff_ranking_<target>.csv,
         results/cliff_ranking_<target>_summary.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SIM_CUT = 0.9  # MoleculeACE default
LOG_FOLD_CUT = 1.0  # fold change > 10
# Fingerprint-only cuts for the sensitivity analysis. Clearly labelled and reported beside the
# MoleculeACE definitions, never in place of them: 0.9 is the strict reading of "structurally
# similar", 0.8 and 0.7 exist because 0.9 may leave too few pairs to score.
STRICT_CUTS = (0.9, 0.8, 0.7)
TARGETS = ("tg2", "cdk2", "maob")


def potency(target):
    """pIC50 per canonical SMILES, from the same loaders that built the labels."""
    if target == "tg2":
        from src.data import tg2
        return tg2.combined("regression").set_index("smiles").y
    from src.data import chembl
    raw = chembl.load_raw(f"data/raw/chembl/{target}_activities.csv")
    return chembl.regression(raw).set_index("smiles").y


def generic_scaffold(smiles):
    """MoleculeACE takes the generic (atom- and bond-agnostic) Murcko scaffold, falling back to the
    plain scaffold when generalising fails, and fingerprints that."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(scaf))
    except Exception:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles)


def fingerprints(smiles, radius=2, n_bits=1024):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    smiles = list(smiles)
    if not smiles:
        return np.zeros((0, n_bits), dtype=np.uint8)
    out = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s) if s else None
        out.append(np.zeros(n_bits, dtype=np.uint8) if mol is None
                   else gen.GetFingerprintAsNumPy(mol).astype(np.uint8))
    return np.stack(out)


def tanimoto_rows(fp_a, fp_b):
    """Tanimoto of every row of fp_a against every row of fp_b, as float32."""
    a, b = fp_a.astype(np.float32), fp_b.astype(np.float32)
    inter = a @ b.T
    union = a.sum(1)[:, None] + b.sum(1)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def _edit_distance(a, b, max_k):
    """Levenshtein distance, two-row DP, abandoned as soon as it cannot come in at or below max_k.
    Returns max_k + 1 when it gives up, which is all the caller needs."""
    if abs(len(a) - len(b)) > max_k:
        return max_k + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        if min(cur) > max_k:
            return max_k + 1
        prev = cur
    return prev[-1]


def _char_counts(strings, alphabet):
    out = np.zeros((len(strings), len(alphabet)), dtype=np.int16)
    index = {c: i for i, c in enumerate(alphabet)}
    for r, s in enumerate(strings):
        for ch in s:
            j = index.get(ch)
            if j is not None:
                out[r, j] += 1
    return out


def levenshtein_similar(pos, neg, cut=SIM_CUT, block=256):
    """Pairs whose normalised SMILES Levenshtein similarity (1 - distance / longer length) reaches
    `cut` -- MoleculeACE's third similarity test, which it ORs with the two fingerprint tests.

    Implemented here rather than by installing an edit-distance library, so the environment every
    committed result was produced under stays untouched. Two vectorised bounds prune first and an
    exact DP decides the survivors:

      length    similarity >= cut needs |len(a) - len(b)| <= (1 - cut) * max(len)
      counts    one edit changes the character histogram by at most 2, so the distance is at
                least half the total histogram difference

    Both are lower bounds on the distance, so neither can discard a pair that qualifies.
    """
    pos, neg = list(pos), list(neg)
    if not pos or not neg:
        return set()
    alphabet = sorted({c for s in pos + neg for c in s})
    cp, cn = _char_counts(pos, alphabet), _char_counts(neg, alphabet)
    len_p = np.array([len(s) for s in pos])
    len_n = np.array([len(s) for s in neg])
    out = set()
    for start in range(0, len(pos), block):
        sl = slice(start, start + block)
        longer = np.maximum(len_p[sl][:, None], len_n[None, :])
        budget = np.floor((1 - cut) * longer).astype(int)  # largest distance that still qualifies
        len_gap = np.abs(len_p[sl][:, None] - len_n[None, :])
        hist_gap = np.abs(cp[sl][:, None, :] - cn[None, :, :]).sum(axis=2) // 2
        cand = (len_gap <= budget) & (hist_gap <= budget)
        for i, j in zip(*np.nonzero(cand)):
            k = int(budget[i, j])
            if _edit_distance(pos[start + i], neg[j], k) <= k:
                out.add((start + i, j))
    return out


def discordant_pairs(mols, block=512, levenshtein=True):
    """Every (potent, not-potent) pair MoleculeACE would call structurally similar, with a column
    per test so either definition can be scored afterwards.

    Only discordant pairs can be scored by ranking, so similarity is computed for those alone
    rather than over the full n^2 matrix.
    """
    cols = ["smiles_pos", "smiles_neg", "similarity", "by_morgan", "by_scaffold", "by_levenshtein"]
    empty = pd.DataFrame(columns=cols)
    pos = mols[mols.y == 1].reset_index(drop=True)
    neg = mols[mols.y == 0].reset_index(drop=True)
    if pos.empty or neg.empty:  # a single-class fold has no discordant pair to rank
        return empty
    fp_p, fp_n = fingerprints(pos.smiles), fingerprints(neg.smiles)
    sc_p = fingerprints([generic_scaffold(s) for s in pos.smiles])
    sc_n = fingerprints([generic_scaffold(s) for s in neg.smiles])
    lev = levenshtein_similar(pos.smiles, neg.smiles) if levenshtein else set()
    rows = []
    for start in range(0, len(pos), block):
        sl = slice(start, start + block)
        morgan = tanimoto_rows(fp_p[sl], fp_n)
        scaffold = tanimoto_rows(sc_p[sl], sc_n)
        by_lev = np.zeros(morgan.shape, dtype=bool)
        for i, j in lev:
            if start <= i < start + morgan.shape[0]:
                by_lev[i - start, j] = True
        keep = (morgan >= SIM_CUT) | (scaffold >= SIM_CUT) | by_lev
        i, j = np.nonzero(keep)
        rows.append(pd.DataFrame({
            "smiles_pos": pos.smiles.values[start + i], "smiles_neg": neg.smiles.values[j],
            "similarity": np.maximum(morgan, scaffold)[i, j],
            "by_morgan": morgan[i, j] >= SIM_CUT, "by_scaffold": scaffold[i, j] >= SIM_CUT,
            "by_levenshtein": by_lev[i, j]}))
    return pd.concat(rows, ignore_index=True) if rows else empty


def build_pairs(target, mols, pic50):
    pairs = discordant_pairs(mols)
    pairs["pic50_pos"] = pairs.smiles_pos.map(pic50)
    pairs["pic50_neg"] = pairs.smiles_neg.map(pic50)
    pairs["log_fold"] = (pairs.pic50_pos - pairs.pic50_neg).abs()
    # a censored negative ('>X uM') has no pIC50, so MoleculeACE's potency test cannot be applied
    pairs["potency_known"] = pairs.log_fold.notna()
    pairs["is_cliff"] = pairs.potency_known & (pairs.log_fold > LOG_FOLD_CUT)
    pairs.insert(0, "target", target)
    pairs.to_csv(RESULTS / f"cliff_pairs_{target}.csv", index=False)
    return pairs


def score(target, mols, pic50):
    """Pairwise ranking accuracy per model x arm x fold, over three kinds of discordant pair.

    Pairs are enumerated inside each test fold rather than taken from the global inventory, because
    the comparison needs a control at a matched potency gap:

      cliff          similar AND fold change > 10   -- the MoleculeACE case
      near_threshold similar AND fold change <= 10  -- the two molecules straddle the 1 uM cut by
                                                       less than 10x, so the labels are nearly
                                                       arbitrary; a floor, not a fair control
      distant        not similar AND fold change > 10 -- same potency gap as a cliff, different
                                                         chemistry: the control that isolates what
                                                         structural similarity costs

    Everything is reported under both similarity definitions, so the effect of the third test is
    visible rather than argued about:
      two_test    Morgan or generic-scaffold Tanimoto, the definition the first run used
      three_test  those two OR normalised SMILES Levenshtein -- MoleculeACE's full rule
    """
    preds_file = {"tg2": "sota_preds.csv"}.get(target, f"sota_preds_{target}.csv")
    preds = pd.read_csv(RESULTS / preds_file)
    smi = mols.smiles.tolist()
    fp = dict(zip(smi, fingerprints(smi)))
    sc = dict(zip(smi, fingerprints([generic_scaffold(s) for s in smi])))
    label = dict(zip(mols.smiles, mols.y))
    rows = []
    for (model, split, seed), g in preds.groupby(["model", "split", "seed"], sort=True):
        g = g.drop_duplicates("smiles")
        known = [s for s in g.smiles if s in label and not np.isnan(pic50.get(s, np.nan))]
        pos = [s for s in known if label[s] == 1]
        neg = [s for s in known if label[s] == 0]
        if not pos or not neg:
            continue
        p = g.set_index("smiles").p
        morgan = tanimoto_rows(np.stack([fp[s] for s in pos]), np.stack([fp[s] for s in neg]))
        scaffold = tanimoto_rows(np.stack([sc[s] for s in pos]), np.stack([sc[s] for s in neg]))
        sim = np.maximum(morgan, scaffold)
        gap = np.abs(np.array([pic50[s] for s in pos])[:, None]
                     - np.array([pic50[s] for s in neg])[None, :])
        pp, pn = p.loc[pos].to_numpy()[:, None], p.loc[neg].to_numpy()[None, :]
        correct = np.where(pp > pn, 1.0, np.where(pp == pn, 0.5, 0.0))
        big = gap > LOG_FOLD_CUT
        by_lev = np.zeros(sim.shape, dtype=bool)
        for i, j in levenshtein_similar(pos, neg):
            by_lev[i, j] = True
        # Primary definitions first, then the stricter fingerprint-only sensitivity analysis.
        # Dropping the scaffold and Levenshtein tests makes "similar" mean what a cheminformatician
        # would usually mean by it, at the cost of far fewer pairs; the lower cuts exist because at
        # 0.9 there are too few fingerprint-similar pairs to score (23 on TG2 in the inventory).
        definitions = [("two_test", sim >= SIM_CUT),
                       ("three_test", (sim >= SIM_CUT) | by_lev)]
        definitions += [(f"morgan_only_{c:g}", morgan >= c) for c in STRICT_CUTS]
        for definition, similar in definitions:
            kinds = {"cliff": similar & big, "near_threshold": similar & ~big,
                     "distant": ~similar & big}
            for kind, mask in kinds.items():
                if mask.sum():
                    i, j = np.nonzero(mask)
                    rows.append({"target": target, "model": model, "split": split, "seed": seed,
                                 "definition": definition,
                                 "pair_kind": kind, "n_pairs": int(mask.sum()),
                                 "ranking_accuracy": float(correct[mask].mean()),
                                 # molecules, not pairs: one molecule appears in many pairs
                                 "n_molecules": len(set(np.array(pos)[i]) | set(np.array(neg)[j]))})
    rows = pd.DataFrame(rows)
    if rows.empty:
        print(f"{target}: no pair has both molecules in the same test fold")
        return rows
    rows.to_csv(RESULTS / f"cliff_ranking_{target}.csv", index=False)
    summary = (rows.groupby(["target", "definition", "model", "split", "pair_kind"])
               .agg(folds=("seed", "nunique"), pairs_total=("n_pairs", "sum"),
                    pairs_per_fold=("n_pairs", "median"),
                    ranking_accuracy=("ranking_accuracy", "mean"),
                    ranking_sd=("ranking_accuracy", "std")).reset_index())
    summary.to_csv(RESULTS / f"cliff_ranking_{target}_summary.csv", index=False)
    return rows


def holm(pvals):
    """Step-down Holm, same as scripts/analyse_sota.py, returned in input order."""
    out, running = np.empty(len(pvals)), 0.0
    for rank, i in enumerate(np.argsort(pvals)):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        out[i] = running
    return out


def tests(target, rows):
    """Cliff against distant, paired per fold. The fold is the independent unit: one molecule sits
    in many pairs, so pairs are not independent, but each fold contributes a single number."""
    from scipy.stats import spearmanr, ttest_rel
    wide = rows.pivot_table(index=["definition", "split", "model", "seed"], columns="pair_kind",
                            values="ranking_accuracy")
    out = []
    for (definition, split), g in wide.groupby(level=["definition", "split"]):
        block = []
        for model, gm in g.groupby(level="model"):
            d = gm[["cliff", "distant"]].dropna()
            if len(d) < 3:
                continue
            t = ttest_rel(d.cliff, d.distant)
            block.append({"target": target, "definition": definition, "split": split,
                          "model": model, "folds": len(d),
                          "acc_cliff": d.cliff.mean(), "acc_distant": d.distant.mean(),
                          "delta": d.cliff.mean() - d.distant.mean(), "p_ttest": t.pvalue})
        if not block:
            continue
        block = pd.DataFrame(block)
        block["p_holm"] = holm(block.p_ttest.to_numpy())
        # does overall ranking predict cliff ranking? one correlation per arm, over models
        rho = spearmanr(block.acc_distant, block.acc_cliff) if len(block) >= 4 else None
        block["rank_rho_models"] = rho.statistic if rho else np.nan
        block["rank_rho_p"] = rho.pvalue if rho else np.nan
        out.append(block)
    if not out:
        return pd.DataFrame()
    out = pd.concat(out, ignore_index=True)
    out.to_csv(RESULTS / f"cliff_ranking_{target}_tests.csv", index=False)
    return out


def run(target):
    splits = pd.read_csv(RESULTS / f"splits_{target}.csv", low_memory=False)
    mols = splits[["smiles", "y"]].drop_duplicates()
    if mols.smiles.duplicated().any():
        raise SystemExit(f"{target}: a molecule carries two labels in splits_{target}.csv")
    pic50 = potency(target)
    pairs = build_pairs(target, mols, pic50)
    n_cliff, n_unknown = int(pairs.is_cliff.sum()), int((~pairs.potency_known).sum())
    print(f"{target}: {len(pairs)} similar discordant pairs, {n_cliff} cliffs, "
          f"{len(pairs) - n_cliff - n_unknown} near-threshold non-cliffs, "
          f"{n_unknown} undecidable (censored, no pIC50)")
    only_lev = pairs.by_levenshtein & ~pairs.by_morgan & ~pairs.by_scaffold
    print(f"  similarity test that admitted each pair: morgan {int(pairs.by_morgan.sum())}, "
          f"scaffold {int(pairs.by_scaffold.sum())}, levenshtein {int(pairs.by_levenshtein.sum())}"
          f" ({int(only_lev.sum())} admitted by levenshtein alone, "
          f"{int((only_lev & pairs.is_cliff).sum())} of them cliffs)")
    if n_cliff < 100:
        print(f"  WARNING {target}: fewer than 100 cliff pairs -- too few to claim a difference")
    rows = score(target, mols, pic50.to_dict())
    if rows.empty:
        return
    print(f"  {len(rows)} scored model x arm x fold x definition x pair-kind rows")
    t = tests(target, rows)
    for definition, g in t.groupby("definition") if not t.empty else []:
        worse = int((g.delta < 0).sum())
        pairs = rows[(rows.definition == definition) & (rows.pair_kind == "cliff")].n_pairs
        print(f"  [{definition:15s}] cliff worse than distant in {worse}/{len(g)} model-arm "
              f"comparisons, {int((g.p_holm < 0.05).sum())} significant after Holm, "
              f"median {pairs.median():.0f} cliff pairs/fold")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", action="append", choices=TARGETS, default=None)
    for t in ap.parse_args().target or TARGETS:
        run(t)
