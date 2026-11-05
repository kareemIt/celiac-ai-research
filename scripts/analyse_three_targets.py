"""Which Stage 1 findings replicate across TG2, CDK2 and MAO-B?

Stage 5's core analysis. Three questions, each answered by combining the targets rather than by
eyeballing three tables:

  1. Does publication-held-out evaluation reduce performance on every target?
     Read from results/split_penalty_<target>.csv (scripts/analyse_split_penalty.py).

  2. Does the per-model PENALTY ranking replicate? TG2 and CDK2 agree perfectly, which looked like
     a finding until MAO-B was run. Reported as a pairwise rank correlation so the disagreement is
     visible rather than averaged away.

  3. Does any model beat the tuned random forest consistently under publication shift? Each
     target gives one paired fold-level mean difference; those are combined by inverse-variance
     weighting. The same contrast is run on the scaffold arm, because a model that wins in both
     regimes is simply a better model -- only a win that is specific to the publication arm
     supports "generalizes better under publication shift".

Cochran's Q is reported with every pooled estimate. A fixed-effect pooled mean assumes one true
effect across targets; where Q is large against its 2 degrees of freedom that assumption fails and
the pooled number should not be quoted. The three targets also do not use the same publication
protocol -- TG2 is leave-one-publication-out, the replications are grouped K-fold -- so a pooled
estimate mixes protocols. Both limitations are printed with the output, not buried here.

Reads only committed result files; fits nothing.

Reproduce: python scripts/analyse_three_targets.py
  -> results/three_targets_{penalty,pooled}.csv
"""
import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

from scripts.analyse_sota import BASELINE, ORDER, TARGETS

PENALTY_IN = "results/split_penalty_{target}.csv"
OUT_PENALTY = "results/three_targets_penalty.csv"
OUT_POOLED = "results/three_targets_pooled.csv"
# The publication-held-out arm each target was actually run under. TG2 keeps its published
# leave-one-publication-out; the replications use grouped K-fold (docs/replication_feasibility.md).
PUB_ARM = {"tg2": "paper_lopo", "cdk2": "paper_grouped", "maob": "paper_grouped"}
SCAFFOLD_ARM = "scaffold_pub"


def paired_deltas(sota, split, model, base=BASELINE):
    """Per-fold AUROC difference between `model` and `base` on one target and split.

    Pairing on fold index is valid here, and only here: within one target and split, fold i is
    the same held-out molecules for every model. (Across splits it is not, which is why
    scripts/analyse_split_penalty.py uses an unpaired test.)
    """
    d = sota[(sota["split"] == split) & sota["model"].isin([model, base])]
    wide = d.pivot_table(index="seed", columns="model", values="auroc").dropna()
    if model not in wide or base not in wide or len(wide) < 3:
        return None
    return (wide[model] - wide[base]).to_numpy()


def pool(per_target):
    """Inverse-variance fixed-effect mean of per-target paired differences.

    per_target: {target: (mean, standard error, n_folds)}. Targets with a zero standard error are
    dropped rather than given infinite weight.
    """
    usable = {t: v for t, v in per_target.items() if v[1] > 0}
    if len(usable) < 2:
        return None
    est = np.array([v[0] for v in usable.values()])
    w = np.array([1 / v[1] ** 2 for v in usable.values()])
    pooled = float((w * est).sum() / w.sum())
    se = float(np.sqrt(1 / w.sum()))
    return {
        "pooled": pooled, "se": se,
        "ci_lo": pooled - 1.96 * se, "ci_hi": pooled + 1.96 * se,
        "p": float(2 * norm.sf(abs(pooled / se))),
        "q": float((w * (est - pooled) ** 2).sum()), "df": len(usable) - 1,
        "targets": ",".join(usable),
    }


def holm(out):
    out = out.sort_values("p").reset_index(drop=True)
    running = 0.0
    for i, row in out.iterrows():
        running = max(running, min(1.0, (len(out) - i) * row["p"]))
        out.loc[i, "p_holm"] = running
    return out


def penalties(targets):
    """One column per target of the per-model publication penalty."""
    cols = {}
    for t in targets:
        path = PENALTY_IN.format(target=t)
        if os.path.exists(path):
            cols[t] = pd.read_csv(path).set_index("model")["penalty"]
    return pd.DataFrame(cols).dropna()


def rank_agreement(tab):
    rows = []
    for a, b in itertools.combinations(tab.columns, 2):
        r = spearmanr(tab[a], tab[b])
        rows.append({"target_a": a, "target_b": b, "models": len(tab),
                     "spearman": r.statistic, "p": r.pvalue})
    return pd.DataFrame(rows)


def contrasts(targets, arm):
    """Every entrant against the random forest, pooled across targets, for one arm."""
    from scripts.run_sota import paths

    sotas = {}
    for t in targets:
        _, sota_path, _ = paths(t)
        if os.path.exists(sota_path):
            sotas[t] = pd.read_csv(sota_path)

    models = sorted({m for d in sotas.values() for m in d["model"]} - {BASELINE},
                    key=lambda m: ORDER.index(m) if m in ORDER else 99)
    rows = []
    for model in models:
        per = {}
        for t, sota in sotas.items():
            split = PUB_ARM[t] if arm == "publication" else SCAFFOLD_ARM
            v = paired_deltas(sota, split, model)
            if v is not None:
                per[t] = (v.mean(), v.std(ddof=1) / np.sqrt(len(v)), len(v))
        p = pool(per)
        if p is None:
            continue
        rows.append({"arm": arm, "model": model, "k": len(per), **p,
                     **{f"delta_{t}": v[0] for t, v in per.items()}})
    return holm(pd.DataFrame(rows)) if rows else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=list(TARGETS))
    args = ap.parse_args()

    tab = penalties(args.targets)
    print("\n## 1. Publication penalty per model and target (AUROC)\n")
    print(tab.round(3).to_string())
    worse = int((tab < 0).all(axis=1).sum())
    print(f"\n  {worse}/{len(tab)} models lose AUROC on every one of "
          f"{len(tab.columns)} targets")
    print(f"  mean penalty: " + "  ".join(f"{t} {tab[t].mean():+.3f}" for t in tab))

    agree = rank_agreement(tab)
    print("\n## 2. Does the penalty RANKING replicate?\n")
    for _, r in agree.iterrows():
        print(f"  {r.target_a:<5} vs {r.target_b:<5} Spearman "
              f"{r.spearman:+.3f}  p={r.p:.3f}")
    if (agree["p"] < 0.05).any() and not (agree["p"] < 0.05).all():
        print("\n  Mixed. A ranking that agrees on two targets and not the third is not a"
              "\n  replicated ranking; the agreement is reported, not claimed.")

    pooled = []
    for arm in ("publication", "scaffold"):
        c = contrasts(args.targets, arm)
        if c.empty:
            continue
        pooled.append(c)
        print(f"\n## 3. {arm.capitalize()} arm: entrant minus tuned random forest, "
              f"pooled across targets\n")
        print(f"{'model':<18}{'pooled':>8}{'95% CI':>20}{'p':>9}{'p_holm':>9}"
              f"{'Q':>7}{'df':>4}   per target")
        for _, r in c.sort_values("pooled", ascending=False).iterrows():
            per = "  ".join(f"{t} {r[f'delta_{t}']:+.3f}" for t in args.targets
                            if f"delta_{t}" in r and pd.notna(r[f"delta_{t}"]))
            print(f"{r['model']:<18}{r['pooled']:>+8.3f}"
                  f"{f'[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]':>20}"
                  f"{r['p']:>9.4f}{r['p_holm']:>9.3f}{r['q']:>7.1f}{int(r['df']):>4}   {per}")

    print("\n  Assumptions. The pooled mean is fixed-effect: it assumes one true effect across"
          "\n  targets, which a large Q against its df contradicts. The targets also do not share"
          "\n  a publication protocol (TG2 leave-one-out, replications grouped K-fold), so a"
          "\n  pooled estimate mixes protocols. Holm is across entrants within an arm.")

    os.makedirs("results", exist_ok=True)
    tab.to_csv(OUT_PENALTY)
    if pooled:
        pd.concat(pooled).to_csv(OUT_POOLED, index=False)
    print(f"\n-> {OUT_PENALTY}, {OUT_POOLED}")


if __name__ == "__main__":
    main()
