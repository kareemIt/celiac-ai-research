"""Jev-style option-logit probe for TG2 potency (IC50 <= 1 uM), compared with fingerprint RF on the
same scaffold splits as run_baselines.py.

Variants per prompt ("smiles", "smiles+name"):
  jev_*        zero-shot: P(potent) from Qwen's option log-probs, averaged over both option orders
  jev_*+platt  one-feature logistic calibration of that score, fit on each seed's train fold

Qwen scores are cached in results/jev_tg2_scores.csv; pass --rescore to recompute.
"""
import argparse
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from scripts.run_baselines import summarize
from src.data import chem, tg2
from src.evaluation import metrics

SCORES = "results/jev_tg2_scores.csv"
QUESTION = "Is this compound a potent inhibitor of human tissue transglutaminase 2 (TG2)?"
POTENT, WEAK = "Potent TG2 inhibitor (IC50 <= 1 uM)", "Not potent (IC50 > 1 uM)"


def score(table, names):
    from src.models.jev_readout import build_prompt, option_logprobs
    from src.models.qwen import load_qwen_4bit

    model, tok = load_qwen_4bit()
    model.eval()
    rows, t0 = [], time.perf_counter()
    for kind in ["smiles", "smiles+name"]:
        for s in table["smiles"]:
            state = f"Compound SMILES: {s}"
            if kind == "smiles+name" and isinstance(names.get(s), str):
                state += f"\nCompound name: {names[s]}"
            # Both orders: LLM option choice is position-biased; averaging cancels a constant bias.
            for order, opts in enumerate([[POTENT, WEAK], [WEAK, POTENT]]):
                lp = option_logprobs(model, tok, build_prompt(tok, state, QUESTION, opts), 2)
                logodds = (lp[0] - lp[1]) if order == 0 else (lp[1] - lp[0])  # log P(potent) - log P(not)
                rows.append({"prompt": kind, "smiles": s, "order": order, "logodds": float(logodds)})
        print(f"  scored {kind}: {time.perf_counter() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(SCORES, index=False)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main(seeds, rescore, data="pubchem"):
    d = tg2.load_raw()
    table = tg2.combined("classification") if data == "combined" else tg2.classification(d)
    if rescore or not os.path.exists(SCORES):
        score(table, d.groupby("smiles")["Compound_Name"].first().to_dict())
    sc = pd.read_csv(SCORES)
    y, smiles = table["y"].to_numpy(), table["smiles"].tolist()

    print(f"\n## Position bias check (all {len(table)} molecules, single fixed order)")
    lo = {}
    for kind, g in sc.groupby("prompt", sort=False):
        piv = g.pivot(index="smiles", columns="order", values="logodds").loc[smiles]
        for order in (0, 1):
            m = metrics.classification(y, sigmoid(piv[order].to_numpy()))
            print(f"  {kind:12s} potent listed {'first ' if order == 0 else 'second'}: mean P(potent) "
                  f"{sigmoid(piv[order]).mean():.3f} (true rate {y.mean():.3f}), AUROC {m['auroc']:.3f}")
        lo[kind] = piv.mean(axis=1).to_numpy()

    rows = []
    for seed in range(seeds):
        tr, _, te = chem.scaffold_split(smiles, seed=seed)
        for kind, x in lo.items():
            platt = LogisticRegression(C=1e4).fit(x[tr, None], y[tr])
            for name, p in [(f"jev_{kind}", sigmoid(x[te])), (f"jev_{kind}+platt", platt.predict_proba(x[te, None])[:, 1])]:
                rows.append({"model": name, "seed": seed, **metrics.classification(y[te], p)})
    out = pd.DataFrame(rows)
    out.to_csv("results/jev_probe_tg2.csv", index=False)

    base = pd.read_csv(f"results/baselines_tg2{'' if data == 'pubchem' else '_' + data}.csv")
    base = base[(base.task == "classification") & (base.seed < seeds)]
    both = pd.concat([base[base.model.isin(["prior", "rf"])], out])[out.columns]
    print(f"\n## TG2 potency: Jev-style readout vs baselines (scaffold split, {seeds} seeds)\n"
          f"{summarize(both, ['model']).to_string()}")
    rf = base[base.model == "rf"]
    for name in out.model.unique():
        m = out[out.model == name].merge(rf, on="seed", suffixes=("", "_rf"))
        diffs = []
        for metric, hb in [("auroc", True), ("ece", False), ("brier", False)]:
            dd = (m[metric] - m[f"{metric}_rf"]).dropna()
            diffs.append(f"Δ{metric} {dd.mean():+.3f} ± {dd.std():.3f} ({((dd > 0) if hb else (dd < 0)).sum()}/{len(dd)} better)")
        print(f"  {name:22s} vs RF: " + ", ".join(diffs))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--data", default="pubchem", choices=["pubchem", "combined"])
    a = ap.parse_args()
    main(a.seeds, a.rescore, a.data)
