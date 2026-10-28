"""Did three identical runs of the whole pipeline agree, end to end?

scripts/check_determinism.py shows that a short training loop is bit-exact once determinism is
enabled. This checks the claim where it matters: three complete runs of run_jev_qlora.py on the same
seed and split, compared on everything a result depends on -- the per-epoch log-odds, the epoch
validation picked, the saved adapter weights, and the headline metrics.

Usage: python scripts/compare_repro.py [--tags repro1 repro2 repro3]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from safetensors.torch import load_file

OUT = "results/repro_check.csv"


def adapter_diff(tag_a, tag_b, seed=0):
    """Largest absolute disagreement between two saved LoRA adapters, or None if either is missing."""
    paths = [f"checkpoints/jev_qlora_{t}/seed{seed}/adapter_model.safetensors" for t in (tag_a, tag_b)]
    if not all(os.path.exists(p) for p in paths):
        return None
    a, b = (load_file(p) for p in paths)
    if a.keys() != b.keys():
        return np.inf
    return max(float((a[k].float() - b[k].float()).abs().max()) for k in a)


def main(tags):
    rows, ref = [], tags[0]
    res = {t: pd.read_csv(f"results/jev_qlora_tg2_{t}.csv") for t in tags}
    pre = {t: pd.read_csv(f"results/jev_qlora_preds_{t}.csv") for t in tags}
    epo = {t: pd.read_csv(f"results/jev_qlora_epochs_{t}.csv") for t in tags}

    print("## Headline metrics per run\n")
    cols = ["seed", "best_epoch", "train_s", "auroc", "auprc", "mcc", "ece", "brier"]
    print(pd.concat([res[t].assign(run=t)[["run"] + cols] for t in tags])
          .to_string(index=False, float_format="%.4f"))

    print(f"\n## Every other run against {ref}\n")
    print(f"{'run':<10}{'same best epoch':>18}{'max |Δ logodds|':>18}{'max |Δ epoch curve|':>22}"
          f"{'max |Δ adapter|':>18}{'Δ AUROC':>10}")
    for t in tags[1:]:
        same_ep = bool((res[ref].best_epoch.values == res[t].best_epoch.values).all())
        a = pre[ref].sort_values(["fold", "smiles"]).logodds.to_numpy()
        b = pre[t].sort_values(["fold", "smiles"]).logodds.to_numpy()
        dz = float(np.abs(a - b).max()) if a.shape == b.shape else np.inf
        ca = epo[ref].sort_values(["epoch", "fold", "smiles"]).logodds.to_numpy()
        cb = epo[t].sort_values(["epoch", "fold", "smiles"]).logodds.to_numpy()
        dc = float(np.abs(ca - cb).max()) if ca.shape == cb.shape else np.inf
        dw = adapter_diff(ref, t)
        da = float(res[t].auroc.mean() - res[ref].auroc.mean())
        print(f"{t:<10}{str(same_ep):>18}{dz:>18.3e}{dc:>22.3e}"
              f"{'n/a' if dw is None else f'{dw:.3e}':>18}{da:>+10.4f}")
        rows.append({"run": t, "vs": ref, "same_best_epoch": same_ep, "max_abs_dlogodds": dz,
                     "max_abs_dcurve": dc, "max_abs_dadapter": dw, "d_auroc": da})

    out = pd.DataFrame(rows)
    exact = bool(((out.max_abs_dlogodds == 0) & (out.max_abs_dcurve == 0) & out.same_best_epoch).all()
                 and (out.max_abs_dadapter.fillna(0) == 0).all())
    print(f"\n## Verdict: {'BIT-EXACT across all runs' if exact else 'runs still differ'}")
    if not exact:
        print("  Residual variation is real and must be reported as the noise floor of any comparison.")
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["repro1", "repro2", "repro3"])
    main(ap.parse_args().tags)
