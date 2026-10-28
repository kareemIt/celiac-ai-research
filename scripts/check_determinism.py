"""Can this machine run the same QLoRA configuration twice and get the same answer?

Two supposedly identical lambda = 0 runs differed by 0.028 mean AUROC with per-seed swings to 0.18
(docs/forgetting.md), which is as large as the KL-anchor effect we were trying to measure. This
script localizes that, cheaply, before any more 3-hour runs:

  stage 1  the same forward pass, repeated          -> is *inference* reproducible?
  stage 2  the same short training run, repeated    -> is *training* reproducible, and if not,
                                                       at which optimizer step do they first part?

It calls the real build_inputs / train_epoch / log_odds from run_jev_qlora, not a copy, so a pass
here is a statement about the actual experiment. Run it with and without --deterministic.

Usage: python scripts/check_determinism.py [--steps 64] [--repeats 3] [--kl 1.0] [--deterministic]
"""
import argparse
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import torch

from scripts.run_jev_qlora import OPTIMS, build_inputs, load_trainable, log_odds, reset_adapter, train_epoch
from src.data import chem, tg2

OUT = "results/determinism_check.csv"


def flat_weights(model):
    return torch.cat([p.detach().float().flatten()
                      for n, p in sorted(model.named_parameters()) if p.requires_grad]).cpu().numpy()


def one_run(model, ctx, y, tr, eval_idx, kl, seed, optim):
    """Reset the adapter and train `tr` once, exactly as run_jev_qlora would."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    reset_adapter(model)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = OPTIMS[optim](params)
    model.train()
    losses = train_epoch(model, opt, params, tr, rng, ctx, y, kl)
    w, z = flat_weights(model), log_odds(model, [ctx["P"][k] for k in eval_idx], ctx["digit_ids"])
    del opt
    return losses, w, z


def report(name, ref, others, unit=""):
    """Max absolute deviation of each repeat from the first, and whether anything moved at all."""
    diffs = [float(np.abs(o - ref).max()) for o in others]
    verdict = "IDENTICAL" if max(diffs, default=0.0) == 0.0 else "DIFFERS"
    print(f"  {name:<34}{verdict:>11}   max |delta| = {max(diffs, default=0.0):.3e}{unit}")
    return diffs


def main(steps, repeats, kl, data, seed, optim):
    table = tg2.combined("classification") if data == "combined" else tg2.classification(tg2.load_raw())
    raw = tg2.load_raw()
    names = raw.groupby("smiles")["Compound_Name"].first().to_dict()
    smiles, y = table["smiles"].tolist(), table["y"].to_numpy()
    tr, va, te = chem.scaffold_split(smiles, seed=seed)
    tr, eval_idx = tr[:steps], te[:24]

    model, tok = load_trainable()
    ctx = build_inputs(tok, smiles, names, y, kl)
    rows = []

    print(f"\n## Stage 1: inference only, base model (adapter at B=0), {repeats} repeats")
    reset_adapter(model)
    model.train()  # dropout is active in train mode but log_odds switches to eval, as in training
    z0 = [log_odds(model, [ctx["P"][k] for k in eval_idx], ctx["digit_ids"]) for _ in range(repeats)]
    d0 = report("log-odds of the same molecules", z0[0], z0[1:])
    rows += [{"stage": "inference", "repeat": i + 1, "max_abs_diff": v} for i, v in enumerate(d0)]

    print(f"\n## Stage 2: {steps} training examples, kl={kl:g}, optimizer {optim}, {repeats} repeats")
    runs = [one_run(model, ctx, y, tr, eval_idx, kl, seed, optim) for _ in range(repeats)]
    losses, weights, zs = zip(*runs)
    dl = report("per-example training loss", losses[0], losses[1:])
    dw = report("LoRA weights after training", weights[0], weights[1:])
    dz = report("log-odds after training", zs[0], zs[1:])
    for i in range(repeats - 1):
        rows += [{"stage": "train_loss", "repeat": i + 2, "max_abs_diff": dl[i]},
                 {"stage": "train_weights", "repeat": i + 2, "max_abs_diff": dw[i]},
                 {"stage": "train_logodds", "repeat": i + 2, "max_abs_diff": dz[i]}]

    if max(dl, default=0.0) > 0:
        first = [int(np.argmax(np.abs(l - losses[0]) > 0)) for l in losses[1:] if np.abs(l - losses[0]).max() > 0]
        print(f"\n  first example whose loss differs: {first} of {len(tr)} "
              f"(optimizer step {[f // 8 + 1 for f in first]} of {len(tr) // 8 + 1})")
        print("  A divergence at example 0 is a kernel; a later one is accumulated drift.")

    print(f"\n## Consequence for a measured AUROC")
    print(f"  log-odds spread after {steps} examples: {max(dz, default=0.0):.3e}; "
          f"the KL effect we are chasing is 0.031 AUROC")
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")
    return max(dl + dw + dz, default=0.0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=64, help="training examples per repeat")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--kl", type=float, default=1.0)
    ap.add_argument("--data", default="combined", choices=["pubchem", "combined"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--optim", default="paged8bit", choices=sorted(OPTIMS))
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--loose", action="store_true",
                    help="with --deterministic, let nondeterministic kernels run instead of raising")
    a = ap.parse_args()
    if a.deterministic:
        from src.models import determinism
        print("determinism:", determinism.enable(seed=a.seed, strict=not a.loose), flush=True)
    main(a.steps, a.repeats, a.kl, a.data, a.seed, a.optim)
