"""Approach D: LLM-as-Jev fine-tuning of Qwen3.5-9B (4-bit QLoRA) for TG2 potency (IC50 <= 1 uM),
on the same 10 scaffold splits as run_baselines.py, compared seed-by-seed with fingerprint RF.

Same prompt/readout as run_jev_probe.py (SMILES + name variant). Training objective = the LLM-as-Jev
tree loss, which for two single-digit options reduces to cross-entropy over the logits of "1" vs "2"
right after the "Best answer: [" prefill. Option order is shuffled per example. Epoch chosen by valid Brier.
shortcut: no KL anchors (paper's λ=0 ablation); add them if the adapter is reused for general chat.

Usage: python scripts/run_jev_qlora.py [--seeds 10] [--epochs 4]
"""
import argparse
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import bitsandbytes as bnb
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer

from scripts.run_baselines import summarize
from scripts.run_jev_probe import POTENT, QUESTION, WEAK, sigmoid
from src.data import chem, tg2
from src.evaluation import metrics
from src.models.jev_readout import build_prompt
from src.models.qwen import load_qwen_4bit

OUT = "results/jev_qlora_tg2.csv"
PREDS = "results/jev_qlora_preds.csv"
ACCUM, LR = 8, 1e-4  # LoRA lr from LLM-as-Jev


def option_logits(model, ids, digit_ids):
    """Logits of option numbers "1","2" at the token after the prefill. Only the last position is computed."""
    return model(input_ids=ids, logits_to_keep=1).logits[0, -1, digit_ids].float()


@torch.no_grad()
def log_odds(model, prompts, digit_ids):
    """log P(potent) - log P(not), averaged over both option orders (as in the zero-shot probe).
    Returned unsquashed so post-hoc calibration can rescale it later."""
    model.eval()
    lo = []
    for pf, ps in prompts:  # potent listed first / second
        a, b = option_logits(model, pf, digit_ids), option_logits(model, ps, digit_ids)
        lo.append(((a[0] - a[1]) + (b[1] - b[0])).item() / 2)
    model.train()
    return np.array(lo)


def main(seeds, epochs, data="pubchem"):
    d = tg2.load_raw()
    table = tg2.combined("classification") if data == "combined" else tg2.classification(d)
    names = d.groupby("smiles")["Compound_Name"].first().to_dict()  # ChEMBL rows have no name
    smiles, y = table["smiles"].tolist(), table["y"].to_numpy()

    model, tok = load_qwen_4bit()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                             target_modules="all-linear", task_type="CAUSAL_LM"))
    digit_ids = tok.convert_tokens_to_ids(["1", "2"])

    def ids(s, potent_first):
        state = f"Compound SMILES: {s}" + (f"\nCompound name: {names[s]}" if isinstance(names.get(s), str) else "")
        opts = [POTENT, WEAK] if potent_first else [WEAK, POTENT]
        return tok(build_prompt(tok, state, QUESTION, opts), add_special_tokens=False,
                   return_tensors="pt")["input_ids"].cuda()

    P = [(ids(s, True), ids(s, False)) for s in smiles]
    os.makedirs(f"checkpoints/jev_qlora_{data}", exist_ok=True)
    rows, preds = [], []
    for seed in range(seeds):
        tr, va, te = chem.scaffold_split(smiles, seed=seed)
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        for m in model.modules():  # fresh adapter per seed: B=0 so training starts from the base model
            if isinstance(m, LoraLayer):
                m.reset_lora_parameters("default", True)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = bnb.optim.PagedAdamW8bit(params, lr=LR)
        torch.cuda.reset_peak_memory_stats()
        t0, best = time.perf_counter(), (np.inf, epochs - 1, None)
        model.train()
        for epoch in range(epochs):
            order = rng.permutation(tr)
            for i, j in enumerate(order):
                pf = rng.random() < 0.5
                gold = 0 if (y[j] == 1) == pf else 1  # index of the correct option number
                loss = F.cross_entropy(option_logits(model, P[j][0 if pf else 1], digit_ids)[None],
                                       torch.tensor([gold], device="cuda")) / ACCUM
                loss.backward()
                if (i + 1) % ACCUM == 0 or i == len(order) - 1:
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
            if len(va) >= 5:
                vb = np.mean((y[va] - sigmoid(log_odds(model, [P[k] for k in va], digit_ids))) ** 2)
                if vb < best[0]:
                    best = (vb, epoch, {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad})
                print(f"  seed {seed} epoch {epoch + 1}: valid Brier {vb:.3f} ({time.perf_counter() - t0:.0f}s)", flush=True)
        if best[2] is not None:  # restore best epoch's adapter
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if n in best[2]:
                        p.copy_(best[2][n])
        z_va, z_te = (log_odds(model, [P[k] for k in idx], digit_ids) for idx in (va, te))
        p_te = sigmoid(z_te)
        adapter = f"checkpoints/jev_qlora_{data}/seed{seed}"
        model.save_pretrained(adapter)  # LoRA weights only, ~170 MB per seed
        for fold, idx, z in [("valid", va, z_va), ("test", te, z_te)]:
            preds.append(pd.DataFrame({"seed": seed, "fold": fold, "smiles": [smiles[k] for k in idx],
                                       "y": y[idx], "logodds": z}))
        pd.concat(preds).to_csv(PREDS.replace(".csv", f"_{data}.csv"), index=False)
        rows.append({"model": "jev_qlora", "seed": seed, "best_epoch": best[1] + 1,
                     "train_s": round(time.perf_counter() - t0), "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                     **metrics.classification(y[te], p_te)})
        print(f"  seed {seed}: test AUROC {rows[-1]['auroc']:.3f}, best epoch {best[1] + 1}", flush=True)
        pd.DataFrame(rows).to_csv(OUT.replace(".csv", f"_{data}.csv") if data != "pubchem" else OUT,
                                  index=False)  # incremental: partial results survive a crash
        del opt, best

    out = pd.DataFrame(rows)
    base = pd.read_csv(f"results/baselines_tg2{'' if data == 'pubchem' else '_' + data}.csv")
    base = base[(base.task == "classification") & (base.model.isin(["prior", "rf"])) & (base.seed < seeds)]
    zs = pd.read_csv("results/jev_probe_tg2.csv")
    zs = zs[(zs.model == "jev_smiles+name") & (zs.seed < seeds)].assign(model="jev_zeroshot")
    cols = ["model", "seed", "auroc", "auprc", "mcc", "ece", "brier"]
    print(f"\n## TG2 potency: LLM-as-Jev QLoRA vs baselines ({seeds} seeds)\n"
          f"{summarize(pd.concat([base, zs, out])[cols], ['model']).to_string()}")
    print(f"best epochs {out.best_epoch.value_counts().to_dict()}, train {out.train_s.mean():.0f}s/seed, "
          f"peak VRAM {out.peak_gib.max():.2f} GiB")
    for ref_name, ref in [("RF", base[base.model == "rf"]), ("zero-shot", zs)]:
        m = out.merge(ref, on="seed", suffixes=("", "_ref"))
        parts = []
        for metric, hb in [("auroc", True), ("mcc", True), ("ece", False), ("brier", False)]:
            dd = (m[metric] - m[f"{metric}_ref"]).dropna()
            parts.append(f"Δ{metric} {dd.mean():+.3f} ± {dd.std():.3f} ({((dd > 0) if hb else (dd < 0)).sum()}/{len(dd)} better)")
        print(f"  QLoRA vs {ref_name}: " + ", ".join(parts))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--data", default="pubchem", choices=["pubchem", "combined"])
    a = ap.parse_args()
    main(a.seeds, a.epochs, a.data)
