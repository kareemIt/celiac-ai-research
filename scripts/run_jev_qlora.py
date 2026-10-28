"""Approach D: LLM-as-Jev fine-tuning of Qwen3.5-9B (4-bit QLoRA) for TG2 potency (IC50 <= 1 uM),
on the same 10 scaffold splits as run_baselines.py, compared seed-by-seed with fingerprint RF.

Same prompt/readout as run_jev_probe.py (SMILES + name variant). Training objective = the LLM-as-Jev
tree loss, which for two single-digit options reduces to cross-entropy over the logits of "1" vs "2"
right after the "Best answer: [" prefill. Option order is shuffled per example. Epoch chosen by valid Brier.

--kl adds the paper's KL anchors (src/models/jev_anchors.py) against the frozen base model, which is
this same model with the LoRA adapter switched off. --kl 0 is the paper's ablation and takes the
original cheap code path, so it reproduces the earlier run exactly. Valid and test log-odds are
written out after every epoch, including epoch 0 (adapter at B=0, i.e. the untouched base model), so
the forgetting curve can be read off afterwards without retraining.

Usage: python scripts/run_jev_qlora.py [--seeds 10] [--epochs 3] [--data combined] [--kl 1.0]
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
from src.models.jev_anchors import anchor_loss
from src.models.jev_readout import PREFILL, build_prompt
from src.models.qwen import chat_prompt, load_qwen_4bit

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


def main(seeds, epochs, data="pubchem", kl=0.0):
    tag = f"{data}_kl{kl:g}"
    out_csv = f"results/jev_qlora_tg2_{tag}.csv"
    preds_csv = f"results/jev_qlora_preds_{tag}.csv"
    epochs_csv = f"results/jev_qlora_epochs_{tag}.csv"

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

    # The anchors need positions the prompt alone does not reach, so training on kl > 0 runs on
    # prompt + the gold answer "k]". Window indices below are derived in docs/forgetting.md.
    ans = {k: tok(f"{k}]", add_special_tokens=False)["input_ids"] for k in (1, 2)}
    A = len(ans[1])
    assert len(ans[2]) == A and [ans[k][0] for k in (1, 2)] == digit_ids, "option digit is not its own token"
    tpl = chat_prompt(tok, "probe")
    n_prefill = (len(tok(tpl + PREFILL, add_special_tokens=False)["input_ids"])
                 - len(tok(tpl, add_special_tokens=False)["input_ids"]))
    KEEP = A + n_prefill + 1
    DECISION, CONTEXT = n_prefill, [0, A + n_prefill]  # reply start and the position after "]"

    def gold_of(j, potent_first):
        return 0 if (y[j] == 1) == potent_first else 1  # index of the correct option number

    E = None
    if kl > 0:
        def ext(prompt_ids, gold):
            tail = torch.tensor([ans[gold + 1]], device="cuda")
            return torch.cat([prompt_ids, tail], dim=1)
        E = [(ext(P[j][0], gold_of(j, True)), ext(P[j][1], gold_of(j, False))) for j in range(len(smiles))]

    ckpt_dir = f"checkpoints/jev_qlora_{tag}"
    os.makedirs(ckpt_dir, exist_ok=True)
    rows, preds, curve = [], [], []
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

        def snapshot(epoch):
            """Valid/test log-odds after `epoch` epochs. epoch 0 is the base model (adapter at B=0)."""
            z = {f: log_odds(model, [P[k] for k in idx], digit_ids) for f, idx in [("valid", va), ("test", te)]}
            for fold, idx in [("valid", va), ("test", te)]:
                curve.append(pd.DataFrame({"seed": seed, "epoch": epoch, "fold": fold,
                                           "smiles": [smiles[k] for k in idx], "y": y[idx],
                                           "logodds": z[fold]}))
            return z["valid"]

        model.train()
        snapshot(0)
        for epoch in range(epochs):
            order = rng.permutation(tr)
            for i, j in enumerate(order):
                pf = rng.random() < 0.5
                gold = gold_of(j, pf)
                target = torch.tensor([gold], device="cuda")
                if kl > 0:
                    lg = model(input_ids=E[j][0 if pf else 1], logits_to_keep=KEEP).logits[0]
                    with torch.no_grad(), model.disable_adapter():
                        ref = model(input_ids=E[j][0 if pf else 1], logits_to_keep=KEEP).logits[0]
                    loss = F.cross_entropy(lg[DECISION, digit_ids].float()[None], target)
                    loss = (loss + kl * anchor_loss(lg, ref, DECISION, CONTEXT, digit_ids)) / ACCUM
                else:
                    loss = F.cross_entropy(option_logits(model, P[j][0 if pf else 1], digit_ids)[None],
                                           target) / ACCUM
                loss.backward()
                if (i + 1) % ACCUM == 0 or i == len(order) - 1:
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
            z_va = snapshot(epoch + 1)
            if len(va) >= 5:
                vb = np.mean((y[va] - sigmoid(z_va)) ** 2)
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
        model.save_pretrained(f"{ckpt_dir}/seed{seed}")  # LoRA weights only, ~170 MB per seed
        for fold, idx, z in [("valid", va, z_va), ("test", te, z_te)]:
            preds.append(pd.DataFrame({"seed": seed, "fold": fold, "smiles": [smiles[k] for k in idx],
                                       "y": y[idx], "logodds": z}))
        pd.concat(preds).to_csv(preds_csv, index=False)
        pd.concat(curve).to_csv(epochs_csv, index=False)
        rows.append({"model": f"jev_qlora_kl{kl:g}", "seed": seed, "best_epoch": best[1] + 1,
                     "train_s": round(time.perf_counter() - t0), "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                     **metrics.classification(y[te], p_te)})
        print(f"  seed {seed}: test AUROC {rows[-1]['auroc']:.3f}, best epoch {best[1] + 1}", flush=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)  # incremental: partial results survive a crash
        del opt, best

    out = pd.DataFrame(rows)
    base = pd.read_csv(f"results/baselines_tg2{'' if data == 'pubchem' else '_' + data}.csv")
    base = base[(base.task == "classification") & (base.model.isin(["prior", "rf"])) & (base.seed < seeds)]
    zs = pd.read_csv("results/jev_probe_tg2.csv")
    zs = zs[(zs.model == "jev_smiles+name") & (zs.seed < seeds)].assign(model="jev_zeroshot")
    cols = ["model", "seed", "auroc", "auprc", "mcc", "ece", "brier"]
    print(f"\n## TG2 potency: LLM-as-Jev QLoRA (lambda={kl:g}) vs baselines ({seeds} seeds)\n"
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
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--data", default="pubchem", choices=["pubchem", "combined"])
    ap.add_argument("--kl", type=float, default=0.0, help="KL-anchor strength; the paper's default is 1.0")
    a = ap.parse_args()
    main(a.seeds, a.epochs, a.data, a.kl)
