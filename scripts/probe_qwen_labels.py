"""Direct label-leakage probe: does base Qwen3.5-9B *recall* these molecules' published TG2 IC50s?

The QLoRA result in docs/jev_qlora.md (AUROC 0.850) is only evidence of structure-activity learning
if the base model cannot already retrieve the answers from its pretraining corpus. The TG2 papers
are public, so the question is empirical. This asks the base model -- no LoRA, no fine-tuning -- for
each molecule's IC50 as a number and correlates the answers with the published truth.

Three prompt arms, every molecule in all three, one pass:

  smiles           canonical SMILES only
  smiles+name      plus the compound name, the retrieval key run_jev_qlora.py actually sends for
                   398 of 631 molecules (see results/sota_leakage.csv)
  smiles_shuffled  a valid but NON-canonical SMILES of the same molecule, asserted to round-trip

The third arm is the discriminator. A model reasoning from structure is indifferent to how the
structure is written; a model keyed on the exact string as it appears in PubChem/ChEMBL/papers is
not. So `smiles` >> `smiles_shuffled` is a retrieval signature, while `smiles` ~= `smiles_shuffled`
points at structural inference (or at nothing).

Answers are elicited as free text, with UNKNOWN explicitly permitted: forcing a guess would
manufacture a correlation out of general reasoning, and the abstention rate is itself a finding.
Raw generations are cached so the analysis is re-runnable, and auditable, without the GPU.

Reproduce: python scripts/probe_qwen_labels.py            # ~45 min on one RTX 5070 Ti
           python scripts/probe_qwen_labels.py --analyse  # tables only, from the cache
Raw: results/qwen_label_probe_raw.csv   Tables: results/qwen_label_probe.csv
"""
import argparse
import os
import re
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from src.data import tg2

RAW = "results/qwen_label_probe_raw.csv"
OUT = "results/qwen_label_probe.csv"
ARMS = ("smiles", "smiles+name", "smiles_shuffled")
MAX_NEW = 24

QUESTION = (
    "What is the half-maximal inhibitory concentration (IC50) of this compound against human "
    "tissue transglutaminase 2 (TG2)?\n"
    "If you know the published value, answer with the number and its unit, for example "
    "'IC50 = 250 nM'.\n"
    "If you do not know it, answer exactly UNKNOWN. Do not guess or explain."
)

# nanomolar per unit prefix; the model is free to answer in any of them
UNITS = {"p": 1e-3, "n": 1.0, "u": 1e3, "μ": 1e3, "µ": 1e3, "m": 1e6, "": 1e9}
# A whole numeric token, optionally in scientific or "x 10^k" form, followed by a unit. The
# lookbehind matters: without it "1.4e+04 nM" matches its own tail as "04 nM" and silently reports
# 4 nM instead of refusing to parse.
NUMBER = re.compile(
    r"(?<![\w.])"                                  # not starting mid-number
    r"(\d*\.?\d+(?:[eE][-+]?\d+)?)"                # 250, 0.25, 1.5e-8
    r"\s*(?:[x×*]\s*10\s*\^?\s*([-+]?\d+))?"       # optional  x 10^-8
    r"\s*(p|n|u|μ|µ|m)?M\b",
    re.IGNORECASE)


def shuffled_smiles(smiles):
    """A valid non-canonical writing of the same molecule, deterministic per molecule."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    alt = Chem.MolToSmiles(mol, rootedAtAtom=mol.GetNumAtoms() // 2, canonical=False)
    # the control is only a control if it is the same molecule
    assert Chem.MolToSmiles(Chem.MolFromSmiles(alt)) == Chem.MolToSmiles(mol), smiles
    return alt


def parse_nm(text):
    """The first concentration in the answer, in nM. None for UNKNOWN or no parsable value."""
    if text is None:
        return None
    if "unknown" in text.lower():
        return None
    m = NUMBER.search(re.sub(r"(?<=\d),(?=\d\d\d)", "", text))  # 1,000 nM -> 1000 nM
    if not m:
        return None
    value, exponent, prefix = float(m.group(1)), m.group(2), (m.group(3) or "").lower()
    if value <= 0:
        return None
    nm = value * 10 ** int(exponent or 0) * UNITS[prefix]
    # 10 pM to 10 mM. Outside that an answer is not a plausible enzyme IC50, so treat it as
    # unparsable rather than let a stray "2024" or a molar-scale answer drive the correlation
    return nm if 1e-2 <= nm <= 1e7 else None


def generate(table, names):
    import torch

    from src.models.determinism import enable
    from src.models.qwen import chat_prompt, load_qwen_4bit

    # strict=False: 4-bit NF4 dequantization is outside torch's determinism contract anyway
    # (src/models/determinism.py documents this), and strict would raise on it
    settings = enable(seed=0, strict=False)
    print(f"determinism: {settings}", flush=True)

    model, tok = load_qwen_4bit()
    model.eval()
    alt = {s: shuffled_smiles(s) for s in table["smiles"]}

    rows, t0 = [], time.perf_counter()
    for arm in ARMS:
        for i, s in enumerate(table["smiles"]):
            state = f"Compound SMILES: {alt[s] if arm == 'smiles_shuffled' else s}"
            if arm == "smiles+name" and isinstance(names.get(s), str):
                state += f"\nCompound name: {names[s]}"
            prompt = chat_prompt(tok, f"{state}\n\n{QUESTION}")
            ids = tok(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
            with torch.no_grad():
                gen = model.generate(**ids, max_new_tokens=MAX_NEW, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            text = tok.decode(gen[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            rows.append({"arm": arm, "smiles": s, "answer": text.replace("\n", " ")})
            if (i + 1) % 100 == 0:
                print(f"  {arm}: {i + 1}/{len(table)}  {time.perf_counter() - t0:.0f}s", flush=True)
        print(f"  {arm} done: {time.perf_counter() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(RAW, index=False)
    print(f"-> {RAW}")


def boot_spearman(x, y, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    if len(x) < 8:
        return np.nan, np.nan
    vals = [stats.spearmanr(x[i], y[i]).statistic
            for i in (rng.integers(0, len(x), len(x)) for _ in range(n))]
    return tuple(np.nanpercentile(vals, [2.5, 97.5]))


def analyse():
    raw = pd.read_csv(RAW)
    truth = tg2.combined("regression").set_index("smiles")["y"]          # pIC50, exact values only
    label = tg2.combined("classification").set_index("smiles")["y"]      # potent at 1 uM
    raw["nm"] = raw["answer"].map(parse_nm)
    raw["pic50_pred"] = 9 - np.log10(raw["nm"])                          # nM -> pIC50
    raw["pic50_true"] = raw["smiles"].map(truth)
    raw["y"] = raw["smiles"].map(label)

    rows = []
    for arm, g in raw.groupby("arm", sort=False):
        answered = g[g["nm"].notna()]
        both = answered[answered["pic50_true"].notna()]
        rho = stats.spearmanr(both["pic50_pred"], both["pic50_true"]) if len(both) >= 8 else None
        lo, hi = boot_spearman(both["pic50_pred"].to_numpy(), both["pic50_true"].to_numpy())
        # a memorized value should land very close to the published one
        near = (both["pic50_pred"] - both["pic50_true"]).abs() < 0.1
        # chance rate for that, from the same predictions paired to shuffled truths
        rng = np.random.default_rng(0)
        pred, true = both["pic50_pred"].to_numpy(), both["pic50_true"].to_numpy()
        chance = np.mean([np.mean(np.abs(pred - rng.permutation(true)) < 0.1)
                          for _ in range(200)]) if len(both) >= 8 else np.nan
        # the 1 uM decision the project actually makes, ranked by the model's own number so it is
        # comparable with the zero-shot AUROC of 0.648 in docs/jev_probe.md
        cls = answered[answered["y"].notna()]
        auroc = roc_auc_score(cls["y"].to_numpy(), cls["pic50_pred"].to_numpy()) \
            if cls["y"].nunique() == 2 else np.nan
        rows.append({
            "arm": arm, "n": len(g), "answered": len(answered),
            "answered_frac": len(answered) / len(g),
            "n_vs_truth": len(both),
            "spearman": rho.statistic if rho else np.nan,
            "spearman_p": rho.pvalue if rho else np.nan,
            "spearman_lo": lo, "spearman_hi": hi,
            "pred_pic50_median": answered["pic50_pred"].median(),
            "pred_pic50_iqr": answered["pic50_pred"].quantile(0.75) - answered["pic50_pred"].quantile(0.25),
            "distinct_answers": answered["nm"].nunique(),
            "near_exact_frac": near.mean() if len(both) else np.nan,
            "near_exact_chance": chance,
            "auroc_1um": auroc,
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)

    pd.set_option("display.width", 200)
    print("\n## Can base Qwen3.5-9B recall these IC50s?\n")
    print(out[["arm", "n", "answered", "answered_frac", "n_vs_truth", "spearman",
               "spearman_lo", "spearman_hi", "spearman_p"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n## Answer distribution, and whether recalled values match published ones\n")
    print(out[["arm", "distinct_answers", "pred_pic50_median", "pred_pic50_iqr",
               "near_exact_frac", "near_exact_chance", "auroc_1um"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n## Most common answers per arm\n")
    for arm, g in raw.groupby("arm", sort=False):
        top = g["answer"].str.slice(0, 48).value_counts().head(3)
        print(f"  {arm}:")
        for text, n in top.items():
            print(f"    {n:4d}x  {text!r}")

    # the retrieval signature: canonical string helps, shuffled string does not
    s, sh = out.set_index("arm").loc["smiles"], out.set_index("arm").loc["smiles_shuffled"]
    print(f"\n  canonical vs shuffled SMILES: Δspearman {s.spearman - sh.spearman:+.3f}, "
          f"Δanswered {s.answered_frac - sh.answered_frac:+.3f}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyse", action="store_true", help="skip generation, use the cached answers")
    ap.add_argument("--limit", type=int, help="smoke test: first N molecules only")
    a = ap.parse_args()

    if not a.analyse:
        table = tg2.combined("classification")
        if a.limit:
            table = table.head(a.limit)
        raw_p = tg2.load_raw()
        generate(table, raw_p.groupby("smiles")["Compound_Name"].first().to_dict())
    analyse()
