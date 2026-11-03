"""State-of-the-art model comparison on TG2 potency, on the folds exported by export_splits.py.

Every entrant reads results/splits_tg2.csv rather than re-deriving a split, gets the same fixed
budget of hyperparameter configurations, is selected on its validation fold and is scored once on
test. Metrics come from src.evaluation.metrics, unchanged, so every number in this project is
computed by the same code.

Usage:
  python scripts/run_sota.py --model rf --split scaffold --seeds 10
  python scripts/run_sota.py --model all --split scaffold --seeds 20
  python scripts/run_sota.py --model rf --split paper_lopo          # all 11 folds

Writes results/sota_tg2.csv (one row per model/split/seed) and results/sota_preds.csv (one row per
test prediction, for the similarity-band and calibration analyses). Both are merged on
model+split+seed, so re-running one entrant replaces only its own rows.

Selection metric: validation AUROC, matching scripts/run_tuned_baselines.py. The protocol in
docs/state_of_the_art_comparison.md proposed validation Brier; matching the incumbent baseline was
judged more important than the original choice, since the baseline's published number is the thing
every entrant is compared against. Validation Brier is recorded alongside so the effect of that
choice can be checked later.
"""
import argparse
import itertools
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from src.data import chem
from src.evaluation import metrics

SPLITS = "results/splits_tg2.csv"
OUT = "results/sota_tg2.csv"
PREDS = "results/sota_preds.csv"
BUDGET = 12  # configurations per entrant per fold; the existing tuned RF grid is exactly 12


# ---------------------------------------------------------------- features

def morgan(smiles):
    """uint8, exactly as src.data.chem returns it. Casting to float32 changes where lbfgs stops and
    moved logistic regression's Brier by 3e-5 against the committed baseline, so it is not done."""
    return chem.morgan(smiles)


def rdkit2d(smiles):
    """RDKit's 2D descriptor block, NaN/inf-cleaned. The same features Chemprop-RDKit concatenates
    onto its learned graph vector, so the descriptor baselines and Chemprop see the same block."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    names = [n for n, _ in Descriptors.descList]
    calc = Chem.Descriptors.CalcMolDescriptors
    rows = [calc(Chem.MolFromSmiles(s)) for s in smiles]
    X = pd.DataFrame(rows)[names].to_numpy(dtype=np.float64)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def morgan_rdkit(smiles):
    return np.hstack([morgan(smiles).astype(np.float32), rdkit2d(smiles)])


FEATURES = {"morgan": morgan, "rdkit2d": rdkit2d, "morgan+rdkit2d": morgan_rdkit}


# ---------------------------------------------------------------- grids

def expand(params):
    keys = list(params)
    return [dict(zip(keys, v)) for v in itertools.product(*(params[k] for k in keys))]


def sklearn_grids():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    # identical to scripts/run_tuned_baselines.py, so a reproduction gate is meaningful
    return {
        "logreg": (LogisticRegression, "morgan",
                   {"C": [0.01, 0.1, 1.0, 10.0], "max_iter": [5000]}),
        "rf": (RandomForestClassifier, "morgan",
               {"n_estimators": [500], "max_depth": [None, 8], "n_jobs": [-1],
                "min_samples_leaf": [1, 2, 5], "max_features": ["sqrt", 0.1]}),
        "xgb": (XGBClassifier, "morgan",
                {"n_estimators": [300, 800], "learning_rate": [0.05, 0.1], "max_depth": [3, 6],
                 "subsample": [0.8], "colsample_bytree": [0.5], "tree_method": ["hist"]}),
    }


# ---------------------------------------------------------------- entrants

def fit_sklearn(name, fold, seed):
    """Tune over the model's grid on valid, refit nothing, score test once.

    Ties go to the first configuration, so grids are ordered simplest-first and tuning cannot
    quietly prefer a more complex model for no gain.
    """
    from sklearn.linear_model import LogisticRegression
    cls, feat, params = sklearn_grids()[name]
    X = FEATURES[feat](fold["smiles"])
    tr, va, te = fold["idx"]
    y = fold["y"]

    best = None
    for cfg in expand(params):
        m = cls(**cfg) if cls is LogisticRegression else cls(random_state=seed, **cfg)
        m.fit(X[tr], y[tr])
        pv = m.predict_proba(X[va])[:, 1]
        score = _valid_score(y[va], pv)
        if best is None or score[0] > best[0][0]:
            best = (score, cfg, m)
    (va_auc, va_brier), cfg, m = best
    return m.predict_proba(X[te])[:, 1], {"valid_auroc": va_auc, "valid_brier": va_brier,
                                          "config": _fmt(cfg), "n_configs": len(expand(params)),
                                          "features": feat, "n_params": np.nan}


CHECKPOINTS = {"chemberta_ft": "DeepChem/ChemBERTa-77M-MTR",
               "chemberta_frozen": "DeepChem/ChemBERTa-77M-MTR",
               "molformer_frozen": "ibm-research/MoLFormer-XL-both-10pct"}


def _embed(name, smiles, pooling):
    """Frozen sentence-style embedding of each SMILES from a pretrained chemical language model."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    ckpt = CHECKPOINTS[name]
    kw = {"trust_remote_code": True} if "MoLFormer" in ckpt else {}
    tok = AutoTokenizer.from_pretrained(ckpt, **kw)
    model = AutoModel.from_pretrained(ckpt, **kw).eval()
    dev = "cuda" if _cuda() else "cpu"
    model.to(dev)
    out = []
    with torch.no_grad():
        for i in range(0, len(smiles), 64):
            enc = tok(smiles[i:i + 64], return_tensors="pt", padding=True, truncation=True,
                      max_length=512).to(dev)
            h = model(**enc).last_hidden_state
            if pooling == "cls":
                out.append(h[:, 0].float().cpu())
            else:  # mean over real tokens only; padding would otherwise shrink short molecules
                m = enc["attention_mask"].unsqueeze(-1).float()
                out.append(((h * m).sum(1) / m.sum(1)).float().cpu())
    n_params = sum(p.numel() for p in model.parameters())
    del model
    if _cuda():
        import torch as _t
        _t.cuda.empty_cache()
    return torch.cat(out).numpy(), n_params


def fit_frozen_lm(name, fold, seed):
    """Frozen chemical-LM embedding + a linear head, which is the protocol Praski et al. used and
    the same shape as this project's existing frozen-Qwen probe."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    tr, va, te = fold["idx"]
    y = fold["y"]
    grid = [(pool, C) for pool in ("cls", "mean")
            for C in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)]
    cache, best, n_params = {}, None, np.nan
    for pool, C in grid:
        if pool not in cache:
            cache[pool], n_params = _embed(name, fold["smiles"], pool)
        Z = StandardScaler().fit(cache[pool][tr]).transform(cache[pool])
        m = LogisticRegression(C=C, max_iter=5000).fit(Z[tr], y[tr])
        score = _valid_score(y[va], m.predict_proba(Z[va])[:, 1])
        if best is None or score[0] > best[0][0]:
            best = (score, f"pooling={pool}, C={C}", m, Z)
    (va_auc, va_brier), cfg, m, Z = best
    return m.predict_proba(Z[te])[:, 1], {"valid_auroc": va_auc, "valid_brier": va_brier,
                                          "config": cfg, "n_configs": len(grid),
                                          "features": CHECKPOINTS[name], "n_params": n_params}


def fit_chemberta_ft(name, fold, seed):
    """Full fine-tune of ChemBERTa-2 with a 2-class head. 12 configurations: 4 learning rates x 3
    epoch budgets, each trained from the pretrained checkpoint, selected on validation AUROC."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ckpt = CHECKPOINTS[name]
    tok = AutoTokenizer.from_pretrained(ckpt)
    enc = tok(fold["smiles"], return_tensors="pt", padding=True, truncation=True, max_length=512)
    ids, mask = enc["input_ids"], enc["attention_mask"]
    y = torch.tensor(fold["y"], dtype=torch.long)
    tr, va, te = fold["idx"]
    dev = "cuda" if _cuda() else "cpu"

    def probs(model, idx):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(idx), 128):
                j = idx[i:i + 128]
                lo = model(input_ids=ids[j].to(dev), attention_mask=mask[j].to(dev)).logits
                out.append(lo.softmax(-1)[:, 1].float().cpu())
        return torch.cat(out).numpy()

    # 12 configurations = 4 learning rates x 3 epoch budgets, but a run to 40 epochs passes through
    # 10 and 20 on the way, so each rate is trained once and scored at each budget. Identical to 12
    # separate runs given the seed, at a third of the cost.
    budgets = (10, 20, 40)
    best, n_params = None, np.nan
    for lr in (1e-5, 3e-5, 1e-4, 3e-4):
        torch.manual_seed(seed)
        model = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=2).to(dev)
        n_params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        loader = DataLoader(TensorDataset(ids[tr], mask[tr], y[tr]), batch_size=32, shuffle=True,
                            generator=torch.Generator().manual_seed(seed))
        for epoch in range(1, max(budgets) + 1):
            model.train()
            for bi, bm, by in loader:
                opt.zero_grad()
                model(input_ids=bi.to(dev), attention_mask=bm.to(dev),
                      labels=by.to(dev)).loss.backward()
                opt.step()
            if epoch in budgets:
                score = _valid_score(y[va].numpy(), probs(model, va))
                if best is None or score[0] > best[0][0]:
                    best = (score, f"lr={lr}, epochs={epoch}", probs(model, te))
        del model, opt
        if _cuda():
            torch.cuda.empty_cache()
    (va_auc, va_brier), cfg, p_test = best
    return p_test, {"valid_auroc": va_auc, "valid_brier": va_brier, "config": cfg,
                    "n_configs": len(budgets) * 4, "features": ckpt, "n_params": n_params}


def fit_subprocess(name, fold, seed):
    """Entrants that need their own virtualenv (chemprop, chemeleon, chemberta, molformer).

    The fold is handed over as a CSV and the probabilities come back as a CSV, so the child process
    never sees a split it could re-derive differently.
    """
    env_py = os.environ.get(f"SOTA_{name.upper()}_PYTHON") or _env_python(name)
    if not env_py or not os.path.exists(env_py):
        raise RuntimeError(f"no interpreter for {name}; run scripts/setup_sota_envs.sh "
                           f"or set SOTA_{name.upper()}_PYTHON")
    work = f"results/_sota_tmp/{name}_{fold['split']}_{seed}"
    os.makedirs(work, exist_ok=True)
    tr, va, te = fold["idx"]
    frame = pd.DataFrame({"smiles": fold["smiles"], "y": fold["y"]})
    for part, idx in (("train", tr), ("valid", va), ("test", te)):
        frame.iloc[idx].to_csv(f"{work}/{part}.csv", index=False)
    # features come from src.data.chem here, so the child env cannot drift from the baselines'
    # featurization -- it only ever consumes arrays it was handed
    np.savez(f"{work}/features.npz", morgan=morgan(fold["smiles"]),
             rdkit2d=rdkit2d(fold["smiles"]), train=tr, valid=va, test=te)

    runner = {"chemprop": "_run_chemprop.py", "chemprop_rdkit": "_run_chemprop.py",
              "chemeleon": "_run_chemprop.py", "tabpfn": "_run_tabpfn.py"}[name]
    cmd = [env_py, os.path.join(os.path.dirname(__file__), runner),
           "--work", work, "--model", name, "--seed", str(seed), "--budget", str(BUDGET)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"{name} failed on {fold['split']} seed {seed}:\n"
                           f"{res.stdout[-3000:]}\n{res.stderr[-3000:]}")
    out = pd.read_csv(f"{work}/pred.csv")
    info = pd.read_csv(f"{work}/info.csv").iloc[0].to_dict()
    return out["p"].to_numpy(), info


def _env_python(name):
    """setup_sota_envs.sh builds these outside the repository, since the checkout is on /mnt/c."""
    base = {"chemprop": "chemprop", "chemprop_rdkit": "chemprop", "chemeleon": "chemprop",
            "tabpfn": "tabpfn"}[name]
    root = os.environ.get("SOTA_ENV_ROOT", os.path.expanduser("~/sota-envs"))
    for cand in (f"{root}/{base}/bin/python", f"envs/{base}/bin/python"):
        if os.path.exists(cand):
            return os.path.abspath(cand)
    return None


ENTRANTS = {
    "logreg": fit_sklearn, "rf": fit_sklearn, "xgb": fit_sklearn,
    "tabpfn": fit_subprocess,
    "chemprop": fit_subprocess, "chemprop_rdkit": fit_subprocess, "chemeleon": fit_subprocess,
    "chemberta_ft": fit_chemberta_ft,
    "chemberta_frozen": fit_frozen_lm, "molformer_frozen": fit_frozen_lm,
}
DEFAULT_ORDER = ["logreg", "rf", "xgb", "tabpfn", "chemprop", "chemprop_rdkit", "chemeleon",
                 "chemberta_frozen", "chemberta_ft"]


# ---------------------------------------------------------------- harness

def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _valid_score(y, p):
    """(AUROC, Brier) on the validation fold. A single-class valid fold gives no AUROC, so it
    falls back to 0.5 rather than crashing the whole sweep -- and that happens on LOPO folds."""
    from sklearn.metrics import brier_score_loss, roc_auc_score
    auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else 0.5
    return float(auc), float(brier_score_loss(y, p))


def _fmt(cfg):
    skip = ("n_jobs", "max_iter", "tree_method")
    return ", ".join(f"{k}={v}" for k, v in cfg.items() if k not in skip)


def load_folds(split, seeds, path=SPLITS):
    """Yield one dict per fold, carrying the molecules and the index arrays into them."""
    d = pd.read_csv(path)
    d = d[d.split_kind == split]
    if d.empty:
        raise SystemExit(f"no rows for split {split!r} in {path}")
    available = sorted(d.seed.unique())
    want = available if seeds is None else [s for s in available if s < seeds]
    for seed in want:
        g = d[d.seed == seed]
        order = {"train": 0, "valid": 1, "test": 2}
        g = g.assign(_o=g.fold.map(order)).sort_values(["_o"], kind="stable").reset_index(drop=True)
        idx = tuple(np.where(g.fold.to_numpy() == f)[0] for f in ("train", "valid", "test"))
        # int(), not the numpy int64 pandas gives: torch.manual_seed rejects it with
        # "an integer is required", which only shows up inside an entrant that seeds torch
        yield {"split": split, "seed": int(seed), "smiles": g["smiles"].tolist(),
               "y": g["y"].to_numpy(), "idx": idx,
               "paper": g["held_out_paper"].iloc[0] if "held_out_paper" in g else np.nan}


def peak_vram():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 2**20
    except Exception:
        pass
    return np.nan


def run(model, split, seeds):
    rows, preds = [], []
    for fold in load_folds(split, seeds):
        te = fold["idx"][2]
        y_te = fold["y"][te]
        try:
            import torch
            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        except Exception:
            pass
        t0 = time.perf_counter()
        p, info = ENTRANTS[model](model, fold, fold["seed"])
        secs = time.perf_counter() - t0
        p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)

        rows.append({"model": model, "split": split, "seed": fold["seed"],
                     **metrics.classification(y_te, p),
                     "seconds": round(secs, 2), "peak_vram_mib": info.get("peak_vram_mib",
                                                                          peak_vram()),
                     "valid_auroc": info.get("valid_auroc"), "valid_brier": info.get("valid_brier"),
                     "n_configs": info.get("n_configs"), "features": info.get("features"),
                     "n_params": info.get("n_params"), "config": info.get("config"),
                     "held_out_paper": fold["paper"], "n_test": len(te), "n_train": len(fold["idx"][0])})
        preds.append(pd.DataFrame({"model": model, "split": split, "seed": fold["seed"],
                                   "smiles": [fold["smiles"][i] for i in te], "y": y_te, "p": p}))
        print(f"  {model:<17} {split:<13} seed {fold['seed']:>2}: "
              f"AUROC {rows[-1]['auroc']:.3f}  Brier {rows[-1]['brier']:.3f}  {secs:.1f}s",
              flush=True)
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


def merge(path, new, keys):
    """Replace this model/split/seed's rows and keep every other row already in the file."""
    if os.path.exists(path):
        old = pd.read_csv(path)
        drop = old.merge(new[keys].drop_duplicates(), on=keys, how="left", indicator=True)
        old = old[drop["_merge"].to_numpy() == "left_only"]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all")
    ap.add_argument("--split", default="scaffold",
                    choices=["scaffold", "random", "scaffold_pub", "paper_lopo"])
    ap.add_argument("--seeds", type=int, default=None, help="use seeds < N; default all exported")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    models = DEFAULT_ORDER if args.model == "all" else args.model.split(",")
    unknown = [m for m in models if m not in ENTRANTS]
    if unknown:
        raise SystemExit(f"unknown model(s) {unknown}; choose from {sorted(ENTRANTS)}")

    failed = {}
    for model in models:
        try:
            rows, preds = run(model, args.split, args.seeds)
        except Exception as e:  # one entrant's environment must not take the others down
            print(f"  {model}: FAILED -- {e}", file=sys.stderr, flush=True)
            failed[model] = str(e).splitlines()[0]
            continue
        merge(args.out, rows, ["model", "split", "seed"])
        merge(PREDS, preds, ["model", "split", "seed"])
        print(f"{model}: AUROC {rows.auroc.mean():.3f} +- {rows.auroc.std():.3f} "
              f"over {len(rows)} folds\n", flush=True)

    if failed:
        print("FAILED: " + "; ".join(f"{k} ({v})" for k, v in failed.items()), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
