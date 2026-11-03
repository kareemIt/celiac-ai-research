"""Chemprop / CheMeleon entrants, run inside the chemprop virtualenv by scripts/run_sota.py.

Three models share this file because they differ only in what the message-passing block starts from
and what extra features the readout sees:

  chemprop        D-MPNN trained from scratch on the molecular graph
  chemprop_rdkit  the same, with RDKit 2D descriptors concatenated onto the graph vector
  chemeleon       the message-passing block initialised from the CheMeleon foundation weights
                  (Zenodo 15460715, arXiv:2506.15792), then fine-tuned end to end

Descriptors are the array the parent computed with src.data.chem, so chemprop_rdkit and the
descriptor baselines see the same block. The parent also owns the split: this script never sees
anything but the three folds it is handed.

Each model gets 12 configurations, selected on validation AUROC, and the winner is scored once on
test -- the same rule every other entrant follows.
"""
import argparse
import itertools
import logging
import os
import warnings

import numpy as np
import pandas as pd

os.environ.setdefault("TQDM_DISABLE", "1")
warnings.filterwarnings("ignore")
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("chemprop").setLevel(logging.ERROR)

CHEMELEON_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"
MAX_EPOCHS = 50
PATIENCE = 10


def grid(model):
    """12 configurations each, over the axes that matter for that model.

    For chemprop the message-passing block is learned from scratch, so capacity and regularisation
    are what matter. For chemeleon the block arrives pretrained at d_h=2048, depth=6, and the thing
    that decides whether fine-tuning helps or destroys it is the learning rate -- chemprop's default
    max_lr of 1e-3 gave AUROC 0.789 with Brier 0.378, worse calibration than the trivial prior, so
    the rate is tuned rather than assumed. Equal budget, different axes, both recorded.
    """
    if model == "chemeleon":
        space = {"max_lr": [1e-4, 3e-4, 1e-3], "ffn_hidden_dim": [300, 1200],
                 "dropout": [0.0, 0.2]}
    else:
        space = {"depth": [3, 4], "d_h": [300, 600], "dropout": [0.0, 0.2, 0.35]}
    keys = list(space)
    return [dict(zip(keys, v)) for v in itertools.product(*(space[k] for k in keys))]


def chemeleon_block():
    """Download once to ~/.chemprop, then build a BondMessagePassing with the pretrained weights.
    Mirrors what `chemprop train --from-foundation CHEMELEON` does."""
    import torch
    from chemprop import nn
    cache = os.path.expanduser("~/.chemprop")
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, "chemeleon_mp.pt")
    if not os.path.exists(path):
        from urllib.request import urlretrieve
        urlretrieve(CHEMELEON_URL, path)
    ckpt = torch.load(path, weights_only=True)
    block = nn.BondMessagePassing(**ckpt["hyper_parameters"])
    block.load_state_dict(ckpt["state_dict"])
    return block, ckpt["hyper_parameters"]


def build_datasets(model, frames, descriptors):
    from chemprop import data, featurizers
    if model == "chemeleon":
        # CheMeleon was pretrained on the V2 atom featurisation and is invalid with any other
        feat = featurizers.SimpleMoleculeMolGraphFeaturizer(
            atom_featurizer=featurizers.MultiHotAtomFeaturizer.v2())
    else:
        feat = featurizers.SimpleMoleculeMolGraphFeaturizer()

    dsets = {}
    for part, frame in frames.items():
        x_d = descriptors[part] if model == "chemprop_rdkit" else None
        dps = [data.MoleculeDatapoint.from_smi(
                   s, np.array([float(y)]), x_d=None if x_d is None else x_d[i])
               for i, (s, y) in enumerate(zip(frame["smiles"], frame["y"]))]
        dsets[part] = data.MoleculeDataset(dps, feat)

    if model == "chemprop_rdkit":
        # fitted on train only; valid and test are transformed with the training statistics
        scaler = dsets["train"].normalize_inputs("X_d")
        dsets["valid"].normalize_inputs("X_d", scaler)
        dsets["test"].normalize_inputs("X_d", scaler)
    return dsets, feat


def fit_once(model, dsets, feat, cfg, seed):
    """Train one configuration and return (valid probabilities, test probabilities, n_params)."""
    import tempfile

    import lightning.pytorch as pl
    import torch
    from chemprop import data, models, nn
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    pl.seed_everything(seed, workers=True)
    x_d_dim = dsets["train"][0].x_d.shape[0] if dsets["train"][0].x_d is not None else 0

    lr = {}
    if model == "chemeleon":
        mp, _ = chemeleon_block()
        ffn = nn.BinaryClassificationFFN(input_dim=mp.output_dim + x_d_dim,
                                         hidden_dim=cfg["ffn_hidden_dim"], dropout=cfg["dropout"])
        lr = {"max_lr": cfg["max_lr"], "init_lr": cfg["max_lr"] / 10,
              "final_lr": cfg["max_lr"] / 10}
    else:
        mp = nn.BondMessagePassing(d_v=feat.atom_fdim, d_e=feat.bond_fdim, d_h=cfg["d_h"],
                                   depth=cfg["depth"], dropout=cfg["dropout"])
        ffn = nn.BinaryClassificationFFN(input_dim=mp.output_dim + x_d_dim, dropout=cfg["dropout"])

    # batch_norm stays off: it is opt-in in chemprop's own CLI, and forcing it on left CheMeleon
    # badly miscalibrated. drop_last matches what that CLI pairs with the flag.
    net = models.MPNN(mp, nn.MeanAggregation(), ffn, batch_norm=False,
                      metrics=[nn.BinaryAUROC()], **lr)
    n_params = sum(p.numel() for p in net.parameters())

    loaders = {p: data.build_dataloader(d, batch_size=64, shuffle=(p == "train"), seed=seed,
                                        num_workers=0, drop_last=False)
               for p, d in dsets.items()}
    # Lightning's EarlyStopping stops training but does NOT restore the best weights, so predicting
    # straight after fit() scores a model up to PATIENCE epochs past its best validation loss. That
    # left CheMeleon at Brier 0.378, worse than the trivial prior. Checkpoint the best epoch and
    # reload it before predicting.
    with tempfile.TemporaryDirectory() as ckpt_dir:
        best_ckpt = ModelCheckpoint(dirpath=ckpt_dir, monitor="val_loss", mode="min", save_top_k=1)
        trainer = pl.Trainer(max_epochs=MAX_EPOCHS, accelerator="auto", devices=1, logger=False,
                             enable_progress_bar=False, enable_model_summary=False,
                             deterministic=False,
                             callbacks=[best_ckpt,
                                        EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                                      mode="min")])
        trainer.fit(net, loaders["train"], loaders["valid"])
        net = models.MPNN.load_from_checkpoint(best_ckpt.best_model_path)
        out = {}
        for part in ("valid", "test"):
            p = torch.cat(trainer.predict(net, loaders[part])).flatten().float().cpu().numpy()
            out[part] = np.clip(p, 0.0, 1.0)
    return out["valid"], out["test"], n_params


def valid_score(y, p):
    from sklearn.metrics import brier_score_loss, roc_auc_score
    auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else 0.5
    return float(auc), float(brier_score_loss(y, np.clip(p, 0, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--budget", type=int, default=12)
    args = ap.parse_args()

    import torch

    frames = {p: pd.read_csv(f"{args.work}/{p}.csv") for p in ("train", "valid", "test")}
    f = np.load(f"{args.work}/features.npz")
    descriptors = {p: f["rdkit2d"][f[p]] for p in ("train", "valid", "test")}

    dsets, feat = build_datasets(args.model, frames, descriptors)
    y_valid = frames["valid"]["y"].to_numpy()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    configs = grid(args.model)[:args.budget]
    best, n_params = None, np.nan
    for cfg in configs:
        pv, pt, n_params = fit_once(args.model, dsets, feat, cfg, args.seed)
        score = valid_score(y_valid, pv)
        if best is None or score[0] > best[0][0]:
            best = (score, cfg, pt)

    (va_auc, va_brier), cfg, p_test = best
    pd.DataFrame({"p": p_test}).to_csv(f"{args.work}/pred.csv", index=False)
    pd.DataFrame([{
        "valid_auroc": va_auc, "valid_brier": va_brier,
        "config": ", ".join(f"{k}={v}" for k, v in cfg.items()),
        "n_configs": len(configs),
        "features": "graph+rdkit2d" if args.model == "chemprop_rdkit" else "graph",
        "n_params": n_params,
        "peak_vram_mib": (torch.cuda.max_memory_allocated() / 2**20
                          if torch.cuda.is_available() else np.nan),
    }]).to_csv(f"{args.work}/info.csv", index=False)


if __name__ == "__main__":
    main()
