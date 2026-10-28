"""Cache frozen Qwen embeddings for TG2 molecules and GDSC2 drugs into data/processed/.

GDSC2 drugs are embedded with three prompts to separate what SMILES vs. the drug's name contributes.
"""
import argparse
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import gdsc2, tg2
from src.training import embed

PROMPTS = {
    "qwen_gdsc2_drugs": "Molecule SMILES: {smiles}",
    "qwen_gdsc2_drugs_name": "Drug: {name}. SMILES: {smiles}",
    "qwen_gdsc2_drugs_nameonly": "Drug: {name}",
}

# batch=1 by default: with padding, GPU kernels pick different reduction orders per batch shape, so the
# same molecule embedded in a different batch shifts by ~1e-3 and moves downstream metrics by ~0.03.
ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=1, help="1 = deterministic (no padding); >1 is faster but varies")
BATCH = ap.parse_args().batch

os.makedirs("data/processed", exist_ok=True)
t = tg2.regression(tg2.load_raw())["smiles"].tolist()  # classification uses the same 305 molecules
drugs = gdsc2.load().drop_duplicates("Drug_ID")
names, smiles = drugs["Drug_ID"].tolist(), drugs["Drug"].tolist()

texts = [f"Molecule SMILES: {s}" for s in t]
for p in PROMPTS.values():
    texts += [p.format(name=n, smiles=s) for n, s in zip(names, smiles)]

t0 = time.perf_counter()
e = embed.embed_texts(texts, batch=BATCH)
embed.save("data/processed/qwen_tg2.npz", t, {l: v[:len(t)] for l, v in e.items()})
for i, f in enumerate(PROMPTS):
    a = len(t) + i * len(names)
    embed.save(f"data/processed/{f}.npz", names, {l: v[a:a + len(names)] for l, v in e.items()})
print(f"Embedded {len(t)} TG2 molecules + {len(names)} GDSC2 drugs x {len(PROMPTS)} prompts, "
      f"batch={BATCH}, {time.perf_counter() - t0:.0f}s")
