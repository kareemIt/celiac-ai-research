"""Frozen Qwen features for molecules: mean-pooled hidden states from several layers."""
import numpy as np
import torch

from src.models.qwen import load_qwen_4bit

LAYERS = (8, 16, 24, 32)  # 32 = last layer; intermediate layers are often better features for probing


@torch.no_grad()
def embed_texts(texts, layers=LAYERS, batch=16):
    model, tok = load_qwen_4bit()
    model.eval()
    tok.padding_side = "right"  # causal model: right padding leaves real-token states untouched
    out = {l: [] for l in layers}
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors="pt", padding=True).to("cuda")
        hs = model.model(**enc, output_hidden_states=True).hidden_states  # skip lm_head (248k-vocab logits)
        mask = enc["attention_mask"].unsqueeze(-1).to(hs[0].dtype)
        for l in layers:
            out[l].append(((hs[l] * mask).sum(1) / mask.sum(1)).float().cpu().numpy())
    return {l: np.concatenate(v) for l, v in out.items()}


def save(path, keys, embs):
    np.savez(path, keys=np.asarray(keys, dtype=str), **{f"layer{l}": v for l, v in embs.items()})


def load(path):
    z = np.load(path)
    return z["keys"], {int(k[5:]): z[k] for k in z.files if k.startswith("layer")}
