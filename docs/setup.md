# Environment setup and validation (measured 2026-10-26)

## Hardware (verified)
| | Measured |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti, 16,303 MiB (15.89 GiB usable to PyTorch), compute capability 12.0 (Blackwell) |
| Driver | 617.42 (Windows), CUDA driver API up to 13.4 |
| CPU | AMD Ryzen 7 7800X3D (8C/16T) |
| RAM | 31.7 GB, of which WSL gets 24 GB + 16 GB swap (`C:\Users\karee\.wslconfig`) |
| Board | ASUS B650E MAX GAMING WIFI |
| Disks | C: WD 1 TB SSD (WSL disk lives here). D: 1 TB **HDD**. E: 119 GB SSD |

## OS and WSL
- Windows 11 Home build 26300. WSL 2.1.5, kernel 5.15.146.1, systemd on.
- Distro **Ubuntu 22.04.5 LTS** (pre-existing, reused). GPU via WSL passthrough (`/usr/lib/wsl/lib/libcuda.so`). **No Linux NVIDIA driver and no CUDA toolkit installed**, since PyTorch wheels bundle the CUDA runtime.
- Added: `build-essential` (gcc 11.4, needed by Triton) and `uv` 0.13.6 (user-level).

## Python stack (`requirements.txt` is the full pin list)
Python 3.12.16 venv at `.venv`. Key versions: torch **2.14.1+cu130** (CUDA 13.0), triton 3.8.0, transformers 5.21.0, accelerate 1.15.0, peft 0.21.2, bitsandbytes 0.50.2, flash-linear-attention 0.6.0, datasets 5.1.0, rdkit 2026.9.1, xgboost 3.4.1, scikit-learn 1.9.1, numpy 2.5.4, pandas 3.1.0, PyTDC 0.3.6, nvidia-ml-py 13.615.71.

Rebuild: `bash scripts/setup_env.sh`.

## Model
| | |
|---|---|
| Checkpoint | `Qwen/Qwen3.5-9B`, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| License | Apache-2.0 |
| Size | 9.65B params total (multimodal). **8.95B text model** loaded via `AutoModelForCausalLM` (vision tower skipped) |
| Download | 19.3 GB bf16 safetensors (4 shards, SHA-256 verified) in `~/.cache/huggingface/hub` |
| Architecture | 32 layers: 24 Gated DeltaNet (linear attention) + 8 full attention, hidden size 4096, vocab 248,320 |
| Quantization | bitsandbytes **4-bit NF4**, double quantization, bf16 compute. Embeddings and lm_head stay bf16 |

## Measured results
**`scripts/check_gpu.py`**: PASS. GPU matmul matches CPU. A 200-step MLP training run took the loss from 7.75 to ~0. Peak 80 MiB.

**`scripts/test_qwen.py`**: PASS (offline, `HF_HUB_OFFLINE=1`)
- Load 9.9 s (warm disk cache; 34.9 s cold). VRAM after load **7.12 GiB**, peak during generation **7.21 GiB**.
- First generate call 14.6 s (Triton JIT/autotune, once per process). Steady state **21.1 tok/s** (80 tokens in 3.8 s, greedy).

**`scripts/test_qlora.py`**: PASS (synthetic data, infrastructure only)
- Frozen 4-bit backbone + LoRA r=16, α=32, dropout 0.05 on all linear layers: **43.3M trainable (0.48%)**.
- Gradient checkpointing, batch 1 × accumulation 4, max length 64, PagedAdamW8bit, lr 2e-4, 30 steps.
- Loss 2.89 → 0.0001. LoRA weights changed (max |ΔW| 4.5e-3). **Peak VRAM 7.66 GiB.** 50.9 s (1.70 s/step).
- Adapter saved to `checkpoints/qlora_smoke` and reloaded bit-identical. Held-out synthetic accuracy: base 0% → trained 100% → reloaded 100%.

That leaves ~8 GiB of VRAM headroom, so longer sequences (512–1024 tokens) and larger batches are realistic next.

## Commands
```bash
cd ~/projects/celiac-ai-research && source .venv/bin/activate
python scripts/check_gpu.py
python scripts/test_qwen.py
python scripts/test_qlora.py       # log: results/qlora_smoke.log
bash scripts/download_data.sh && pytest
```

## Known issues / notes
- `causal_conv1d` is not installed, so transformers uses its slower PyTorch fallback for the short conv in DeltaNet layers. It needs a CUDA-toolkit build. Speed is acceptable without it, so it was deferred.
- Do not call `peft.prepare_model_for_kbit_training` on this model. It upcasts the 248k-vocab embedding and lm_head to fp32 (~8 GB extra). `test_qlora.py` does the needed steps manually.
- From Windows PowerShell, prefer `wsl -d Ubuntu-22.04 --cd <dir> -e <cmd>` or run scripts. `wsl -- '<cmd with $VARS or quotes>'` gets re-expanded.
- WSL 2.1.5 is from early 2024. It works, so `wsl --update` was not run.
