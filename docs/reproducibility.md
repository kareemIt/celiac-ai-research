# Why two identical runs disagreed, and what fixed it (run 2026-10-28)

Reproduce: `python scripts/check_determinism.py --steps 64 --repeats 3 --kl 1 [--deterministic]`
and `python scripts/compare_repro.py`.
Tables: `results/determinism_check.csv`, `results/repro_check.csv`.

## The problem

Two runs of the *same* λ = 0 configuration on the same splits differed by 0.028 mean AUROC, with
per-seed swings to 0.18 ([`forgetting.md`](forgetting.md)). That is as large as the KL-anchor effect
the experiment was built to measure, so no conclusion from it could stand.

## Environment

| | |
|---|---|
| Python 3.12.16 | NumPy 2.5.4 |
| PyTorch 2.14.1+cu130 (CUDA 13.0, cuDNN 9.24) | transformers 5.21.0, peft 0.21.2, bitsandbytes 0.50.2 |
| RTX 5070 Ti, compute capability 12.0 | single GPU, single process, one venv |

TF32 matmul was already off. `torch.use_deterministic_algorithms` was off, cuDNN determinism off,
`CUBLAS_WORKSPACE_CONFIG` unset.

## The audit

| candidate source | how it is controlled | verdict |
|---|---|---|
| library versions | one venv, unchanged between the two runs | not a factor |
| train/valid/test split | `chem.scaffold_split(smiles, seed)`, pure function of the SMILES list | not a factor |
| example order, option-order shuffle | `np.random.default_rng(seed)`, re-created per seed | not a factor |
| LoRA initialization | `reset_lora_parameters` after `torch.manual_seed(seed)` | not a factor |
| LoRA dropout (p = 0.05) | torch RNG, seeded per seed | not a factor |
| epoch selection | validation Brier — deterministic *given* the predictions | downstream, not a source |
| 4-bit NF4 dequantization | — | **not a factor** (see stage 1) |
| 8-bit paged optimizer | — | **amplifier, not cause** |
| backward-pass kernel selection | nothing; PyTorch picks the fastest kernel | **the cause** |

### Stage 1 — inference is already bit-exact

Repeating the same forward pass over 24 molecules on the base model, with no flags set at all,
reproduces the log-odds **exactly** (max |Δ| = 0). So 4-bit dequantization, the attention and Gated
DeltaNet kernels, and the readout are all deterministic as shipped. Whatever is moving, it is not
inference.

### Stage 2 — training diverges at the first optimizer step

Gradients accumulate over 8 examples before `opt.step()` fires. Running 64 training examples three
times, with no flags:

| configuration | per-example loss | LoRA weights | log-odds | first example that differs |
|---|---|---|---|---|
| default, PagedAdamW8bit | 1.14e-1 | 2.71e-2 | 1.25e-1 | **8** |
| default, AdamW8bit | 1.42e-1 | 2.48e-2 | 1.25e-1 | **8** |
| default, fp32 AdamW | 1.18e-1 | 8.15e-4 | 1.88e-1 | **8** |
| `CUBLAS_WORKSPACE_CONFIG=:4096:8` only, PagedAdamW8bit | 1.67e-1 | 2.13e-2 | 1.25e-1 | **8** |
| **`use_deterministic_algorithms(True)` + workspace, PagedAdamW8bit** | **0** | **0** | **0** | — |
| **same, fp32 AdamW** | **0** | **0** | **0** | — |

Read the "first example that differs" column first. **Examples 0 to 7 are identical in every
configuration, and example 8 is the first forward after the first weight update.** The forward pass
and the loss are reproducible; the *accumulated gradient* is not. The divergence is in the backward
pass, in kernels PyTorch selects for speed and which reduce in a nondeterministic order.

Two things this rules out:

- **The optimizer is not the cause.** Plain fp32 AdamW diverges at exactly the same example. It is
  an *amplifier*: 8-bit state turns a 8.1e-4 weight disagreement into 2.5e-2, roughly thirty times
  larger, because a tiny difference crosses a quantization boundary and snaps to a different level.
- **The cuBLAS workspace alone is not enough.** Setting `CUBLAS_WORKSPACE_CONFIG` without
  `use_deterministic_algorithms` changes nothing.

### Nothing had to be waived

`enable(strict=True)` calls `torch.use_deterministic_algorithms(True, warn_only=False)`, which
*raises* on any operation lacking a deterministic implementation. **It never raised.** Every
operation in this model's forward and backward has a deterministic kernel available — the default
build simply does not use them. That is the cleanest possible outcome: determinism here costs
nothing in coverage, and no result depends on an operation we had to exempt.

## What is now enabled

`src/models/determinism.py`, applied by `--deterministic` on `run_jev_qlora.py`:

```
CUBLAS_WORKSPACE_CONFIG = :4096:8     PYTHONHASHSEED = <seed>
torch/numpy/random seeded             cudnn.deterministic = True, benchmark = False
tf32 off (matmul and cudnn)           use_deterministic_algorithms(True, warn_only=False)
```

The original **PagedAdamW8bit optimizer is kept**, since it is reproducible once the flags are set.
That matters for comparability: the new runs differ from the old ones in determinism alone, not in
the optimizer as well.

## End-to-end check

The short probe is not the experiment. Three complete runs of `run_jev_qlora.py` on seed 0 —
3 epochs, λ = 1, 631 molecules, the full pipeline including validation-based epoch selection and
checkpoint saving:

| run | best epoch | train_s | AUROC | AUPRC | MCC | ECE | Brier |
|---|---|---|---|---|---|---|---|
| repro1 | 2 | 1180 | 0.7862 | 0.8125 | 0.3395 | 0.2828 | 0.2623 |
| repro2 | 2 | 1172 | 0.7862 | 0.8125 | 0.3395 | 0.2828 | 0.2623 |
| repro3 | 2 | 1144 | 0.7862 | 0.8125 | 0.3395 | 0.2828 | 0.2623 |

| run vs repro1 | same best epoch | max abs Δ log-odds | max abs Δ epoch curve | max abs Δ adapter | Δ AUROC |
|---|---|---|---|---|---|
| repro2 | yes | 0 | 0 | 0 | 0 |
| repro3 | yes | 0 | 0 | 0 | 0 |

**Bit-exact**, including the saved LoRA weights compared tensor by tensor. Determinism costs about
10% in wall clock (1165 s/seed against 1068 s before), which is a cheap price for an experiment
whose previous conclusion was destroyed by its own noise.

## Consequences for the earlier results

- The λ = 0 versus λ = 1 comparison in [`forgetting.md`](forgetting.md) was run without any of this.
  Its stated noise floor (−0.028 ± 0.089 between identical configurations) was real, and the
  +0.031 effect it measured was correctly reported as indistinguishable from it.
- Under determinism, repeating a configuration adds no information: two runs are the same run. The
  remaining spread across seeds is split and initialization variance, which is what the paired
  per-seed test was always meant to isolate. **Replicates are therefore unnecessary**, and λ = 0
  versus λ = 1 becomes a clean comparison of the objective alone.
- Everything measured before today keeps its numbers; what changes is that new runs are repeatable.

## What the deterministic rerun found

The λ = 0 versus λ = 1 comparison was repeated under this regime, 10 seeds each, identical splits
(`--tag det_kl0` / `det_kl1`). With the noise floor at exactly zero, the anchors give **+0.038
overall AUROC, 8/10 seeds, p = 0.127** — still not significant, now for an honest reason: the
remaining uncertainty is test-set sampling, not the machine. The novel-chemistry effect is
inconclusive (+0.035, CI [−0.107, +0.225]) and the forgetting the experiment set out to prevent did
not reproduce. Full result in [`forgetting.md`](forgetting.md); KL experiments are stopped.
