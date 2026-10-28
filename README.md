# celiac-ai-research

Local Qwen (Qwen3.5-9B, 4-bit QLoRA) research environment for drug-response prediction (GDSC2) and TG2/celiac inhibitor modelling. Everything runs on the local RTX 5070 Ti. No hosted model APIs.

## Quick start (inside WSL Ubuntu)
```bash
cd ~/projects/celiac-ai-research
bash scripts/setup_env.sh            # venv + pinned packages (requirements.txt)
source .venv/bin/activate
python scripts/check_gpu.py          # GPU + real training step
hf download Qwen/Qwen3.5-9B          # ~19 GB into ~/.cache/huggingface
python scripts/test_qwen.py          # 4-bit load + local inference
python scripts/test_qlora.py         # QLoRA smoke test (synthetic data)
bash scripts/download_data.sh        # GDSC2 (TDC) + TG2 CSVs (PubChem export)
python scripts/fetch_chembl_tg2.py   # human TG2 activities from ChEMBL (CHEMBL2730)
python scripts/check_sources.py      # do ChEMBL and PubChem agree? see docs/data.md
python scripts/audit_splits.py       # scaffold-split leakage audit, see docs/data.md
pytest                               # data + metrics + calibration tests
python scripts/run_baselines.py --data combined        # fingerprint baselines, see docs/baselines.md
python scripts/run_tuned_baselines.py --data combined  # same, tuned on each seed's valid fold
python scripts/run_calibration.py --data combined      # temperature/Platt/isotonic, see docs/calibration.md
python scripts/embed_molecules.py    # cache frozen Qwen embeddings
python scripts/run_qwen_probe.py     # frozen Qwen + heads vs baselines, see docs/qwen_probe.md
python scripts/run_jev_probe.py      # zero-shot Jev-style option readout on TG2, see docs/jev_probe.md
python scripts/run_jev_qlora.py --data combined   # LLM-as-Jev QLoRA fine-tune, see docs/jev_qlora.md
python scripts/calibrate_qlora.py    # calibrate the fine-tuned model from its saved log-odds
python scripts/compare.py A.csv modelA B.csv modelB   # paired significance test between two runs
```

**Start with [`docs/summary.md`](docs/summary.md)** — current state, what's established, open questions.

## Layout
- `src/models/qwen.py`: 4-bit NF4 loader (GPU-only, refuses CPU fallback)
- `src/data/`: `gdsc2.py`, `tg2.py`, `chembl.py`, `chem.py` (largest-fragment canonical SMILES, Morgan FPs, scaffold/group splits)
- `src/evaluation/`: `metrics.py` (AUROC/AUPRC/MCC/ECE/Brier/log loss), `calibration.py` (temperature, Platt, isotonic — all fit on valid only)
- `docs/data.md`: **data provenance, unit validation, leakage audit — read before trusting any number**
- `docs/calibration.md`: why post-hoc calibration fails under scaffold splits
- `docs/setup.md`: environment and measured results. `docs/jev_research.md`: Jev findings. `docs/research_plan.md`: datasets and architecture plan

## TG2 dataset

631 molecules with a potency label (IC50 ≤ 1 µM), from the PubChem export of Wichka & Lai (2024)
merged with ChEMBL target CHEMBL2730. Units are µM, confirmed against the paper. Censored `>`
measurements are kept as negatives where the bound proves non-potency. See `docs/data.md`.

Synthetic data in `test_qlora.py` only validates infrastructure. Its results are not research findings.
