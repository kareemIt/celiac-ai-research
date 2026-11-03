#!/usr/bin/env bash
# Run every state-of-the-art entrant on every protocol, serially.
#
# Serial on purpose: wall-clock per fold and peak VRAM are reported metrics, and two models sharing
# the GPU would make both numbers meaningless.
#
# Order puts the cheap and the decisive first, so a partial run is still useful: the baselines are
# the bar, and the scaffold split is the primary evaluation. Each entrant's rows are merged into
# results/sota_tg2.csv by model+split+seed, so re-running one replaces only its own rows and an
# interrupted run can be resumed by re-invoking this script.
#
# Usage: bash scripts/run_sota_all.sh [scaffold|pub|all]   (default all)
set -uo pipefail

PY="${SOTA_PYTHON:-$HOME/projects/celiac-ai-research/.venv/bin/python}"
SEEDS="${SOTA_SEEDS:-20}"
LOG="results/sota_run.log"
WHICH="${1:-all}"

run() {  # run <split> <seeds-or-empty> <model>
  local split="$1" seeds="$2" model="$3"
  local args=(--model "$model" --split "$split")
  [[ -n "$seeds" ]] && args+=(--seeds "$seeds")
  echo "=== $(date +%H:%M:%S)  $model  $split" | tee -a "$LOG"
  if ! "$PY" scripts/run_sota.py "${args[@]}" >>"$LOG" 2>&1; then
    # one entrant's environment must not take the rest of the sweep down; the failure is in the log
    echo "!!! $(date +%H:%M:%S)  $model $split FAILED (see $LOG)" | tee -a "$LOG"
  fi
}

MODELS=(logreg,rf,xgb chemberta_frozen tabpfn chemprop chemprop_rdkit chemeleon chemberta_ft)

echo "### sweep started $(date)" | tee -a "$LOG"

if [[ "$WHICH" == "scaffold" || "$WHICH" == "all" ]]; then
  for m in "${MODELS[@]}"; do run scaffold "$SEEDS" "$m"; done
fi

if [[ "$WHICH" == "pub" || "$WHICH" == "all" ]]; then
  # the two publication arms cover the same 444 molecules, so splitting strategy is the only
  # variable between them
  for m in "${MODELS[@]}"; do run scaffold_pub "$SEEDS" "$m"; done
  for m in "${MODELS[@]}"; do run paper_lopo "" "$m"; done
fi

echo "### sweep finished $(date)" | tee -a "$LOG"
"$PY" scripts/analyse_sota.py --split scaffold --figures 2>&1 | tail -40
