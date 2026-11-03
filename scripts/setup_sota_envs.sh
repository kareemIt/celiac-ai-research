#!/usr/bin/env bash
# Build the isolated virtualenvs the state-of-the-art comparison needs.
#
# Why separate envs: chemprop pins its own lightning/torch, and tabpfn pins scikit-learn. Installing
# either into the project .venv could move scikit-learn or torch under the existing random forest
# and QLoRA results, which must stay reproducible. ChemBERTa and MoLFormer need nothing new and run
# in the project .venv directly.
#
# Envs live outside the repository (default ~/sota-envs) because the checkout is on /mnt/c, where
# multi-GB venvs are slow and would land in git status. scripts/run_sota.py looks for them there,
# or at $SOTA_<MODEL>_PYTHON if you put them somewhere else.
#
# Usage: bash scripts/setup_sota_envs.sh [chemprop|tabpfn|all]
set -euo pipefail

ROOT="${SOTA_ENV_ROOT:-$HOME/sota-envs}"
WHICH="${1:-all}"
mkdir -p "$ROOT"

command -v uv >/dev/null || { echo "uv not found; this project's .venv was built with it" >&2; exit 1; }

build() {
  local name="$1"; shift
  local dir="$ROOT/$name"
  echo "=== $name -> $dir"
  uv venv --python 3.12 "$dir"
  VIRTUAL_ENV="$dir" uv pip install --python "$dir/bin/python" "$@"
  "$dir/bin/python" -c "import sys; print(name_ok := sys.version)" >/dev/null
  echo "=== $name ready: $dir/bin/python"
}

if [[ "$WHICH" == "chemprop" || "$WHICH" == "all" ]]; then
  # chemprop 2.x brings its own torch + lightning; CheMeleon loads through the same package
  build chemprop "chemprop>=2.2,<3" "rdkit" "pandas" "scikit-learn"
fi

if [[ "$WHICH" == "tabpfn" || "$WHICH" == "all" ]]; then
  build tabpfn "tabpfn" "rdkit" "pandas" "scikit-learn"
fi

echo
echo "Record these in docs; scripts/run_sota.py finds them automatically under $ROOT:"
for d in "$ROOT"/*/; do
  [[ -x "$d/bin/python" ]] || continue
  printf '  %-12s %s\n' "$(basename "$d")" "$("$d/bin/python" --version)"
done
