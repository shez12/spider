#!/usr/bin/env bash
# 0b-adept on another machine: original SimToolReal (left Sharpa hand, original task YAML incl. domain randomisation,
# 10 s, native tolerance curriculum 0.075 -> x0.9 when episodes average >=3 goals -> floor 0.01), mustard bottle,
# ADEPT observations (64 points), from scratch, seed 42. Runs up to the original SAPG max_epochs (1e6) and stops
# when the tolerance curriculum reaches its floor (tolerance_floor_reached.json); each tolerance step is checkpointed.
# Layout under $WORK: newsimtoolreal/simtoolreal, spider/experiments/mustard_simtoolreal, spider/examples/arm_hand.
# Usage: PYTHON=/path/to/isaaclab/env/bin/python bash run_0b_adept_remote.sh [output_dir] [extra train args...]
set -euo pipefail
WORK=${WORK:-/home/ecomo/work}
PYTHON=${PYTHON:?set PYTHON to the Isaac Sim 5.1 / Isaac Lab python}
OUT=${1:-$WORK/outputs/pristine_mustard_adept64_s42}
shift || true
export SIMTOOLREAL_PRISTINE=$WORK/newsimtoolreal/simtoolreal
export OMNI_KIT_ACCEPT_EULA=YES OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PYTHON")")/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$(dirname "$OUT")"
# Original assets are referenced relative to the SimToolReal checkout.
cd "$SIMTOOLREAL_PRISTINE"
exec "$PYTHON" "$WORK/spider/experiments/mustard_simtoolreal/train_pristine_mustard.py" --headless \
  --num-envs "${NUM_ENVS:-8184}" --epochs "${EPOCHS:-1000000}" --seed "${SEED:-42}" \
  --object-repr adept --pc-points 64 --stop-at-tolerance-floor --wandb-project "${WANDB_PROJECT:-}" --output "$OUT" "$@"
