#!/usr/bin/env bash
set -euo pipefail
cd /home/szc/work/spider
export LD_LIBRARY_PATH="/home/szc/.conda/envs/simtoolreal_origin_isaacsim/lib:${LD_LIBRARY_PATH:-}"
export OMNI_KIT_ACCEPT_EULA=YES
export TMPDIR=/home/szc/work/spider/outputs/mustard_isaac_runtime/tmp
export XDG_CACHE_HOME=/home/szc/work/spider/outputs/mustard_isaac_runtime/xdg_cache
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="/home/szc/work/spider/third_party/simtoolreal:/home/szc/work/spider/third_party/simtoolreal/rl_games:/home/szc/work/spider:${PYTHONPATH:-}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
entry=experiments/mustard_simtoolreal/run.py
if [[ "${1:-}" == "build-reset-bank" ]]; then
  entry=experiments/mustard_simtoolreal/build_diverse_reset_bank.py
  shift
fi
if [[ "${1:-}" == "balance-evaluate" ]]; then
  entry=experiments/mustard_simtoolreal/evaluate_balanced_bc.py
  shift
fi
if [[ "${1:-}" == "critic" ]]; then
  entry=experiments/mustard_simtoolreal/train_critic.py
  shift
fi
if [[ "${1:-}" == "scratch" ]]; then
  entry=experiments/mustard_simtoolreal/train_scratch.py
  shift
fi
if [[ "${1:-}" == "train" ]]; then
  entry=experiments/mustard_simtoolreal/train_staged.py
  shift
fi
if [[ "${1:-}" == "warmup" ]]; then
  entry=experiments/mustard_simtoolreal/train_actor_warmup.py
  shift
fi
if [[ "${1:-}" == "bc" ]]; then
  entry=experiments/mustard_simtoolreal/train_bc.py
  shift
fi
if [[ "${1:-}" == "evaluate" ]]; then
  entry=experiments/mustard_simtoolreal/evaluate_actor_video.py
  shift
fi
exec /home/szc/.conda/envs/simtoolreal_origin_isaacsim/bin/python "$entry" "$@"
