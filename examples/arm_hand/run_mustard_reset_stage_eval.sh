#!/usr/bin/env bash
set -euo pipefail
cd /home/szc/work/spider
export LD_LIBRARY_PATH="/home/szc/.conda/envs/simtoolreal_origin_isaacsim/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/szc/work/spider/third_party/simtoolreal:/home/szc/work/spider/third_party/simtoolreal/rl_games:/home/szc/work/spider:${PYTHONPATH:-}"
export OMNI_KIT_ACCEPT_EULA=YES
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export TMPDIR=/home/szc/work/spider/outputs/mustard_isaac_runtime/tmp
export XDG_CACHE_HOME=/home/szc/work/spider/outputs/mustard_isaac_runtime/xdg_cache
exec /home/szc/.conda/envs/simtoolreal_origin_isaacsim/bin/python experiments/mustard_simtoolreal/evaluate_reset_stages.py "$@"
