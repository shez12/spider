#!/usr/bin/env bash
set -euo pipefail
cd /home/szc/work/spider
export LD_LIBRARY_PATH="/home/szc/.conda/envs/simtoolreal_origin_isaacsim/lib:${LD_LIBRARY_PATH:-}"
# Matches the existing installed environment's activation configuration.
export OMNI_KIT_ACCEPT_EULA=YES
export OMNI_KIT_CACHE_PATH=/home/szc/work/spider/outputs/mustard_isaac_runtime/cache
export TMPDIR=/home/szc/work/spider/outputs/mustard_isaac_runtime/tmp
export XDG_CACHE_HOME=/home/szc/work/spider/outputs/mustard_isaac_runtime/xdg_cache
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
mkdir -p "$OMNI_KIT_CACHE_PATH" "$TMPDIR" "$XDG_CACHE_HOME"
exec /home/szc/.conda/envs/simtoolreal_origin_isaacsim/bin/python examples/arm_hand/train_mustard_actor_isaac.py "$@"
