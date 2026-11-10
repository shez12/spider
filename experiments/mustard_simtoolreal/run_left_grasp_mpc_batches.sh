#!/usr/bin/env bash
# Scaled left-hand MPPI grasp generation in the unmodified SimToolReal env, then the left reset bank.
# Usage: bash run_left_grasp_mpc_batches.sh SEED1 SEED2 ...   (outputs/left_grasp_mpc_v2_s<SEED>)
set -euo pipefail
export LD_LIBRARY_PATH="/home/szc/.conda/envs/simtoolreal_origin_isaacsim/lib:${LD_LIBRARY_PATH:-}" OMNI_KIT_ACCEPT_EULA=YES
export TMPDIR=/home/szc/work/spider/outputs/mustard_isaac_runtime/tmp XDG_CACHE_HOME=/home/szc/work/spider/outputs/mustard_isaac_runtime/xdg_cache
P=/home/szc/.conda/envs/simtoolreal_origin_isaacsim/bin/python
S=/home/szc/work/spider/experiments/mustard_simtoolreal
O=/home/szc/work/spider/outputs
cd /home/szc/work/newsimtoolreal/simtoolreal
runs=()
for seed in "$@"; do
  out=$O/left_grasp_mpc_v2_s$seed
  [ -f $out/report.json ] || $P -u $S/left_grasp_mpc.py --headless --groups 32 --samples 256 --seed $seed --output $out > $O/logs/left_grasp_mpc_v2_s$seed.log 2>&1
  runs+=($out)
done
$P $S/left_reset_bank.py build --runs $O/left_grasp_mpc_v1 "${runs[@]}" --output $O/left_reset_bank_v1/reset_bank.npz
