#!/usr/bin/env bash
# Rotation-curriculum comparison in the unmodified SimToolReal (left hand + mustard, ADEPT obs):
# A = from scratch, S = + MPPI state bank (delta first goals, trivial goals resampled); seeds 42 and 7.
set -euo pipefail
export LD_LIBRARY_PATH="/home/szc/.conda/envs/simtoolreal_origin_isaacsim/lib:${LD_LIBRARY_PATH:-}" OMNI_KIT_ACCEPT_EULA=YES
export TMPDIR=/home/szc/work/spider/outputs/mustard_isaac_runtime/tmp XDG_CACHE_HOME=/home/szc/work/spider/outputs/mustard_isaac_runtime/xdg_cache
P=/home/szc/.conda/envs/simtoolreal_origin_isaacsim/bin/python
T=/home/szc/work/spider/experiments/mustard_simtoolreal/train_pristine_mustard.py
O=/home/szc/work/spider/outputs
cd /home/szc/work/newsimtoolreal/simtoolreal
for seed in 42 7; do
  nohup $P $T --headless --num-envs 8184 --epochs 12000 --seed $seed --object-repr adept --pc-points 64 --rot-curriculum \
    --output $O/pristine_mustard_adept64_rotcurr_A_s$seed > $O/logs/pristine_mustard_adept64_rotcurr_A_s$seed.log 2>&1 &
  sleep 60
  nohup $P $T --headless --num-envs 8184 --epochs 12000 --seed $seed --object-repr adept --pc-points 64 --rot-curriculum \
    --state-bank $O/left_reset_bank_v1/reset_bank.npz --bank-delta-first-goal \
    --output $O/pristine_mustard_adept64_rotcurr_S_s$seed > $O/logs/pristine_mustard_adept64_rotcurr_S_s$seed.log 2>&1 &
  sleep 60
done
