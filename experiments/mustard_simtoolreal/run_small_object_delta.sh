#!/usr/bin/env bash
# Object-size ablation: the delta2 pipeline (MPPI state bank -> SAPG with bank resets + delta first goals, no rotation
# curriculum) and a same-seed from-scratch control, on the mustard bottle uniformly scaled to 0.6x
# (assets/object_mustard_s060, from make_scaled_object.py). Unmodified SimToolReal, left hand, ADEPT obs, seed 42.
set -euo pipefail
export LD_LIBRARY_PATH="/home/szc/.conda/envs/simtoolreal_origin_isaacsim/lib:${LD_LIBRARY_PATH:-}" OMNI_KIT_ACCEPT_EULA=YES
export TMPDIR=/home/szc/work/spider/outputs/mustard_isaac_runtime/tmp XDG_CACHE_HOME=/home/szc/work/spider/outputs/mustard_isaac_runtime/xdg_cache
P=/home/szc/.conda/envs/simtoolreal_origin_isaacsim/bin/python
S=/home/szc/work/spider/experiments/mustard_simtoolreal
O=/home/szc/work/spider/outputs
OBJ=$S/assets/object_mustard_s060
TRAIN="$P $S/train_pristine_mustard.py --headless --num-envs 8184 --epochs 20000 --seed 42 --object-repr adept --pc-points 64 --object-dir $OBJ"
cd /home/szc/work/newsimtoolreal/simtoolreal
A=$O/pristine_mustard060_adept64_scratch_s42
[ -d $A ] || nohup $TRAIN --output $A > $O/logs/pristine_mustard060_adept64_scratch_s42.log 2>&1 &
runs=()
for seed in 1 2 3 4; do
  out=$O/left_grasp_mpc_m060_s$seed
  [ -f $out/report.json ] || $P -u $S/left_grasp_mpc.py --headless --groups 32 --samples 256 --seed $seed --object-dir $OBJ --output $out > $O/logs/left_grasp_mpc_m060_s$seed.log 2>&1
  runs+=($out)
done
B=$O/left_reset_bank_m060/reset_bank.npz
$P $S/left_reset_bank.py build --runs "${runs[@]}" --output $B > $O/logs/left_reset_bank_m060.log 2>&1
nohup $TRAIN --state-bank $B --bank-delta-first-goal --output $O/pristine_mustard060_adept64_mpcbank_delta2_s42 > $O/logs/pristine_mustard060_adept64_mpcbank_delta2_s42.log 2>&1 &
