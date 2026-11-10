#!/usr/bin/env bash
set -euo pipefail
exec bash /home/szc/work/spider/examples/arm_hand/run_local_simtoolreal.sh bc \
  --headless \
  --num-envs 8184 --block-size 1364 \
  --expected-trajectories 10 \
  --dataset /home/szc/work/spider/outputs/mustard_bc_eligible_v1/native_bc_dataset.npz \
  --source /home/szc/work/spider/outputs/mustard_bc_eligible_v1/source_manifest.json \
  --waypoints /home/szc/work/spider/outputs/mustard_bc_eligible_v1/waypoints.json \
  --epochs 4000 --learning-rate 0.0001 \
  --learner-device cuda:0 --batch-sequences 128 \
  --sequence-length 16 --burn-in 16 \
  --checkpoint-every 100 --video-every 100 --video-length 600 \
  --wandb-project simtoolreal-warmstart \
  --output /home/szc/work/spider/outputs/mustard_bc_native_eligible10_4000_v2 \
  "$@"
