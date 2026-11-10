#!/usr/bin/env bash
set -euo pipefail
exec bash /home/szc/work/spider/examples/arm_hand/run_local_simtoolreal.sh critic \
  --headless --num-envs 8184 --block-size 1364 \
  --checkpoint /home/szc/work/spider/outputs/mustard_classified_20261102_v2/bc_carry200/actor_init.pt \
  --resume-training /home/szc/work/spider/outputs/mustard_diverse_reset_ab_v1/mixed_reset/sapg_latest.pth \
  --dataset /home/szc/work/spider/outputs/mustard_classified_20261102_v2/selected/native_bc_dataset.npz \
  --waypoints /home/szc/work/spider/outputs/mustard_classified_20261102_v2/selected/waypoints.json \
  --native-object-reset --native-robot-reset --multi-target --multi-target-level 4 \
  --diverse-reset-bank /home/szc/work/spider/outputs/mustard_diverse_reset_bank_v2/reset_bank.npz \
  --fixed-difficulty --episode-seconds 16 \
  --critic-epochs 683 --joint-epochs 200000 --run-epochs-limit 2000 \
  --actor-lr 0.00001 --actor-clip 0.05 --critic-lr 0.00005 \
  --checkpoint-every 500 --video-every 500 --video-length 960 \
  --wandb-project simtoolreal-warmstart \
  --output /home/szc/work/spider/outputs/mustard_diverse_reset_v2_native_robot \
  "$@"
