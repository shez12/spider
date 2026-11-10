#!/usr/bin/env bash
# v4b: v4 + first goal blended from the demo target toward the native absolute
# goal with the same alpha. Resumes v4's own checkpoint and keeps its learned
# exploration (no log-std override).
set -euo pipefail
exec bash /home/szc/work/spider/examples/arm_hand/run_local_simtoolreal.sh critic \
  --headless --num-envs 8184 --block-size 1364 \
  --checkpoint /home/szc/work/spider/outputs/mustard_classified_20261102_v2/bc_carry200/actor_init.pt \
  --resume-training /home/szc/work/spider/outputs/mustard_v4_snapshot_e13500/sapg_latest.pth \
  --dataset /home/szc/work/spider/outputs/mustard_classified_20261102_v2/selected/native_bc_dataset.npz \
  --waypoints /home/szc/work/spider/outputs/mustard_classified_20261102_v2/selected/waypoints.json \
  --native-object-reset --native-robot-reset --multi-target --multi-target-level 4 \
  --diverse-reset-bank /home/szc/work/spider/outputs/mustard_diverse_reset_bank_v2/reset_bank.npz \
  --native-tolerance --episode-seconds 16 \
  --pose-curriculum --pose-goal-curriculum --pose-alpha 0 --pose-step 0.1 --pose-threshold 0.5 --pose-window 500 \
  --actor-lr 0.0001 --adaptive-kl 0.016 --actor-clip 0.1 \
  --critic-epochs 683 --joint-epochs 200000 --run-epochs-limit 4000 --critic-lr 0.00005 \
  --checkpoint-every 500 --video-every 500 --video-length 960 \
  --wandb-project simtoolreal-warmstart \
  --output /home/szc/work/spider/outputs/mustard_pose_goal_curriculum_v4b \
  "$@"
