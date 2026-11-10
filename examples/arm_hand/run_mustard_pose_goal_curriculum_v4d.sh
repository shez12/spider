#!/usr/bin/env bash
# v4d: same as v4c (gate 0.15, resume v4b epoch 14500) but with the corrected
# non-overlapping gate: one decision per 500 fresh frontier episodes. v4c used a
# sliding window that could advance on noise; its output is kept as
# outputs/mustard_pose_goal_curriculum_v4c_slidinggate.
set -euo pipefail
exec bash /home/szc/work/spider/examples/arm_hand/run_mustard_pose_goal_curriculum_v4b.sh \
  --resume-training /home/szc/work/spider/outputs/mustard_v4b_snapshot_e14500/sapg_latest.pth \
  --pose-threshold 0.15 \
  --output /home/szc/work/spider/outputs/mustard_pose_goal_curriculum_v4d \
  "$@"
