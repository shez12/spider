#!/usr/bin/env bash
# v4c: v4b with the pose/goal curriculum gate lowered from 0.5 to 0.15 (v4b's
# demo-start frontier rate plateaued at ~0.17). Resumes v4b epoch 14500.
set -euo pipefail
exec bash /home/szc/work/spider/examples/arm_hand/run_mustard_pose_goal_curriculum_v4b.sh \
  --resume-training /home/szc/work/spider/outputs/mustard_v4b_snapshot_e14500/sapg_latest.pth \
  --pose-threshold 0.15 \
  --output /home/szc/work/spider/outputs/mustard_pose_goal_curriculum_v4c \
  "$@"
