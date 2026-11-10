#!/usr/bin/env bash
set -euo pipefail
exec bash /home/szc/work/spider/examples/arm_hand/run_local_simtoolreal.sh critic \
  --headless --num-envs 8184 --block-size 1364 \
  --checkpoint /home/szc/work/spider/outputs/mustard_classified_20261102_v2/bc_carry200/actor_init.pt \
  --critic-initialization /home/szc/work/spider/outputs/mustard_unfreeze_native_reset_init_v1/critic_latest.pth \
  --start-unfrozen --critic-epochs 683 --joint-epochs 200000 \
  --dataset /home/szc/work/spider/outputs/mustard_classified_20261102_v2/selected/native_bc_dataset.npz \
  --waypoints /home/szc/work/spider/outputs/mustard_classified_20261102_v2/selected/waypoints.json \
  --actor-lr 0.00001 --actor-clip 0.05 --critic-lr 0.00005 \
  --native-object-reset --native-robot-reset --native-tolerance \
  --multi-target --multi-target-level 4 --restore-native-curriculum --episode-seconds 16 \
  --checkpoint-every 1000 --video-every 1000 --video-length 960 \
  --wandb-project simtoolreal-warmstart \
  --output /home/szc/work/spider/outputs/mustard_unfreeze_native_robot_tolerance_v1 \
  "$@"
