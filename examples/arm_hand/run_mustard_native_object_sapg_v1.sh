#!/usr/bin/env bash
set -euo pipefail
exec bash /home/szc/work/spider/examples/arm_hand/run_local_simtoolreal.sh critic \
  --headless --num-envs 8184 --block-size 1364 \
  --checkpoint /home/szc/work/spider/outputs/mustard_classified_20261102_v2/bc_carry200/actor_init.pt \
  --resume-training /home/szc/work/spider/outputs/mustard_native_object_from_checkpoint_20261103/sapg_latest.pth \
  --dataset /home/szc/work/spider/outputs/mustard_classified_20261102_v2/selected/native_bc_dataset.npz \
  --waypoints /home/szc/work/spider/outputs/mustard_classified_20261102_v2/selected/waypoints.json \
  --critic-epochs 683 --joint-epochs 200000 --episode-seconds 16 \
  --actor-lr 0.00001 --actor-clip 0.05 --critic-lr 0.00005 \
  --multi-target --multi-target-level 4 --native-object-reset --restore-native-curriculum \
  --curriculum-interval 500 --curriculum-min-episodes 1000 \
  --checkpoint-every 1000 --video-every 1000 --video-length 960 \
  --wandb-project simtoolreal-warmstart \
  --output /home/szc/work/spider/outputs/mustard_native_object_sapg_v1 \
  "$@"
