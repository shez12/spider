#!/usr/bin/env bash
set -euo pipefail
exec bash /home/szc/work/spider/examples/arm_hand/run_local_simtoolreal.sh critic \
  --headless --num-envs 8184 --block-size 1364 \
  --checkpoint /home/szc/work/spider/outputs/mustard_bc100_carry_lr1e5_pilot200_v1/actor_init.pt \
  --resume-training /home/szc/work/spider/outputs/mustard_multitarget_from_v2_epoch5000/sapg_latest.pth \
  --dataset /home/szc/work/spider/outputs/mustard_continuous_teacher100_v2/native_bc_dataset.npz \
  --waypoints /home/szc/work/spider/experiments/mustard_simtoolreal/waypoints100.json \
  --critic-epochs 200 --joint-epochs 200000 --episode-seconds 16 \
  --actor-lr 0.00001 --actor-clip 0.05 --critic-lr 0.00005 \
  --multi-target --restore-native-curriculum --preuploaded-history \
  --curriculum-interval 500 --curriculum-min-episodes 1000 \
  --checkpoint-every 1000 --video-every 1000 --video-length 960 \
  --wandb-project simtoolreal-warmstart \
  --output /home/szc/work/spider/outputs/mustard_warmstart_sapg_monotonic_v4 \
  "$@"
