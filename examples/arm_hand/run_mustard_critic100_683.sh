#!/usr/bin/env bash
set -euo pipefail
exec bash /home/szc/work/spider/examples/arm_hand/run_local_simtoolreal.sh critic \
  --headless --num-envs 8184 --block-size 1364 \
  --checkpoint /home/szc/work/spider/outputs/mustard_bc100_carry_lr1e5_pilot200_v1/actor_init.pt \
  --dataset /home/szc/work/spider/outputs/mustard_continuous_teacher100_v2/native_bc_dataset.npz \
  --waypoints /home/szc/work/spider/experiments/mustard_simtoolreal/waypoints100.json \
  --episode-seconds 16 --critic-epochs 683 --critic-lr 0.00005 \
  --checkpoint-every 100 --video-every 200 --video-length 960 \
  --wandb-project simtoolreal-warmstart \
  --output /home/szc/work/spider/outputs/mustard_critic_demo16s_683_v1 \
  "$@"
