#!/usr/bin/env bash
set -euo pipefail
exec bash /home/szc/work/spider/examples/arm_hand/run_local_simtoolreal.sh critic \
  --headless --num-envs 8184 --block-size 1364 \
  --checkpoint /home/szc/work/spider/outputs/mustard_bc100_carry_lr1e5_pilot200_v1/actor_init.pt \
  --critic-initialization /home/szc/work/spider/outputs/mustard_critic_demo16s_683_v1/critic_calibrated.pth \
  --dataset /home/szc/work/spider/outputs/mustard_continuous_teacher100_v2/native_bc_dataset.npz \
  --waypoints /home/szc/work/spider/experiments/mustard_simtoolreal/waypoints100.json \
  --episode-seconds 16 --sigma-scale 0.25 \
  --critic-epochs 200 --critic-lr 0.00005 \
  --joint-epochs 200000 --actor-lr 0.00001 --actor-clip 0.05 \
  --curriculum-interval 500 --curriculum-min-episodes 1000 --curriculum-success 0.70 \
  --checkpoint-every 1000 --video-every 1000 --video-length 960 \
  --wandb-project simtoolreal-warmstart \
  --output /home/szc/work/spider/outputs/mustard_warmstart_sapg_lownoise_v2 \
  "$@"
