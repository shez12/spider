# Mustard bottle: 100 validated spatial trajectories

`mustard_valid100.py` samples additional candidates until exactly 100 have passed
both geometric screening and two GPU dynamic rollouts. A candidate is not a
successful trajectory merely because it was generated or passed IK.

The source is `dexycb_01_right_006_mustard_bottle_20200709_143211_00` from the
original batch-20 selection. Spatial augmentation uses shared XY offsets up to
5 cm and yaw up to 10 degrees, with original object initial XY as anchor. Height,
timing, original references and interaction geometry are preserved. The five
previously screened mustard variants are reused; their physics is trained fresh.
Further variants use a per-candidate deterministic seed sequence (20261026, id).

IK uses eight CPU workers and the existing solver. Screening follows
`spatial_augmentation.py`: initial palm <1 cm, each tip <3 cm, mean wrist <=8 cm,
robot/table and hand/self penetration <1 mm, joint limits, full-reference object
footprint within table with 1 mm clearance. No failed IK enters GPU optimization.

GPU optimization uses original SPIDER sampling, 512 samples, seed 0, at most
200 rounds, and plateau stopping after 30 rounds without significant improvement
(absolute 0.01, relative 0.001). Only robot bodies have gravcomp=1. Object gravity
and physical contacts are enabled; no extra hold or cost-threshold stop is added.

A successful result must pass BOTH the actual selected GPU rollout and a fresh
GPU replay of the selected controls: object position RMSE <3 cm, final position
error <3 cm, final rotation error <20 degrees, arm/table, hand/table and hand/self
penetration each <1 mm. These are simulation criteria, not hardware validation;
collision checks use saved 10 ms frames. The fresh replay is not an imposed object
trajectory or CPU re-simulation. No retention hold is required.

Output `outputs/mustard_spatial_valid100/accepted.json` lists only accepted
trajectories and videos. `summary.json` shows generated, screened, evaluated and
accepted counts separately. Failed candidates remain in their folders. Resume
the same command to reuse completed work and continue checkpoints. The generator
samples more candidates whenever necessary; no finite sample count guarantees
that 100 successes will be achievable.

`timing.json` records wall time from request start (2026-10-26 16:10:49 +08:00),
pipeline run durations, summed per-candidate IK compute time, GPU optimization
time, and audit/render time. Summed IK time is CPU work across concurrent workers
and is not the elapsed wall time. Until completion, timings and acceptance counts
are provisional.

Physics defaults to two independent optimizer processes (`--physics-workers 2`),
each retaining 512 parallel samples, its own CUDA stream/context, seed, checkpoint
and files. The coordinator alone updates aggregate reports every 30 seconds.
`parallelism.json` records concurrency. Summed physics times overlap and are not
elapsed wall time. Dispatch reserves remaining acceptance slots, so no more than
the requested 100 accepted candidates are launched near completion. Unrelated
GPU workloads are not stopped; speedup depends on actual shared GPU throughput.
The coordinator rereads `physics_workers` from `parallelism.json` every 30 seconds
and accepts values 1 through 15. Increasing launches more independent candidates;
decreasing lets existing candidates finish before reducing concurrency. This
changes scheduling only. On coordinator replacement, existing optimizer PIDs can
be adopted through per-case `adopt_optimizer_pid` without resetting GPU state.

## MPS execution (2026-10-27)

The production ten-worker run now uses NVIDIA MPS at
`/tmp/spider-mps-mustard-1005`. Only the optimizer subprocess receives
`CUDA_MPS_PIPE_DIRECTORY`; model, loss, SPIDER update, samples, iteration limits,
stopping rules and two-rollout acceptance are unchanged. `parallelism.json`
stores `mps_pipe_dir`; each state records `gpu_execution=cuda_mps`. The control
daemon is in a private per-project directory; GPU compute mode is not changed.
Check its actual client list, since a bad pipe can cause silent CUDA fallback.

Matched ten-process benchmarks with a warmup barrier excluded initialization:
20 repeats of six control frames took 10.036 s without MPS and 3.946 s with MPS
(2.54x throughput). Two repeats of the full 174-frame trajectory took 30.903 s
without MPS and 9.105 s with MPS (3.39x). Each client retained 512 worlds. All
ten full-trajectory outputs in each mode passed the original physical gate.
These are fixed-control simulation tests, not an end-to-end 3.39x guarantee.

Raw results are in `outputs/mustard_parallel_{baseline10,mps10}[_full]/`;
physical audits are summarized in `outputs/mustard_mps_validation.json`.
The Nsight trace `outputs/mustard_profile_before.nsys-rep` and CPU call profile
show the six-frame workload spent about 11 ms submitting work and 514 ms waiting
for GPU completion. A busy CPU core alone did not establish a CPU compute
bottleneck. The original CUDA graph experiment was not adopted.

`checkpoint_before_mps.pt` preserves each migrated case's prior checkpoint.
Profiling pauses are recorded separately in `timing.json`; total wall time
includes them. To restart after a reboot, first start a per-project MPS daemon
with the same pipe directory, then run with `--physics-workers 10 --mps-pipe-dir
/tmp/spider-mps-mustard-1005`. MPS is a driver scheduling change; every candidate
still has to pass the same selected-rollout and independent-replay checks.

Run from the repository:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -u -m examples.arm_hand.mustard_valid100 --physics-workers 2
```
