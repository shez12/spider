# GPU optimization until cost plateaus

## Full-rollout graph experiment (2026-10-27)

Subsequent Nsight profiling identified substantial GPU waiting rather than
Python computation in a representative six-frame rollout (submission ~11 ms,
synchronization ~514 ms). Matched MPS benchmarks then improved full-trajectory
ten-process throughput 3.39x, and the production mustard pipeline uses MPS.
See `MUSTARD_VALID100.md` for configuration, measured scope and validation.

Physics, costs, random sampling and SPIDER weighting already execute on CUDA.
Python submits approximately 870 captured physics steps per 174-frame rollout.
Scalar decisions, original optimizer diagnostics (`cpu().numpy()`), checkpoint
serialization and scheduling remain on the host. CPU utilization alone cannot
distinguish kernel submission from busy waiting for GPU work. Geometry audit and
video export occur after optimization, not at each physics step.

An isolated benchmark using a previously validated candidate-4 checkpoint,
512 worlds, and the same targets/costs measured eager execution at 15.774 s and
full-rollout CUDA Graph at 16.205 s, with 174.822 s initial graph construction.
This one-repeat shared-GPU test did not show a speedup. All graph results were
finite, but the strict cost/final-state equivalence gate failed (max cost
difference 0.609; max final qpos difference 0.0121). These are simulation
comparisons, not evidence of hardware performance or deterministic equivalence.

Results: `outputs/mustard_cuda_graph_benchmark_valid/benchmark.json`.
The earlier zero-offset stress test also produced nonfinite/unstable candidates
and is not a valid acceleration result. Production jobs retain their original
execution. `--benchmark-rollout` is an opt-in isolated experiment;
`--rollout-cuda-graph` is blocked for production until validation passes.

## Training command

Run from `/home/szc/work/spider`:

```bash
conda activate spider-repro
python -u -m examples.arm_hand.optimize_kuka_gpu \
  --input-dir outputs/kuka_sharpa_current_environment \
  --output-dir outputs/kuka_sharpa_gpu \
  --samples 512 --min-iterations 1 --max-iterations 100 --target-cost 100 \
  --patience 30 --relative-tol 0.001 --absolute-tol 0.01
```

Append `--resume` to continue the saved CUDA random state, sampling center and
best parameters. The checkpoint requires identical scene/reference hashes and
sample count and optimizer settings. Updates call the repository's original
`make_optimize_once_fn`, with temperature 1.0 as in the Sharpa demo. The first
sample has zero noise; exploration decays by 0.97 per iteration without a floor.
Older independent-sampler checkpoints cannot be resumed directly.

The default maximum is 100 iterations; best cost strictly below 100 stops
immediately, even before the minimum plateau-check iteration. Otherwise,
convergence means that cumulative improvement of the best cost has not exceeded
`max(absolute_tol, relative_tol * previous_significant_best_cost)` for 30 rounds,
after the configured minimum number of rounds. `result.json` distinguishes
`target_cost`, `cost_plateau` and `maximum_iterations`. This is a stochastic
stopping rule, not proof of a global optimum.

The prepared scene and reference are read unchanged. No extra hold is added.
CUDA is mandatory for both physics and cost evaluation; no CPU fallback runs.
The GPU trajectory uses float32 MuJoCo Warp physics and is not guaranteed to
match CPU MuJoCo replay. Outputs are `checkpoint.pt`, `history.json`,
`trajectory.npz`, and `result.json`. The GPU cost follows the existing tracking
weights, with shortest quaternion angle errors.

## Twenty published dataset trajectories

```bash
python -u -m examples.arm_hand.batch_kuka_gpu
```

This command freezes twenty distinct right-Allegro DexYCB tasks from the pinned
retarget_full manifest, distributed across object categories. References come
from published robot trajectories, not raw human MANO labels. The established
left-hand reflection is applied without scaling, additional translation, or
extra hold. Each case rebuilds its object scene and solves fresh KUKA/Sharpa IK
with the current base `[0, 0.8, -0.53]` and table top z=0.

Each case uses 512 samples, seed 0, at most 100 iterations, and immediate
termination for best cost <100. Plateau stopping is disabled for this batch.
Results, logs and videos are under `outputs/kuka_sharpa_batch20/`; `selected.json`
freezes the cases, `status.json` tracks completed cases and `summary.json` is
written when all cases finish. Re-run the same command after interruption to
skip completed cases and resume existing checkpoints. The absolute cost is a
sum over time, so the same cost threshold is not duration-normalized across
different episodes.
