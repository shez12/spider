# Three-stage KUKA + right Sharpa optimization

```bash
cd /home/szc/work/spider
conda activate spider-repro
python -u -m examples.arm_hand.staged_kuka_gpu --iterations 100 --samples 512
```

All three stages run exactly 100 optimization iterations. Both cost-threshold
and plateau early stopping are disabled. The existing right-hand/table fixture
is reused; wrist, fingertip and object reference arrays remain unchanged, and
no extra hold is appended. The command resumes interrupted stages and skips
stages already completed with the same run settings.

1. `01_hand`: remove KUKA and drive the right hand root as a mocap body. The
   world hand-root transform is computed by inverting the fixed palm-site
   transform, so the palm follows the dataset wrist at every physics step.
   Only 22 hand controls are optimized, using the existing joint IK prior.
   Object motion remains free dynamics,
   with table and hand contacts enabled. This is ideal kinematic wrist support,
   not evidence that an arm can execute the same motion.
2. `02_arm`: restore KUKA and initialize the finger controls from stage 1.
   Optimize only 7 arm controls; all 22 finger command sequences remain fixed.
   The wrist residual weight remains 15.
3. `03_joint`: initialize all controls from stage 2 and optimize all 29 joints.
   All stages use the same noise scale, +/-0.4 offset bounds, and cost weights.
   The dataset reference arrays are never replaced by an earlier stage's
   predicted wrist, fingertip or object motion.

Under `outputs/kuka_sharpa_staged_spider_foam/`, each stage contains the prepared
`scene.xml`, `reference.npz`, `training.log`, `gpu/checkpoint.pt`,
`gpu/trajectory.npz`, `gpu/history.json`, `report.json`, and `best_gpu.mp4`.
`status.json` tracks progress and `comparison.json` accumulates the common
tracking/collision metrics. Compare physical tracking errors across stages.
Regressions relative to
the previous stage are reported explicitly.

Reports check that the stage-2 finger controls are unchanged, measure wrist,
fingertip and object errors, and audit table/self penetration on saved 10 ms
GPU states. These are not continuous collision certificates between frames.
The videos render actual winning candidate states from GPU physics, including
the saved mocap root poses in stage 1. A separate independent replay cost is
recorded because contact-rich float32 trajectories can differ upon replay.

## CUDA stream correction

The GPU adapter now directly calls `spider.optimizers.sampling.make_optimize_once_fn`.
Sampling, normalized top-10% weighting and temperature use that original function;
the adapter uses the pre-existing Sharpa demo's 0.97 noise schedule, with no added
noise floor. The independent sampler has been removed. The added fingertip and
table-penetration losses are disabled for this pipeline; physical table contacts
remain enabled. This restores the original sampling algorithm, not the entire
upstream task/reward pipeline: the prepared KUKA scene, demo tracking cost and
prescribed-wrist hand stage remain adapters.

PyTorch and Warp now share one explicit non-default CUDA stream. Previously
they used different default streams despite zero-copy shared controls/states,
which could cause unsynchronized reads. New checkpoints record the stream and
export policies. Resuming older checkpoints directly is rejected; parameters
can be imported using `--warm-start` and their costs recomputed. Previous GPU
cost histories should not be treated as equivalent to the corrected pipeline.
