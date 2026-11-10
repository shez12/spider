# One-trajectory backend pilot

Output: `outputs/mustard_backend_comparison_v1`.

Both runs start from the same frozen mustard candidate 4 scene/reference, with
seed 0, 512 samples, ten control knots, 200 fixed rounds and the original
`spider.optimizers.sampling.make_optimize_once_fn`. Both use the same control
bounds, noise schedule, offset bounds, cost terms, 2 ms physics step and 10 ms
control interval. Cost and convergence thresholds do not stop this pilot early.
No existing optimized checkpoint initializes either backend.

The Isaac adapter reuses the existing SimToolReal right-hand robot and object
assets. Its native PPO/SAPG learner, policy observation/reward computations,
action EMA, control retiming and episode resets are bypassed during optimization.
The source table URDF and source PD gains, effort bounds, armature, passive
damping and friction-loss values are applied to the benchmark environment.
Robot gravity is disabled; object gravity remains active. The global 53 cm
coordinate offset and object-root recentering are inverted in exported states.
Initial robot site FK must agree with source FK within 1 mm before timing runs.

This is a backend comparison, not a claim that the dynamics are identical:
PhysX implicit drives, friction laws, solver and collision import differ from
MuJoCo. The exact same source geometry audit is applied to each backend's saved
states, with no CPU dynamics replay. Both the selected rollout and a separately
reset GPU replay must satisfy the existing acceptance limits: object position
RMSE < 3 cm, final position < 3 cm, final orientation < 20 degrees, and arm/table,
hand/table and hand/self penetration < 1 mm. Saved 10 ms states do not establish
continuous collision clearance between frames.

The pilot runs Isaac initialization/nominal-rollout smoke checks first, then
MuJoCo and Isaac 200-round optimization sequentially. Existing training and
augmentation jobs remain active, so timings include shared-machine contention.
`status.json` records stage and final metrics. Each backend retains checkpoints,
history, source-frame trajectories and both geometric validation reports.
End-to-end elapsed time includes process startup and audits; optimizer-loop
timing is reported separately. Videos are excluded equally from throughput.

One trajectory with one seed gives either zero or one valid output per backend.
The derived valid/hour value is a pilot rate, not a statistically established
acceptance rate or a campaign completion estimate. Migration decisions need
both comparable runtime and preserved physical success.
