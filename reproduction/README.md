# Local restoration

Restored `current-test-snapshot.tar.gz` onto base commit
`44717007de41cbef7565dff7ff9f4453557a2d3d`.

`relocation_manifest.json` records original archive member hashes and restored
hashes after replacing `/home/yimu12/work/spider` with `/home/szc/work/spider`
in text files. The archive is preserved unchanged. Saved experimental metrics
are historical, not newly reproduced. Original report fingerprints remain
historical; use the relocation manifest for local file provenance.

The partial processed dataset from the snapshot is preserved in
`snapshot-retarget_full/` so the official downloader can verify a fresh complete
release at `example_datasets/retarget_full`.

The downloader now hashes using the standard library without importing the
conversion runtime. No optimization algorithm has been additionally changed.

Dataset scope: pinned retarget/retarget_full release and full original DexYCB,
as specified in AGENT_REPRODUCTION_HANDOFF.md. Both datasets are stored under `/data4t/spider-datasets`, with repository
symlinks at `example_datasets/retarget_full` and `example_datasets/raw/dexycb`.
The destination had 1.8 TiB free before download.

The captured pip freeze referenced in the handoff was not supplied, and no
existing spider Conda environment was found. `.venv-download` is only for data
transfer; numerical reproduction needs a separately validated runtime.

The frozen-reference rerun accepts `--output-dir`, allowing a new-machine run
without overwriting the historical 100- or 512-sample outputs. While the full
processed release is still downloading, the copied scene in the new output is
rebased to the identical mesh subset under `snapshot-retarget_full/`.

## Current-machine verification

The `spider-repro` environment uses Python 3.12.3, PyTorch 2.14.1+cu130,
NumPy 2.5.4, SciPy 1.18.1, and MuJoCo 3.7.0. The host GPU is an NVIDIA RTX
6000D; the optimizer still deliberately uses its configured CPU sampling path.

`outputs/repro_spider_repro_512` is a fresh frozen-reference optimization on
this machine. It reached cost 135.9508, lifted the object 0.22215 m, finished
14.50 mm from the position target, and retained the object. Its 51.20-degree
orientation error means it fails the same complete pose criterion as the old
512-sample result. Continuous replay and the extra hold passed; the replayed
state error is exactly zero at the saved precision.

Two fresh 100-sample runs were identical to each other (cost 463.4457), but
differed from the historical 100-sample run (cost 162.5064). The frozen baseline
metrics still match to 1e-12, and the historical controls replay with zero state
error on this machine. This isolates the difference to cross-machine sampling,
not scene relocation or MuJoCo replay.

## Supplied KUKA iiwa14 + Sharpa validation

`examples/arm_hand/validate_kuka_sharpa.py` imports the supplied combined URDF
through MuJoCo, preserves its 7 iiwa and 22 Sharpa joint order and limits, adds
position servos and tracking sites, and reruns the frozen MANO mustard-bottle
reference. The source URDF is not modified.

Run the full validation with:

```bash
conda run -n spider-repro python examples/arm_hand/validate_kuka_sharpa.py \
  --description /home/szc/work/play2perfect/assets/urdf/kuka_sharpa_description \
  --output-dir outputs/kuka_sharpa_validation \
  --samples 512 --iterations 20 --threads 8 --ik-stride 2
```

The model and continuous-control pipeline validate successfully: the generated
scene has 29 actuators, deterministic replay has zero qpos error, all actuator
forces remain within their supplied limits, and all seven iiwa joints remain
below the supplied 10 rad/s rating. The retargeting IK has 3.35 mm mean raw
fingertip error and 0.67 mm maximum reported self penetration.

The frozen manipulation task does **not** pass with this controller and budget.
The optimized trajectory lifts the object only 0.81 mm against a 236.12 mm
reference lift, ends 245.43 mm from the reference object position, and fails
both pose tracking and retention. This negative result is retained rather than
being presented as a successful grasp. Detailed evidence is in
`outputs/kuka_sharpa_validation/report.json`, `model_audit.json`, and
`kuka_sharpa_replay.mp4`.

The KUKA base pose is calibrated once to the first reference wrist pose. KUKA
link contacts are disabled because the supplied description does not include a
matching fixture or table; Sharpa-object and object-floor contacts remain
active. The result therefore validates this simulated arm-hand/control setup,
not hardware execution or collision-free arm motion around a real fixture.

### Closer-base rerun

The follow-up run fixes the KUKA base at `[-0.55, -0.02, 0]` m, 0.447 m from
the initial object position, using the new `--base-position X Y Z` option. Its
artifacts are in `outputs/kuka_sharpa_close_validation`.

This placement improves the lift from 0.81 mm to 132.53 mm and reduces final
position error from 245.43 mm to 125.91 mm. The 0.5-second retention check now
passes with 7.80 mm object-to-palm relative displacement. Complete pose tracking
still fails: the final orientation error is 68.91 degrees and the position
error remains above the 30 mm threshold. All seven iiwa joints stay within the
300 N m torque and 10 rad/s velocity ratings, and continuous replay again has
zero saved-state error.

### Play2Perfect-relative current environment

`outputs/kuka_sharpa_current_environment` keeps the frozen DexYCB wrist,
fingertip, and object arrays unchanged while using the Play2Perfect-relative
layout: a 0.475 x 0.4 x 0.3 m table with its top at dataset z=0, and the KUKA
base at `[0, 0.8, -0.53]` m. It also uses the Play2Perfect arm default pose,
per-joint stiffness/damping, palm position offset, and the fixed Sharpa mounting
rotation required by the dataset wrist frame.

At 512 samples and 20 iterations, the object lifts 183.19 mm and produces 3657
hand-object contact samples. The final position error is 109.64 mm and final
orientation error is 70.73 degrees. During the additional 0.5-second hold the
arm and object descend 83.95 mm together; relative object-to-palm motion is only
4.13 mm, but final height falls just below the retention threshold. Pose,
retention, and complete-task criteria therefore remain false. Continuous replay
matches saved states exactly, and KUKA torque and velocity stay within ratings.

## Right-hand and table-contact correction

The left-hand foam-brick continuation was stopped at iteration 163 (best cost
67.6177). Its initial IK palm error was 71.49 mm, and link 5 intersected the
table by 161.69 mm because the old arm contact masks disabled table contacts.
Historical runs remain unchanged and are not collision-valid demonstrations.

`prepare_kuka_right.py` imports the preserved right Sharpa Wave model and the
KUKA-only URDF, avoiding left-hand mass merged into the final arm link. It reads
the original published Allegro-right reference without reflection, translation,
scaling, or extra hold. It uses explicit robot-table contacts, native hand
self-contact exclusions, a 5 mm robot-table contact buffer and stiffer table
contact parameters. Initial IK uses deterministic multiple restarts, stronger
palm alignment, table/self-penetration penalties, and an unfiltered initial
pose. `geometry_report.json` gates GPU optimization on initial alignment and
table clearance. New audited GPU scenes also penalize robot-table penetration.

The corrected foam-brick fixture is in
`outputs/kuka_sharpa_right_foam_clearance`. Initial palm error is 2.93 mm;
thumb/index/middle/ring tip errors are 27.01/2.18/0.004/9.27 mm. The different
hand morphology still leaves an initial thumb mismatch. Full-path raw IK mean
tip error is 3.07 mm and maximum fitted robot-table penetration is 0.135 mm.

A 5-round GPU verification, warm-started from the preceding right-hand test,
has zero detected arm-table and hand-table penetration in the saved 10 ms
states; maximum hand self-penetration is 0.826 mm. It does not lift the object
successfully. See `gpu/tracking_collision_report.json` for measured tracking
and collision results. This corrects the model/geometry setup, not a completed
manipulation task. Old checkpoints cannot be treated as optimized results for
the corrected scene. `source_vs_right_ik.mp4` under
`outputs/kuka_sharpa_right_foam` is a geometric IK comparison, not a physics
rollout; the clearance video in the corrected fixture is a saved GPU rollout.
