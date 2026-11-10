# Spatial augmentation pilot

Run `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u -m
examples.arm_hand.spatial_augmentation --ik-workers 8` from the repository.
Outputs are under `outputs/kuka_sharpa_spatial100`; reruns resume saved case
states and optimizer checkpoints. Do not run two instances against this folder.
IK screening runs in eight independent CPU processes with the same solver;
this does not migrate SciPy IK to GPU or change its objective. Physics retains
512 parallel GPU samples per optimization task. Worker logs are in each
variant's `ik.log`; completed variants are reused when restarting.

The selection is the original batch-20 selection, with five variants per source,
seed 0, uniform XY offsets in [-0.05, 0.05] m and yaw in [-10, 10] degrees.
Following paper 2609.07747v3 Appendix D, one shared transform is applied to
wrist, fingertips, object positions and orientations before IK. Rotation is about
the fixed arm base. The workspace anchor for each source is its original initial
object XY, rather than one common anchor across all tasks. Height, timing,
objects and original source references are unchanged. No mirroring or extra hold.

All 100 source references are generated first. IK uses the existing solver with
stride 2 and eight deterministic first-frame restarts. Geometric acceptance
requires the existing initial alignment gate (palm <1 cm, each fingertip <3 cm),
robot/table penetration <1 mm, mean wrist error <=8 cm (paper threshold),
hand self penetration <1 mm, joint limits, and object collision-geometry AABB
footprints within the table rectangle with 1 mm clearance at every reference
frame. This table rule is conservative, including during airborne motion.
Fingertip tracking error is also reported. This is project screening, not an
exact reproduction of the paper's complete retargeting method.

After IK screening, accepted variants receive original-SPIDER GPU sampling
optimization with robot-only gravcomp=1, free gravitational objects, 512 samples,
seed 0, max 200 rounds, 30-round plateau stopping, relative tolerance 0.001 and
absolute tolerance 0.01. No cost-threshold stopping. The existing original-20 GPU
batch is allowed to finish before this physical stage starts.

Physical success is measured on saved winning GPU states after optimization:
object position RMSE <3 cm, final position error <3 cm, final orientation error
<20 degrees, and arm/table, hand/table, hand/self penetration each <1 mm.
It is not a claim of hardware success or sustained retention; no hold is added.
Contacts are audited at 10 ms saved frames, not continuously between frames.
Videos show actual selected GPU states, not a new CPU physics replay.

`summary.json` records screened/pass counts, IK pass rate among screened cases,
physical success rate among evaluated cases (errors count as unsuccessful), and
validated success fraction among all 100 generated candidates. Pending candidates
are reported separately; partial rates are not final rates. Per-source counts,
augmentation parameters, source hashes, audits, checkpoints and videos are kept.
