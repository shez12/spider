# SPIDER hand-retargeting conversation and reproduction handoff

Prepared 2026-10-22, Asia/Shanghai. This file describes the actual local workspace and saved measurements, including failures. Read this before changing code or rerunning experiments.

## 1. Current objective and user constraints

The user wants human or another robot's hand-only demonstrations retargeted to their dexterous hand mounted on a robot arm, manipulating a free physical object. The current experiment uses original DexYCB human annotations to control a left Sharpa HA4 plus a generic seven-joint arm.

**Keep repository optimization methods unchanged.** New standalone adapters, test scripts, separate robot variants, and output directories are appropriate. Preserve existing results. Do not silently retune weights, friction, torque, reference timing, seed, or model geometry when comparing sample budgets. The latest requested comparison changed only the sampling budget from 512 to 100 per iteration.

This experiment is trajectory optimization, not PPO training. It imports the repository's sampling update with a CPU MuJoCo rollout adapter and experiment-specific objectives. It is not a reproduction of every stage of the paper's full pipeline.

## 2. Summary of the overall conversation

1. Reviewed SPIDER, Conda setup, dataset access, and quick-start examples. Discussed the supplied paper `/home/yimu12/Documents/spider.pdf`: retargeting combines geometric initialization with physics-based optimization; the test runs do not train PPO.
2. Built hand-only dataset reference → generic articulated arm plus hand examples. Checked hand mounting and visible object geometry. Recorded source and dynamic-result videos together. Demonstrated cross-embodiment transfer, including Allegro → Xhand.
3. Investigated sampling ablations, including reducing samples to zero and testing one sample. Saved reports show non-finite updates for one sample in both tested sampling implementations. The elite reward standard deviation can be ill-defined for a singleton elite set; do not assume every positive sample count is valid.
4. Imported the user's Yimu hand, diagnosed coupling, torque, contacts/friction, grasping and slipping. Tested a separate fully actuated distal-joint variant and torque/friction variants. These experiments did not establish reliable transfer for the difficult can task. More torque or independently actuated distal joints alone should not be represented as a general solution.
5. Imported supplied left Sharpa HA4, preserving native geometry/servo behavior while applying force limits from its matching URDF. Sharpa generally grasped better in these tests, but object orientation and retention still depend on the task. Also imported the supplied Wuji hand; a saved transfer lifts and subsequently loses the grasp.
6. Downloaded and verified the complete `retarget/retarget_full` release. Tested three harder Sharpa cases: scissors, mug and power drill. All retained the object in their hold checks; only drill passed the chosen position-plus-orientation criterion. Speed ratings were exceeded in some cases.
7. Ran ten more distinct object cases with Sharpa. Seven retained the object, but only mustard bottle and pudding box passed the chosen pose-plus-retention criteria. See the exact table below.
8. Discussed human MANO data. Recommended DexYCB first, then HO-3D/OakInk-Image; OakInk2 is an option for later complex/bimanual work. Downloaded the original full DexYCB dataset, separately from the processed robot trajectories.
9. Implemented an original-human DexYCB → Sharpa example. Uses supplied 21-joint hand annotations, not a robot reference and not MANO PCA decoding. Preserves `pose_m` as provenance. MANO surface model files were not installed; the video shows original RGB and a human skeleton.
10. Corrected an initially mistaken upright-object frame assumption by fitting the table plane from aligned depth. The incorrect run is retained only as alignment debugging output. The corrected 512-sample result lifts and holds the object but misses its orientation.
11. Reran with 100 samples, freezing the exact scene, IK reference and initial state. The same-seed run performs worse and drops the object during the extra hold. This is the latest completed experiment.

## 3. Workspace, environment and code preservation

- Repository: `/home/yimu12/work/spider`
- Upstream: <https://github.com/facebookresearch/spider>
- Current base Git commit: `44717007de41cbef7565dff7ff9f4453557a2d3d`
- Conda environment: `spider`
- Python executable: `/home/yimu12/miniconda3/envs/spider/bin/python`
- Python version at handoff: `3.12.15`
- Existing environment is the best route to repeat the saved numerical result.

The working tree is **not a clean upstream checkout**. Earlier compatibility work modified:

- `spider/io.py`
- `spider/preprocess/ik_fast.py`
- `spider/simulators/mjwp.py`

There are new, untracked `spider/robot_model.py`, `examples/run_arm_hand.py`, `examples/test_sampling_ablation.py`, `examples/arm_hand/`, and robot asset directories. Do not discard these. The later comparisons fingerprinted these existing files and verified no further method changes during each experiment. “Method unchanged” does not mean the working tree never changed throughout this whole conversation.

Supporting preservation files beside this document:

- `reproduction/spider-pip-freeze.txt`: captured installed Python packages.
- `reproduction/preexisting-core-changes.patch`: tracked core-file diff against the base commit; does not include untracked files.
- `reproduction/snapshot_sha256.json`: checksums of files in the snapshot.
- `reproduction/current-test-snapshot.tar.gz`: local code, Sharpa assets, required scene templates and their mesh dependencies, selected source-scene metadata, frozen current references/results/videos, and the release revision receipt. This is a **local reproduction supplement**, not a full dataset archive or a standalone source checkout. Do not redistribute user-supplied models without appropriate permission.

A fresh upstream clone alone cannot run these commands: the local adapters and generated templates must also be restored. For a new checkout, first clone/check out the base commit, then extract the supplement into that checkout. The supplement already contains the three locally modified core files; do not apply the diff again on top of those copies.

```bash
git clone https://github.com/facebookresearch/spider.git /path/to/spider
cd /path/to/spider
git checkout 44717007de41cbef7565dff7ff9f4453557a2d3d
tar -xzf /path/to/current-test-snapshot.tar.gz -C /path/to/spider
```

Saved XMLs and report metadata contain absolute `/home/yimu12/work/spider/...` paths. For exact local reproduction retain this location. On another machine, rebase those XML compiler/mesh paths and metadata paths to the new checkout; recompute checksums and document that relocation. The Python wrappers generally derive the root from `__file__`, but saved scenes and frozen metadata do not automatically relocate.

### Environment setup on a new machine

Do not reinstall the working environment just to rerun a test. The repository's `requirements.txt` differs from the captured runtime, including older Torch/NumPy pins and different simulator sources. For a new environment, use the captured package list as the authoritative version record; review its local editable-install paths before installing it. A reasonable setup sequence is:

```bash
conda create -n spider python=3.12 pip
conda activate spider
python -m pip install --upgrade pip
# Inspect/rebase local editable paths in the captured freeze before using it.
python -m pip install -r docs/reproduction/spider-pip-freeze.txt
python -m pip install --no-deps -e .
```

Selected versions actually observed:

| Package | Version |
|---|---|
| numpy | 2.5.4 |
| scipy | 1.18.1 |
| torch | 2.14.1 |
| mujoco | 3.7.0 |
| mujoco-warp | 3.7.0.1 |
| warp-lang | 1.12.1 |
| trimesh | 5.1.1 |
| opencv-python | 5.0.0.93 |
| imageio | 2.38.1 |
| imageio-ffmpeg | 0.6.0 |
| PyYAML | 6.0.3 |
| huggingface-hub | 2.2.0 |

Optimization uses CPU rollouts and needs no Warp GPU execution. Rendering uses EGL and needs a working graphics driver. In this Codex sandbox, EGL rendering required an escalated tool invocation to access the GPU; the same shell command works normally outside the sandbox. Keep `OPENBLAS_NUM_THREADS=4` for optimization. Avoid unrequested package upgrades: contact-rich results are sensitive to runtime changes.

## 4. Dataset A: processed SPIDER full release

This contains already-retargeted robot trajectories. It must not be confused with the original human MANO dataset.

- Website: <https://huggingface.co/datasets/retarget/retarget_full>
- Exact revision: `1bb88b0c32b33eb868def58333a516ad2f2aba30`
- Extracted location: `example_datasets/retarget_full`
- Archive size: `10,229,431,953` bytes
- Archive SHA256: `5e690d3a699efbb3fca3304925723f2fa034877cac1a421125f8eebfca19eca3`
- Extracted size: about 17 GB
- Trajectories: 7,876 successful right-hand trajectories
- Datasets: DexYCB 1,111; HOT3D v2 5,479; HRDexDB 973; OakInk 313
- Robot counts: Allegro 2,270; Xhand 2,048; Inspire 1,946; Sharpa 1,612
- Verification: 89,902 indexed file checksums and 89,905 remote file identities
- Receipt: `outputs/sharpa_complex/full_dataset_download.json`
- Downloader state: `example_datasets/.retarget_full-download/`

Download the same revision from repository root:

```bash
conda activate spider
python -m examples.lifted_bench.download_release \
  --output-dir example_datasets/retarget_full \
  --revision 1bb88b0c32b33eb868def58333a516ad2f2aba30
```

The official repository downloader checks the archive and extracted files. It keeps transfer state in a hidden sibling directory. Use a new directory if its saved revision differs. `examples/arm_hand/download_full_parallel.py` was an acceleration/resume helper after the normal downloader had initialized its state; it is not a standalone fresh-download entry point.

The release is single-right-hand/rigid-object data and contains no source robot arm. Release validation is state replay; original collision geometry equivalence is not established. A saved state trajectory is not proof that original controls reproduce it. Our target tests independently replay their generated controls.

## 5. Dataset B: original full DexYCB human data

- Official project/download page: <https://dex-ycb.github.io/>
- Official toolkit/data-field documentation: <https://github.com/NVlabs/dex-ycb-toolkit>
- Full Google Drive archive ID: `18fD8RtWJM_fBi3bsuQTzdjIKxdRYtl57`
- File: `dex-ycb-20210415.tar.gz`
- Exact size: `127,611,213,470` bytes (~127.6 GB decimal / 118.85 GiB)
- Local calculated SHA256: `f5b8d68cefde1cba71590ce1f2ed15ff611e0bf6001faea7e4396b5684942f8e`
- Directory: `/home/yimu12/work/spider/example_datasets/raw/dexycb`
- **Current status: complete**, all 10 subjects, 581,968 label files
- Receipt: `example_datasets/raw/dexycb/download_status.json`
- Archive digest file: `example_datasets/raw/dexycb/archive_sha256.txt`
- Archive: `example_datasets/raw/dexycb/downloads/dex-ycb-20210415.tar.gz`

The SHA256 above was calculated locally; do not present it as a checksum published by DexYCB. The downloader checked exact HTTP byte ranges and sizes, assembled the archive, extracted it with gzip integrity checking, and validated directory layout plus an annotation sample. It did not individually verify every NPZ against a publisher manifest.

For a new download, install downloader dependencies into its isolated local directory and run the preserved downloader:

```bash
cd /home/yimu12/work/spider
conda activate spider
mkdir -p example_datasets/raw/dexycb/.download-tools
python -m pip install --target example_datasets/raw/dexycb/.download-tools gdown==6.4.2
python -u examples/arm_hand/download_dexycb.py
```

It uses 16 parallel connections with 128 MiB chunks, ten retry attempts per chunk, exact `Content-Range` verification, and retained complete chunks. Output progresses through downloading → assembling → extracting → validating → complete. To resume an interrupted transfer, rerun the same command after confirming no other downloader is active. On the current machine **do not rerun the downloader unnecessarily**: an existing complete archive would be extracted again. No download process needs to remain running now.

Disk use at handoff is about **393 GB including retained chunks, archive and extracted data**. Allow substantial headroom for a new transfer; do not infer extracted size from compressed size. Do not delete retained data without user authorization.

Expected layout:

```text
example_datasets/raw/dexycb/
  20200709-subject-01/
  ... nine other subject directories ...
  calibration/
  models/
  downloads/
  download_status.json
```

Useful separate official downloads, if only assets/calibration are needed:

| Archive | Google Drive ID |
|---|---|
| calibration.tar.gz | `1UAwVKT4Rgb1fLcFoa1o71_-0NtSvvLAQ` |
| models.tar.gz | `1cAzlQBpcTatI5ykYQ8ziQiHLUG_a_UpM` |

For example: `python -m gdown FILE_ID -O /path/to/archive.tar.gz --continue` after installing gdown in the selected Python environment. The downloader for this experiment was isolated instead of modifying the existing environment.

DexYCB's data license is CC BY-NC 4.0; refer to the official project. MANO model files are separately obtained from <https://mano.is.tuebingen.mpg.de/>. Our current skeleton-based example does not require `MANO_LEFT.pkl` or `MANO_RIGHT.pkl`. To decode PCA pose parameters or render the actual hand surface, obtain those model assets through their official process and implement/test that additional loader.

### Exact human sequence and fields

- Sequence: `20200709-subject-01/20200709_143257`
- Camera: `932122060861`
- Original valid frames: **1–69 inclusive**, 30 Hz; frame 0 has an invalid hand annotation
- Object: DexYCB class ID **5**, YCB model **006_mustard_bottle**. Do not confuse class ID 5 with YCB's three-digit model prefix 006.
- `joint_3d`: `(1,21,3)` in camera coordinates, metres
- `pose_y`: `(number_of_objects,3,4)` object-to-camera rotation and translation
- `pose_m`: `(1,51)` MANO pose coefficients and translation; stored for provenance but not decoded in our test
- `meta.yml`: grasped object index, hand side and sequence metadata
- `aligned_depth_to_color_XXXXXX.png`: depth used for table-plane fitting; millimetres converted to metres
- `color_XXXXXX.jpg`: original RGB shown in the comparison video

Joint order: wrist, thumb joints/tip, index joints/tip, middle joints/tip, ring joints/tip, little joints/tip. Matched tip indices are `[4,8,12,16]`.

## 6. Exact reproduction on the current workspace

These commands overwrite their respective output files. Preserve a copy first if retaining a new run separately is important.

### A. Repeat the latest 100-sample comparison, using the frozen 512-sample reference

```bash
cd /home/yimu12/work/spider
conda activate spider
OPENBLAS_NUM_THREADS=4 python -m examples.arm_hand.rerun_mano_sharpa --samples 100
MUJOCO_GL=egl python -m examples.arm_hand.rerun_mano_sharpa --samples 100 --record-only
```

Results: `outputs/mano_to_sharpa_samples_100/`.

This reuses the exact saved `scene.xml`, `reference.npz`, IK initialization and full initial physics state from `outputs/mano_to_sharpa`. It asserts all five nominal-baseline metrics match the original to `1e-12`. It then calls the unchanged optimizer with 100 samples, 20 iterations, seed 0 and eight threads. The unchanged top-10-percent rule means 10 elites rather than 51. No new IK run is needed for this controlled comparison.

### B. Repeat the 512-sample run from the original human sequence

```bash
cd /home/yimu12/work/spider
conda activate spider
OPENBLAS_NUM_THREADS=4 python -m examples.arm_hand.retarget_mano_sharpa
MUJOCO_GL=egl python -m examples.arm_hand.record_mano_sharpa
```

Results: `outputs/mano_to_sharpa/`. This rebuilds the human reference and IK and reruns physics optimization. `samples=512`, iterations and other settings are hardcoded in this original wrapper. The budget-comparison wrapper supports `--samples`.

Dependencies that must already exist, all preserved by the reproduction supplement or full dataset download:

- `outputs/allegro_to_sharpa_left/scene.xml`: generic-arm/Sharpa scene template
- `outputs/allegro_to_sharpa_left/model_audit.json`: URDF torque and speed ratings
- `outputs/sharpa_more/selected.json`: selects the mustard object's processed source scene/geometry
- Referenced processed source scene and meshes from `retarget_full`
- `example_datasets/.retarget_full-download/candidate.json`: validation code reads the pinned revision here
- `outputs/mano_to_sharpa/source_assets/intrinsics.yml`
- `outputs/mano_to_sharpa/source_assets/original_mustard_bottle.obj`
- Existing helper modules under `examples/arm_hand/` and `spider/robot_model.py`; fingerprinting also reads earlier helper files

The fresh conversion can source the original bottle mesh from `RAW/models/006_mustard_bottle/textured_simple.obj` when its copied asset is absent. It still expects the copied intrinsics file. To bootstrap that file after extracting the full dataset:

```bash
mkdir -p outputs/mano_to_sharpa/source_assets
cp example_datasets/raw/dexycb/calibration/intrinsics/932122060861_640x480.yml \
   outputs/mano_to_sharpa/source_assets/intrinsics.yml
```

Do not delete `source_assets` when cleaning outputs. The snapshot already includes both required copies. `/tmp/dexycb-models` and `/tmp/dexycb-calibration` were temporary download staging locations; reproduction should use the persistent copies/full dataset, not rely on `/tmp` surviving.

### C. Repeat the 512-sample optimization with an exactly frozen reference

```bash
OPENBLAS_NUM_THREADS=4 python -m examples.arm_hand.rerun_mano_sharpa --samples 512
MUJOCO_GL=egl python -m examples.arm_hand.rerun_mano_sharpa --samples 512 --record-only
```

This writes `outputs/mano_to_sharpa_samples_512/` and preserves the original 512 directory. It is the appropriate control if comparing optimization alone rather than rebuilding the conversion and IK.

## 7. Conversion and optimization details

### Reference conversion

1. Read the original human `joint_3d` and grasped `pose_y` annotations; discard invalid hand frames.
2. Compare original object geometry to cached processed geometry and account for a fixed mesh-origin translation. The matched vertex distance p99 in this run is effectively zero.
3. Fit the tabletop from first-frame aligned depth using official color intrinsics. Deterministic RANSAC uses seed 0, 8,000 sampled points, 400 candidate planes, 4 mm inlier threshold, background segmentation only, depth 0.5–1.6 m, then SVD refinement. Inlier fraction ~0.634; RMS ~2.07 mm.
4. Align the plane normal with world vertical, place the object in arm workspace, and seat it 2 mm above the simulated table. Apply one consistent rigid placement to all human landmarks and object poses. Do not rescale the human hand.
5. Palm target is the midpoint of wrist and middle MCP; orientation axes are derived from MCP landmarks.
6. Interpolate positions linearly and rotations with Slerp to 100 Hz. Mirror world/local Y consistently for left Sharpa, including object mesh geometry.
7. Match thumb/index/middle/ring; pinky uses a posture prior.

**Trajectory counts:** 69 original frames → 227 interpolated motion poses → append 50 hold poses → 277 reference poses and **276 controls**. The optimized horizon is 2.76 s. Validation adds another 0.5 s of the final command, outside the optimization horizon. The 100-sample rerun preserves these counts.

### Optimization parameters

The complete resolved model, joint/actuator/contact options and optimizer settings are in `outputs/mano_to_sharpa/optimization_parameters.json` and the corresponding 100-sample directory. Those JSONs were exported during this conversation; the original 512 wrapper does not itself regenerate that parameter export.

| Setting | Value |
|---|---|
| Sampling iterations | 20 |
| Samples | 512 original; 100 comparison |
| RNG seed | 0 |
| Decision variables | 10 knots × 29 actuators = 290 |
| Actuators | 7 arm + 22 hand |
| Knot control indices | 0,30,61,91,122,152,183,213,244,275 |
| Interpolation | Linear |
| Gaussian noise std | Arm 0.035 rad; hand 0.12 rad |
| Noise decay | `0.97**i`, i=0…19 |
| Candidate zero | Current center without noise |
| Offset bounds | ±0.4 rad; final commands clipped to actuator limits |
| Elite fraction | Top 10%; 51 elites at 512, 10 at 100 |
| Elite normalization | `(reward - elite_mean)/(elite_std + 0.01)` |
| Softmax temperature | 1.0 |
| Returned solution | Best evaluated candidate, including nominal baseline |
| Backend | CPU MuJoCo rollout, 8 threads; Torch compile disabled |
| Dynamic tracking checkpoints | Every 3 controls, 92 checkpoints |
| Domain randomization | None |

Dynamic cost is `0.5 * sum(weighted_residual**2)`. Residual multipliers, not already-squared cost weights:

| Residual | Multiplier |
|---|---:|
| Palm position, m | 15 |
| Object position, m | 60 |
| Palm rotation, rad | 0.8 |
| Object rotation, rad | 0.3 |
| Hand joint deviation from IK, rad | 0.05 |
| Knot offsets, rad | 0.03 |
| Adjacent-knot differences, rad | 0.02 |

There is no additional terminal multiplier, direct dynamic fingertip objective, explicit contact-force objective, or energy objective in this adapter. Do not read unrelated `Config` defaults as settings actually used here.

IK uses SciPy bounded least squares (`trf`, two-point finite differences, linear loss, `ftol=xtol=gtol=1e-8`, max_nfev=80 per fitted frame), stride 2, 139 fitted poses. Initial arm seed is `[0.5,0.2,-0.4,2.1,0,-2.5,0.3]`, hand seed zero, clipped within limits with 0.0001 rad margin. After the first frame, step bounds are 0.08 rad arm and 0.16 rad finger. IK residual multipliers: palm position 5, palm rotation 1, fingertips 40, hand–object penetration 180 with 1 mm allowance, temporal joint change 0.015. Interpolated joints are smoothed with Gaussian sigma 3 frames. Nominal commands include velocity compensation: hand `q + clip(kv*dq/kp, -.15, .15)`, arm `q + .05*dq`, then actuator-range clipping.

Physics: 2 ms implicitfast, five substeps per 10 ms control, gravity `[0,0,-9.81]`, Newton solver, 40 solver iterations, tolerance `1e-8`. Bottle mass ~0.474314 kg. Explicit hand–object pairs use condim 4 and friction `[1,1,0.1,0,0]`. Finger force bounds come from supplied URDF (about 0.189–3.3 Nm), arm bounds ±100 Nm. Native servo settings are preserved. Speed ratings are checked after simulation, not enforced. The object is free and unactuated, with no weld or reference-pose reset during target replay.

## 8. Current measured results and acceptance criteria

| Metric | 512 samples | 100 samples |
|---|---:|---:|
| Optimized cost | 137.1139 | 162.5064 |
| Object lift | 22.9036 cm | 21.7067 cm |
| Human reference lift | 23.6118 cm | same |
| Final position error | 11.2206 mm | 46.0202 mm |
| Object-position RMSE | 9.7498 mm | 12.3604 mm |
| Final orientation error | 50.9400° | 69.7687° |
| Extra-hold object displacement relative to palm | 5.5681 mm | 239.2009 mm |
| Retention | Pass | Fail / drops |
| Complete pose + retention | Fail | Fail |
| Within every supplied motor speed rating | No | No |

Nominal baseline cost is 4152.596957304195 in both comparisons. This is a single-seed observation, not a statistical sample-budget conclusion.

Acceptance criteria used by the test: position error <30 mm, final rotation error <20°, extra-hold relative displacement <30 mm, and retained lift >100 mm. Dynamic control replay is checked with atol/rtol `1e-9`. Neither current run passes the complete criterion, even though 512 lifts and retains the bottle. Torque limits hold; speed ratings can be exceeded. IK has no self-collision penalty, with ~12 mm maximum raw self-penetration reported. The arm is procedural, with inferred mounting/palm alignment, not calibrated hardware. These are integration experiments, not hardware-ready solutions.

The directory `outputs/mano_to_sharpa_alignment_debug` is an **invalid earlier scene alignment**, assuming the initially lying bottle was upright. Do not use it as the final result, training target, or baseline. The current run uses depth-derived tabletop alignment and retains the lying-to-lift motion.

### Important current files

| Path under repository | Purpose |
|---|---|
| `examples/arm_hand/retarget_mano_sharpa.py` | Human conversion, IK, 512-sample run |
| `examples/arm_hand/rerun_mano_sharpa.py` | Frozen-reference sample-budget comparison |
| `examples/arm_hand/record_mano_sharpa.py` | Original RGB + human skeleton + target dynamics video |
| `examples/arm_hand/run_sharpa_complex.py` | Existing build/IK/replay/hold validation functions |
| `examples/arm_hand/try_sharpa.py` | Existing sampling rollout/objective adapter |
| `spider/optimizers/sampling.py` | Imported repository sampling update |
| `outputs/mano_to_sharpa/comparison.mp4` | 512-sample video |
| `outputs/mano_to_sharpa_samples_100/comparison.mp4` | Latest 100-sample video |
| `outputs/mano_to_sharpa_samples_100/comparison_512.json` | Direct numerical comparison |
| `reference.npz` in each case | Frozen joint IK, nominal controls, goals, full initial state |
| `trajectory.npz` | Optimized controls and resulting states, 10 ms control / 2 ms physics |
| `human_reference.npz` | Human joints, object trajectory, raw frame IDs, source pose_m |
| `conversion.json` | Input checksums, sequence and frame/mesh/table mapping |
| `retargeting.json` | IK errors and self-penetration report |
| `report.json` | Metrics, hold check, ratings, method fingerprints |

The original RGB and human skeleton panels are references; they freeze during hold phases. Only the right-hand panel is the controlled target physics. Never describe the skeleton as a reconstructed MANO surface mesh.

## 9. Earlier demonstrations and results

User-provided models:

- Yimu source ZIP: `/home/yimu12/Downloads/yimu_v2_hands_20260922.zip`
- Sharpa source: `/home/yimu12/work/simtoolreal/assets/urdf/left_sharpa_ha4/`
- Wuji source: `/home/yimu12/work/wuji-description/hand2/hand2_2/`
- Imported models: `spider/assets/robots/yimu_v2`, `yimu_v2_active_distal`, `sharpa_ha4`, `wuji_hand2_2`

The current-test supplement preserves Sharpa assets, not every historical hand/source archive. To rerun all historical tests, preserve the full existing workspace or separately copy those models and output dependencies. User-provided source paths may not exist on another machine.

| Output directory | Experiment |
|---|---|
| `outputs/arm_hand`, `arm_hand_external` | Early arm+hand tracking tests |
| `outputs/allegro_to_xhand` | Cross-hand reference → arm+Xhand |
| `outputs/sampling_ablation` | Sample-count failure tests, including one sample |
| `outputs/allegro_to_yimu_v2`, `_slow` | Yimu with arm |
| `outputs/yimu_active_distal` | Separate uncoupled/actuated distal-joint variant |
| `outputs/yimu_torque_sweep` | Yimu torque/contact experiments |
| `outputs/xhand_coke_to_yimu`, `_slow_hold` | Different source-hand/object transfer |
| `outputs/xhand_coke_to_wuji` | Wuji transfer |
| `outputs/allegro_to_sharpa_left` | Sharpa model audit and foundational scene template |
| `outputs/sharpa_complex` | Three harder processed-reference cases |
| `outputs/sharpa_more` | Ten further processed-reference cases |

Earlier three-case results:

| Object | Final position error | Rotation error | Retained |
|---|---:|---:|---|
| Scissors | 9.0 mm | 32.2° | Yes |
| Mug | 16.5 mm | 50.9° | Yes |
| Power drill | 2.6 mm | 16.6° | Yes |

Ten-more results:

| Object | Position error | Rotation error | Retained | Complete criterion |
|---|---:|---:|---|---|
| Banana | 24.3 mm | 29.3° | Yes | Fail |
| Bowl | 262.8 mm | 66.3° | No | Fail |
| Sugar box | 6.5 mm | 93.4° | Yes | Fail |
| Mustard bottle | 7.1 mm | 12.8° | Yes | Pass |
| Bleach bottle | 21.6 mm | 57.9° | Yes | Fail |
| Pitcher | 29.2 mm | 39.3° | Yes | Fail |
| Wood block | 159.8 mm | 92.7° | No | Fail |
| Pudding box | 9.2 mm | 16.2° | Yes | Pass |
| Gelatin box | 348.7 mm | 123.8° | No | Fail |
| Clamp | 11.9 mm | 32.6° | Yes | Fail |

These used robot-hand references from the processed release, unlike the current human-reference experiment. Some speed ratings were exceeded; a criterion pass is not a hardware validation. Selection was frozen before target tests, with ten distinct objects and maximum source net rotation per object among available Allegro/DexYCB episodes.

Original command entry points for those batches:

```bash
OPENBLAS_NUM_THREADS=4 python -m examples.arm_hand.run_sharpa_complex
MUJOCO_GL=egl python -m examples.arm_hand.record_sharpa_complex
OPENBLAS_NUM_THREADS=4 python -m examples.arm_hand.run_sharpa_more
MUJOCO_GL=egl python -m examples.arm_hand.record_sharpa_more
```

Both batch runners reuse completed case reports. Preserving outputs means rerunning the command may skip already completed work; choose a deliberate new-output strategy if a genuinely new experiment is requested. Current MANO/frozen-budget wrappers instead overwrite their target output files.

## 10. Checks for the next agent

1. Read this file and the current reports before drawing conclusions.
2. Check `conda activate spider` resolves to the expected environment and preserve its runtime versions.
3. Check original DexYCB status is complete and do not repeat the 127.6 GB download unnecessarily.
4. Distinguish raw human data from already-retargeted robot trajectories.
5. Preserve the original 512 result and use the frozen-reference wrapper for controlled sample-budget comparisons.
6. Check method fingerprints in reports; existing local compatibility edits must not be silently reverted.
7. Validate dynamic control replay and the extra hold, not only a posed/rendered hand or the final optimization cost.
8. Report rotation errors, speed-rating violations, self-collision limitations and failures honestly.
9. If moving workspaces, resolve saved absolute paths before debugging optimization behavior.
10. Do not assume changing Yimu torque/coupling or increasing sample count guarantees success. Further experiments require explicit scope and separate outputs.

No PPO policy was trained by the current tests. No new test run was needed to create this handoff; its numbers come from saved, independently replay-verified reports.
