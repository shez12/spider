"""Retarget original DexYCB human hand annotations to the existing Sharpa arm."""

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import trimesh
import yaml
from scipy.spatial.transform import Rotation, Slerp

from examples.arm_hand.run_sharpa_complex import ROOT, build, fit, validate
from examples.arm_hand.run_sharpa_more import hashes
from examples.arm_hand.try_sharpa import S, mirror_quat, optimize

OUT = ROOT / "outputs/mano_to_sharpa"
RAW = ROOT / "example_datasets/raw/dexycb"
SEQ = RAW / "20200709-subject-01/20200709_143257"
CAMERA = "932122060861"


def convert():
    """Extract human keypoints directly; never read robot reference trajectories."""
    meta = yaml.safe_load((SEQ / "meta.yml").read_text())
    assert meta["mano_sides"] == ["right"]
    assert meta["ycb_ids"][meta["ycb_grasp_ind"]] == 5
    frames, joints, objects, poses, inputs = [], [], [], [], {}
    for path in sorted((SEQ / CAMERA).glob("labels_*.npz")):
        with np.load(path) as f:
            j = f["joint_3d"][0]
            if (j == -1).any() or not np.isfinite(j).all():
                continue
            frames.append(int(path.stem[-6:]))
            joints.append(j)
            objects.append(f["pose_y"][meta["ycb_grasp_ind"]])
            poses.append(f["pose_m"][0])
            inputs[str(path.relative_to(ROOT))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    frames, joints, objects = np.array(frames), np.array(joints), np.array(objects)
    assert np.all(np.diff(frames) == 1)
    # The original and cached object meshes must agree up to a fixed translation.
    selected = next(
        r
        for r in json.loads((ROOT / "outputs/sharpa_more/selected.json").read_text())
        if r["label"] == "mustard_bottle"
    )
    scene = ROOT / "example_datasets/retarget_full" / selected["scene"]
    cached = trimesh.load(
        ROOT
        / "example_datasets/retarget_full/processed/dexycb_lifted/assets/objects"
        / selected["task"]
        / "visual.obj",
        force="mesh",
    )
    original_path = OUT / "source_assets/original_mustard_bottle.obj"
    original_path.parent.mkdir(exist_ok=True)
    if not original_path.exists():
        candidates = [
            RAW / "models/006_mustard_bottle/textured_simple.obj",
            Path("/tmp/dexycb-models/models/006_mustard_bottle/textured_simple.obj"),
        ]
        shutil.copy2(next(p for p in candidates if p.exists()), original_path)
    original = trimesh.load(original_path, force="mesh")
    shift = original.bounds.mean(axis=0) - cached.bounds.mean(axis=0)
    np.testing.assert_allclose(original.extents, cached.extents, atol=0.0001)
    # Validate mesh-frame translation against sampled vertices in both directions.
    from scipy.spatial import cKDTree

    distances = cKDTree(cached.vertices).query(original.vertices[::10] - shift)[0]
    if np.quantile(distances, 0.99) > 0.002:
        raise RuntimeError("Object mesh frame mismatch")
    # Estimate the actual table plane from aligned depth and camera intrinsics.
    intr = yaml.load(
        (OUT / "source_assets/intrinsics.yml").read_text(), Loader=yaml.FullLoader
    )["color"]
    depth = (
        cv2.imread(
            str(SEQ / CAMERA / f"aligned_depth_to_color_{frames[0]:06d}.png"), -1
        )
        * 0.001
    )
    with np.load(SEQ / CAMERA / f"labels_{frames[0]:06d}.npz") as first:
        mask = (depth > 0.5) & (depth < 1.6) & (first["seg"] == 0)
    v, u = np.where(mask)
    dz = depth[v, u]
    points = np.c_[
        (u - intr["ppx"]) * dz / intr["fx"], (v - intr["ppy"]) * dz / intr["fy"], dz
    ]
    rng = np.random.default_rng(0)
    points = points[rng.choice(len(points), 8000, replace=False)]
    best = np.zeros(len(points), bool)
    for _ in range(400):
        pa, pb, pc = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(pb - pa, pc - pa)
        if np.linalg.norm(normal) < 1e-8:
            continue
        normal /= np.linalg.norm(normal)
        candidate = abs(points @ normal - normal @ pa) < 0.004
        if candidate.sum() > best.sum():
            best = candidate
    center = points[best].mean(0)
    _, _, vh = np.linalg.svd(points[best] - center, full_matrices=False)
    normal = vh[-1]
    if normal @ center > 0:
        normal = -normal
    offset = -normal @ center
    assert best.mean() > 0.4
    axis_x = np.array([1.0, 0.0, 0.0]) - normal[0] * normal
    axis_x /= np.linalg.norm(axis_x)
    axis_y = np.cross(normal, axis_x)
    align = np.stack([axis_x, axis_y, normal])
    r = align[None] @ objects[:, :, :3]
    initial = objects[0, :, 3]
    signed = (original.vertices @ objects[0, :, :3].T + initial) @ normal + offset
    assert -0.01 < signed.min() < 0.02
    translation = np.array([-0.10, 0.02, 0.0]) - align @ initial
    translation[2] = offset + 0.002 - signed.min()
    joints = joints @ align.T + translation
    p = objects[:, :, 3] @ align.T + translation + np.einsum("tij,j->ti", r, shift)
    plane = {
        "normal_camera": normal.tolist(),
        "offset_m": float(offset),
        "inlier_fraction": float(best.mean()),
        "rms_m": float(np.sqrt(np.mean((points[best] @ normal + offset) ** 2))),
        "initial_mesh_clearance_before_seating_m": float(signed.min()),
        "initial_mesh_clearance_m": 0.002,
    }
    z = joints[:, 9] - joints[:, 0]
    z /= np.linalg.norm(z, axis=1)[:, None]
    y = joints[:, 5] - joints[:, 13]
    x = np.cross(y, z)
    x /= np.linalg.norm(x, axis=1)[:, None]
    y = np.cross(z, x)
    palm_rotation = np.stack([x, y, z], axis=2)
    palm = (joints[:, 0] + joints[:, 9]) / 2
    t = (frames - frames[0]) / 30.0
    grid = np.arange(0, t[-1] + 1e-8, 0.01)

    def linear(a):
        shape = a.shape[1:]
        return np.column_stack(
            [np.interp(grid, t, v) for v in a.reshape(len(t), -1).T]
        ).reshape((len(grid),) + shape)

    jr = linear(joints)
    pr = linear(palm)
    oq = Slerp(t, Rotation.from_matrix(r))(grid).as_quat()[:, [3, 0, 1, 2]]
    wq = Slerp(t, Rotation.from_matrix(palm_rotation))(grid).as_quat()[:, [3, 0, 1, 2]]
    op = linear(p)
    ref = {
        "wrist_pos": pr @ S,
        "wrist_quat": mirror_quat(wq),
        "fingertips": jr[:, [4, 8, 12, 16]] @ S,
        "object_qpos": np.c_[op @ S, mirror_quat(oq)],
    }
    for key, value in ref.items():
        ref[key] = np.concatenate([value, np.repeat(value[-1:], 50, axis=0)])
    np.savez(
        OUT / "human_reference.npz",
        joints=jr,
        object_qpos=np.c_[op, oq],
        time=grid,
        frame_ids=frames,
        pose_m=poses,
        raw_camera_joints=np.array(
            [
                np.load(SEQ / CAMERA / f"labels_{f:06d}.npz")["joint_3d"][0]
                for f in frames
            ]
        ),
    )
    np.savez(OUT / "source_reference.npz", **ref)
    conversion = {
        "sequence": str(SEQ),
        "camera": CAMERA,
        "source": "original DexYCB human MANO annotations; joint_3d",
        "source_frame_ids": frames.tolist(),
        "source_fps": 30,
        "control_fps": 100,
        "source_duration_s": float(t[-1]),
        "resampled_frames": len(grid),
        "mesh_frame_translation_m": shift.tolist(),
        "mesh_vertex_distance_p99_m": float(np.quantile(distances, 0.99)),
        "alignment": "Depth-fitted table plane establishes gravity; global rigid placement only; no human hand scaling",
        "table_plane": plane,
        "palm_frame": "midpoint of wrist and middle MCP; axes from index/ring/middle MCP landmarks",
        "reference_render": "21-joint human skeleton, not MANO surface mesh; separately licensed model not installed",
        "source_sha256": inputs,
    }
    (OUT / "conversion.json").write_text(json.dumps(conversion, indent=2) + "\n")
    return scene, ref, conversion


def main():
    """Run unchanged IK and sampling optimization, then independent replay checks."""
    OUT.mkdir(exist_ok=True)
    before = hashes()
    scene, ref, conversion = convert()
    model, _ = build(scene, OUT)
    ref = fit(model, ref, OUT, 2)
    optimize(
        SimpleNamespace(output_dir=OUT, samples=512, iterations=20, threads=8),
        model,
        ref,
    )
    report = validate(
        OUT,
        model,
        {
            "dataset": "DexYCB original human annotations",
            "task": SEQ.name,
            "robot_type": "human_MANO",
            "label": "mustard_bottle",
        },
    )
    assert before == hashes()
    report.update(
        method_files_unchanged=True, method_sha256=before, conversion=conversion
    )
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
