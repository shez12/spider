"""Retarget three harder full-release cases to the supplied left Sharpa hand."""

import argparse
import copy
import hashlib
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

from examples.arm_hand.test_active_distal import fingerprint
from examples.arm_hand.try_sharpa import S, mirror_quat, optimize
from examples.run_arm_hand import quaternion_error

ROOT = Path(__file__).resolve().parents[2]
FINGERS = ["thumb", "index", "middle", "ring"]


def build(source_scene, output):
    """Keep the supplied Sharpa and substitute a consistently mirrored object."""
    template = ROOT / "outputs/allegro_to_sharpa_left/scene.xml"
    tree = ET.parse(template)
    root = tree.getroot()
    world = root.find("worldbody")
    old = world.find("body[@name='right_object']")
    old_names = {n.get("name") for n in old.iter() if n.get("name")}
    old_meshes = {n.get("mesh") for n in old.findall("geom") if n.get("mesh")}
    world.remove(old)
    source = ET.parse(source_scene)
    obj = copy.deepcopy(source.find("worldbody/body[@name='right_object']"))
    wanted = {n.get("mesh") for n in obj.findall("geom") if n.get("mesh")}
    for node in list(obj):
        if node.get("pos"):
            node.set(
                "pos", " ".join(map(str, S @ np.fromstring(node.get("pos"), sep=" ")))
            )
        if node.get("quat"):
            node.set(
                "quat",
                " ".join(
                    map(str, mirror_quat(np.fromstring(node.get("quat"), sep=" ")))
                ),
            )
    obj.find("geom[@name='right_object_visual']").set("rgba", ".85 .27 .06 1")
    world.append(obj)
    asset = root.find("asset")
    for node in list(asset):
        if node.get("name") in old_meshes:
            asset.remove(node)
    meshroot = (source_scene.parent / source.find("compiler").get("meshdir")).resolve()
    for node in source.findall("asset/mesh"):
        if node.get("name") not in wanted:
            continue
        node = copy.deepcopy(node)
        node.set("file", str(meshroot / node.get("file")))
        scale = np.fromstring(node.get("scale", "1 1 1"), sep=" ") * [1, -1, 1]
        node.set("scale", " ".join(map(str, scale)))
        asset.append(node)
    contact = root.find("contact")
    for node in list(contact):
        if any(v in old_names for v in node.attrib.values()):
            contact.remove(node)
    object_names = {n.get("name") for n in obj.findall("geom")}
    floors = []
    for node in source.findall("contact/pair"):
        names = [node.get("geom1"), node.get("geom2")]
        if "floor" in names and any(n in object_names for n in names):
            contact.append(copy.deepcopy(node))
            floors.append(node)
    assert len(floors) == len(obj.findall("geom")) - 1
    for geom in world.findall(".//geom"):
        if geom.get("name", "").startswith("collision_sharpa_"):
            for og in obj.findall("geom"):
                if og.get("name") != "right_object_visual":
                    ET.SubElement(
                        contact,
                        "pair",
                        geom1=geom.get("name"),
                        geom2=og.get("name"),
                        condim="4",
                        friction="1 1 0.1 0 0",
                    )
    ET.indent(root)
    tree.write(output / "scene.xml", encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(output / "scene.xml"))
    original = mujoco.MjModel.from_xml_path(str(template))
    for field in [
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_forcerange",
        "actuator_ctrlrange",
    ]:
        np.testing.assert_array_equal(getattr(model, field), getattr(original, field))
    source_model = mujoco.MjModel.from_xml_path(str(source_scene))
    np.testing.assert_allclose(
        model.body_mass[model.body("right_object").id],
        source_model.body_mass[source_model.body("right_object").id],
        rtol=1e-6,
    )
    return model, source_model


def reference(source_model, source_file, output, hold):
    """Read only a source-hand trajectory and mirror it for the left hand."""
    with np.load(source_file) as f:
        q = f["qpos"].reshape(-1, source_model.nq)
    original_frames = len(q)
    q = np.vstack([q, np.repeat(q[-1:], round(hold / 0.01), axis=0)])
    np.savez(
        output / "chosen_sample.npz",
        qpos=q,
        control_dt=0.01,
        original_frames=original_frames,
    )
    data = mujoco.MjData(source_model)
    positions, quats, tips = [], [], []
    for pose in q:
        data.qpos[:] = pose
        mujoco.mj_forward(source_model, data)
        site = data.site("right_palm")
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, site.xmat)
        positions.append(site.xpos.copy())
        quats.append(quat.copy())
        tips.append(
            [
                data.site(f"track_hand_right_{finger}_tip").xpos.copy()
                for finger in FINGERS
            ]
        )
    objects = q[:, -7:].copy()
    objects[:, :3] = objects[:, :3] @ S
    objects[:, 3:] = mirror_quat(objects[:, 3:])
    ref = {
        "wrist_pos": np.array(positions) @ S,
        "wrist_quat": mirror_quat(np.array(quats)),
        "fingertips": np.array(tips) @ S,
        "object_qpos": objects,
    }
    np.savez(output / "source_reference.npz", **ref)
    return ref, original_frames


class InitialGeometryRejected(RuntimeError):
    """Necessary initial-frame checks fail before the remaining IK is solved."""
    def __init__(self,report):
        super().__init__('Initial-frame geometry fails the unchanged final IK gate')
        self.report=report


def fit(model, ref, output, stride, initial_arm_pose=None, first_frame_restarts=1,reject_bad_initial=False):
    """Use the prior wrist/fingertip/penetration IK recipe with native limits."""
    low, high = model.actuator_ctrlrange.T
    arm_seed = (
        np.asarray(initial_arm_pose, dtype=float)
        if initial_arm_pose is not None
        else np.array([0.5, 0.2, -0.4, 2.1, 0, -2.5, 0.3])
    )
    if arm_seed.shape != (7,):
        raise ValueError(f"initial_arm_pose must have shape (7,), got {arm_seed.shape}")
    seed = np.r_[arm_seed, np.zeros(22)]
    seed = np.clip(seed, low + 0.0001, high - 0.0001)
    data = mujoco.MjData(model)
    hand_ids = [
        g
        for g in range(model.ngeom)
        if (model.geom(g).name or "").startswith("collision_sharpa_")
    ]
    hand_index = {g: i for i, g in enumerate(hand_ids)}
    robot_ids = [g for g in range(model.ngeom)
                 if (model.geom(g).name or "").startswith(("collision_sharpa_", "collision_kuka_"))]
    robot_index = {g: i for i, g in enumerate(robot_ids)}
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    object_ids = {
        g
        for g in range(model.ngeom)
        if (model.geom(g).name or "").startswith("right_object_")
        and model.geom(g).name != "right_object_visual"
    }
    grid = np.arange(len(ref["wrist_pos"]))
    indices = np.unique(np.r_[grid[::stride], grid[-1]])
    fitted, errors, penetrations = [], [], []
    for i in indices:
        previous = seed.copy()

        def residual(joints, i=i, previous=previous):
            data.qpos[:29] = joints
            data.qpos[-7:] = ref["object_qpos"][i]
            # Residuals need only position sensors and contact geometry. The
            # velocity, actuation and acceleration solvers cannot affect them.
            mujoco.mj_fwdPosition(model, data)
            mujoco.mj_sensorPos(model, data)
            penetration = np.zeros(len(hand_ids))
            self_penetration = np.zeros(len(hand_ids))
            table_penetration = np.zeros(len(robot_ids))
            for c in data.contact:
                a, b = int(c.geom1), int(c.geom2)
                if a in hand_index and b in hand_index:
                    depth = max(0.0, -c.dist - 0.0005)
                    self_penetration[hand_index[a]] = max(self_penetration[hand_index[a]], depth)
                    self_penetration[hand_index[b]] = max(self_penetration[hand_index[b]], depth)
                if a == floor_id and b in robot_index:
                    table_penetration[robot_index[b]] = max(table_penetration[robot_index[b]], -c.dist)
                if b == floor_id and a in robot_index:
                    table_penetration[robot_index[a]] = max(table_penetration[robot_index[a]], -c.dist)
                if a in hand_index and b in object_ids:
                    penetration[hand_index[a]] = max(
                        penetration[hand_index[a]], -c.dist - 0.001
                    )
                if b in hand_index and a in object_ids:
                    penetration[hand_index[b]] = max(
                        penetration[hand_index[b]], -c.dist - 0.001
                    )
            return np.r_[
                (100 if i == 0 else 5) * (data.sensordata[:3] - ref["wrist_pos"][i]),
                (5 if i == 0 else 1) * quaternion_error(data.sensordata[3:7], ref["wrist_quat"][i]),
                40
                * (data.sensordata[14:26].reshape(4, 3) - ref["fingertips"][i]).ravel(),
                180 * penetration,
                300 * self_penetration,
                500 * table_penetration,
                0.015 * (joints - previous),
            ]

        delta = stride * np.r_[np.full(7, 0.04), np.full(22, 0.08)]
        bounds = (
            (
                np.maximum(low + 0.0001, seed - delta),
                np.minimum(high - 0.0001, seed + delta),
            )
            if i
            else (low + 0.0001, high - 0.0001)
        )
        starts = [seed.copy()]
        if i == 0 and first_frame_restarts > 1:
            rng = np.random.default_rng(0)
            for _ in range(first_frame_restarts - 1):
                candidate = seed.copy()
                candidate[:7] = rng.uniform(low[:7]+0.001, high[:7]-0.001)
                starts.append(candidate)
        best_fit = None
        for start in starts:
            result = least_squares(residual, start, bounds=bounds,
                                   max_nfev=300 if i == 0 and first_frame_restarts > 1 else 80)
            if best_fit is None or result.cost < best_fit.cost:
                best_fit = result
        seed = best_fit.x
        residual(seed)
        if i==0 and reject_bad_initial:
            palm_error=float(np.linalg.norm(data.sensordata[:3]-ref['wrist_pos'][0]))
            tip_errors=np.linalg.norm(data.sensordata[14:26].reshape(4,3)-ref['fingertips'][0],axis=1)
            table_depth=max([0.]+[-float(c.dist) for c in data.contact if
                (int(c.geom1)==floor_id and int(c.geom2) in robot_index) or
                (int(c.geom2)==floor_id and int(c.geom1) in robot_index)])
            self_depth=max([0.]+[-float(c.dist) for c in data.contact if
                int(c.geom1) in hand_index and int(c.geom2) in hand_index])
            alignment=bool(palm_error<.01 and np.max(tip_errors)<.03)
            if not alignment or table_depth>=.001 or self_depth>=.001:
                raise InitialGeometryRejected(dict(ik_pass=False,ready_for_gpu=False,ik_not_solved=True,
                    initial_alignment_pass=alignment,initial_palm_error_m=palm_error,
                    initial_fingertip_errors_m=tip_errors.tolist(),initial_robot_table_penetration_m=table_depth,
                    initial_hand_self_penetration_m=self_depth,
                    reason='Initial frame fails necessary unchanged geometry gate; remaining frames skipped'))
        fitted.append(seed.copy())
        errors.append(
            np.linalg.norm(
                data.sensordata[14:26].reshape(4, 3) - ref["fingertips"][i], axis=1
            )
        )
        penetrations.append(
            max(
                [0.0]
                + [
                    -float(c.dist)
                    for c in data.contact
                    if all(
                        (model.geom(int(g)).name or "").startswith("collision_sharpa_")
                        for g in [c.geom1, c.geom2]
                    )
                ]
            )
        )
        if i % 60 == 0:
            print(f"IK {i}/{len(grid)}", flush=True)
    joints = np.column_stack(
        [np.interp(grid, indices, row) for row in np.array(fitted).T]
    )
    joints = gaussian_filter1d(joints, sigma=3, axis=0, mode="nearest")
    # Start at the solved initial pose; filtering must not move the starting
    # hand away from the dataset-relative object pose.
    joints[0] = fitted[0]
    velocity = np.gradient(joints, 0.01, axis=0)
    kp = model.actuator_gainprm[:, 0]
    kv = -model.actuator_biasprm[:, 2]
    commands = joints + np.clip(kv * velocity / kp, -0.15, 0.15)
    commands[:, :7] = joints[:, :7] + 0.05 * velocity[:, :7]
    commands = np.clip(commands, low, high)
    data = mujoco.MjData(model)
    data.qpos[:29] = joints[0]
    data.qpos[-7:] = ref["object_qpos"][0]
    next_q = np.r_[joints[1], ref["object_qpos"][1]]
    mujoco.mj_differentiatePos(model, data.qvel, 0.01, data.qpos, next_q)
    data.ctrl[:] = commands[0]
    mujoco.mj_forward(model, data)
    initial_palm_error = float(np.linalg.norm(data.sensordata[:3] - ref["wrist_pos"][0]))
    initial_tip_errors = np.linalg.norm(data.sensordata[14:26].reshape(4, 3) - ref["fingertips"][0], axis=1)
    max_table_penetration = 0.0
    for frame, pose in enumerate(joints):
        data.qpos[:29] = pose
        data.qpos[-7:] = ref["object_qpos"][frame]
        mujoco.mj_forward(model, data)
        for contact in data.contact:
            a, b = int(contact.geom1), int(contact.geom2)
            if (a == floor_id and b in robot_index) or (b == floor_id and a in robot_index):
                max_table_penetration = max(max_table_penetration, -float(contact.dist))
    # Restore the initial state after the full-path geometry audit.
    data.qpos[:29] = joints[0]
    data.qpos[-7:] = ref["object_qpos"][0]
    mujoco.mj_forward(model, data)
    initial = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS))
    mujoco.mj_getState(model, data, initial, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    ref.update(joint_ik=joints, ctrl_nominal=commands[1:], initial_state=initial)
    np.savez(output / "reference.npz", **ref)
    (output / "retargeting.json").write_text(
        json.dumps(
            {
                "mean_raw_ik_fingertip_error_m": float(np.mean(errors)),
                "max_raw_self_penetration_m": float(max(penetrations)),
                "ik_stride": stride,
                "matched_fingers": FINGERS,
                "pinky": "posture prior only",
                "initial_palm_error_m": initial_palm_error,
                "initial_fingertip_errors_m": initial_tip_errors.tolist(),
                "max_fitted_robot_table_penetration_m": max_table_penetration,
                "initial_alignment_pass": bool(initial_palm_error < 0.01 and np.max(initial_tip_errors) < 0.03),
                "table_clearance_pass": bool(max_table_penetration < 0.001),
            },
            indent=2,
        )
        + "\n"
    )
    return ref


def validate(output, model, row, extra_hold=True):
    """Verify dynamics, object orientation and grasp retention beyond the end."""
    ref = np.load(output / "reference.npz")
    t = np.load(output / "trajectory.npz")
    data = mujoco.MjData(model)
    probe = mujoco.MjData(model)
    mujoco.mj_setState(
        model, data, t["initial_state"], mujoco.mjtState.mjSTATE_FULLPHYSICS
    )
    for control, expected in zip(t["ctrl"], t["qpos"], strict=True):
        data.ctrl[:] = control
        for _ in range(5):
            mujoco.mj_step(model, data)
        np.testing.assert_allclose(data.qpos, expected, atol=1e-9, rtol=1e-9)

    def relative():
        probe.qpos[:] = data.qpos
        mujoco.mj_forward(model, probe)
        site = probe.site("sharpa_palm")
        p = site.xpos.copy()
        return p, site.xmat.reshape(3, 3).T @ (data.qpos[-7:-4] - p)

    final = data.qpos[-7:-4].copy()
    wrist0, relative0 = relative()
    hold = []
    for _ in range(250 if extra_hold else 0):
        mujoco.mj_step(model, data)
        hold.append(data.qpos.copy())
    wrist1, relative1 = relative()
    if extra_hold:
        np.savez(
            output / "hold_check.npz", qpos=hold, ctrl=data.ctrl.copy(), physics_dt=0.002
        )
    angle = np.linalg.norm(
        quaternion_error(t["qpos"][:, -4:], ref["object_qpos"][1:, 3:]), axis=1
    )
    report = json.loads((output / "metrics.json").read_text())
    report.update(
        case=row,
        object_orientation_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(angle**2)))),
        final_object_orientation_error_deg=float(np.rad2deg(angle[-1])),
        hold={
            "duration_s": 0.5,
            "object_orientation_change_deg": float(
                np.rad2deg(
                    np.linalg.norm(quaternion_error(data.qpos[-4:], t["qpos"][-1, -4:]))
                )
            ),
            "final_orientation_error_deg": float(
                np.rad2deg(
                    np.linalg.norm(
                        quaternion_error(data.qpos[-4:], ref["object_qpos"][-1, 3:])
                    )
                )
            ),
            "height_change_m": float(data.qpos[-5] - final[2]),
            "object_displacement_m": float(np.linalg.norm(data.qpos[-7:-4] - final)),
            "object_relative_palm_displacement_m": float(
                np.linalg.norm(relative1 - relative0)
            ),
            "wrist_displacement_m": float(np.linalg.norm(wrist1 - wrist0)),
        },
    )
    report["pose_tracking_pass"] = bool(
        report["optimized"]["final_object_error_m"] < 0.03
        and report["final_object_orientation_error_deg"] < 20
    )
    report["retention_pass"] = bool(
        report["hold"]["object_relative_palm_displacement_m"] < 0.03
        and data.qpos[-5] - ref["object_qpos"][0, 2] > 0.1
    )
    report["task_success"] = report["pose_tracking_pass"] and report["retention_pass"]
    if not extra_hold:
        report.pop("hold")
        report.pop("retention_pass")
        report["task_success"] = report["pose_tracking_pass"]
        report["evaluation_scope"] = "saved reference trajectory only; no extra hold"
    audit = json.loads(
        (ROOT / "outputs/allegro_to_sharpa_left/model_audit.json").read_text()
    )
    report["speed_within_supplied_ratings"] = bool(
        (
            np.array(report["max_motor_speed_rad_s"])
            <= audit["motor_velocity_ratings_rad_s"]
        ).all()
    )
    assert (
        np.array(report["max_motor_torque_nm"])
        <= np.array(audit["motor_effort_limits_nm"]) + 1e-9
    ).all()
    report["source_revision"] = json.loads(
        (ROOT / "example_datasets/.retarget_full-download/candidate.json").read_text()
    )["data_revision"]
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    """Run independent cases without changing existing methods or native limits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=ROOT / "example_datasets/retarget_full",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/sharpa_complex"
    )
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--ik-stride", type=int, default=2)
    args = parser.parse_args()
    if args.samples < 20 or args.ik_stride < 1:
        parser.error("samples>=20, ik-stride>=1 required")
    selected = json.loads((args.output_dir / "selected.json").read_text())
    before = fingerprint()
    before["examples/arm_hand/try_sharpa.py"] = hashlib.sha256(
        (ROOT / "examples/arm_hand/try_sharpa.py").read_bytes()
    ).hexdigest()
    results = []
    for row, label in zip(selected, ["scissors", "mug", "power_drill"], strict=True):
        output = args.output_dir / label
        output.mkdir(exist_ok=True)
        if (output / "report.json").exists():
            results.append(json.loads((output / "report.json").read_text()))
            continue
        print("CASE:", label, row["task"], flush=True)
        start = time.perf_counter()
        source_scene = args.dataset_dir / row["scene"]
        model, source_model = build(source_scene, output)
        ref, frames = reference(
            source_model, source_scene.parent / "0/trajectory_mjwp.npz", output, 0.5
        )
        ref = fit(model, ref, output, args.ik_stride)
        optimize(
            SimpleNamespace(
                output_dir=output,
                samples=args.samples,
                iterations=args.iterations,
                threads=8,
            ),
            model,
            ref,
        )
        report = validate(output, model, row)
        report["original_frames"] = frames
        report["source_scene"] = str(source_scene.resolve())
        report["elapsed_s"] = time.perf_counter() - start
        after = fingerprint()
        after["examples/arm_hand/try_sharpa.py"] = hashlib.sha256(
            (ROOT / "examples/arm_hand/try_sharpa.py").read_bytes()
        ).hexdigest()
        assert before == after
        report["method_files_unchanged"] = True
        report["method_sha256"] = before
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        results.append(report)
        (args.output_dir / "comparison.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )
        print(
            json.dumps(
                {
                    "case": label,
                    "optimized": report["optimized"],
                    "orientation_error_deg": report[
                        "final_object_orientation_error_deg"
                    ],
                    "hold": report["hold"],
                    "success": report["task_success"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
