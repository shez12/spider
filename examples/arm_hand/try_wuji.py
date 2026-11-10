"""Use the supplied Wuji hand in a separate Xhand-to-arm transfer test."""

import argparse
import copy
import hashlib
import json
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.rollout
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from examples.arm_hand.test_active_distal import fingerprint
from examples.run_arm_hand import quaternion_error
from spider.config import Config
from spider.optimizers.sampling import make_optimize_once_fn

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path("/home/yimu12/work/wuji-description/hand2/hand2_2/body")
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
NATIVE_SITES = [
    "r_thumb_tip",
    "r_index_finger_tip",
    "r_middle_finger_tip",
    "r_ring_finger_tip",
    "r_pinky_tip",
]
CANONICAL = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])


def build(output):
    """Copy source data and attach the native hand without changing its motors."""
    assets = ROOT / "spider/assets/robots/wuji_hand2_2"
    original = assets / "original"
    for directory in ["mjcf", "urdf", "meshes/right"]:
        (original / directory).mkdir(parents=True, exist_ok=True)
    for directory, filename in [("mjcf", "right.xml"), ("urdf", "right.urdf")]:
        shutil.copy2(SOURCE / directory / filename, original / directory / filename)
    shutil.copytree(
        SOURCE / "meshes/right", original / "meshes/right", dirs_exist_ok=True
    )
    native = ET.parse(original / "mjcf/right.xml").getroot()
    original_model = mujoco.MjModel.from_xml_path(str(original / "mjcf/right.xml"))
    tree = ET.parse(ROOT / "outputs/xhand_coke_to_yimu/yimu/scene.xml")
    root = tree.getroot()
    root.set("model", "wuji_hand2_2_arm_coke")
    old_option = root.find("option")
    root.remove(old_option)
    option = copy.deepcopy(native.find("option"))
    option.set("integrator", "implicitfast")
    root.insert(1, option)
    world = root.find("worldbody")
    end = world.find(".//body[@name='arm_link_6']")
    old_palm = end.find("body[@name='right_palm']")
    old_names = {node.get("name") for node in old_palm.iter() if node.get("name")}
    end.remove(old_palm)
    palm = copy.deepcopy(native.find("worldbody/body"))
    palm.set("childclass", "wuji_native")
    # Native mount origin defines the wrist. Fingers point along native -z;
    # native +x runs toward the index, and native +y is the flexion direction.
    quat = Rotation.from_matrix(
        Rotation.from_euler("y", np.pi / 2).as_matrix() @ CANONICAL.T
    ).as_quat()[[3, 0, 1, 2]]
    palm.set("pos", ".125 0 0")
    palm.set("quat", " ".join(map(str, quat)))
    ET.SubElement(
        palm,
        "site",
        name="wuji_palm",
        pos="0 0 0",
        quat="0 0.7071067811865476 0.7071067811865476 0",
        size=".003",
        group="3",
    )
    for i, geom in enumerate(palm.findall(".//geom")):
        if int(geom.get("contype", "1")) or int(geom.get("conaffinity", "1")):
            geom.set("name", f"collision_wuji_{geom.get('mesh', str(i))}_{i}")
    end.append(palm)
    defaults = ET.SubElement(root.find("default"), "default", {"class": "wuji_native"})
    for node in native.find("default"):
        defaults.append(copy.deepcopy(node))
    for node in native.find("asset"):
        node = copy.deepcopy(node)
        if node.tag == "mesh":
            node.set(
                "file", str((original / "meshes/right" / node.get("file")).resolve())
            )
        root.find("asset").append(node)
    actuators = root.find("actuator")
    for node in list(actuators)[7:]:
        actuators.remove(node)
    for node in native.find("actuator"):
        actuators.append(copy.deepcopy(node))
    root.remove(root.find("equality"))
    contacts = root.find("contact")
    for node in list(contacts):
        if any(v in old_names for v in node.attrib.values()):
            contacts.remove(node)
    for node in native.find("contact"):
        contacts.append(copy.deepcopy(node))
    obj = world.find("body[@name='right_object']")
    assert (
        len(
            [
                node
                for node in contacts.findall("pair")
                if node.get("geom1") == "floor" or node.get("geom2") == "floor"
            ]
        )
        >= 9
    )
    for geom in palm.findall(".//geom"):
        if geom.get("name", "").startswith("collision_wuji_"):
            for object_geom in obj.findall("geom"):
                if object_geom.get("name") != "right_object_visual":
                    ET.SubElement(
                        contacts,
                        "pair",
                        geom1=geom.get("name"),
                        geom2=object_geom.get("name"),
                        condim="4",
                        friction="1 1 0.1 0 0",
                    )
    root.remove(root.find("sensor"))
    sensors = ET.SubElement(root, "sensor")
    for name, typ in [("wuji_palm", "site"), ("right_object", "xbody")]:
        ET.SubElement(sensors, "framepos", objtype=typ, objname=name)
        ET.SubElement(sensors, "framequat", objtype=typ, objname=name)
    for name in NATIVE_SITES[:4]:
        ET.SubElement(sensors, "framepos", objtype="site", objname=name)
    for node in native.find("sensor"):
        sensors.append(copy.deepcopy(node))
    ET.indent(root)
    tree.write(output / "scene.xml", encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(output / "scene.xml"))
    assert (model.nq, model.nu, model.neq) == (34, 27, 0)
    for field in [
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_ctrlrange",
        "actuator_forcerange",
    ]:
        np.testing.assert_array_equal(
            getattr(model, field)[7:], getattr(original_model, field)
        )
    for j in range(original_model.njnt):
        target = model.joint(original_model.joint(j).name).id
        for field in ["jnt_axis", "jnt_range", "jnt_actfrcrange"]:
            np.testing.assert_array_equal(
                getattr(model, field)[target], getattr(original_model, field)[j]
            )
        for field in ["dof_armature", "dof_damping", "dof_frictionloss"]:
            np.testing.assert_allclose(
                getattr(model, field)[model.jnt_dofadr[target]],
                getattr(original_model, field)[original_model.jnt_dofadr[j]],
            )
    # Separate fixed-base adapter; copied files in original/ remain byte-identical.
    standalone = copy.deepcopy(native)
    standalone.find("compiler").set(
        "meshdir", str((original / "meshes/right").resolve())
    )
    base = standalone.find("worldbody/body")
    ET.SubElement(
        base,
        "site",
        name="wuji_palm",
        pos="0 0 0",
        quat="0 0.7071067811865476 0.7071067811865476 0",
        size=".003",
        group="3",
    )
    ET.indent(standalone)
    ET.ElementTree(standalone).write(assets / "right.xml", encoding="unicode")
    ratings = {
        j.get("name"): j.find("limit").attrib
        for j in ET.parse(original / "urdf/right.urdf").findall("joint")
        if j.find("limit") is not None
    }
    report = {
        "native_nq": original_model.nq,
        "native_motors": original_model.nu,
        "native_mass_kg": float(original_model.body_mass.sum()),
        "native_xml_sha256": hashlib.sha256(
            (original / "mjcf/right.xml").read_bytes()
        ).hexdigest(),
        "actuator_force_limits_nm": original_model.actuator_forcerange.tolist(),
        "joint_force_limits_nm": original_model.jnt_actfrcrange.tolist(),
        "velocity_ratings_rad_s": [
            float(ratings[original_model.joint(int(j)).name]["velocity"])
            for j in original_model.actuator_trnid[:, 0]
        ],
        "canonical_wrist": "native r_mount origin; canonical x=native +y, y=native +x, z=native -z",
        "tracking_sites": NATIVE_SITES,
        "integrator": "implicitfast for arm comparison; native fixed-base adapter retains supplied RK4",
        "native_motor_properties_preserved": True,
        "speed_limit_enforced": False,
    }
    (output / "model_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return model


def retarget(args, model):
    """Fit the same four Cartesian fingertips and wrist as the earlier tests."""
    source = np.load(args.reference_dir / "prepared/source_landmarks.npz")
    positions, quats, tips, objects = [
        source[k] for k in ["wrist_pos", "wrist_quat", "fingertips", "object_qpos"]
    ]
    low, high = model.actuator_ctrlrange.T
    seed = np.zeros(model.nu)
    seed[:7] = [-0.6, -0.5, 0.3, 1.6, -0.3, -0.6, -0.3]
    seed = np.clip(seed, low + 0.0001, high - 0.0001)
    data = mujoco.MjData(model)
    hand_geoms = [
        i
        for i in range(model.ngeom)
        if (model.geom(i).name or "").startswith("collision_wuji_")
    ]
    hand_index = {g: i for i, g in enumerate(hand_geoms)}
    object_geoms = {
        i
        for i in range(model.ngeom)
        if (model.geom(i).name or "").startswith("right_object_")
        and model.geom(i).name != "right_object_visual"
    }
    stride = max(1, int(args.time_scale))
    indices = np.unique(np.r_[np.arange(0, len(positions), stride), len(positions) - 1])
    fitted, errors, self_penetrations, wrists, wrist_quats = [], [], [], [], []
    for i in indices:
        previous = seed.copy()

        def residual(joints, i=i, previous=previous):
            data.qpos[:27] = joints
            data.qpos[-7:] = objects[i]
            mujoco.mj_forward(model, data)
            penetration = np.zeros(len(hand_geoms))
            for c in data.contact:
                g1, g2 = int(c.geom1), int(c.geom2)
                if g1 in hand_index and g2 in object_geoms:
                    penetration[hand_index[g1]] = max(
                        penetration[hand_index[g1]], -c.dist - 0.001
                    )
                if g2 in hand_index and g1 in object_geoms:
                    penetration[hand_index[g2]] = max(
                        penetration[hand_index[g2]], -c.dist - 0.001
                    )
            return np.r_[
                5 * (data.sensordata[:3] - positions[i]),
                quaternion_error(data.sensordata[3:7], quats[i]),
                40 * (data.sensordata[14:26].reshape(4, 3) - tips[i]).ravel(),
                180 * penetration,
                0.015 * (joints - previous),
            ]

        delta = stride * np.r_[np.full(7, 0.04), np.full(20, 0.08)]
        bounds = (
            (
                np.maximum(low + 0.0001, seed - delta),
                np.minimum(high - 0.0001, seed + delta),
            )
            if i
            else (low + 0.0001, high - 0.0001)
        )
        seed = least_squares(residual, seed, bounds=bounds, max_nfev=80).x
        residual(seed)
        fitted.append(seed.copy())
        wrists.append(data.sensordata[:3].copy())
        wrist_quats.append(data.sensordata[3:7].copy())
        errors.append(
            np.linalg.norm(data.sensordata[14:26].reshape(4, 3) - tips[i], axis=1)
        )
        self_penetrations.append(
            max(
                [0.0]
                + [
                    -float(c.dist)
                    for c in data.contact
                    if all(
                        (model.geom(int(g)).name or "").startswith("collision_wuji_")
                        for g in [c.geom1, c.geom2]
                    )
                ]
            )
        )
        if i % 60 == 0:
            print(f"Wuji IK {i}/{len(positions)}", flush=True)
    joints = np.column_stack(
        [
            np.interp(np.arange(len(positions)), indices, row)
            for row in np.array(fitted).T
        ]
    )
    joints = gaussian_filter1d(joints, sigma=3, axis=0, mode="nearest")
    wrists = np.column_stack(
        [
            np.interp(np.arange(len(positions)), indices, row)
            for row in np.array(wrists).T
        ]
    )
    wrist_quats = Slerp(
        indices, Rotation.from_quat(np.array(wrist_quats)[:, [1, 2, 3, 0]])
    )(np.arange(len(positions))).as_quat()[:, [3, 0, 1, 2]]
    velocity = np.gradient(joints, 0.01, axis=0)
    kp = model.actuator_gainprm[:, 0]
    kv = -model.actuator_biasprm[:, 2]
    commands = joints + np.clip(kv * velocity / kp, -0.15, 0.15)
    commands[:, :7] = joints[:, :7] + 0.05 * velocity[:, :7]
    commands = np.clip(commands, low, high)
    data = mujoco.MjData(model)
    data.qpos[:27] = joints[0]
    data.qpos[-7:] = objects[0]
    next_q = np.r_[joints[1], objects[1]]
    mujoco.mj_differentiatePos(model, data.qvel, 0.01, data.qpos, next_q)
    data.ctrl[:] = commands[0]
    mujoco.mj_forward(model, data)
    initial = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS))
    mujoco.mj_getState(model, data, initial, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    ref = {
        "wrist_pos": wrists,
        "wrist_quat": wrist_quats,
        "source_wrist_pos": positions,
        "fingertips": tips,
        "object_qpos": objects,
        "joint_ik": joints,
        "ctrl_nominal": commands[1:],
        "initial_state": initial,
    }
    np.savez(args.output_dir / "reference.npz", **ref)
    (args.output_dir / "retargeting.json").write_text(
        json.dumps(
            {
                "source_hand": "xhand",
                "matched_fingers": FINGERS[:4],
                "pinky": "posture prior only; no fifth-finger target in prior adapter recipe",
                "mean_raw_ik_fingertip_error_m": float(np.mean(errors)),
                "max_raw_self_penetration_m": float(max(self_penetrations)),
                "reference_frames": len(positions),
                "wrist_target": "fitted IK wrist, same convention as Yimu tests",
                "note": "Native fingertip sites retained. No hand-self collision penalty added; penetration is reported.",
            },
            indent=2,
        )
        + "\n"
    )
    return ref


def optimize(args, model, ref):
    """Run the unchanged sampling update and existing tracking-weight recipe."""
    steps = len(ref["ctrl_nominal"])
    nu = model.nu
    robot_nq = model.nq - 7
    interpolation = np.stack(
        [
            np.interp(np.arange(steps), np.linspace(0, steps - 1, 10).astype(int), row)
            for row in np.eye(10)
        ],
        axis=1,
    )
    checkpoints = np.arange(0, steps, 3)
    workers = [mujoco.MjData(model) for _ in range(args.threads)]
    with mujoco.rollout.Rollout(nthread=args.threads) as engine:

        def evaluate(parameters, full=False):
            offsets = np.asarray(parameters).reshape(-1, 10, nu)
            commands = np.clip(
                ref["ctrl_nominal"] + np.einsum("tk,bkj->btj", interpolation, offsets),
                *model.actuator_ctrlrange.T,
            )
            states, sensors = engine.rollout(
                model,
                workers,
                ref["initial_state"],
                np.repeat(commands, 5, axis=1),
                initial_warmstart=np.zeros(model.nv),
            )
            states = states[:, 4::5]
            sensors = sensors[:, ::5]
            wrist = sensors[:, checkpoints, :3] - ref["wrist_pos"][:-1][checkpoints]
            obj = (
                states[:, checkpoints, 1 + robot_nq : 4 + robot_nq]
                - ref["object_qpos"][1:][checkpoints, :3]
            )
            rot = quaternion_error(
                sensors[:, checkpoints, 3:7],
                np.broadcast_to(
                    ref["wrist_quat"][:-1][checkpoints],
                    sensors[:, checkpoints, 3:7].shape,
                ),
            )
            objrot = quaternion_error(
                states[:, checkpoints, 4 + robot_nq : 8 + robot_nq],
                np.broadcast_to(
                    ref["object_qpos"][1:][checkpoints, 3:],
                    states[:, checkpoints, 4 + robot_nq : 8 + robot_nq].shape,
                ),
            )
            finger = (
                states[:, checkpoints, 8 : 1 + robot_nq]
                - ref["joint_ik"][1:, 7:][checkpoints]
            )
            residual = np.concatenate(
                [
                    (15 * wrist).reshape(len(offsets), -1),
                    (60 * obj).reshape(len(offsets), -1),
                    (0.8 * rot).reshape(len(offsets), -1),
                    (0.3 * objrot).reshape(len(offsets), -1),
                    (0.05 * finger).reshape(len(offsets), -1),
                    (0.03 * offsets).reshape(len(offsets), -1),
                    (0.02 * np.diff(offsets, axis=1)).reshape(len(offsets), -1),
                ],
                axis=1,
            )
            costs = 0.5 * np.sum(residual**2, axis=1)
            return (costs, states, sensors, commands) if full else costs

        baseline = evaluate(np.zeros((10, nu)), True)
        best = {"cost": float(baseline[0][0]), "parameters": np.zeros((10, nu))}
        config = Config(
            device="cpu",
            num_samples=args.samples,
            use_torch_compile=False,
            knot_steps=1,
            temperature=1.0,
            num_trace_uniform_samples=0,
            num_trace_topk_samples=0,
        )
        scale = np.r_[np.full(7, 0.035), np.full(20, 0.12)]
        config.noise_scale = torch.tensor(
            np.broadcast_to(scale, (args.samples, 10, nu)).copy()
        )
        config.noise_scale[0] = 0
        torch.manual_seed(0)

        def rollout(config, env, candidates, ref_slice, env_param):
            del config, env, ref_slice, env_param
            candidates = candidates.clamp(-0.4, 0.4)
            costs = evaluate(candidates.numpy())
            i = int(np.argmin(costs))
            if costs[i] < best["cost"]:
                best.update(
                    cost=float(costs[i]), parameters=candidates[i].numpy().copy()
                )
            return (
                candidates,
                torch.from_numpy(-costs),
                torch.zeros(len(candidates), dtype=torch.bool),
                {},
            )

        update = make_optimize_once_fn(rollout)
        center = torch.zeros((10, nu), dtype=torch.float64)
        start = time.perf_counter()
        for i in range(args.iterations):
            center, _, _ = update(
                config, None, center, (), sample_params={"global_noise_scale": 0.97**i}
            )
            assert torch.isfinite(center).all()
            print(
                f"Wuji sampling {i + 1}/{args.iterations}: cost={best['cost']:.3f}, elapsed={time.perf_counter() - start:.1f}s",
                flush=True,
            )
        optimized = evaluate(best["parameters"], True)

    def metrics(value):
        cost, state, sensor, _ = value
        obj = state[0, :, 1 + robot_nq : 4 + robot_nq]
        return {
            "cost": float(cost[0]),
            "object_lift_m": float(obj[-1, 2] - ref["object_qpos"][0, 2]),
            "final_object_error_m": float(
                np.linalg.norm(obj[-1] - ref["object_qpos"][-1, :3])
            ),
            "object_rmse_m": float(
                np.sqrt(
                    np.mean(np.sum((obj - ref["object_qpos"][1:, :3]) ** 2, axis=1))
                )
            ),
            "wrist_rmse_m": float(
                np.sqrt(
                    np.mean(
                        np.sum((sensor[0, :, :3] - ref["wrist_pos"][:-1]) ** 2, axis=1)
                    )
                )
            ),
        }

    np.savez(
        args.output_dir / "trajectory.npz",
        initial_state=ref["initial_state"],
        ctrl=optimized[3][0],
        qpos=optimized[1][0, :, 1 : 1 + model.nq],
        qvel=optimized[1][0, :, 1 + model.nq :],
        control_dt=0.01,
        physics_dt=0.002,
    )
    report = {
        "hand": "right Wuji hand2_2",
        "finger_motors": 20,
        "arm_joints": 7,
        "couplings": 0,
        "baseline": metrics(baseline),
        "optimized": metrics(optimized),
        "reference_lift_m": float(ref["object_qpos"][-1, 2] - ref["object_qpos"][0, 2]),
        "object_mass_kg": float(model.body_mass[model.body("right_object").id]),
        "samples": args.samples,
        "iterations": args.iterations,
        "seed": 0,
        "source": "Xhand fair_mon Coke example; same retimed reference as Yimu",
        "source_reference_dir": str(args.reference_dir.resolve()),
        "time_scale": args.time_scale,
        "hold_reference_seconds": 0.5,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def validate(output):
    """Independently replay controls and measure an additional stationary hold."""
    model = mujoco.MjModel.from_xml_path(str(output / "scene.xml"))
    data = mujoco.MjData(model)
    trajectory = np.load(output / "trajectory.npz")
    mujoco.mj_setState(
        model, data, trajectory["initial_state"], mujoco.mjtState.mjSTATE_FULLPHYSICS
    )
    dofs = model.jnt_dofadr[model.actuator_trnid[7:, 0]]
    force = np.zeros(20)
    speeds = np.zeros(20)
    for ctrl, expected in zip(trajectory["ctrl"], trajectory["qpos"], strict=True):
        data.ctrl[:] = ctrl
        for _ in range(5):
            mujoco.mj_step(model, data)
            force = np.maximum(force, np.abs(data.actuator_force[7:]))
            speeds = np.maximum(speeds, np.abs(data.qvel[dofs]))
        np.testing.assert_allclose(data.qpos, expected, atol=1e-9, rtol=1e-9)
    final = data.qpos[-7:-4].copy()
    probe = mujoco.MjData(model)

    def wrist_state():
        probe.qpos[:] = data.qpos
        mujoco.mj_forward(model, probe)
        site = probe.site("wuji_palm")
        position = site.xpos.copy()
        relative = site.xmat.reshape(3, 3).T @ (data.qpos[-7:-4] - position)
        return position, relative

    wrist_before, relative_before = wrist_state()
    qpos = []
    for _ in range(250):
        mujoco.mj_step(model, data)
        qpos.append(data.qpos.copy())
    np.savez(
        output / "hold_check.npz", qpos=qpos, ctrl=data.ctrl.copy(), physics_dt=0.002
    )
    wrist_after, relative_after = wrist_state()
    audit = json.loads((output / "model_audit.json").read_text())
    limits = np.array(audit["actuator_force_limits_nm"])[:, 1]
    assert (force <= limits + 1e-9).all()
    check = {
        "continuous_replay_verified": True,
        "extra_hold_s": 0.5,
        "wrist_displacement_during_hold_m": float(
            np.linalg.norm(wrist_after - wrist_before)
        ),
        "object_relative_to_palm_displacement_during_hold_m": float(
            np.linalg.norm(relative_after - relative_before)
        ),
        "object_height_end_of_trajectory_m": float(final[2]),
        "object_height_after_hold_m": float(data.qpos[-5]),
        "height_change_during_hold_m": float(data.qpos[-5] - final[2]),
        "object_displacement_during_hold_m": float(
            np.linalg.norm(data.qpos[-7:-4] - final)
        ),
        "max_motor_torque_nm": force.tolist(),
        "max_motor_speed_rad_s": speeds.tolist(),
        "speed_within_native_urdf_ratings": bool(
            (speeds <= audit["velocity_ratings_rad_s"]).all()
        ),
        "speed_limit_enforced": False,
    }
    report = json.loads((output / "report.json").read_text())
    ref = np.load(output / "reference.npz")
    report["trajectory_lift_criterion_met"] = bool(
        report["optimized"]["object_lift_m"] > 0.1
        and report["optimized"]["final_object_error_m"] < 0.03
    )
    report["extended_hold_success"] = bool(
        data.qpos[-5] - ref["object_qpos"][0, 2] > 0.1
        and check["object_displacement_during_hold_m"] < 0.03
    )
    report["task_success"] = (
        report["trajectory_lift_criterion_met"] and report["extended_hold_success"]
    )
    report["hold_check"] = check
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    """Run the separate Wuji adapter without changing existing method files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/xhand_coke_to_wuji"
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=ROOT / "outputs/xhand_coke_to_yimu_slow_hold",
    )
    parser.add_argument("--time-scale", type=float, default=3.0)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.samples < 20:
        parser.error("At least 20 samples required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    before = fingerprint()
    model = build(args.output_dir)
    ref = retarget(args, model)
    optimize(args, model, ref)
    report = validate(args.output_dir)
    assert before == fingerprint()
    report["method_files_unchanged"] = True
    report["method_sha256"] = before
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
