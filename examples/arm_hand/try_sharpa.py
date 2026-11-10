"""Test the supplied left Sharpa hand using unchanged SPIDER sampling code."""

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
from scipy.spatial.transform import Rotation

from examples.arm_hand.test_active_distal import fingerprint
from examples.run_arm_hand import quaternion_error
from spider.config import Config
from spider.optimizers.sampling import make_optimize_once_fn

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path("/home/yimu12/work/simtoolreal/assets/urdf/left_sharpa_ha4")
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
S = np.diag([1.0, -1.0, 1.0])


def mirror_quat(quat):
    """Reflect a proper rotation in world and local y coordinates."""
    rot = Rotation.from_quat(np.asarray(quat)[..., [1, 2, 3, 0]])
    return Rotation.from_matrix(S @ rot.as_matrix() @ S).as_quat()[..., [3, 0, 1, 2]]


def build(output):
    """Import native geometry and servos; use effort limits from matching URDF."""
    assets = ROOT / "spider/assets/robots/sharpa_ha4"
    original = assets / "original"
    original.mkdir(parents=True, exist_ok=True)
    for name in ["left_sharpa_ha4_v2_1.xml", "left_sharpa_ha4_v2_1.urdf"]:
        shutil.copy2(SOURCE / name, original / name)
    shutil.copytree(SOURCE / "meshes", original / "meshes", dirs_exist_ok=True)
    # MuJoCo accepts the missing whitespace here; ElementTree requires it.
    native = ET.fromstring(
        (original / "left_sharpa_ha4_v2_1.xml").read_text().replace('"kp=', '" kp=')
    )
    native_model = mujoco.MjModel.from_xml_path(
        str(original / "left_sharpa_ha4_v2_1.xml")
    )
    native_data = mujoco.MjData(native_model)
    mujoco.mj_forward(native_model, native_data)
    urdf = ET.parse(original / "left_sharpa_ha4_v2_1.urdf")
    limits = {
        j.get("name"): dict(j.find("limit").attrib)
        for j in urdf.findall("joint")
        if j.find("limit") is not None
    }
    # Start with the established generic arm and published coffee-can scene.
    tree = ET.parse(ROOT / "outputs/allegro_to_yimu_v2/arm_hand_scene.xml")
    root = tree.getroot()
    root.set("model", "left_sharpa_arm_dataset_test")
    root.find("option").set("timestep", ".002")
    root.find("option").set("iterations", "40")
    world = root.find("worldbody")
    arm = world.find("body[@name='arm_mount']")
    armpos = np.fromstring(arm.get("pos"), sep=" ")
    armpos[1] *= -1
    arm.set("pos", " ".join(map(str, armpos)))
    end = arm.find(".//body[@name='arm_link_6']")
    old_palm = end.find("body[@name='right_palm']")
    removed_names = {node.get("name") for node in old_palm.iter() if node.get("name")}
    end.remove(old_palm)
    palm = copy.deepcopy(native.find("worldbody/body"))
    palm.set("pos", "0.16 0.01 0")
    palm.set("quat", "0.7071067811865476 0 0.7071067811865476 0")
    # R_y(90)*[0,-.01,.06] + [ .16,.01,0 ] = [.22,0,0].
    ET.SubElement(
        palm, "site", name="sharpa_palm", pos="0 -0.01 0.06", size=".004", group="3"
    )
    pad_sites = {}
    for finger in FINGERS:
        body = palm.find(f".//body[@name='left_{finger}_DP']")
        gid = native_model.geom(f"left_{finger}_elastomer").id
        mid = native_model.geom_dataid[gid]
        verts = native_model.mesh_vert[
            native_model.mesh_vertadr[mid] : native_model.mesh_vertadr[mid]
            + native_model.mesh_vertnum[mid]
        ]
        worldverts = (
            verts @ native_data.geom_xmat[gid].reshape(3, 3).T
            + native_data.geom_xpos[gid]
        )
        bid = native_model.body(f"left_{finger}_DP").id
        local = (worldverts - native_data.xpos[bid]) @ native_data.xmat[bid].reshape(
            3, 3
        )
        low, high = local.min(0), local.max(0)
        point = (low + high) / 2
        point[1] = high[1]
        pad_sites[finger] = point.tolist()
        ET.SubElement(
            body,
            "site",
            name=f"sharpa_{finger}_tip",
            pos=" ".join(map(str, point)),
            size=".003",
            group="3",
        )
    for i, g in enumerate(palm.findall(".//geom")):
        if int(g.get("contype", "1")) or int(g.get("conaffinity", "1")):
            g.set("name", "collision_sharpa_" + g.get("name", str(i)))
    end.append(palm)
    for default in native.findall("default/default"):
        root.find("default").append(copy.deepcopy(default))
    for mesh in native.findall("asset/mesh"):
        mesh = copy.deepcopy(mesh)
        mesh.set("file", str((original / "meshes" / mesh.get("file")).resolve()))
        root.find("asset").append(mesh)
    actuator = root.find("actuator")
    for node in list(actuator)[7:]:
        actuator.remove(node)
    effort = []
    speeds = []
    for node in native.find("actuator"):
        node = copy.deepcopy(node)
        limit = limits[node.get("joint")]
        force = float(limit["effort"])
        node.set("forcerange", f"{-force} {force}")
        node.set("forcelimited", "true")
        effort.append(force)
        speeds.append(float(limit["velocity"]))
        actuator.append(node)
    if root.find("equality") is not None:
        root.remove(root.find("equality"))
    contact = root.find("contact")
    for node in list(contact):
        if any(value in removed_names for value in node.attrib.values()):
            contact.remove(node)
    # Preserve explicit floor-object pairs: source geoms use zero collision masks.
    floor_pairs = [
        node for node in contact.findall("pair") if "right_object_" in str(node.attrib)
    ]
    assert len(floor_pairs) >= 9, "Dataset object-floor contact pairs must remain"
    for node in native.findall("contact/exclude"):
        contact.append(copy.deepcopy(node))
    # Reflect the free can's local geometry and inertial orientation, preserving
    # mass and shape up to reflection, consistently with the left-hand task.
    obj = world.find("body[@name='right_object']")
    object_meshes = set()
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
        if node.get("mesh"):
            object_meshes.add(node.get("mesh"))
    for mesh in root.findall("asset/mesh"):
        if mesh.get("name") in object_meshes:
            scale = np.fromstring(mesh.get("scale", "1 1 1"), sep=" ") * [1, -1, 1]
            mesh.set("scale", " ".join(map(str, scale)))
    # Explicit pair parameters match the preceding hand comparisons.
    for hand_geom in palm.findall(".//geom"):
        if hand_geom.get("name", "").startswith("collision_sharpa_"):
            for geom in obj.findall("geom"):
                if geom.get("name") != "right_object_visual":
                    ET.SubElement(
                        contact,
                        "pair",
                        geom1=hand_geom.get("name"),
                        geom2=geom.get("name"),
                        condim="4",
                        friction="1 1 0.1 0 0",
                    )
    root.remove(root.find("sensor"))
    sensor = ET.SubElement(root, "sensor")
    ET.SubElement(sensor, "framepos", objtype="site", objname="sharpa_palm")
    ET.SubElement(sensor, "framequat", objtype="site", objname="sharpa_palm")
    ET.SubElement(sensor, "framepos", objtype="xbody", objname="right_object")
    ET.SubElement(sensor, "framequat", objtype="xbody", objname="right_object")
    for finger in FINGERS[:4]:
        ET.SubElement(
            sensor, "framepos", objtype="site", objname=f"sharpa_{finger}_tip"
        )
    ET.indent(root)
    tree.write(output / "scene.xml", encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(output / "scene.xml"))
    assert (model.nq, model.nu, model.neq) == (36, 29, 0)
    # Export a separate fixed-base hand adapter with the same sites and effort
    # limits. It does not add floating-base motors or modify the supplied files.
    hand = copy.deepcopy(native)
    hand.set("model", "left_sharpa_ha4_effort_limited")
    hand.find("compiler").set("meshdir", str((original / "meshes").resolve()))
    hand_palm = copy.deepcopy(palm)
    hand_palm.set("pos", "0 0 0")
    hand_palm.set("quat", "1 0 0 0")
    hand_world = hand.find("worldbody")
    hand_world.remove(hand_world.find("body"))
    hand_world.append(hand_palm)
    hand.remove(hand.find("actuator"))
    hand_actuators = ET.SubElement(hand, "actuator")
    for node in list(actuator)[7:]:
        hand_actuators.append(copy.deepcopy(node))
    ET.SubElement(
        hand, "option", timestep="0.002", iterations="40", integrator="implicitfast"
    )
    ET.indent(hand)
    ET.ElementTree(hand).write(assets / "left.xml", encoding="unicode")
    standalone = mujoco.MjModel.from_xml_path(str(assets / "left.xml"))
    assert (standalone.nq, standalone.nu, standalone.neq) == (22, 22, 0)
    np.testing.assert_allclose(standalone.body_mass.sum(), native_model.body_mass.sum())
    for j in range(native_model.njnt):
        target = model.joint(native_model.joint(j).name).id
        np.testing.assert_allclose(model.jnt_range[target], native_model.jnt_range[j])
        for field in ["dof_damping", "dof_armature", "dof_frictionloss"]:
            np.testing.assert_allclose(
                getattr(model, field)[model.jnt_dofadr[target]],
                getattr(native_model, field)[native_model.jnt_dofadr[j]],
            )
    report = {
        "source_xml_sha256": hashlib.sha256(
            (original / "left_sharpa_ha4_v2_1.xml").read_bytes()
        ).hexdigest(),
        "native_finger_joints": 22,
        "native_mass_kg": float(native_model.body_mass.sum()),
        "motor_effort_limits_nm": effort,
        "motor_velocity_ratings_rad_s": speeds,
        "effort_limits_source": "matching supplied URDF; supplied XML omits force limits",
        "pad_tracking_sites_local": pad_sites,
        "canonical_wrist_pos_native": [0, -0.01, 0.06],
        "task_transform": "Reflect world and object-local y; left task from right Allegro episode",
        "speed_limit_enforced": False,
    }
    (output / "model_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return model


def retarget(model, output):
    """Fit mirrored dataset palm/fingertips with the prior Cartesian IK recipe."""
    original = np.load(ROOT / "outputs/allegro_to_yimu_v2/source_landmarks.npz")
    positions = original["wrist_pos"] @ S
    quats = mirror_quat(original["wrist_quat"])
    tips = original["fingertips"] @ S
    objects = original["object_qpos"].copy()
    objects[:, :3] = objects[:, :3] @ S
    objects[:, 3:] = mirror_quat(objects[:, 3:])
    data = mujoco.MjData(model)
    seed = np.zeros(model.nu)
    seed[:7] = [0.5, 0.2, -0.4, 2.1, 0, -2.5, 0.3]
    low, high = model.actuator_ctrlrange.T
    seed = np.clip(seed, low + 0.0001, high - 0.0001)
    solutions = []
    errors = []
    selfpenetrations = []
    for i, (p, q, t) in enumerate(zip(positions, quats, tips, strict=True)):
        previous = seed.copy()

        def residual(joints, p=p, q=q, t=t, previous=previous, i=i):
            data.qpos[:29] = joints
            data.qpos[-7:] = objects[i]
            mujoco.mj_forward(model, data)
            penetration = np.zeros(60)
            k = 0
            for c in data.contact:
                names = [model.geom(int(g)).name or "" for g in [c.geom1, c.geom2]]
                if any(n.startswith("collision_sharpa_") for n in names) and any(
                    n.startswith("right_object_") for n in names
                ):
                    penetration[min(k, 59)] = max(0.0, -c.dist - 0.001)
                    k += 1
            return np.r_[
                5 * (data.sensordata[:3] - p),
                quaternion_error(data.sensordata[3:7], q),
                40 * (data.sensordata[14:26].reshape(4, 3) - t).ravel(),
                180 * penetration,
                0.015 * (joints - previous),
            ]

        delta = np.r_[np.full(7, 0.04), np.full(22, 0.08)]
        bounds = (
            (
                np.maximum(low + 0.0001, seed - delta),
                np.minimum(high - 0.0001, seed + delta),
            )
            if i
            else (low + 0.0001, high - 0.0001)
        )
        fit = least_squares(residual, seed, bounds=bounds, max_nfev=80)
        seed = fit.x
        residual(seed)
        solutions.append(seed.copy())
        errors.append(np.linalg.norm(data.sensordata[14:26].reshape(4, 3) - t, axis=1))
        selfpenetrations.append(
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
        if i % 40 == 0:
            print(f"IK {i}/{len(positions)}", flush=True)
    joints = gaussian_filter1d(np.array(solutions), sigma=3, axis=0, mode="nearest")
    velocity = np.gradient(joints, 0.01, axis=0)
    kp = model.actuator_gainprm[:, 0]
    kv = -model.actuator_biasprm[:, 2]
    controls = np.clip(joints + np.clip(kv * velocity / kp, -0.15, 0.15), low, high)
    controls[:, :7] = np.clip(joints[:, :7] + 0.05 * velocity[:, :7], low[:7], high[:7])
    data = mujoco.MjData(model)
    data.qpos[:29] = joints[0]
    data.qpos[-7:] = objects[0]
    data.qvel[:29] = velocity[0]
    for i in range(3):
        data.qvel[-6 + i] = (objects[1, i] - objects[0, i]) / 0.01
    data.ctrl[:] = controls[0]
    mujoco.mj_forward(model, data)
    initial = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS))
    mujoco.mj_getState(model, data, initial, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    np.savez(
        output / "reference.npz",
        wrist_pos=positions,
        wrist_quat=quats,
        fingertips=tips,
        object_qpos=objects,
        joint_ik=joints,
        ctrl_nominal=controls[1:],
        initial_state=initial,
    )
    report = {
        "mean_raw_ik_fingertip_error_m": float(np.mean(errors)),
        "max_raw_ik_self_penetration_m": float(max(selfpenetrations)),
        "frames": len(positions),
        "matched_fingers": FINGERS[:4],
        "pinky": "No source fingertip; temporal posture prior only",
        "reference": "Published Allegro episode, mirrored for supplied left hand; no Sharpa trajectory loaded",
        "limitations": "Tracking-site and wrist frame inferred from meshes. Same geometric IK recipe; self-contact is reported, not penalized.",
    }
    (output / "retargeting.json").write_text(json.dumps(report, indent=2) + "\n")
    return dict(np.load(output / "reference.npz"))


def optimize(args, model, ref):
    """Run the existing SPIDER sampling update with continuous 2 ms physics."""
    steps = len(ref["ctrl_nominal"])
    nu = model.nu
    robot_nq = model.nq - 7
    knots = 10
    grid = np.arange(steps)
    indices = np.linspace(0, steps - 1, knots).astype(int)
    interpolation = np.stack(
        [np.interp(grid, indices, row) for row in np.eye(knots)], axis=1
    )
    checkpoints = np.arange(0, steps, 3)
    workers = [mujoco.MjData(model) for _ in range(args.threads)]
    with mujoco.rollout.Rollout(nthread=args.threads) as engine:

        def evaluate(parameters, full=False):
            offsets = np.asarray(parameters).reshape(-1, knots, nu)
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

        zeros = np.zeros((knots, nu))
        baseline = evaluate(zeros, True)
        best = {"cost": float(baseline[0][0]), "parameters": zeros.copy()}
        config = Config(
            device="cpu",
            num_samples=args.samples,
            use_torch_compile=False,
            knot_steps=1,
            temperature=1.0,
            num_trace_uniform_samples=0,
            num_trace_topk_samples=0,
        )
        scale = np.r_[np.full(7, 0.035), np.full(22, 0.12)]
        config.noise_scale = torch.tensor(
            np.broadcast_to(scale, (args.samples, knots, nu)).copy()
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
        center = torch.zeros((knots, nu), dtype=torch.float64)
        for i in range(args.iterations):
            center, _, _ = update(
                config, None, center, (), sample_params={"global_noise_scale": 0.97**i}
            )
            assert torch.isfinite(center).all()
            print(
                f"Sampling {i + 1}/{args.iterations}: cost={best['cost']:.3f}",
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
    data = mujoco.MjData(model)
    mujoco.mj_setState(
        model, data, ref["initial_state"], mujoco.mjtState.mjSTATE_FULLPHYSICS
    )
    replay = []
    maxforce = np.zeros(22)
    maxspeed = np.zeros(22)
    handcontacts = 0
    for command in optimized[3][0]:
        data.ctrl[:] = command
        for _ in range(5):
            mujoco.mj_step(model, data)
            maxforce = np.maximum(maxforce, np.abs(data.actuator_force[7:]))
            maxspeed = np.maximum(maxspeed, np.abs(data.qvel[7:29]))
            handcontacts += sum(
                any(
                    (model.geom(int(g)).name or "").startswith("collision_sharpa_")
                    for g in [c.geom1, c.geom2]
                )
                and any(
                    (model.geom(int(g)).name or "").startswith("right_object_")
                    for g in [c.geom1, c.geom2]
                )
                for c in data.contact
            )
        replay.append(data.qpos.copy())
    np.testing.assert_allclose(
        replay, optimized[1][0, :, 1 : 1 + model.nq], atol=1e-9, rtol=1e-9
    )
    report = {
        "hand": "left Sharpa HA4",
        "finger_motors": 22,
        "arm_joints": 7,
        "baseline": metrics(baseline),
        "optimized": metrics(optimized),
        "samples": args.samples,
        "iterations": args.iterations,
        "reference_lift_m": float(ref["object_qpos"][-1, 2] - ref["object_qpos"][0, 2]),
        "object_mass_kg": float(model.body_mass[model.body("right_object").id]),
        "max_motor_torque_nm": maxforce.tolist(),
        "max_motor_speed_rad_s": maxspeed.tolist(),
        "hand_object_contact_samples": int(handcontacts),
        "continuous_replay_verified": True,
        "limitations": "Single seed and fixed budget; generic arm, inferred mounting/tracking frame, mirrored task, URDF torque limits, no explicit speed limiter.",
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main():
    """Build, retarget, optimize and verify without editing existing methods."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/allegro_to_sharpa_left"
    )
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.samples < 20:
        parser.error("At least 20 samples required")
    before = fingerprint()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    model = build(args.output_dir)
    ref = retarget(model, args.output_dir)
    optimize(args, model, ref)
    assert before == fingerprint()
    (args.output_dir / "method_integrity.json").write_text(
        json.dumps(
            {
                "unchanged": True,
                "sha256": before,
                "elapsed_s": time.perf_counter() - start,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
