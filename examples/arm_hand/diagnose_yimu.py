"""Controlled timestep and stationary-grasp ablations for the imported hand."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from spider.robot_model import actuator_qpos_indices, project_joint_equalities


def telemetry(model, data):
    """Read contacts and limits without altering simulation state."""
    fingers = set()
    force = 0.0
    self_contacts = 0
    for contact_id, contact in enumerate(data.contact):
        names = [model.geom(int(g)).name or "" for g in [contact.geom1, contact.geom2]]
        hand = [n for n in names if n.startswith("collision_hand_")]
        if len(hand) == 2:
            self_contacts += 1
        if hand and any(n.startswith("right_object_") for n in names):
            fingers.add(hand[0].split("_")[3])
            wrench = np.zeros(6)
            mujoco.mj_contactForce(model, data, contact_id, wrench)
            force += abs(wrench[0])
    return fingers, force, self_contacts


def replay(folder, dt, iterations=None):
    """Replay identical 100 Hz controls with a chosen physics substep."""
    model = mujoco.MjModel.from_xml_path(str(folder / "arm_hand_scene.xml"))
    model.opt.timestep = dt
    if iterations is not None:
        model.opt.iterations = iterations
    data = mujoco.MjData(model)
    t = np.load(folder / "trajectory.npz")
    mujoco.mj_setState(
        model, data, t["initial_state"], mujoco.mjtState.mjSTATE_FULLPHYSICS
    )
    qpos, max_force, max_speed, linkage, contact_steps = [], 0.0, 0.0, 0.0, 0
    saturation = np.zeros(17)
    dofs = model.jnt_dofadr[model.actuator_trnid[7:, 0]]
    for command in t["ctrl"]:
        data.ctrl[:] = command
        for _ in range(round(0.01 / dt)):
            mujoco.mj_step(model, data)
            max_force = max(max_force, float(np.abs(data.actuator_force[7:]).max()))
            max_speed = max(max_speed, float(np.abs(data.qvel[dofs]).max()))
            linkage = max(
                linkage,
                float(
                    np.abs(
                        project_joint_equalities(model, data.qpos.copy()) - data.qpos
                    ).max()
                ),
            )
            saturation += np.abs(data.actuator_force[7:]) > 0.159
            contact_steps += bool(telemetry(model, data)[0])
        qpos.append(data.qpos.copy())
    qpos = np.array(qpos)
    ref = np.load(folder / "target_ik_reference.npz")
    np.savez(
        folder / f"replay_dt_{dt:g}_iterations_{model.opt.iterations}.npz",
        qpos=qpos,
        ctrl=t["ctrl"],
        initial_state=t["initial_state"],
    )
    return {
        "dt": dt,
        "solver_iterations": int(model.opt.iterations),
        "object_lift_m": float(qpos[-1, -5] - ref["initial_qpos"][-5]),
        "object_rmse_m": float(
            np.sqrt(
                np.mean(
                    np.sum((qpos[:, -7:-4] - ref["object_qpos"][:, :3]) ** 2, axis=1)
                )
            )
        ),
        "final_object_error_m": float(
            np.linalg.norm(qpos[-1, -7:-4] - ref["object_qpos"][-1, :3])
        ),
        "max_motor_torque_nm": max_force,
        "max_motor_speed_rad_s": max_speed,
        "max_coupling_error_rad": linkage,
        "contact_time_s": contact_steps * dt,
        "motor_saturation_fraction": dict(
            zip(
                [model.actuator(i).name for i in range(7, 24)],
                (saturation / (len(t["ctrl"]) * round(0.01 / dt))).tolist(),
                strict=True,
            )
        ),
    }


def grasp(folder, frame, closure, pinky, output):
    """Hold a grasp, then command a slow 5 cm wrist lift with ordinary servos."""
    model = mujoco.MjModel.from_xml_path(str(folder / "arm_hand_scene.xml"))
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)
    ref = np.load(folder / "target_ik_reference.npz")
    indices = actuator_qpos_indices(model)
    # Independent experiment starts at an IK pose with a free object. No states
    # are overwritten after initialization; the arm remains servo controlled.
    data.qpos[:28] = ref["target_joint_ik"][frame]
    data.qpos[-7:] = (
        ref["initial_qpos"][-7:] if frame == 0 else ref["object_qpos"][frame - 1]
    )
    mujoco.mj_forward(model, data)
    command = data.qpos[indices].copy()
    for i in range(7, model.nu):
        name = model.joint(int(model.actuator_trnid[i, 0])).name
        if name in [f"right_{f}-{j}" for f in range(2, 6) for j in [2, 3]] or name in [
            "right_1-3",
            "right_1-4",
            "right_1-5",
        ]:
            command[i] += closure
    if pinky:
        command[-3:] = command[-6:-3]
    command = np.clip(command, *model.actuator_ctrlrange.T)
    palm = model.site("right_palm").id
    origin_object = data.qpos[-7:-4].copy()
    hold_contacts = set()
    hold_normal, max_speed, max_torque, self_contacts = 0.0, 0.0, 0.0, 0
    history, controls = [], []
    arm_target = command[:7].copy()
    ik_data = mujoco.MjData(model)
    dofs = model.jnt_dofadr[model.actuator_trnid[7:, 0]]
    start_p = start_r = hold_object = None
    for step in range(300):
        if step == 100:
            mujoco.mj_forward(model, data)
            start_p = data.site_xpos[palm].copy()
            start_r = data.site_xmat[palm].reshape(3, 3).copy()
            hold_object = data.qpos[-7:-4].copy()
        if step >= 100:
            desired = start_p + [0, 0, 0.05 * min(1.0, (step - 100) / 150)]

            def residual(angles, desired=desired, start_r=start_r, previous=arm_target):
                ik_data.qpos[:] = data.qpos
                ik_data.qpos[:7] = angles
                mujoco.mj_forward(model, ik_data)
                return np.r_[
                    20 * (ik_data.site_xpos[palm] - desired),
                    (ik_data.site_xmat[palm].reshape(3, 3) - start_r).ravel(),
                    0.001 * (angles - previous),
                ]

            arm_target = least_squares(
                residual, arm_target, bounds=model.actuator_ctrlrange[:7].T, max_nfev=15
            ).x
            command[:7] = arm_target
        data.ctrl[:] = command
        for _ in range(5):
            mujoco.mj_step(model, data)
            fingers, normal, sc = telemetry(model, data)
            if 75 <= step < 100:
                hold_contacts.update(fingers)
                hold_normal += normal / 125
            self_contacts = max(self_contacts, sc)
            max_speed = max(max_speed, float(np.abs(data.qvel[dofs]).max()))
            max_torque = max(max_torque, float(np.abs(data.actuator_force[7:]).max()))
        history.append(data.qpos.copy())
        controls.append(command.copy())
    mujoco.mj_forward(model, data)
    result = {
        "frame": frame,
        "closure_offset_rad": closure,
        "pinky_follows_ring": pinky,
        "hold_contact_fingers": sorted(hold_contacts),
        "hold_mean_sum_normal_force_n": hold_normal,
        "max_self_contacts": self_contacts,
        "hold_object_displacement_m": float(
            np.linalg.norm(hold_object - origin_object)
        ),
        "wrist_lift_m": float(data.site_xpos[palm, 2] - start_p[2]),
        "object_lift_after_hold_m": float(data.qpos[-5] - hold_object[2]),
        "max_motor_speed_rad_s": max_speed,
        "max_motor_torque_nm": max_torque,
    }
    name = f"grasp_f{frame}_c{closure:g}_p{int(pinky)}"
    np.savez(
        output / f"{name}.npz", qpos=history, ctrl=controls, dt=0.002, control_dt=0.01
    )
    return result


def geometry(folder):
    """Inspect initial IK self-penetrations and native parameter preservation."""
    model = mujoco.MjModel.from_xml_path(str(folder / "arm_hand_scene.xml"))
    native_path = (
        Path(__file__).resolve().parents[2]
        / "spider/assets/robots/yimu_v2/original/yimu_v2_right/yimu_v2_right.xml"
    )
    native = mujoco.MjModel.from_xml_path(str(native_path))
    for joint in range(native.njnt):
        target = model.joint("right_" + native.joint(joint).name).id
        for field in ["jnt_range", "jnt_axis"]:
            np.testing.assert_allclose(
                getattr(model, field)[target], getattr(native, field)[joint]
            )
        a, b = model.jnt_dofadr[target], native.jnt_dofadr[joint]
        for field in ["dof_armature", "dof_damping", "dof_frictionloss"]:
            np.testing.assert_allclose(
                getattr(model, field)[a], getattr(native, field)[b]
            )
    for actuator in range(native.nu):
        target = model.actuator("right_" + native.actuator(actuator).name).id
        for field in [
            "actuator_gainprm",
            "actuator_biasprm",
            "actuator_forcerange",
            "actuator_ctrlrange",
        ]:
            np.testing.assert_allclose(
                getattr(model, field)[target], getattr(native, field)[actuator]
            )
    data = mujoco.MjData(model)
    ref = np.load(folder / "target_ik_reference.npz")
    frames = []
    for frame in [0, 30, 60, 65, 70]:
        data.qpos[:28] = ref["target_joint_ik"][frame]
        data.qpos[-7:] = (
            ref["initial_qpos"][-7:] if frame == 0 else ref["object_qpos"][frame - 1]
        )
        mujoco.mj_forward(model, data)
        contacts = []
        for contact in data.contact:
            names = [
                model.geom(int(g)).name or "" for g in [contact.geom1, contact.geom2]
            ]
            if all(n.startswith("collision_hand_") for n in names):
                contacts.append(
                    {"geoms": names, "penetration_m": max(0.0, -float(contact.dist))}
                )
        frames.append({"frame": frame, "self_contacts": contacts})
    return {
        "native_joint_and_motor_parameters_preserved": True,
        "ik_pose_self_contacts": frames,
    }


def main():
    """Save controlled comparisons without modifying the original trial."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trial", type=Path, default=Path("outputs/allegro_to_yimu_v2")
    )
    args = parser.parse_args()
    output = args.trial / "diagnosis"
    output.mkdir(exist_ok=True)
    report = {
        "replay": [replay(args.trial, dt) for dt in [0.01, 0.002, 0.001]]
        + [replay(args.trial, 0.002, iterations=40)],
        "geometry": geometry(args.trial),
        "stationary_grasp": [],
    }
    for frame in [0, 30, 60, 65, 70]:
        for closure in [0.0, 0.1, 0.3]:
            for pinky in [False, True] if frame <= 60 else [True]:
                result = grasp(args.trial, frame, closure, pinky, output)
                report["stationary_grasp"].append(result)
                print(json.dumps(result), flush=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["replay"], indent=2), flush=True)


if __name__ == "__main__":
    main()
