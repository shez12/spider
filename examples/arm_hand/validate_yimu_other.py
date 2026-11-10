"""Verify source-transfer replay and an extra half-second stationary hold."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def validate(output):
    """Verify dynamic replay, measure speed, and check grasp retention."""
    model = mujoco.MjModel.from_xml_path(str(output / "yimu/scene.xml"))
    data = mujoco.MjData(model)
    trajectory = np.load(output / "yimu/trajectory.npz")
    mujoco.mj_setState(
        model, data, trajectory["initial_state"], mujoco.mjtState.mjSTATE_FULLPHYSICS
    )
    dofs = model.jnt_dofadr[model.actuator_trnid[7:, 0]]
    max_speed = 0.0
    for command, expected in zip(trajectory["ctrl"], trajectory["qpos"], strict=True):
        data.ctrl[:] = command
        for _ in range(5):
            mujoco.mj_step(model, data)
            max_speed = max(max_speed, float(np.abs(data.qvel[dofs]).max()))
        np.testing.assert_allclose(data.qpos, expected, atol=1e-9, rtol=1e-9)
    final = data.qpos[-7:-4].copy()
    qpos, contacts = [], set()
    for _ in range(250):
        mujoco.mj_step(model, data)
        qpos.append(data.qpos.copy())
        for contact in data.contact:
            names = [
                model.geom(int(g)).name or "" for g in [contact.geom1, contact.geom2]
            ]
            if any(n.startswith("right_object_") for n in names) and any(
                n.startswith("collision_hand_") for n in names
            ):
                contacts.update(n for n in names if n.startswith("collision_hand_"))
    np.savez(
        output / "hold_check.npz", qpos=qpos, physics_dt=0.002, ctrl=data.ctrl.copy()
    )
    check = {
        "extra_hold_s": 0.5,
        "object_height_end_of_trajectory_m": float(final[2]),
        "object_height_after_hold_m": float(data.qpos[-5]),
        "height_change_during_hold_m": float(data.qpos[-5] - final[2]),
        "object_displacement_during_hold_m": float(
            np.linalg.norm(data.qpos[-7:-4] - final)
        ),
        "max_actuated_finger_speed_rad_s": max_speed,
        "rated_speed_rad_s": 3.456,
        "speed_limit_enforced": False,
        "speed_within_rating": bool(max_speed <= 3.456),
        "hold_contact_geoms": sorted(contacts),
        "continuous_replay_verified": True,
    }
    (output / "hold_check.json").write_text(json.dumps(check, indent=2) + "\n")
    report = json.loads((output / "report.json").read_text())
    with np.load(output / "prepared/target_ik_reference.npz") as ref:
        initial_z = float(ref["initial_qpos"][-5])
    report["trajectory_lift_criterion_met"] = bool(
        report["optimized"]["object_lift_m"] > 0.1
        and report["optimized"]["final_object_error_m"] < 0.03
    )
    report["extended_hold_success"] = bool(
        data.qpos[-5] - initial_z > 0.1
        and check["object_displacement_during_hold_m"] < 0.03
    )
    report["task_success"] = (
        report["trajectory_lift_criterion_met"] and report["extended_hold_success"]
    )
    report["hold_check"] = check
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    """Validate a saved transfer without rerunning optimization."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/xhand_coke_to_yimu"
    )
    args = parser.parse_args()
    print(json.dumps(validate(args.output_dir)["hold_check"], indent=2))


if __name__ == "__main__":
    main()
