"""Compare stationary grasp retention using coupled and active distal models."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from examples.arm_hand.diagnose_yimu import telemetry

ROOT = Path(__file__).resolve().parents[2]


def run(trial, output, variant, closure, distal_offset):
    """Replay identical arm commands with matched original finger commands."""
    model = mujoco.MjModel.from_xml_path(str(output / variant / "scene.xml"))
    ref = np.load(trial / "target_ik_reference.npz")
    data = mujoco.MjData(model)
    data.qpos[:28] = ref["target_joint_ik"][70]
    data.qpos[-7:] = ref["object_qpos"][69]
    initial = data.qpos.copy()
    # Reuse the previously diagnosed 5 cm lift commands, identically for both
    # models. These are ordinary arm actuator commands, not pose overwrites.
    source = np.load(trial / "diagnosis/grasp_f70_c0.3_p1.npz")
    controls = source["ctrl"].copy()
    if closure != 0.3:
        controls = np.load(trial / f"diagnosis/grasp_f70_c{closure:g}_p1.npz")[
            "ctrl"
        ].copy()
        controls[:, :7] = source["ctrl"][:, :7]
    if variant == "active_distal":
        addresses = model.jnt_qposadr[model.actuator_trnid[24:, 0]]
        distal = ref["target_joint_ik"][70, addresses] + distal_offset
        controls = np.column_stack([controls, np.broadcast_to(distal, (300, 4))])
    controls = np.clip(controls, *model.actuator_ctrlrange.T)
    contacts, normal, max_torque = set(), 0.0, 0.0
    history = []
    hold_z = None
    palm = model.site("right_palm").id
    for step, command in enumerate(controls):
        if step == 100:
            hold_z = data.qpos[-5]
            mujoco.mj_forward(model, data)
            wrist_z = data.site_xpos[palm, 2]
        data.ctrl[:] = command
        for _ in range(5):
            mujoco.mj_step(model, data)
            fingers, force, _ = telemetry(model, data)
            if 75 <= step < 100:
                contacts.update(fingers)
                normal += force / 125
            max_torque = max(max_torque, float(np.abs(data.actuator_force[7:]).max()))
        history.append(data.qpos.copy())
    mujoco.mj_forward(model, data)
    result = {
        "variant": variant,
        "frame": 70,
        "closure_offset_rad": closure,
        "distal_offset_rad": distal_offset if variant == "active_distal" else None,
        "hold_contact_fingers": sorted(contacts),
        "hold_mean_sum_normal_force_n": normal,
        "object_lift_after_hold_m": float(data.qpos[-5] - hold_z),
        "wrist_lift_m": float(data.site_xpos[palm, 2] - wrist_z),
        "max_motor_torque_nm": max_torque,
    }
    stem = f"{variant}_closure{closure:g}_distal{distal_offset:g}"
    np.savez(
        output / "grasp" / f"{stem}.npz",
        initial_qpos=initial,
        qpos=history,
        ctrl=controls,
        control_dt=0.01,
        physics_dt=0.002,
    )
    return result


def main():
    """Run a matched arm-lift command set with and without distal actuation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trial", type=Path, default=ROOT / "outputs/allegro_to_yimu_v2"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/yimu_active_distal"
    )
    args = parser.parse_args()
    (args.output_dir / "grasp").mkdir(exist_ok=True)
    reports = []
    for closure in [0.0, 0.1, 0.3]:
        for variant in ["coupled", "active_distal"]:
            for distal in [0.0] if variant == "coupled" else [-0.3, 0.0, 0.3]:
                result = run(args.trial, args.output_dir, variant, closure, distal)
                reports.append(result)
                print(json.dumps(result), flush=True)
    (args.output_dir / "grasp/comparison.json").write_text(
        json.dumps(reports, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
