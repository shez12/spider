"""Independently replay a saved Yimu arm trial and report physical limits."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from spider.robot_model import project_joint_equalities


def main():
    """Verify command replay, measure constraints, and optionally record video."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/allegro_to_yimu_v2")
    )
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    folder = args.output_dir.resolve()
    model = mujoco.MjModel.from_xml_path(str(folder / "arm_hand_scene.xml"))
    data = mujoco.MjData(model)
    trajectory = np.load(folder / "trajectory.npz", allow_pickle=False)
    mujoco.mj_setState(
        model, data, trajectory["initial_state"], mujoco.mjtState.mjSTATE_FULLPHYSICS
    )
    max_force = max_speed = linkage = replay_error = 0.0
    contacts = 0
    motor_dofs = model.jnt_dofadr[model.actuator_trnid[7:, 0]]
    assert model.nu == 24 and model.neq == 4
    np.testing.assert_allclose(
        model.actuator_forcerange[7:], np.tile([-0.16, 0.16], (17, 1))
    )
    for command, expected in zip(trajectory["ctrl"], trajectory["qpos"], strict=True):
        data.ctrl[:] = command
        mujoco.mj_step(model, data)
        max_force = max(max_force, float(np.abs(data.actuator_force[7:]).max()))
        max_speed = max(max_speed, float(np.abs(data.qvel[motor_dofs]).max()))
        linkage = max(
            linkage,
            float(
                np.abs(
                    project_joint_equalities(model, data.qpos.copy()) - data.qpos
                ).max()
            ),
        )
        replay_error = max(replay_error, float(np.abs(data.qpos - expected).max()))
        for contact in data.contact:
            names = [
                model.geom(int(g)).name or "" for g in [contact.geom1, contact.geom2]
            ]
            if any(n.startswith("collision_hand_") for n in names) and any(
                n.startswith("right_object_") for n in names
            ):
                contacts += 1
    assert np.isfinite(data.qpos).all()
    assert replay_error < 1e-9, replay_error
    metrics = json.loads((folder / "metrics.json").read_text())
    report = {
        "continuous_replay_max_qpos_error": replay_error,
        "max_motor_torque_nm": max_force,
        "max_motor_speed_rad_s": max_speed,
        "rated_motor_speed_rad_s": 3.456,
        "speed_limit_enforced": False,
        "max_passive_joint_error_rad": linkage,
        "hand_object_contacts": contacts,
        "physics_timestep_s": float(model.opt.timestep),
        "native_adapter_timestep_s": 0.002,
        "object_mass_kg": float(model.body_mass[model.body("right_object").id]),
        "task_success": bool(
            metrics["optimized"]["object_lift_m"] > 0.25
            and metrics["optimized"]["final_object_error_m"] < 0.05
        ),
        "limitations": "Coarse 0.01 s arm-demo physics; speed cap is not implemented. Simulation integration does not validate hardware readiness.",
    }
    (folder / "physical_validation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    if args.video:
        from examples.arm_hand.dataset_demo import render

        provenance = json.loads((folder / "reference_provenance.json").read_text())
        if provenance.get("time_scale", 1) != 1:
            raise ValueError("Use the original-speed trial for this recorder")
        source_file = Path(provenance["source_file"])
        source_model = mujoco.MjModel.from_xml_path(
            str(source_file.parent.parent / "scene.xml")
        )
        with np.load(source_file, allow_pickle=False) as source:
            source_qpos = source["qpos"].reshape(-1, source_model.nq)[
                provenance["start_frame"] :
            ]
        render(
            model,
            source_model,
            {},
            source_qpos,
            trajectory["initial_state"],
            trajectory["ctrl"],
            folder,
        )


if __name__ == "__main__":
    main()
