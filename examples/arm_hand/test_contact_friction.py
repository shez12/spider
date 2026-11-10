"""Sweep only explicit hand-object sliding friction with unchanged controls."""

import json
from pathlib import Path

import mujoco
import numpy as np

from examples.arm_hand.test_active_distal import fingerprint

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/yimu_active_distal"


def run(variant, mu, static):
    """Measure retention while preserving all non-friction parameters."""
    model = mujoco.MjModel.from_xml_path(str(OUTPUT / variant / "scene.xml"))
    pairs = []
    for pair in range(model.npair):
        names = [
            model.geom(int(g)).name or ""
            for g in [model.pair_geom1[pair], model.pair_geom2[pair]]
        ]
        if any(n.startswith("collision_hand_") for n in names) and any(
            n.startswith("right_object_") for n in names
        ):
            pairs.append(pair)
    assert len(pairs) == 198
    np.testing.assert_allclose(
        model.pair_friction[pairs], np.tile([1, 1, 0.1, 0, 0], (len(pairs), 1))
    )
    model.pair_friction[pairs, :2] = mu
    data = mujoco.MjData(model)
    if static:
        record = np.load(OUTPUT / "grasp" / f"{variant}_closure0.3_distal0.npz")
        data.qpos[:] = record["initial_qpos"]
    else:
        record = np.load(OUTPUT / variant / "trajectory.npz")
        mujoco.mj_setState(
            model, data, record["initial_state"], mujoco.mjtState.mjSTATE_FULLPHYSICS
        )
    initial_z = float(data.qpos[-5])
    settled_z = initial_z
    history = []
    observed = set()
    for step, command in enumerate(record["ctrl"]):
        if static and step == 100:
            settled_z = float(data.qpos[-5])
        data.ctrl[:] = command
        for _ in range(5):
            mujoco.mj_step(model, data)
            for contact in data.contact:
                names = [
                    model.geom(int(g)).name or ""
                    for g in [contact.geom1, contact.geom2]
                ]
                if any(n.startswith("collision_hand_") for n in names) and any(
                    n.startswith("right_object_") for n in names
                ):
                    observed.add(tuple(np.round(contact.friction, 8)))
        history.append(data.qpos.copy())
    result = {
        "variant": variant,
        "test": "stationary_grasp" if static else "optimized_full_trajectory",
        "sliding_friction": mu,
        "object_lift_m": float(data.qpos[-5] - settled_z),
        "observed_contact_friction": sorted(observed),
    }
    np.savez(
        OUTPUT / "friction" / f"{variant}_mu{mu:g}_static{int(static)}.npz",
        qpos=history,
        ctrl=record["ctrl"],
        physics_dt=0.002,
        control_dt=0.01,
    )
    return result


def main():
    """Save model-only friction sensitivity experiments for both hand variants."""
    before = fingerprint()
    (OUTPUT / "friction").mkdir(exist_ok=True)
    reports = []
    for variant in ["coupled", "active_distal"]:
        for static in [True, False]:
            for mu in [0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 5.0]:
                report = run(variant, mu, static)
                reports.append(report)
                print(json.dumps(report), flush=True)
    assert before == fingerprint()
    (OUTPUT / "friction/report.json").write_text(
        json.dumps(
            {
                "results": reports,
                "method_files_unchanged": True,
                "notes": "Only the first two explicit hand-object pair friction coefficients change in memory. Torsional friction 0.1, condim 4, geometry, 0.843 kg object, torque and controls remain unchanged. No optimizer rerun; high friction values are sensitivity probes, not measured material properties.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
