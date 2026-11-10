"""Test only higher Yimu finger torque limits with unchanged optimization code."""

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from examples.arm_hand.test_active_distal import fingerprint, optimize

ROOT = Path(__file__).resolve().parents[2]


def prepare(source, destination, torque):
    """Change only the 17 finger force limits in a separate source scene."""
    destination.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(source / "arm_hand_scene.xml")
    root = tree.getroot()
    motors = list(root.find("actuator"))
    assert len(motors) == 24 and len(root.find("equality")) == 4
    for motor in motors[7:]:
        motor.set("forcerange", f"{-torque} {torque}")
        motor.set("forcelimited", "true")
    ET.indent(root)
    tree.write(destination / "arm_hand_scene.xml", encoding="unicode")
    for name in ["trajectory.npz", "target_ik_reference.npz"]:
        shutil.copy2(source / name, destination / name)
    original = mujoco.MjModel.from_xml_path(str(source / "arm_hand_scene.xml"))
    variant = mujoco.MjModel.from_xml_path(str(destination / "arm_hand_scene.xml"))
    for field in [
        "body_mass",
        "body_inertia",
        "jnt_range",
        "dof_armature",
        "dof_damping",
        "dof_frictionloss",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_ctrlrange",
        "pair_friction",
        "eq_data",
    ]:
        np.testing.assert_array_equal(getattr(original, field), getattr(variant, field))
    np.testing.assert_allclose(
        variant.actuator_forcerange[7:], np.tile([-torque, torque], (17, 1))
    )
    return destination


def main():
    """Run equal-budget torque ablations while preserving original models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/yimu_torque_sweep"
    )
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.samples < 20:
        parser.error("At least 20 samples required")
    before = fingerprint()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    source = ROOT / "outputs/allegro_to_yimu_v2"
    reports = []
    for torque in [0.16, 0.45, 1.0, 1.864]:
        name = f"torque_{torque:g}"
        baseline = ROOT / "outputs/yimu_active_distal/coupled"
        # The prior coupled experiment has exactly the same settings and seed.
        if (
            torque == 0.16
            and args.samples == 512
            and args.iterations == 20
            and baseline.exists()
        ):
            shutil.copytree(baseline, output / name, dirs_exist_ok=True)
            report = json.loads((output / name / "metrics.json").read_text())
            report["reused_verified_baseline"] = True
        else:
            trial = prepare(source, output / "inputs" / name, torque)
            report = optimize(
                SimpleNamespace(
                    trial=trial,
                    output_dir=output,
                    threads=args.threads,
                    samples=args.samples,
                    iterations=args.iterations,
                    seed=0,
                ),
                name,
                False,
            )
        report["finger_torque_limit_nm"] = torque
        report["task_success"] = bool(
            report["optimized"]["object_lift_m"] > 0.25
            and report["optimized"]["final_object_error_m"] < 0.05
        )
        reports.append(report)
        assert before == fingerprint()
        (output / "comparison.json").write_text(
            json.dumps(
                {
                    "comparison": reports,
                    "method_files_unchanged": True,
                    "method_sha256": before,
                    "notes": "Only 17 finger force limits change; all four passive couplings retained. Same seed, initial state, saved nominal controls, geometry, friction, gains, object and matched sampling budget. Models use 2 ms physics. Higher limits are hypothetical and do not represent the supplied hardware. No new IK fit or speed limiter.",
                },
                indent=2,
            )
            + "\n"
        )
        print(json.dumps(report), flush=True)
    # Also isolate grasp retention with the prior 5 cm wrist-lift commands.
    from examples.arm_hand.compare_distal_grasp import run

    (output / "grasp").mkdir(exist_ok=True)
    grasp_reports = []
    for torque in [0.16, 0.45, 1.0, 1.864]:
        for closure in [0.0, 0.1, 0.3]:
            grasp_reports.append(
                run(source, output, f"torque_{torque:g}", closure, 0.0)
            )
    (output / "grasp/comparison.json").write_text(
        json.dumps(grasp_reports, indent=2) + "\n"
    )
    assert before == fingerprint()


if __name__ == "__main__":
    main()
