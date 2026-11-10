"""Regression checks for hand-only reference tracking through actual dynamics."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np


class DirectShootingTest(unittest.TestCase):
    """Verify improvement and independent command replay, not pose playback."""

    def test_optimization_and_continuous_replay(self):
        """Ensure optimized controls improve tracking and reproduce saved states."""
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            subprocess.run(
                [
                    sys.executable,
                    str(root / "examples/run_arm_hand.py"),
                    "--synthetic",
                    "--output-dir",
                    directory,
                    "--iterations",
                    "8",
                    "--no-video",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads((output / "metrics.json").read_text())
            self.assertLess(
                report["optimized"]["cost"], report["baseline"]["cost"] * 0.1
            )
            self.assertLess(report["optimized"]["wrist_rmse_m"], 0.003)
            with np.load(output / "hand_only_reference.npz") as reference:
                self.assertEqual(
                    set(reference.files),
                    {"time", "wrist_pos", "wrist_quat", "finger_qpos"},
                )
                self.assertEqual(reference["finger_qpos"].shape[1], 16)
            model = mujoco.MjModel.from_xml_path(str(output / "arm_hand_scene.xml"))
            self.assertEqual(model.nu, 23)
            data = mujoco.MjData(model)
            with np.load(output / "trajectory.npz") as trajectory:
                mujoco.mj_setState(
                    model,
                    data,
                    trajectory["initial_state"],
                    mujoco.mjtState.mjSTATE_FULLPHYSICS,
                )
                replay = []
                for control in trajectory["ctrl"]:
                    data.ctrl[:] = control
                    mujoco.mj_step(model, data)
                    replay.append(data.qpos.copy())
                np.testing.assert_allclose(
                    replay, trajectory["qpos"], atol=1e-10, rtol=1e-10
                )

    def test_dataset_object_lift_and_mount(self):
        """Check real dataset provenance, physical object lift and flange alignment."""
        root = Path(__file__).resolve().parents[2]
        source = (
            root
            / "example_datasets/arm_hand_source/processed/dexycb_lifted/allegro/right/dexycb_01_right_002_master_chef_can_20200709_141754_00/scene.xml"
        )
        if not source.exists():
            self.skipTest("Run examples/arm_hand/download_example.py first")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            subprocess.run(
                [
                    sys.executable,
                    str(root / "examples/run_arm_hand.py"),
                    "--output-dir",
                    directory,
                    "--iterations",
                    "12",
                    "--no-video",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads((output / "metrics.json").read_text())
            self.assertGreater(report["optimized"]["object_lift_m"], 0.25)
            self.assertLess(report["optimized"]["final_object_error_m"], 0.025)
            self.assertLess(
                report["optimized"]["object_rmse_m"],
                report["baseline"]["object_rmse_m"],
            )
            provenance = json.loads((output / "reference_provenance.json").read_text())
            self.assertEqual(provenance["repo"], "retarget/retarget_full")
            self.assertEqual(provenance["trajectory"], "trajectory_mjwp.npz")
            model = mujoco.MjModel.from_xml_path(str(output / "arm_hand_scene.xml"))
            palm_id = model.body("right_palm").id
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, model.body_quat[palm_id])
            # Fingers point forward along the forearm, not sideways. The palm's
            # mounting face is 65 mm beyond the arm's final capsule endpoint.
            np.testing.assert_allclose(
                rotation.reshape(3, 3) @ [0, 0, 1], [1, 0, 0], atol=1e-8
            )
            np.testing.assert_allclose(
                model.body_pos[palm_id] + rotation.reshape(3, 3) @ [0, 0, -0.095],
                [0.125, 0, 0],
                atol=1e-8,
            )
            object_id = model.body("right_object").id
            self.assertGreater(model.body_mass[object_id], 0)
            self.assertEqual(
                model.jnt_type[model.body_jntadr[object_id]], mujoco.mjtJoint.mjJNT_FREE
            )
            self.assertEqual(model.neq, 0)
            self.assertEqual(model.nu, 23)
            for joint_id in model.actuator_trnid[:, 0]:
                self.assertNotEqual(model.jnt_bodyid[joint_id], object_id)
            data = mujoco.MjData(model)
            with np.load(output / "trajectory.npz") as trajectory:
                mujoco.mj_setState(
                    model,
                    data,
                    trajectory["initial_state"],
                    mujoco.mjtState.mjSTATE_FULLPHYSICS,
                )
                replay, hand_object_contacts = [], 0
                for control in trajectory["ctrl"]:
                    data.ctrl[:] = control
                    mujoco.mj_step(model, data)
                    replay.append(data.qpos.copy())
                    for contact in data.contact:
                        names = [
                            model.geom(int(g)).name
                            for g in [contact.geom1, contact.geom2]
                        ]
                        if any(n.startswith("collision_hand_") for n in names) and any(
                            n.startswith("right_object_") for n in names
                        ):
                            hand_object_contacts += 1
                self.assertGreater(hand_object_contacts, 0)
                np.testing.assert_allclose(
                    replay, trajectory["qpos"], atol=1e-10, rtol=1e-10
                )
                np.testing.assert_array_equal(data.xfrc_applied, 0)

    def test_cross_hand_sampling(self):
        """Transfer a 16-DoF Allegro reference to 12-DoF Xhand with sampling."""
        root = Path(__file__).resolve().parents[2]
        source = (
            root
            / "example_datasets/arm_hand_source/processed/dexycb_lifted/allegro/right/dexycb_01_right_002_master_chef_can_20200709_141754_00/scene.xml"
        )
        if not source.exists():
            self.skipTest("Download the source Allegro episode first")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            subprocess.run(
                [
                    sys.executable,
                    str(root / "examples/run_arm_hand.py"),
                    "--target-hand",
                    "xhand",
                    "--optimizer",
                    "sampling",
                    "--num-samples",
                    "1024",
                    "--iterations",
                    "10",
                    "--output-dir",
                    directory,
                    "--no-video",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads((output / "metrics.json").read_text())
            mapping = json.loads((output / "cross_hand_mapping.json").read_text())
            self.assertEqual(report["optimizer"], "sampling")
            self.assertEqual(mapping["source_finger_dofs"], 16)
            self.assertEqual(mapping["target_finger_dofs"], 12)
            self.assertFalse(mapping["target_trajectory_loaded"])
            self.assertGreater(report["optimized"]["object_lift_m"], 0.25)
            self.assertLess(report["optimized"]["final_object_error_m"], 0.05)
            self.assertLess(
                report["optimized"]["cost"], report["baseline"]["cost"] * 0.1
            )
            with (
                np.load(output / "hand_only_reference.npz") as original,
                np.load(output / "target_ik_reference.npz") as target,
            ):
                self.assertEqual(original["finger_qpos"].shape[1], 16)
                self.assertEqual(target["finger_qpos"].shape[1], 12)
                np.testing.assert_array_equal(
                    original["object_qpos"], target["object_qpos"]
                )
            model = mujoco.MjModel.from_xml_path(str(output / "arm_hand_scene.xml"))
            self.assertEqual(model.nu, 19)
            self.assertEqual(model.nq, 26)
            self.assertGreater(model.body("right_object").mass, 0)
            self.assertEqual(model.neq, 0)
            data = mujoco.MjData(model)
            with np.load(output / "trajectory.npz") as trajectory:
                mujoco.mj_setState(
                    model,
                    data,
                    trajectory["initial_state"],
                    mujoco.mjtState.mjSTATE_FULLPHYSICS,
                )
                replay = []
                for command in trajectory["ctrl"]:
                    data.ctrl[:] = command
                    mujoco.mj_step(model, data)
                    replay.append(data.qpos.copy())
                np.testing.assert_allclose(
                    replay, trajectory["qpos"], atol=1e-10, rtol=1e-10
                )


if __name__ == "__main__":
    unittest.main()
