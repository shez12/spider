"""Record the original sample beside continuous left-Sharpa control replay."""

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    """Render synchronized source and mirrored-task target with honest labels."""
    output = ROOT / "outputs/allegro_to_sharpa_left"
    source_dir = (
        ROOT
        / "example_datasets/arm_hand_source/processed/dexycb_lifted/allegro/right/dexycb_01_right_002_master_chef_can_20200709_141754_00"
    )
    source_model = mujoco.MjModel.from_xml_path(str(source_dir / "scene.xml"))
    source_model.geom_rgba[source_model.geom("right_object_visual").id] = [
        0.85,
        0.27,
        0.06,
        1,
    ]
    with np.load(source_dir / "0/trajectory_mjwp.npz") as original:
        source_qpos = original["qpos"].reshape(-1, source_model.nq)
    model = mujoco.MjModel.from_xml_path(str(output / "scene.xml"))
    actual, shown, source = (
        mujoco.MjData(model),
        mujoco.MjData(model),
        mujoco.MjData(source_model),
    )
    trajectory = np.load(output / "trajectory.npz")
    mujoco.mj_setState(
        model, actual, trajectory["initial_state"], mujoco.mjtState.mjSTATE_FULLPHYSICS
    )
    cameras = []
    for azimuth in [130, 230]:
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [-0.13, 0.02 if azimuth == 130 else -0.02, 0.24]
        camera.distance, camera.azimuth, camera.elevation = 1.15, azimuth, -20
        cameras.append(camera)
    with (
        mujoco.Renderer(source_model, 480, 640) as left,
        mujoco.Renderer(model, 480, 640) as right,
        imageio.get_writer(output / "comparison.mp4", fps=50) as video,
    ):
        for i, command in enumerate(trajectory["ctrl"]):
            actual.ctrl[:] = command
            for _ in range(5):
                mujoco.mj_step(model, actual)
            np.testing.assert_allclose(
                actual.qpos, trajectory["qpos"][i], atol=1e-9, rtol=1e-9
            )
            if i % 2:
                continue
            source.qpos[:] = source_qpos[i + 1]
            mujoco.mj_forward(source_model, source)
            shown.qpos[:], shown.qvel[:] = actual.qpos, actual.qvel
            mujoco.mj_forward(model, shown)
            left.update_scene(source, camera=cameras[0])
            right.update_scene(shown, camera=cameras[1])
            panels = [left.render().copy(), right.render().copy()]
            for panel, label in zip(
                panels,
                [
                    "DATASET: right Allegro, original task",
                    "RESULT: left Sharpa + arm, mirrored task",
                ],
                strict=True,
            ):
                cv2.rectangle(panel, (0, 0), (640, 65), (255, 255, 255), -1)
                cv2.putText(
                    panel, label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2
                )
                cv2.putText(
                    panel,
                    f"t={actual.time:.2f}s | same can mass: 0.843 kg",
                    (12, 51),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.47,
                    (0, 0, 0),
                    1,
                )
            video.append_data(np.concatenate(panels, axis=1))
    print(output / "comparison.mp4")


if __name__ == "__main__":
    main()
