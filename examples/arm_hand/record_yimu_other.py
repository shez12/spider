"""Record the Xhand coke sample beside continuous Yimu control replay."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    """Render synchronized source and target command replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/xhand_coke_to_yimu"
    )
    args = parser.parse_args()
    output = args.output_dir
    report = json.loads((output / "report.json").read_text())
    source_dir = Path(report["source_scene"]).parent
    source_model = mujoco.MjModel.from_xml_path(str(source_dir / "scene.xml"))
    # Display-only changes make the source hand/object visible on the dark
    # published floor. No simulation state or contact property is changed.
    source_model.vis.headlight.active = 1
    source_model.vis.headlight.ambient[:] = [.5, .5, .5]
    source_model.vis.headlight.diffuse[:] = [.7, .7, .7]
    floor = source_model.geom("floor").id
    source_model.geom_matid[floor] = -1
    source_model.geom_rgba[floor] = [.86, .88, .9, 1]
    source_model.geom_rgba[source_model.geom("right_object_visual").id] = [
        0.85,
        0.27,
        0.06,
        1,
    ]
    with np.load(output / "chosen_sample.npz") as original:
        source_qpos = original["qpos"]
    model = mujoco.MjModel.from_xml_path(str(output / "yimu/scene.xml"))
    actual, shown, source = (
        mujoco.MjData(model),
        mujoco.MjData(model),
        mujoco.MjData(source_model),
    )
    trajectory = np.load(output / "yimu/trajectory.npz")
    mujoco.mj_setState(
        model, actual, trajectory["initial_state"], mujoco.mjtState.mjSTATE_FULLPHYSICS
    )
    cameras = []
    for azimuth in [130, 130]:
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [-0.13, 0.02 if azimuth == 130 else -0.02, 0.24]
        camera.distance, camera.azimuth, camera.elevation = 1.15, azimuth, -20
        cameras.append(camera)
    with (
        mujoco.Renderer(source_model, 480, 640) as left,
        mujoco.Renderer(model, 480, 640) as right,
        imageio.get_writer(output / "comparison.mp4", fps=50) as video,
    ):
        count = len(trajectory["ctrl"])
        hold = np.load(output / "hold_check.npz")
        commands = np.vstack(
            [trajectory["ctrl"], np.repeat(trajectory["ctrl"][-1:], 50, axis=0)]
        )
        for i, command in enumerate(commands):
            actual.ctrl[:] = command
            for _ in range(5):
                mujoco.mj_step(model, actual)
            np.testing.assert_allclose(
                actual.qpos,
                trajectory["qpos"][i]
                if i < count
                else hold["qpos"][(i - count + 1) * 5 - 1],
                atol=1e-9,
                rtol=1e-9,
            )
            if i % 2:
                continue
            source.qpos[:] = source_qpos[min(i + 1, len(source_qpos) - 1)]
            mujoco.mj_forward(source_model, source)
            shown.qpos[:], shown.qvel[:] = actual.qpos, actual.qvel
            mujoco.mj_forward(model, shown)
            left.update_scene(source, camera=cameras[0])
            right.update_scene(shown, camera=cameras[1])
            panels = [left.render().copy(), right.render().copy()]
            for panel, label in zip(
                panels,
                [
                    (
                        "REFERENCE: Xhand, 3x slow + hold"
                        if report.get("time_scale", 1) > 1
                        else "DATASET: right Xhand, Coke task"
                    ),
                    "RESULT: original Yimu + arm",
                ],
                strict=True,
            ):
                cv2.rectangle(panel, (0, 0), (640, 65), (255, 255, 255), -1)
                cv2.putText(
                    panel, label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2
                )
                cv2.putText(
                    panel,
                    f"t={actual.time:.2f}s | "
                    + (
                        "EXTRA HOLD; reference frozen"
                        if i >= count
                        else "same can mass: 0.299 kg"
                    ),
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
