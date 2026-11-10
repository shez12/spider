"""Record harder Sharpa trials with source, control replay and extra hold."""

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


def record(output, label):
    """Render verified continuous dynamics beside the source right-hand sample."""
    report = json.loads((output / "report.json").read_text())
    source_model = mujoco.MjModel.from_xml_path(report["source_scene"])
    source_model.vis.headlight.active = 1
    source_model.vis.headlight.ambient[:] = [0.5, 0.5, 0.5]
    source_model.vis.headlight.diffuse[:] = [0.7, 0.7, 0.7]
    floor = source_model.geom("floor").id
    source_model.geom_matid[floor] = -1
    source_model.geom_rgba[floor] = [0.86, 0.88, 0.9, 1]
    source_model.geom_rgba[source_model.geom("right_object_visual").id] = [
        0.85,
        0.27,
        0.06,
        1,
    ]
    model = mujoco.MjModel.from_xml_path(str(output / "scene.xml"))
    actual, shown, source = (
        mujoco.MjData(model),
        mujoco.MjData(model),
        mujoco.MjData(source_model),
    )
    trajectory = np.load(output / "trajectory.npz")
    hold = np.load(output / "hold_check.npz")
    chosen = np.load(output / "chosen_sample.npz")["qpos"]
    mujoco.mj_setState(
        model, actual, trajectory["initial_state"], mujoco.mjtState.mjSTATE_FULLPHYSICS
    )
    cameras = []
    for azimuth in [130, 230]:
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [
            -0.13,
            0.02 if azimuth == 130 else -0.02,
            0.38 if label == "mug" else 0.28,
        ]
        camera.distance, camera.azimuth, camera.elevation = (
            (1.45 if label == "mug" else 1.2),
            azimuth,
            -20,
        )
        cameras.append(camera)
    count = len(trajectory["ctrl"])
    commands = np.vstack(
        [trajectory["ctrl"], np.repeat(trajectory["ctrl"][-1:], 50, axis=0)]
    )
    with (
        mujoco.Renderer(source_model, 480, 640) as left,
        mujoco.Renderer(model, 480, 640) as right,
        imageio.get_writer(output / "comparison.mp4", fps=50) as writer,
    ):
        for i, command in enumerate(commands):
            actual.ctrl[:] = command
            for _ in range(5):
                mujoco.mj_step(model, actual)
            expected = (
                trajectory["qpos"][i]
                if i < count
                else hold["qpos"][(i - count + 1) * 5 - 1]
            )
            np.testing.assert_allclose(actual.qpos, expected, atol=1e-9, rtol=1e-9)
            if i % 2:
                continue
            source.qpos[:] = chosen[min(i + 1, len(chosen) - 1)]
            mujoco.mj_forward(source_model, source)
            shown.qpos[:], shown.qvel[:] = actual.qpos, actual.qvel
            mujoco.mj_forward(model, shown)
            left.update_scene(source, camera=cameras[0])
            right.update_scene(shown, camera=cameras[1])
            panels = [left.render().copy(), right.render().copy()]
            for panel, title in zip(
                panels,
                [
                    f"REFERENCE: Allegro / {label}",
                    f"RESULT: left Sharpa + arm / {label}",
                ],
                strict=True,
            ):
                cv2.rectangle(panel, (0, 0), (640, 65), (255, 255, 255), -1)
                cv2.putText(
                    panel, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2
                )
                phase = (
                    "EXTRA HOLD; reference frozen"
                    if i >= count
                    else (
                        "held final reference"
                        if i >= report["original_frames"] - 1
                        else "original-speed motion"
                    )
                )
                cv2.putText(
                    panel,
                    f"t={actual.time:.2f}s | {phase}",
                    (12, 51),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 0),
                    1,
                )
            writer.append_data(np.concatenate(panels, axis=1))
    print(output / "comparison.mp4", flush=True)


def main():
    """Record all completed cases and concatenate them into one comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/sharpa_complex"
    )
    args = parser.parse_args()
    clips = []
    for name in ["scissors", "mug", "power_drill"]:
        folder = args.output_dir / name
        if (folder / "report.json").exists():
            if not (folder / "comparison.mp4").exists():
                record(folder, name.replace("_", " "))
            clips.append(folder / "comparison.mp4")
    with imageio.get_writer(args.output_dir / "comparison.mp4", fps=50) as writer:
        for path in clips:
            with imageio.get_reader(path) as reader:
                for frame in reader:
                    writer.append_data(frame)
    print(args.output_dir / "comparison.mp4", flush=True)


if __name__ == "__main__":
    main()
