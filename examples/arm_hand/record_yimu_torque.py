"""Record original versus best higher-torque Yimu command replays."""

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
    """Render synchronized continuous command replays with distinct labels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/yimu_torque_sweep"
    )
    args = parser.parse_args()
    results = json.loads((args.output_dir / "comparison.json").read_text())[
        "comparison"
    ]
    higher = min(
        (r for r in results if r["finger_torque_limit_nm"] > 0.16),
        key=lambda r: r["optimized"]["cost"],
    )
    torque = higher["finger_torque_limit_nm"]
    models, actual, display, trajectories = [], [], [], []
    for name in ["torque_0.16", f"torque_{torque:g}"]:
        model = mujoco.MjModel.from_xml_path(str(args.output_dir / name / "scene.xml"))
        data = mujoco.MjData(model)
        trajectory = np.load(args.output_dir / name / "trajectory.npz")
        mujoco.mj_setState(
            model,
            data,
            trajectory["initial_state"],
            mujoco.mjtState.mjSTATE_FULLPHYSICS,
        )
        models.append(model)
        actual.append(data)
        display.append(mujoco.MjData(model))
        trajectories.append(trajectory)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [-0.13, 0.02, 0.24]
    camera.distance, camera.azimuth, camera.elevation = 1.15, 130, -20
    labels = ["ORIGINAL LIMIT: 0.16 Nm", f"HIGHER LIMIT: {torque:g} Nm"]
    with (
        mujoco.Renderer(models[0], 480, 640) as left,
        mujoco.Renderer(models[1], 480, 640) as right,
        imageio.get_writer(args.output_dir / "comparison.mp4", fps=50) as video,
    ):
        for step in range(len(trajectories[0]["ctrl"])):
            panels = []
            for model, data, shown, trajectory, renderer, label in zip(
                models,
                actual,
                display,
                trajectories,
                [left, right],
                labels,
                strict=True,
            ):
                data.ctrl[:] = trajectory["ctrl"][step]
                for _ in range(5):
                    mujoco.mj_step(model, data)
                np.testing.assert_allclose(
                    data.qpos, trajectory["qpos"][step], atol=1e-9, rtol=1e-9
                )
                if step % 2:
                    continue
                shown.qpos[:], shown.qvel[:] = data.qpos, data.qvel
                mujoco.mj_forward(model, shown)
                renderer.update_scene(shown, camera=camera)
                panel = renderer.render().copy()
                cv2.rectangle(panel, (0, 0), (640, 65), (255, 255, 255), -1)
                cv2.putText(
                    panel, label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 0, 0), 2
                )
                cv2.putText(
                    panel,
                    f"t={data.time:.2f}s | same can, gains, geometry and passive couplings",
                    (12, 51),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 0),
                    1,
                )
                panels.append(panel)
            if panels:
                video.append_data(np.concatenate(panels, axis=1))
    print(args.output_dir / "comparison.mp4")


if __name__ == "__main__":
    main()
