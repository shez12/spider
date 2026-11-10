"""Render and verify a saved Sharpa trajectory without requiring raw DexYCB RGB."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    """Continuously simulate saved controls and render the Sharpa result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=ROOT / "outputs/mano_to_sharpa"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    output = (args.output or input_dir / "sharpa_replay.mp4").resolve()

    scene_text = (input_dir / "scene.xml").read_text()
    dataset = ROOT / "example_datasets/retarget_full"
    snapshot_dataset = ROOT / "reproduction/snapshot-retarget_full"
    if not dataset.exists() and snapshot_dataset.exists():
        scene_text = scene_text.replace(str(dataset), str(snapshot_dataset))
    replay_scene = input_dir / "replay_scene.xml"
    replay_scene.write_text(scene_text)

    model = mujoco.MjModel.from_xml_path(str(replay_scene))
    data = mujoco.MjData(model)
    with np.load(input_dir / "trajectory.npz") as trajectory:
        mujoco.mj_setState(
            model,
            data,
            trajectory["initial_state"],
            mujoco.mjtState.mjSTATE_FULLPHYSICS,
        )
        controls = trajectory["ctrl"]
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [-0.13, 0, 0.38]
        camera.distance = 1.65
        camera.azimuth = 230
        camera.elevation = -20
        output.parent.mkdir(parents=True, exist_ok=True)
        max_error = 0.0
        with (
            mujoco.Renderer(model, 480, 640) as renderer,
            imageio.get_writer(output, fps=50) as writer,
        ):
            for i, command in enumerate(controls):
                data.ctrl[:] = command
                for _ in range(5):
                    mujoco.mj_step(model, data)
                expected = trajectory["qpos"][i]
                max_error = max(
                    max_error, float(np.max(np.abs(data.qpos - expected)))
                )
                if i % 2:
                    continue
                renderer.update_scene(data, camera=camera)
                frame = renderer.render().copy()
                cv2.rectangle(frame, (0, 0), (640, 58), (255, 255, 255), -1)
                phase = "motion"
                cv2.putText(
                    frame,
                    "LEFT SHARPA + ARM (CONTINUOUS PHYSICS REPLAY)",
                    (10, 23),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (0, 0, 0),
                    1,
                )
                cv2.putText(
                    frame,
                    f"t={data.time:.2f}s | {phase}",
                    (10, 46),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.44,
                    (0, 0, 0),
                    1,
                )
                writer.append_data(frame)
    if max_error > 1e-9:
        raise AssertionError(f"Replay diverged: max qpos error {max_error}")
    print(f"Wrote {output}; max qpos replay error={max_error}", flush=True)


if __name__ == "__main__":
    main()
