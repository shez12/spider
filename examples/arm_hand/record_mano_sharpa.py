"""Record original RGB, annotated human skeleton, and dynamic Sharpa replay."""

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

from examples.arm_hand.retarget_mano_sharpa import OUT, ROOT


def main():
    """Render three synchronized panels and verify saved control states."""
    conversion = json.loads((OUT / "conversion.json").read_text())
    selected = next(
        r
        for r in json.loads((ROOT / "outputs/sharpa_more/selected.json").read_text())
        if r["label"] == "mustard_bottle"
    )
    path = ROOT / "example_datasets/retarget_full" / selected["scene"]
    tree = ET.parse(path)
    root = tree.getroot()
    compiler = root.find("compiler")
    compiler.set("meshdir", str((path.parent / compiler.get("meshdir")).resolve()))
    world = root.find("worldbody")
    for node in list(world):
        if node.tag not in ["light", "geom"] and node.get("name") != "right_object":
            world.remove(node)
    for name in ["actuator", "sensor", "contact", "equality", "keyframe", "tendon"]:
        node = root.find(name)
        if node is not None:
            root.remove(node)
    root.find(
        "worldbody/body[@name='right_object']/geom[@name='right_object_visual']"
    ).set("rgba", ".85 .27 .06 1")
    tree.write(OUT / "human_display_scene.xml", encoding="unicode")
    source_model = mujoco.MjModel.from_xml_path(str(OUT / "human_display_scene.xml"))
    source_model.vis.headlight.active = 1
    source_model.vis.headlight.ambient[:] = 0.5
    source = mujoco.MjData(source_model)
    model = mujoco.MjModel.from_xml_path(str(OUT / "scene.xml"))
    actual = mujoco.MjData(model)
    shown = mujoco.MjData(model)
    trajectory = np.load(OUT / "trajectory.npz")
    hold = np.load(OUT / "hold_check.npz")
    human = np.load(OUT / "human_reference.npz")
    mujoco.mj_setState(
        model, actual, trajectory["initial_state"], mujoco.mjtState.mjSTATE_FULLPHYSICS
    )
    cameras = []
    for az in [130, 230]:
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [-0.13, 0, 0.38]
        camera.distance = 1.65
        camera.azimuth = az
        camera.elevation = -20
        cameras.append(camera)
    links = [
        (start + j - 1, start + j) for start in [1, 5, 9, 13, 17] for j in range(1, 4)
    ]
    links += [(0, start) for start in [1, 5, 9, 13, 17]]
    controls = np.vstack(
        [trajectory["ctrl"], np.repeat(trajectory["ctrl"][-1:], 50, axis=0)]
    )
    with (
        mujoco.Renderer(source_model, 480, 640) as left,
        mujoco.Renderer(model, 480, 640) as right,
        imageio.get_writer(OUT / "comparison.mp4", fps=50) as writer,
    ):
        for i, command in enumerate(controls):
            actual.ctrl[:] = command
            for _ in range(5):
                mujoco.mj_step(model, actual)
            expected = (
                trajectory["qpos"][i]
                if i < len(trajectory["ctrl"])
                else hold["qpos"][(i - len(trajectory["ctrl"]) + 1) * 5 - 1]
            )
            np.testing.assert_allclose(actual.qpos, expected, rtol=1e-9, atol=1e-9)
            if i % 2:
                continue
            index = min(i + 1, len(human["joints"]) - 1)
            source.qpos[:] = human["object_qpos"][index]
            mujoco.mj_forward(source_model, source)
            left.update_scene(source, camera=cameras[0])
            joints = human["joints"][index]
            for a, b in links:
                geom = left.scene.geoms[left.scene.ngeom]
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                    np.zeros(3),
                    np.zeros(3),
                    np.eye(3).ravel(),
                    np.array([0.1, 0.4, 0.8, 1.0]),
                )
                mujoco.mjv_connector(
                    geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.004, joints[a], joints[b]
                )
                left.scene.ngeom += 1
            for j in joints:
                geom = left.scene.geoms[left.scene.ngeom]
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.full(3, 0.006),
                    j,
                    np.eye(3).ravel(),
                    np.array([0.9, 0.6, 0.2, 1.0]),
                )
                left.scene.ngeom += 1
            shown.qpos[:] = actual.qpos
            shown.qvel[:] = actual.qvel
            mujoco.mj_forward(model, shown)
            right.update_scene(shown, camera=cameras[1])
            frame_id = min(
                conversion["source_frame_ids"][-1],
                round(conversion["source_frame_ids"][0] + human["time"][index] * 30),
            )
            rgb = cv2.imread(
                str(
                    Path(conversion["sequence"])
                    / conversion["camera"]
                    / f"color_{frame_id:06d}.jpg"
                )
            )
            if rgb is None:
                raise RuntimeError("Missing original RGB frame")
            panels = [
                cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB),
                left.render().copy(),
                right.render().copy(),
            ]
            for panel, title in zip(
                panels,
                [
                    "ORIGINAL HUMAN: DexYCB RGB",
                    "REFERENCE: human 21-joint annotations",
                    "RESULT: left Sharpa + arm (physics)",
                ],
                strict=True,
            ):
                cv2.rectangle(panel, (0, 0), (640, 65), (255, 255, 255), -1)
                cv2.putText(
                    panel, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 1
                )
                phase = (
                    "motion"
                    if i < len(human["joints"]) - 1
                    else (
                        "extra control hold"
                        if i >= len(trajectory["ctrl"])
                        else "reference hold"
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
    print(OUT / "comparison.mp4", flush=True)


if __name__ == "__main__":
    main()
