"""Track a hand-only reference with a simulated 7-DoF arm and Allegro hand.

This standalone demonstration uses MuJoCo's threaded CPU rollouts with SciPy
direct shooting or SPIDER's sampling update. IK initializes controls; optimization
and evaluation use uninterrupted dynamics. Its arm objective is defined here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

# Select the rendering backend before importing MuJoCo's GL context module.
if "--view" not in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import mujoco.rollout
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
TIP_NAMES = [
    "track_hand_right_index_tip",
    "track_hand_right_middle_tip",
    "track_hand_right_ring_tip",
    "track_hand_right_thumb_tip",
]


def make_scene(path: Path, with_arm: bool) -> mujoco.MjModel:
    """Attach the actual Allegro asset to a generic, gravity-compensated arm."""
    root = ET.parse(ROOT / "spider/assets/robots/allegro/right.xml").getroot()
    root.find("compiler").set(
        "meshdir", str(ROOT / "spider/assets/robots/allegro/assets")
    )
    root.find("option").set("integrator", "implicitfast")
    root.find("option").set("timestep", "0.01")
    world, actuators = root.find("worldbody"), root.find("actuator")
    palm = world.find("body")
    for joint in list(palm.findall("joint")):
        palm.remove(joint)
    for actuator in list(actuators)[:6]:
        actuators.remove(actuator)
    ET.SubElement(palm, "site", name="demo_wrist", size="0.006")
    ET.SubElement(
        world,
        "geom",
        name="floor",
        type="plane",
        size="2 2 0.1",
        rgba="0.86 0.88 0.9 1",
        contype="1",
        conaffinity="1",
    )
    ET.SubElement(world, "light", pos="0.5 -1 2", dir="0 0 -1")
    if with_arm:
        world.remove(palm)
        parent = ET.SubElement(world, "body", name="arm_mount", pos="0 0 0.35")
        ET.SubElement(
            parent,
            "geom",
            type="cylinder",
            size="0.10 0.175",
            pos="0 0 -0.175",
            rgba="0.18 0.2 0.23 1",
            contype="0",
            conaffinity="0",
        )
        axes = ["0 0 1", "0 1 0", "1 0 0", "0 1 0", "1 0 0", "0 1 0", "0 0 1"]
        lengths = [0.08, 0.20, 0.08, 0.20, 0.06, 0.10, 0.06]
        for i, (axis, length) in enumerate(zip(axes, lengths, strict=True)):
            body = ET.SubElement(parent, "body", name=f"arm_link_{i}", gravcomp="1")
            if i:
                body.set("pos", f"{lengths[i - 1]} 0 0")
            ET.SubElement(
                body,
                "joint",
                name=f"arm_joint_{i}",
                axis=axis,
                range="-2.7 2.7",
                damping="3",
                armature="0.12",
            )
            ET.SubElement(
                body,
                "geom",
                type="capsule",
                fromto=f"0 0 0 {length} 0 0",
                size="0.035",
                mass="0.65",
                rgba="0.25 0.5 0.72 1",
                contype="1",
                conaffinity="1",
            )
            ET.SubElement(
                body,
                "geom",
                type="sphere",
                size="0.045",
                mass="0.1",
                rgba="0.15 0.2 0.25 1",
                contype="0",
                conaffinity="0",
            )
            actuator = ET.Element(
                "position",
                name=f"arm_target_{i}",
                joint=f"arm_joint_{i}",
                kp="180",
                kv="25",
                ctrlrange="-2.7 2.7",
                forcerange="-100 100",
            )
            actuators.insert(i, actuator)
            parent = body
        # Allegro's palm origin is at the finger roots, not the wrist flange.
        # Its mounting face is at local z=-0.095 m. Rotate the hand so its
        # longitudinal +z axis extends along the forearm's +x axis, then leave
        # room for a visible wrist adapter rather than burying the arm in it.
        palm.set("pos", f"{lengths[-1] + 0.16} 0 0")
        palm.set("quat", "0.7071067812 0 0.7071067812 0")
        ET.SubElement(
            parent,
            "geom",
            name="wrist_adapter_shaft",
            type="capsule",
            fromto=f"{lengths[-1]} 0 0 {lengths[-1] + 0.045} 0 0",
            size="0.025",
            mass="0.08",
            rgba="0.55 0.57 0.6 1",
            contype="0",
            conaffinity="0",
        )
        ET.SubElement(
            parent,
            "geom",
            name="wrist_adapter_flange",
            type="cylinder",
            pos=f"{lengths[-1] + 0.045} 0 0",
            quat="0.7071067812 0 0.7071067812 0",
            size="0.04 0.02",
            mass="0.08",
            rgba="0.55 0.57 0.6 1",
            contype="0",
            conaffinity="0",
        )
        parent.append(palm)
        contact = ET.SubElement(root, "contact")
        # Neighboring joint housings overlap by construction. Distant links
        # still collide; exclude only links separated by a single short link.
        for i in range(5):
            ET.SubElement(
                contact, "exclude", body1=f"arm_link_{i}", body2=f"arm_link_{i + 2}"
            )
    else:
        palm.set("mocap", "true")
    sensors = ET.SubElement(root, "sensor")
    ET.SubElement(sensors, "framepos", objtype="site", objname="demo_wrist")
    ET.SubElement(sensors, "framequat", objtype="site", objname="demo_wrist")
    for name in TIP_NAMES:
        ET.SubElement(sensors, "framepos", objtype="site", objname=name)
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="unicode")
    return mujoco.MjModel.from_xml_path(str(path))


def quaternion_error(actual, target):
    """Return a three-dimensional local orientation error, in radians."""
    a = Rotation.from_quat(np.asarray(actual)[..., [1, 2, 3, 0]].reshape(-1, 4))
    b = Rotation.from_quat(np.asarray(target)[..., [1, 2, 3, 0]].reshape(-1, 4))
    return (b.inv() * a).as_rotvec().reshape(np.asarray(actual).shape[:-1] + (3,))


def make_reference(model, output: Path, duration: float):
    """Generate a wrist/finger demo, containing no arm-joint trajectory."""
    dt = model.opt.timestep
    times = np.arange(0, duration, dt) + dt
    data = mujoco.MjData(model)
    seed = np.array([0, -0.65, 0.1, 1.2, -0.1, -0.55, 0])
    data.qpos[:7] = seed
    data.qpos[7:] = np.clip(0.25, model.jnt_range[7:, 0], model.jnt_range[7:, 1])
    mujoco.mj_forward(model, data)
    center, orientation = data.sensordata[:3].copy(), data.sensordata[3:7].copy()
    phase = 2 * np.pi * times / duration
    wrist_pos = center + np.column_stack(
        [
            -0.035 * (1 - np.cos(phase)),
            0.055 * np.sin(phase),
            0.025 * (1 - np.cos(phase)),
        ]
    )
    r0 = Rotation.from_quat(orientation[[1, 2, 3, 0]])
    rotations = r0 * Rotation.from_rotvec(
        np.column_stack(
            [
                0.10 * np.sin(phase),
                np.zeros_like(phase),
                0.12 * np.sin(phase),
            ]
        )
    )
    wrist_quat = rotations.as_quat()[:, [3, 0, 1, 2]]
    low, high = model.jnt_range[7:].T
    finger_qpos = np.tile(np.clip(0.25, low, high), (len(times), 1))
    for finger in range(4):
        curl = 0.25 + 0.35 * (1 - np.cos(phase + finger * 0.35)) / 2
        finger_qpos[:, 4 * finger + 1 : 4 * finger + 4] = curl[:, None]
    finger_qpos = np.clip(finger_qpos, low + 0.005, high - 0.005)
    ref = {
        "time": times,
        "wrist_pos": wrist_pos,
        "wrist_quat": wrist_quat,
        "finger_qpos": finger_qpos,
    }
    np.savez(output / "hand_only_reference.npz", **ref)
    return ref, seed


def reference_landmarks(model, ref):
    """Evaluate the hand-only reference's fingertip positions."""
    data = mujoco.MjData(model)
    landmarks = []
    for p, q, fingers in zip(
        ref["wrist_pos"], ref["wrist_quat"], ref["finger_qpos"], strict=True
    ):
        data.mocap_pos[0] = p
        data.mocap_quat[0] = q
        data.qpos[:] = fingers
        mujoco.mj_forward(model, data)
        # mocap quaternion is the palm body orientation, same as demo_wrist.
        landmarks.append(data.sensordata[7:].copy())
    return np.array(landmarks)


def initialize_ik(model, ref, seed, knot_indices):
    """Warm-start direct shooting with sequential arm IK, not reference arm joints."""
    data = mujoco.MjData(model)
    values = []
    for index in knot_indices:

        def residual(q, index=index, posture=seed):
            data.qpos[:7] = q
            data.qpos[7:] = ref["finger_qpos"][index]
            mujoco.mj_forward(model, data)
            return np.concatenate(
                [
                    10 * (data.sensordata[:3] - ref["wrist_pos"][index]),
                    quaternion_error(data.sensordata[3:7], ref["wrist_quat"][index]),
                    0.002 * (q - posture),
                ]
            )

        solution = least_squares(residual, seed, bounds=(-2.69, 2.69), max_nfev=80)
        seed = solution.x
        values.append(np.r_[seed, ref["finger_qpos"][index]])
    return np.array(values)


def main():
    """Optimize arm/finger control knots and render continuous physics replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/arm_hand")
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Duration of the optional --synthetic demo; dataset mode uses the complete episode",
    )
    parser.add_argument("--knots", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="Replay dataset motion more slowly; preserves its spatial path and object mass",
    )
    parser.add_argument(
        "--target-hand",
        choices=["allegro", "xhand", "yimu_v2"],
        default="allegro",
        help="Target hand; xhand enables Allegro-to-Xhand landmark retargeting",
    )
    parser.add_argument(
        "--optimizer", choices=["least-squares", "sampling"], default="least-squares"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1024,
        help="Candidates per iteration for the SPIDER sampling optimizer",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use the optional free-space synthetic regression demo instead of the dataset",
    )
    parser.add_argument(
        "--dataset-dir", type=Path, default=ROOT / "example_datasets/arm_hand_source"
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Starting frame in the dataset's recorded hand-only trajectory",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="NPZ: time, wrist_pos, wrist_quat (wxyz), finger_qpos (16)",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="Replay optimized controls in a MuJoCo window after optimization",
    )
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    if args.time_scale < 1:
        parser.error("--time-scale must be at least 1")
    if (args.synthetic or args.reference) and (
        args.target_hand != "allegro" or args.optimizer != "least-squares"
    ):
        parser.error(
            "Cross-hand transfer and sampling are currently supported in dataset mode only"
        )
    if args.duration <= 0 or args.knots < 2 or args.iterations < 1 or args.threads < 1:
        parser.error(
            "duration must be positive; knots >= 2; iterations and threads >= 1"
        )
    if not args.view:
        os.environ.setdefault("MUJOCO_GL", "egl")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not args.synthetic and not args.reference:
        if __package__:
            from .arm_hand.dataset_demo import run
        else:
            from arm_hand.dataset_demo import run
        run(args, make_scene, quaternion_error)
        return
    model = make_scene(output / "arm_hand_scene.xml", True)
    reference_model = make_scene(output / "hand_only_scene.xml", False)
    assert model.nq == model.nu == 23 and reference_model.nq == 16
    if args.reference:
        with np.load(args.reference, allow_pickle=False) as source:
            ref = {
                name: source[name].copy()
                for name in ("time", "wrist_pos", "wrist_quat", "finger_qpos")
            }
        seed = np.array([0, -0.65, 0.1, 1.2, -0.1, -0.55, 0])
        np.savez(output / "hand_only_reference.npz", **ref)
    else:
        ref, seed = make_reference(model, output, args.duration)
    nstep = len(ref["time"])
    if ref["time"].ndim != 1 or not np.isfinite(ref["time"]).all():
        raise ValueError("Reference time must be a finite one-dimensional array")
    for name, width in [("wrist_pos", 3), ("wrist_quat", 4), ("finger_qpos", 16)]:
        if ref[name].shape != (nstep, width) or not np.isfinite(ref[name]).all():
            raise ValueError(
                f"Invalid reference {name}: expected finite ({nstep}, {width})"
            )
    if nstep < args.knots or not np.allclose(np.diff(ref["time"]), model.opt.timestep):
        raise ValueError(
            "Reference needs at least --knots frames and uniform 0.01 s spacing"
        )
    if not np.allclose(np.linalg.norm(ref["wrist_quat"], axis=1), 1, atol=1e-4):
        raise ValueError("Reference wrist quaternions must be normalized (wxyz)")
    if np.any(ref["finger_qpos"] < model.jnt_range[7:, 0]) or np.any(
        ref["finger_qpos"] > model.jnt_range[7:, 1]
    ):
        raise ValueError("Reference finger joints exceed Allegro joint limits")
    reference_tips = reference_landmarks(reference_model, ref)
    knot_indices = np.linspace(0, nstep - 1, args.knots).astype(int)
    knots = initialize_ik(model, ref, seed, knot_indices)
    initial = mujoco.MjData(model)
    initial.qpos[:] = knots[0]
    initial.ctrl[:] = knots[0]
    mujoco.mj_forward(model, initial)
    initial_state = np.empty(
        mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    )
    mujoco.mj_getState(
        model, initial, initial_state, mujoco.mjtState.mjSTATE_FULLPHYSICS
    )
    grid = np.arange(nstep)
    interpolation = np.stack(
        [
            np.interp(grid, knot_indices, np.eye(args.knots)[i])
            for i in range(args.knots)
        ],
        axis=1,
    )
    low, high = model.actuator_ctrlrange.T
    workers = [mujoco.MjData(model) for _ in range(args.threads)]
    checkpoints = np.arange(0, nstep, 4)
    if checkpoints[-1] != nstep - 1:
        checkpoints = np.r_[checkpoints, nstep - 1]
    count = 0
    start = time.perf_counter()
    with mujoco.rollout.Rollout(nthread=args.threads) as engine:

        def evaluate(parameters, return_rollout=False):
            parameters = np.atleast_2d(parameters)
            values = parameters.reshape(-1, args.knots, model.nu)
            controls = np.einsum("tk,bkj->btj", interpolation, values)
            states, sensors = engine.rollout(
                model,
                workers,
                initial_state,
                controls,
                initial_warmstart=np.zeros(model.nv),
            )
            wrist = sensors[:, checkpoints, :3] - ref["wrist_pos"][checkpoints]
            orient = quaternion_error(
                sensors[:, checkpoints, 3:7],
                np.broadcast_to(
                    ref["wrist_quat"][checkpoints], sensors[:, checkpoints, 3:7].shape
                ),
            )
            fingers = (
                states[:, checkpoints, 1 + 7 : 1 + model.nq]
                - ref["finger_qpos"][checkpoints]
            )
            tips = sensors[:, checkpoints, 7:] - reference_tips[checkpoints]
            residuals = np.concatenate(
                [
                    (30 * wrist).reshape(len(values), -1),
                    (2 * orient).reshape(len(values), -1),
                    (0.5 * fingers).reshape(len(values), -1),
                    (15 * tips).reshape(len(values), -1),
                    (0.015 * np.diff(values, axis=1)).reshape(len(values), -1),
                ],
                axis=1,
            )
            if return_rollout:
                return residuals[0], states[0], sensors[0], controls[0]
            return residuals

        def fun(x):
            nonlocal count
            count += 1
            residual = evaluate(x)[0]
            print(
                f"Evaluation {count}: cost={0.5 * residual @ residual:.6f}, elapsed={time.perf_counter() - start:.1f}s",
                flush=True,
            )
            return residual

        def jac(x):
            # Every derivative is measured through a complete physics rollout.
            epsilon = 1e-3
            batch = np.tile(x, (len(x) + 1, 1))
            batch[1:] += epsilon * np.eye(len(x))
            result = evaluate(batch)
            return ((result[1:] - result[0]) / epsilon).T

        baseline = evaluate(knots.ravel(), True)
        solution = least_squares(
            fun,
            knots.ravel(),
            jac=jac,
            bounds=(np.tile(low, args.knots), np.tile(high, args.knots)),
            max_nfev=args.iterations,
            ftol=1e-5,
            xtol=1e-5,
            gtol=1e-5,
            verbose=0,
        )
        optimized = evaluate(solution.x, True)
    _, states, sensors, controls = optimized
    if not np.isfinite(states).all() or not np.all(np.diff(states[:, 0]) > 0):
        raise RuntimeError("Simulation diverged; no valid optimized trajectory")
    qpos = states[:, 1 : 1 + model.nq]
    if np.any(qpos < model.jnt_range[:, 0] - 0.02) or np.any(
        qpos > model.jnt_range[:, 1] + 0.02
    ):
        raise RuntimeError("Executed trajectory exceeds a joint limit")

    def metrics(result):
        residual, _, sensor, _ = result
        return {
            "cost": float(0.5 * residual @ residual),
            "wrist_rmse_m": float(
                np.sqrt(
                    np.mean(np.sum((sensor[:, :3] - ref["wrist_pos"]) ** 2, axis=1))
                )
            ),
            "orientation_rmse_rad": float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            quaternion_error(sensor[:, 3:7], ref["wrist_quat"]) ** 2,
                            axis=1,
                        )
                    )
                )
            ),
            "fingertip_rmse_m": float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            (
                                sensor[:, 7:].reshape(-1, 4, 3)
                                - reference_tips.reshape(-1, 4, 3)
                            )
                            ** 2,
                            axis=2,
                        )
                    )
                )
            ),
        }

    report = {
        "robot": "Generic 7-DoF arm + 16-DoF Allegro",
        "reference": "Wrist pose and finger joints only; no reference arm joints",
        "method": "Bounded direct shooting over interpolated arm/finger position targets; finite-difference simulator Jacobians",
        "baseline": metrics(baseline),
        "optimized": metrics(optimized),
        "optimizer_converged": bool(solution.success),
        "optimizer_message": solution.message,
        "evaluations": int(solution.nfev),
        "elapsed_seconds": time.perf_counter() - start,
        "continuous_control_replay": True,
        "simulation_dt": model.opt.timestep,
        "limitations": "Synthetic tracking demo; gravity-compensated generic arm; no piano, grasping, hardware validation, or collision-avoidance objective",
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez(
        output / "trajectory.npz",
        initial_state=initial_state,
        ctrl=controls,
        qpos=qpos,
        qvel=states[:, 1 + model.nq :],
        time=states[:, 0],
        control_knots=solution.x.reshape(args.knots, model.nu),
    )
    print(json.dumps(report, indent=2), flush=True)
    if not args.no_video:
        render_video(model, reference_model, initial_state, controls, ref, output)
    if args.view:
        from mujoco import viewer as mj_viewer

        data = mujoco.MjData(model)
        with mj_viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                mujoco.mj_setState(
                    model, data, initial_state, mujoco.mjtState.mjSTATE_FULLPHYSICS
                )
                for command in controls:
                    if not viewer.is_running():
                        break
                    tick = time.perf_counter()
                    data.ctrl[:] = command
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    time.sleep(
                        max(0, model.opt.timestep - (time.perf_counter() - tick))
                    )


def render_video(model, reference_model, initial_state, controls, ref, output):
    """Render reference-only hand beside continuously simulated optimized robot."""
    import cv2

    actual, reference = mujoco.MjData(model), mujoco.MjData(reference_model)
    mujoco.mj_setState(
        model, actual, initial_state, mujoco.mjtState.mjSTATE_FULLPHYSICS
    )
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.35, 0, 0.45]
    camera.distance, camera.azimuth, camera.elevation = 1.3, 130, -25
    with (
        mujoco.Renderer(model, height=480, width=640) as renderer,
        mujoco.Renderer(reference_model, height=480, width=640) as ref_renderer,
        imageio.get_writer(output / "comparison.mp4", fps=50) as video,
    ):
        for i, command in enumerate(controls):
            actual.ctrl[:] = command
            mujoco.mj_step(model, actual)
            if i % 2:
                continue
            reference.mocap_pos[0], reference.mocap_quat[0] = (
                ref["wrist_pos"][i],
                ref["wrist_quat"][i],
            )
            reference.qpos[:] = ref["finger_qpos"][i]
            mujoco.mj_forward(reference_model, reference)
            # Refresh derived geometry after mj_step without changing state.
            mujoco.mj_forward(model, actual)
            renderer.update_scene(actual, camera=camera)
            ref_renderer.update_scene(reference, camera=camera)
            frame = np.concatenate(
                [ref_renderer.render().copy(), renderer.render().copy()], axis=1
            )
            cv2.putText(
                frame,
                "REFERENCE: hand only",
                (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (20, 20, 20),
                2,
            )
            cv2.putText(
                frame,
                "OPTIMIZED: arm + hand",
                (660, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (20, 20, 20),
                2,
            )
            video.append_data(frame)
    print(f"Video: {output / 'comparison.mp4'}", flush=True)


if __name__ == "__main__":
    main()
