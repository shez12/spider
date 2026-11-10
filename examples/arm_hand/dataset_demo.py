"""Dataset-derived arm/Allegro grasp-and-lift via continuous physics shooting."""

import copy
import hashlib
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import mujoco.rollout
import numpy as np
from scipy.optimize import least_squares

TASK = "dexycb_01_right_002_master_chef_can_20200709_141754_00"
RELATIVE = Path("processed/dexycb_lifted/allegro/right") / TASK


def prepare(args, make_scene):
    """Preserve the dataset object, hand geometry and explicit contact pairs."""
    output, dataset = args.output_dir.resolve(), args.dataset_dir.resolve()
    source_scene = dataset / RELATIVE / "scene.xml"
    if not source_scene.exists():
        raise FileNotFoundError(
            "Download the dataset episode first: python examples/arm_hand/download_example.py"
        )
    source_model = mujoco.MjModel.from_xml_path(str(source_scene))
    source_data = mujoco.MjData(source_model)
    source_file = dataset / RELATIVE / "0/trajectory_mjwp.npz"
    with np.load(source_file, allow_pickle=False) as trajectory:
        all_qpos = trajectory["qpos"].reshape(-1, source_model.nq)
        all_qvel = trajectory["qvel"].reshape(-1, source_model.nv)
        all_ctrl = trajectory["ctrl"].reshape(-1, source_model.nu)
    if not 0 <= args.start_frame < len(all_qpos) - 10:
        raise ValueError("--start-frame must leave at least 10 dataset frames")
    start = args.start_frame
    original_start = start
    if args.time_scale != 1:
        from scipy.spatial.transform import Rotation, Slerp

        all_qpos, all_qvel, all_ctrl = (
            all_qpos[start:],
            all_qvel[start:],
            all_ctrl[start:],
        )
        original_grid = np.arange(len(all_qpos))
        new_grid = np.linspace(
            0, len(all_qpos) - 1, round((len(all_qpos) - 1) * args.time_scale) + 1
        )
        stretched = np.column_stack(
            [np.interp(new_grid, original_grid, column) for column in all_qpos.T]
        )
        stretched[:, -4:] = Slerp(
            original_grid, Rotation.from_quat(all_qpos[:, -4:][:, [1, 2, 3, 0]])
        )(new_grid).as_quat()[:, [3, 0, 1, 2]]
        all_ctrl = np.column_stack(
            [np.interp(new_grid, original_grid, column) for column in all_ctrl.T]
        )
        all_qpos = stretched
        all_qvel = np.zeros((len(stretched), source_model.nv))
        for index in range(1, len(stretched)):
            mujoco.mj_differentiatePos(
                source_model,
                all_qvel[index],
                0.01,
                stretched[index - 1],
                stretched[index],
            )
        all_qvel[0] = all_qvel[1]
        start = 0
    # Dataset frame start is the initial state. Subsequent frames are targets.
    qpos, _qvel, controls = (
        all_qpos[start + 1 :],
        all_qvel[start + 1 :],
        all_ctrl[start + 1 :],
    )
    palm_id = source_model.body("right_palm").id
    positions, orientations = [], []
    for q in all_qpos[start:]:
        source_data.qpos[:] = q
        mujoco.mj_forward(source_model, source_data)
        positions.append(source_data.xpos[palm_id].copy())
        orientations.append(source_data.xquat[palm_id].copy())
    ref = {
        "time": np.arange(1, len(qpos) + 1) * 0.01,
        "wrist_pos": np.array(positions[1:]),
        "wrist_quat": np.array(orientations[1:]),
        "finger_qpos": qpos[:, 6:22],
        "object_qpos": qpos[:, -7:],
        "finger_ctrl": controls[:, 6:22],
        "initial_qpos": all_qpos[start],
        "initial_qvel": all_qvel[start],
        "initial_wrist_pos": positions[0],
        "initial_wrist_quat": orientations[0],
    }
    np.savez(output / "hand_only_reference.npz", **ref)
    np.savez(output / "source_replay.npz", qpos=all_qpos[start:])
    tree = ET.parse(source_scene)
    root = tree.getroot()
    root.find("compiler").set(
        "meshdir", str((source_scene.parent / "../../../assets").resolve())
    )
    # Match the published scene's integrator/contact geometry. No virtual object
    # actuators, welds, mocap object or externally applied object forces are added.
    root.find("option").set("integrator", "implicitfast")
    world, actuators = root.find("worldbody"), root.find("actuator")
    for body in list(world.findall("body")):
        if body.get("mocap") == "true":
            world.remove(body)
    # Use a colored opaque can rather than an untextured white mesh.
    world.find("body[@name='right_object']/geom[@name='right_object_visual']").set(
        "rgba", "0.85 0.27 0.06 1"
    )
    for geom in root.findall(".//geom"):
        if (
            geom.get("name", "").startswith("right_object_")
            and geom.get("name") != "right_object_visual"
        ):
            geom.set("group", "3")
    palm = world.find("body[@name='right_palm']")
    for joint in list(palm.findall("joint")):
        palm.remove(joint)
    for actuator in list(actuators)[:6]:
        actuators.remove(actuator)
    ET.SubElement(palm, "site", name="demo_wrist", size="0.005", group="3")
    template_path = output / "arm_template.xml"
    make_scene(template_path, True)
    template = ET.parse(template_path).getroot()
    arm = template.find("worldbody/body[@name='arm_mount']")
    arm.set("pos", "-0.55 0.25 0.32")
    end = arm.find(".//body[@name='arm_link_6']")
    template_palm = end.find("body[@name='right_palm']")
    palm.set("pos", template_palm.get("pos"))
    palm.set("quat", template_palm.get("quat"))
    end.remove(template_palm)
    world.remove(palm)
    end.append(palm)
    # Arm must precede the object so qpos order is 7 arm + 16 fingers + free object.
    world.insert(0, arm)
    for index, actuator in enumerate(list(template.find("actuator"))[:7]):
        actuator = copy.deepcopy(actuator)
        actuator.set("kp", "1800")
        actuator.set("kv", "90")
        actuators.insert(index, actuator)
    for exclude in template.findall("contact/exclude"):
        root.find("contact").append(copy.deepcopy(exclude))
    sensors = ET.SubElement(root, "sensor")
    ET.SubElement(sensors, "framepos", objtype="xbody", objname="right_palm")
    ET.SubElement(sensors, "framequat", objtype="xbody", objname="right_palm")
    ET.SubElement(sensors, "framepos", objtype="xbody", objname="right_object")
    ET.SubElement(sensors, "framequat", objtype="xbody", objname="right_object")
    ET.indent(root)
    tree.write(output / "arm_hand_scene.xml", encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(output / "arm_hand_scene.xml"))
    assert model.nu == 23 and model.nq == 30 and model.nv == 29
    provenance = {
        "repo": "retarget/retarget_full",
        "task": TASK,
        "trajectory": "trajectory_mjwp.npz",
        "source_file": str(source_file),
        "source_sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
        "start_frame": original_start,
        "time_scale": args.time_scale,
        "reference_frames": len(qpos),
        "description": "Recorded, physics-retargeted floating-hand and object trajectory; no reference arm joints",
    }
    (output / "reference_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    return model, source_model, ref, all_qpos[start:]


def solve_ik(model, ref, quaternion_error):
    """Solve the dataset's palm-frame targets with the corrected flange transform."""
    data = mujoco.MjData(model)
    positions = np.vstack([ref["initial_wrist_pos"], ref["wrist_pos"]])
    quaternions = np.vstack([ref["initial_wrist_quat"], ref["wrist_quat"]])
    seed = np.array([-0.6, -0.5, 0.3, 1.6, -0.3, -0.6, -0.3])
    solutions = []
    for p, q in zip(positions, quaternions, strict=True):

        def residual(angles, target_p=p, target_q=q, posture=seed):
            data.qpos[:7] = angles
            data.qpos[7:23] = ref["initial_qpos"][6:22]
            data.qpos[-7:] = ref["initial_qpos"][-7:]
            mujoco.mj_forward(model, data)
            return np.r_[
                20 * (data.sensordata[:3] - target_p),
                2 * quaternion_error(data.sensordata[3:7], target_q),
                0.0001 * (angles - posture),
            ]

        result = least_squares(residual, seed, bounds=(-2.69, 2.69), max_nfev=100)
        seed = result.x
        solutions.append(seed.copy())
    return np.array(solutions)


def run(args, make_scene, quaternion_error):
    """Optimize joint commands against a real hand/object dataset reference."""
    output = args.output_dir.resolve()
    model, source_model, ref, source_qpos = prepare(args, make_scene)
    if args.target_hand != "allegro":
        from .cross_hand import retarget

        model, ref = retarget(
            args, model, source_model, ref, source_qpos, quaternion_error
        )
    arm = (
        ref["target_joint_ik"][:, :7]
        if "target_joint_ik" in ref
        else solve_ik(model, ref, quaternion_error)
    )
    robot_nq = model.nq - 7
    initial = mujoco.MjData(model)
    initial.qpos[:] = np.r_[arm[0], ref["initial_qpos"][6:]]
    initial.qvel[:] = np.r_[(arm[1] - arm[0]) / 0.01, ref["initial_qvel"][6:]]
    initial.ctrl[:] = np.r_[arm[0], ref["finger_ctrl"][0]]
    mujoco.mj_forward(model, initial)
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    initial_state = np.empty(nstate)
    mujoco.mj_getState(
        model, initial, initial_state, mujoco.mjtState.mjSTATE_FULLPHYSICS
    )
    nstep = len(ref["time"])
    if args.knots > nstep:
        raise ValueError("--knots cannot exceed the remaining number of dataset frames")
    indices = np.linspace(0, nstep - 1, args.knots).astype(int)
    grid = np.arange(nstep)
    interpolation = np.stack(
        [np.interp(grid, indices, np.eye(args.knots)[i]) for i in range(args.knots)],
        axis=1,
    )
    # Start from arm IK and the original dataset's finger commands. Optimized
    # knot offsets refine both sets of commands without replacing the reference.
    if "target_ctrl_nominal" in ref:
        nominal = ref["target_ctrl_nominal"].copy()
    else:
        nominal = np.column_stack([arm[1:], ref["finger_ctrl"]])
        nominal[:, :7] += np.gradient(arm[1:], 0.01, axis=0) * 0.05
    workers = [mujoco.MjData(model) for _ in range(args.threads)]
    count, start = 0, time.perf_counter()
    checkpoints = np.arange(0, nstep, 3)
    wrist_before_step = np.vstack([ref["initial_wrist_pos"], ref["wrist_pos"][:-1]])
    quat_before_step = np.vstack([ref["initial_wrist_quat"], ref["wrist_quat"][:-1]])
    with mujoco.rollout.Rollout(nthread=args.threads) as engine:

        def evaluate(parameters, full=False):
            parameters = np.atleast_2d(parameters)
            offsets = parameters.reshape(-1, args.knots, model.nu)
            commands = nominal + np.einsum("tk,bkj->btj", interpolation, offsets)
            commands = np.clip(
                commands, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
            )
            states, sensor = engine.rollout(
                model,
                workers,
                initial_state,
                commands,
                initial_warmstart=np.zeros(model.nv),
            )
            # mj_step frame sensors precede integration; returned states follow
            # integration. Align each residual to its actual sample time.
            wrist = sensor[:, checkpoints, :3] - wrist_before_step[checkpoints]
            obj = (
                states[:, checkpoints, 1 + robot_nq : 4 + robot_nq]
                - ref["object_qpos"][checkpoints, :3]
            )
            rot = quaternion_error(
                sensor[:, checkpoints, 3:7],
                np.broadcast_to(
                    quat_before_step[checkpoints], sensor[:, checkpoints, 3:7].shape
                ),
            )
            objrot = quaternion_error(
                states[:, checkpoints, 4 + robot_nq : 8 + robot_nq],
                np.broadcast_to(
                    ref["object_qpos"][checkpoints, 3:],
                    states[:, checkpoints, 4 + robot_nq : 8 + robot_nq].shape,
                ),
            )
            finger = (
                states[:, checkpoints, 8 : 1 + robot_nq]
                - ref["finger_qpos"][checkpoints]
            )
            residual = np.concatenate(
                [
                    (15 * wrist).reshape(len(parameters), -1),
                    (60 * obj).reshape(len(parameters), -1),
                    (0.8 * rot).reshape(len(parameters), -1),
                    (0.3 * objrot).reshape(len(parameters), -1),
                    (0.05 * finger).reshape(len(parameters), -1),
                    (0.03 * offsets).reshape(len(parameters), -1),
                    (0.02 * np.diff(offsets, axis=1)).reshape(len(parameters), -1),
                ],
                axis=1,
            )
            if full:
                return residual[0], states[0], sensor[0], commands[0]
            return residual

        def fun(x):
            nonlocal count
            count += 1
            residual = evaluate(x)[0]
            print(
                f"Evaluation {count}: cost={0.5 * residual @ residual:.5f}, elapsed={time.perf_counter() - start:.1f}s",
                flush=True,
            )
            return residual

        def jac(x):
            epsilon = 0.001
            batch = np.tile(x, (len(x) + 1, 1))
            batch[1:] += epsilon * np.eye(len(x))
            result = evaluate(batch)
            return ((result[1:] - result[0]) / epsilon).T

        zeros = np.zeros(args.knots * model.nu)
        baseline = evaluate(zeros, True)
        if args.optimizer == "sampling":
            from types import SimpleNamespace

            import torch

            from spider.config import Config
            from spider.optimizers.sampling import make_optimize_once_fn

            if args.num_samples < 20:
                raise ValueError(
                    "Use at least 20 candidates; the upstream weight normalizer requires multiple elite rewards"
                )
            torch.manual_seed(0)
            sampling_config = Config(
                device="cpu",
                num_samples=args.num_samples,
                use_torch_compile=False,
                knot_steps=1,
                temperature=1.0,
                num_trace_uniform_samples=0,
                num_trace_topk_samples=0,
            )
            scale = np.r_[np.full(7, 0.035), np.full(model.nu - 7, 0.12)]
            sampling_config.noise_scale = torch.tensor(
                np.broadcast_to(scale, (args.num_samples, args.knots, model.nu)).copy()
            )
            sampling_config.noise_scale[0] = 0
            best = {
                "cost": float(0.5 * baseline[0] @ baseline[0]),
                "parameters": zeros.copy(),
            }

            def sampled_rollout(config, env, candidates, ref_slice, env_param):
                del config, env, ref_slice, env_param
                candidates = candidates.clamp(-0.4, 0.4)
                parameters = candidates.numpy().reshape(len(candidates), -1)
                residual = evaluate(parameters)
                costs = 0.5 * np.sum(residual**2, axis=1)
                index = int(np.argmin(costs))
                if costs[index] < best["cost"]:
                    best.update(
                        cost=float(costs[index]), parameters=parameters[index].copy()
                    )
                return (
                    candidates,
                    torch.from_numpy(-costs),
                    torch.zeros(len(candidates), dtype=torch.bool),
                    {},
                )

            optimize_once = make_optimize_once_fn(sampled_rollout)
            center = torch.zeros((args.knots, model.nu), dtype=torch.float64)
            for iteration in range(args.iterations):
                center, _, _ = optimize_once(
                    sampling_config,
                    None,
                    center,
                    (),
                    sample_params={"global_noise_scale": 0.97**iteration},
                )
                if not torch.isfinite(center).all():
                    raise RuntimeError(
                        "Upstream sampling update returned non-finite controls"
                    )
                print(
                    f"Sampling {iteration + 1}/{args.iterations}: best cost={best['cost']:.5f}",
                    flush=True,
                )
            result = SimpleNamespace(
                x=best["parameters"],
                success=False,
                message="Fixed sampling iteration budget completed; best physically evaluated candidate retained",
            )
        else:
            result = least_squares(
                fun,
                zeros,
                jac=jac,
                bounds=(-0.4, 0.4),
                max_nfev=args.iterations,
                xtol=1e-5,
                ftol=1e-5,
                gtol=1e-5,
            )
        optimized = evaluate(result.x, True)

    def metrics(value):
        residual, executed, _, _ = value
        metric_data = mujoco.MjData(model)
        palm_positions = []
        for state in executed:
            metric_data.qpos[:] = state[1 : 1 + model.nq]
            mujoco.mj_kinematics(model, metric_data)
            palm_positions.append(metric_data.site("right_palm").xpos.copy())
        objects = executed[:, 1 + robot_nq : 4 + robot_nq]
        return {
            "cost": float(0.5 * residual @ residual),
            "wrist_rmse_m": float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            (np.array(palm_positions) - ref["wrist_pos"]) ** 2, axis=1
                        )
                    )
                )
            ),
            "object_rmse_m": float(
                np.sqrt(
                    np.mean(np.sum((objects - ref["object_qpos"][:, :3]) ** 2, axis=1))
                )
            ),
            "object_lift_m": float(objects[-1, 2] - ref["initial_qpos"][-5]),
            "final_object_error_m": float(
                np.linalg.norm(objects[-1] - ref["object_qpos"][-1, :3])
            ),
        }

    residual, states, sensor, commands = optimized
    if not np.isfinite(states).all() or not np.all(np.diff(states[:, 0]) > 0):
        raise RuntimeError("Physics rollout diverged")
    report = {
        "task": TASK,
        "source_hand": "allegro",
        "target_hand": args.target_hand,
        "target_finger_dofs": model.nu - 7,
        "target_finger_joints": robot_nq - 7,
        "optimizer": args.optimizer,
        "num_samples": args.num_samples if args.optimizer == "sampling" else None,
        "time_scale": args.time_scale,
        "reference": "Published DexYCB floating Allegro hand + coffee can; no arm joints",
        "mounting": {
            "xhand": "Canonical palm-site frame aligned with forearm; adapter at x=0.125 m",
            "yimu_v2": "Supplied Yimu palm frame, canonical site and rear mesh mounting face; original finger linkage preserved",
            "allegro": "Allegro rear mounting face at local z=-0.095 m",
        }[args.target_hand],
        "baseline": metrics(baseline),
        "optimized": metrics(optimized),
        "optimizer_converged": bool(result.success),
        "optimizer_message": result.message,
        "reference_lift_m": float(ref["object_qpos"][-1, 2] - ref["initial_qpos"][-5]),
        "continuous_control_replay": True,
        "free_dynamic_object": True,
        "object_actuators": 0,
        "limitations": "Generic gravity-compensated arm; reference is an already-retargeted dataset rollout, not raw human motion",
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez(
        output / "trajectory.npz",
        initial_state=initial_state,
        ctrl=commands,
        qpos=states[:, 1 : 1 + model.nq],
        qvel=states[:, 1 + model.nq :],
        time=states[:, 0],
    )
    print(json.dumps(report, indent=2), flush=True)
    if not args.no_video:
        render(model, source_model, ref, source_qpos, initial_state, commands, output)
    if args.view:
        from mujoco import viewer as mj_viewer

        data = mujoco.MjData(model)
        with mj_viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = [-0.12, 0, 0.25]
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = (
                1.2,
                140,
                -20,
            )
            while viewer.is_running():
                mujoco.mj_resetData(model, data)
                mujoco.mj_setState(
                    model, data, initial_state, mujoco.mjtState.mjSTATE_FULLPHYSICS
                )
                for command in commands:
                    if not viewer.is_running():
                        break
                    tick = time.perf_counter()
                    data.ctrl[:] = command
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    time.sleep(max(0, 0.01 - (time.perf_counter() - tick)))


def render(model, source_model, ref, source_qpos, initial_state, commands, output):
    """Render dataset reference beside actual dynamic arm/object control replay."""
    import cv2

    actual, reference = mujoco.MjData(model), mujoco.MjData(source_model)
    display = mujoco.MjData(model)
    mujoco.mj_setState(
        model, actual, initial_state, mujoco.mjtState.mjSTATE_FULLPHYSICS
    )
    source_model.geom_rgba[source_model.geom("right_object_visual").id] = [
        0.85,
        0.27,
        0.06,
        1,
    ]
    camera = mujoco.MjvCamera()
    target_name = {19: "Xhand", 23: "Allegro", 24: "Yimu v2"}.get(model.nu, "hand")
    camera.lookat[:] = [-0.13, 0.02, 0.24]
    camera.distance, camera.azimuth, camera.elevation = 1.15, 130, -20
    with (
        mujoco.Renderer(model, 480, 640) as renderer,
        mujoco.Renderer(source_model, 480, 640) as ref_renderer,
        imageio.get_writer(output / "comparison.mp4", fps=50) as video,
        imageio.get_writer(output / "chosen_sample.mp4", fps=50) as source_video,
        imageio.get_writer(output / "optimized_result.mp4", fps=50) as result_video,
    ):
        for i, command in enumerate(commands):
            actual.ctrl[:] = command
            mujoco.mj_step(model, actual)
            if i % 2:
                continue
            reference.qpos[:] = source_qpos[i + 1]
            mujoco.mj_forward(source_model, reference)
            # Rendering must not alter the solver's simulation/warm-start state.
            display.qpos[:], display.qvel[:], display.ctrl[:] = (
                actual.qpos,
                actual.qvel,
                actual.ctrl,
            )
            mujoco.mj_forward(model, display)
            renderer.update_scene(display, camera=camera)
            ref_renderer.update_scene(reference, camera=camera)
            panels = [ref_renderer.render().copy(), renderer.render().copy()]
            for panel, label in [
                (panels[0], "DATASET SAMPLE: Allegro hand"),
                (panels[1], f"RESULT: arm + {target_name}"),
            ]:
                cv2.rectangle(panel, (0, 0), (640, 65), (255, 255, 255), -1)
                cv2.putText(
                    panel,
                    label,
                    (18, 27),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (20, 20, 20),
                    2,
                )
                cv2.putText(
                    panel,
                    f"DexYCB coffee-can lift | t = {actual.time:.2f} s",
                    (18, 51),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46,
                    (70, 70, 70),
                    1,
                )
            source_video.append_data(panels[0])
            result_video.append_data(panels[1])
            video.append_data(np.concatenate(panels, axis=1))
    with np.load(output / "trajectory.npz") as saved:
        np.testing.assert_allclose(
            actual.qpos, saved["qpos"][-1], atol=1e-10, rtol=1e-10
        )
    print(f"Video: {output / 'comparison.mp4'}", flush=True)
