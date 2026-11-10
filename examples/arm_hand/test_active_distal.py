"""Compare coupled Yimu fingers with a separately fully actuated model.

Only experimental XML and output files are written. The repository's sampling
optimizer is imported unchanged. This is an actuator/coupling ablation starting
from the same saved IK trajectory, not a new retargeting method.
"""

import argparse
import copy
import hashlib
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.rollout
import numpy as np
import torch

from examples.run_arm_hand import quaternion_error
from spider.config import Config
from spider.optimizers.sampling import make_optimize_once_fn

ROOT = Path(__file__).resolve().parents[2]
METHODS = [
    "spider/io.py",
    "spider/preprocess/ik_fast.py",
    "spider/simulators/mjwp.py",
    "spider/robot_model.py",
    "spider/optimizers/sampling.py",
    "examples/arm_hand/cross_hand.py",
    "examples/arm_hand/dataset_demo.py",
]


def fingerprint():
    """Record current method-file hashes, including pre-existing local edits."""
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in METHODS}


def create_variant(source, output, active):
    """Remove four couplings and append four matched position servos in a copy."""
    original = mujoco.MjModel.from_xml_path(str(source))
    tree = ET.parse(source)
    root = tree.getroot()
    mesh_root = source.parent / root.find("compiler").get("meshdir", ".")
    for mesh in root.findall("asset/mesh"):
        mesh.set("file", str((mesh_root / mesh.get("file")).resolve()))
    root.find("option").set("timestep", "0.002")
    root.find("option").set("iterations", "40")
    if active:
        equality = root.find("equality")
        assert len(equality) == 4
        for joint in list(equality):
            assert joint.tag == "joint" and "follows" in joint.get("name")
            equality.remove(joint)
        root.remove(equality)
        for finger in range(2, 6):
            name = f"right_{finger}-4"
            joint = root.find(f".//joint[@name='{name}']")
            template = copy.deepcopy(
                root.find(f"actuator/general[@joint='right_{finger}-3']")
            )
            template.set("name", name + "_act")
            template.set("joint", name)
            template.set("ctrlrange", joint.get("range"))
            root.find("actuator").append(template)
        addresses = [
            original.joint(f"right_{finger}-4").qposadr[0] for finger in range(2, 6)
        ]
        for key in root.findall("keyframe/key"):
            qpos = np.fromstring(key.get("qpos"), sep=" ")
            key.set("ctrl", key.get("ctrl") + " " + " ".join(map(str, qpos[addresses])))
    ET.indent(root)
    tree.write(output, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(output))
    assert (model.nq, model.nv) == (original.nq, original.nv)
    assert (model.nu, model.neq) == (
        (original.nu + 4, 0) if active else (original.nu, 4)
    )
    for field in [
        "body_mass",
        "body_inertia",
        "jnt_axis",
        "jnt_range",
        "dof_damping",
        "dof_armature",
        "dof_frictionloss",
        "geom_friction",
        "geom_contype",
        "geom_conaffinity",
    ]:
        np.testing.assert_array_equal(getattr(model, field), getattr(original, field))
    for field in [
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_forcerange",
        "actuator_ctrlrange",
    ]:
        np.testing.assert_array_equal(
            getattr(model, field)[: original.nu], getattr(original, field)
        )
    if active:
        np.testing.assert_allclose(
            model.actuator_forcerange[original.nu :], np.tile([-0.16, 0.16], (4, 1))
        )
    return model


def optimize(args, name, active):
    """Use the existing sampling update and prior arm-demo tracking objective."""
    folder = args.output_dir / name
    folder.mkdir(parents=True, exist_ok=True)
    model = create_variant(
        args.trial / "arm_hand_scene.xml", folder / "scene.xml", active
    )
    saved = np.load(args.trial / "trajectory.npz")
    ref = np.load(args.trial / "target_ik_reference.npz")
    initial_state = saved["initial_state"].copy()
    nominal = saved["ctrl"].copy()
    if active:
        addresses = model.jnt_qposadr[model.actuator_trnid[24:, 0]]
        nominal = np.column_stack([nominal, ref["target_joint_ik"][1:, addresses]])
    steps = len(nominal)
    knots = 10
    interpolation = np.stack(
        [
            np.interp(
                np.arange(steps), np.linspace(0, steps - 1, knots).astype(int), row
            )
            for row in np.eye(knots)
        ],
        axis=1,
    )
    checkpoints = np.arange(0, steps, 3)
    wrist_before = np.vstack([ref["initial_wrist_pos"], ref["wrist_pos"][:-1]])
    quat_before = np.vstack([ref["initial_wrist_quat"], ref["wrist_quat"][:-1]])
    workers = [mujoco.MjData(model) for _ in range(args.threads)]
    with mujoco.rollout.Rollout(nthread=args.threads) as engine:

        def evaluate(parameters, full=False):
            offsets = np.asarray(parameters).reshape(-1, knots, model.nu)
            commands = nominal + np.einsum("tk,bkj->btj", interpolation, offsets)
            commands = np.clip(commands, *model.actuator_ctrlrange.T)
            states, sensors = engine.rollout(
                model,
                workers,
                initial_state,
                np.repeat(commands, 5, axis=1),
                initial_warmstart=np.zeros(model.nv),
            )
            # Each command starts at t and ends at t+0.01. Sensor values are
            # pre-integration, so use its first substep and the last state.
            states = states[:, 4::5]
            sensors = sensors[:, ::5]
            wrist = sensors[:, checkpoints, :3] - wrist_before[checkpoints]
            obj = states[:, checkpoints, 29:32] - ref["object_qpos"][checkpoints, :3]
            rot = quaternion_error(
                sensors[:, checkpoints, 3:7],
                np.broadcast_to(
                    quat_before[checkpoints], sensors[:, checkpoints, 3:7].shape
                ),
            )
            objrot = quaternion_error(
                states[:, checkpoints, 32:36],
                np.broadcast_to(
                    ref["object_qpos"][checkpoints, 3:],
                    states[:, checkpoints, 32:36].shape,
                ),
            )
            finger = states[:, checkpoints, 8:29] - ref["finger_qpos"][checkpoints]
            residual = np.concatenate(
                [
                    (15 * wrist).reshape(len(offsets), -1),
                    (60 * obj).reshape(len(offsets), -1),
                    (0.8 * rot).reshape(len(offsets), -1),
                    (0.3 * objrot).reshape(len(offsets), -1),
                    (0.05 * finger).reshape(len(offsets), -1),
                    (0.03 * offsets).reshape(len(offsets), -1),
                    (0.02 * np.diff(offsets, axis=1)).reshape(len(offsets), -1),
                ],
                axis=1,
            )
            costs = 0.5 * np.sum(residual**2, axis=1)
            return (costs, states, sensors, commands) if full else costs

        zeros = np.zeros((knots, model.nu))
        baseline = evaluate(zeros, True)
        best = {"cost": float(baseline[0][0]), "parameters": zeros.copy()}
        torch.manual_seed(args.seed)
        config = Config(
            device="cpu",
            num_samples=args.samples,
            use_torch_compile=False,
            knot_steps=1,
            temperature=1.0,
            num_trace_uniform_samples=0,
            num_trace_topk_samples=0,
        )
        scale = np.r_[np.full(7, 0.035), np.full(model.nu - 7, 0.12)]
        config.noise_scale = torch.tensor(
            np.broadcast_to(scale, (args.samples, knots, model.nu)).copy()
        )
        config.noise_scale[0] = 0

        def sampled_rollout(config, env, candidates, ref_slice, env_param):
            del config, env, ref_slice, env_param
            candidates = candidates.clamp(-0.4, 0.4)
            costs = evaluate(candidates.numpy())
            i = int(np.argmin(costs))
            if costs[i] < best["cost"]:
                best.update(
                    cost=float(costs[i]), parameters=candidates[i].numpy().copy()
                )
            return (
                candidates,
                torch.from_numpy(-costs),
                torch.zeros(len(candidates), dtype=torch.bool),
                {},
            )

        update = make_optimize_once_fn(sampled_rollout)
        center = torch.zeros((knots, model.nu), dtype=torch.float64)
        start = time.perf_counter()
        for iteration in range(args.iterations):
            center, _, _ = update(
                config,
                None,
                center,
                (),
                sample_params={"global_noise_scale": 0.97**iteration},
            )
            assert torch.isfinite(center).all()
            print(
                f"{name}: {iteration + 1}/{args.iterations}, best cost {best['cost']:.3f}, elapsed {time.perf_counter() - start:.1f}s",
                flush=True,
            )
        result = evaluate(best["parameters"], True)

    def metrics(values):
        costs, states, sensors, _ = values
        obj = states[0, :, 29:32]
        return {
            "cost": float(costs[0]),
            "object_lift_m": float(obj[-1, 2] - ref["initial_qpos"][-5]),
            "final_object_error_m": float(
                np.linalg.norm(obj[-1] - ref["object_qpos"][-1, :3])
            ),
            "object_rmse_m": float(
                np.sqrt(np.mean(np.sum((obj - ref["object_qpos"][:, :3]) ** 2, axis=1)))
            ),
            "wrist_rmse_m": float(
                np.sqrt(
                    np.mean(np.sum((sensors[0, :, :3] - wrist_before) ** 2, axis=1))
                )
            ),
        }

    report = {
        "variant": name,
        "motors": model.nu - 7,
        "couplings": model.neq,
        "baseline": metrics(baseline),
        "optimized": metrics(result),
        "samples": args.samples,
        "iterations": args.iterations,
        "seed": args.seed,
        "elapsed_s": time.perf_counter() - start,
    }
    np.savez(
        folder / "trajectory.npz",
        initial_state=initial_state,
        ctrl=result[3][0],
        qpos=result[1][0, :, 1:36],
        qvel=result[1][0, :, 36:],
        control_dt=0.01,
        physics_dt=0.002,
    )
    # Independently verify the recorded commands through continuous substeps.
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, initial_state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    replay = []
    max_torque = 0.0
    for control in result[3][0]:
        data.ctrl[:] = control
        for _ in range(5):
            mujoco.mj_step(model, data)
            max_torque = max(max_torque, float(np.abs(data.actuator_force[7:]).max()))
        replay.append(data.qpos.copy())
    np.testing.assert_allclose(replay, result[1][0, :, 1:36], atol=1e-9, rtol=1e-9)
    report["max_finger_torque_nm"] = max_torque
    report["continuous_replay_verified"] = True
    (folder / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    """Build a separate variant and run matched-budget comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trial", type=Path, default=ROOT / "outputs/allegro_to_yimu_v2"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/yimu_active_distal"
    )
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.samples < 20:
        parser.error("Use at least 20 samples for the unchanged optimizer")
    before = fingerprint()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = [optimize(args, "coupled", False), optimize(args, "active_distal", True)]
    assert fingerprint() == before, "Method file changed during experiment"
    report = {
        "method_files_unchanged_during_experiment": True,
        "method_sha256": before,
        "comparison": reports,
        "limitations": "Single seed, fixed budget; both start from the same coupled IK and previously optimized 24 commands. Additional active joints start by tracking that IK. No new IK fit. Added four motors also add torque capacity. No hardware speed cap. Existing method edits from earlier integration are retained, not reverted.",
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report["comparison"], indent=2), flush=True)


if __name__ == "__main__":
    main()
