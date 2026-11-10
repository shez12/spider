"""Transfer another published hand's coke example to the original Yimu model."""

import argparse
import copy
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from examples.arm_hand.cross_hand import retarget
from examples.arm_hand.download_yimu_other import OUTPUT, RELATIVE, REVISION
from examples.arm_hand.test_active_distal import fingerprint, optimize
from examples.run_arm_hand import quaternion_error

ROOT = Path(__file__).resolve().parents[2]


def assemble(output, source_scene):
    """Reuse the arm, substitute the exact new object, and retain floor pairs."""
    tree = ET.parse(ROOT / "outputs/allegro_to_yimu_v2/arm_hand_scene.xml")
    root = tree.getroot()
    source = ET.parse(source_scene)
    world = root.find("worldbody")
    old_object = world.find("body[@name='right_object']")
    old_names = {node.get("name") for node in old_object.iter() if node.get("name")}
    old_meshes = {
        node.get("mesh") for node in old_object.findall("geom") if node.get("mesh")
    }
    old_palm = world.find(".//body[@name='right_palm']")
    old_names |= {node.get("name") for node in old_palm.iter() if node.get("name")}
    world.remove(old_object)
    obj = copy.deepcopy(source.find("worldbody/body[@name='right_object']"))
    obj.find("geom[@name='right_object_visual']").set("rgba", ".85 .15 .1 1")
    world.append(obj)
    asset = root.find("asset")
    for node in list(asset):
        if node.get("name") in old_meshes or node.get("name", "").startswith("yimu_"):
            asset.remove(node)
    wanted = {node.get("mesh") for node in obj.findall("geom") if node.get("mesh")}
    meshroot = (source_scene.parent / source.find("compiler").get("meshdir")).resolve()
    for mesh in source.findall("asset/mesh"):
        if mesh.get("name") in wanted:
            mesh = copy.deepcopy(mesh)
            mesh.set("file", str(meshroot / mesh.get("file")))
            asset.append(mesh)
    contact = root.find("contact")
    for node in list(contact):
        if any(v in old_names for v in node.attrib.values()):
            contact.remove(node)
    object_names = {node.get("name") for node in obj.findall("geom")}
    floor_pairs = []
    for node in source.findall("contact/pair"):
        names = [node.get("geom1"), node.get("geom2")]
        if any(n in object_names for n in names) and "floor" in names:
            contact.append(copy.deepcopy(node))
            floor_pairs.append(node)
    assert len(floor_pairs) == 9
    root.remove(root.find("equality"))
    root.remove(root.find("sensor"))
    sensor = ET.SubElement(root, "sensor")
    for objname in ["right_palm", "right_object"]:
        ET.SubElement(sensor, "framepos", objtype="xbody", objname=objname)
        ET.SubElement(sensor, "framequat", objtype="xbody", objname=objname)
    root.find("option").set("timestep", ".002")
    root.find("option").set("iterations", "40")
    ET.indent(root)
    tree.write(output / "arm_hand_scene.xml", encoding="unicode")


def main():
    """Retarget and sample optimize using existing methods without editing them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/xhand_coke_to_yimu"
    )
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--hold-seconds", type=float, default=0.0)
    args = parser.parse_args()
    if args.time_scale < 1 or args.hold_seconds < 0:
        parser.error("time-scale >= 1 and hold-seconds >= 0 required")
    before = fingerprint()
    prepared = args.output_dir / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    scene = OUTPUT / RELATIVE / "scene.xml"
    model = mujoco.MjModel.from_xml_path(str(scene))
    source_file = scene.parent / "0/trajectory_mjwp.npz"
    with np.load(source_file) as source:
        qpos = source["qpos"].reshape(-1, model.nq)
        qvel = source["qvel"].reshape(-1, model.nv)
        controls = source["ctrl"].reshape(-1, model.nu)
        times = source["time"].reshape(-1)
    np.testing.assert_allclose(np.diff(times), 0.01, atol=1e-8)
    original_frames = len(qpos)
    if args.time_scale != 1 or args.hold_seconds:
        grid = np.arange(len(qpos))
        stretched = np.linspace(
            0, len(qpos) - 1, round((len(qpos) - 1) * args.time_scale) + 1
        )
        new_qpos = np.column_stack([np.interp(stretched, grid, row) for row in qpos.T])
        new_qpos[:, -4:] = Slerp(
            grid, Rotation.from_quat(qpos[:, -4:][:, [1, 2, 3, 0]])
        )(stretched).as_quat()[:, [3, 0, 1, 2]]
        controls = np.column_stack(
            [np.interp(stretched, grid, row) for row in controls.T]
        )
        hold = round(args.hold_seconds / 0.01)
        qpos = np.vstack([new_qpos, np.repeat(new_qpos[-1:], hold, axis=0)])
        controls = np.vstack([controls, np.repeat(controls[-1:], hold, axis=0)])
        times = np.arange(len(qpos)) * 0.01
        qvel = np.zeros((len(qpos), model.nv))
        for i in range(1, len(qpos)):
            mujoco.mj_differentiatePos(model, qvel[i], 0.01, qpos[i - 1], qpos[i])
        qvel[0] = qvel[1]
    source_data = mujoco.MjData(model)
    positions, quats = [], []
    for q in qpos:
        source_data.qpos[:] = q
        mujoco.mj_forward(model, source_data)
        positions.append(source_data.site("right_palm").xpos.copy())
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, source_data.site("right_palm").xmat)
        quats.append(quat)
    ref = {
        "time": np.arange(1, len(qpos)) * 0.01,
        "wrist_pos": np.array(positions)[1:],
        "wrist_quat": np.array(quats)[1:],
        "finger_qpos": qpos[1:, 6:-7],
        "finger_ctrl": controls[1:, 6:],
        "object_qpos": qpos[1:, -7:],
        "initial_qpos": qpos[0],
        "initial_qvel": qvel[0],
        "initial_wrist_pos": positions[0],
        "initial_wrist_quat": quats[0],
    }
    np.savez(args.output_dir / "chosen_sample.npz", qpos=qpos, time=times)
    np.savez(args.output_dir / "source_reference.npz", **ref)
    assemble(prepared, scene)
    retarget_args = SimpleNamespace(
        output_dir=prepared, target_hand="yimu_v2", time_scale=args.time_scale
    )
    target, ref = retarget(retarget_args, None, model, ref, qpos, quaternion_error)
    mapping_path = prepared / "cross_hand_mapping.json"
    mapping = json.loads(mapping_path.read_text())
    mapping.update(
        source_hand="xhand",
        source_finger_dofs=12,
        extra_target_finger="Source has a pinky, but unchanged four-finger adapter does not fit it; temporal posture prior only",
    )
    mapping_path.write_text(json.dumps(mapping, indent=2) + "\n")
    data = mujoco.MjData(target)
    data.qpos[:] = np.r_[ref["target_joint_ik"][0, :7], ref["initial_qpos"][6:]]
    data.qvel[:] = np.r_[
        (ref["target_joint_ik"][1, :7] - ref["target_joint_ik"][0, :7]) / 0.01,
        ref["initial_qvel"][6:],
    ]
    data.ctrl[:] = ref["target_ctrl_nominal"][0]
    mujoco.mj_forward(target, data)
    initial = np.empty(mujoco.mj_stateSize(target, mujoco.mjtState.mjSTATE_FULLPHYSICS))
    mujoco.mj_getState(target, data, initial, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    np.savez(
        prepared / "trajectory.npz",
        initial_state=initial,
        ctrl=ref["target_ctrl_nominal"],
    )
    np.testing.assert_allclose(
        target.actuator_forcerange[7:], np.tile([-0.16, 0.16], (17, 1))
    )
    result = optimize(
        SimpleNamespace(
            trial=prepared,
            output_dir=args.output_dir,
            threads=8,
            samples=args.samples,
            iterations=args.iterations,
            seed=0,
        ),
        "yimu",
        False,
    )
    reference_lift = float(qpos[-1, -5] - qpos[0, -5])
    result.update(
        source_hand="xhand",
        task="fair_mon/right/coke/0",
        source_repo="retarget/retarget_example",
        source_revision=REVISION,
        source_trajectory_sha256=hashlib.sha256(source_file.read_bytes()).hexdigest(),
        source_scene=str(scene.resolve()),
        reference_lift_m=reference_lift,
        time_scale=args.time_scale,
        added_hold_seconds=args.hold_seconds,
        original_reference_frames=original_frames,
        reference_control_steps=len(qpos) - 1,
        object_mass_kg=float(model.body_mass[model.body("right_object").id]),
        original_yimu_torque_limit_nm=0.16,
        passive_couplings=4,
        method_files_unchanged=before == fingerprint(),
        method_sha256=before,
    )
    result["task_success"] = bool(
        result["optimized"]["object_lift_m"] > 0.1
        and result["optimized"]["final_object_error_m"] < 0.03
    )
    assert result["method_files_unchanged"]
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    from examples.arm_hand.validate_yimu_other import validate

    result = validate(args.output_dir)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
