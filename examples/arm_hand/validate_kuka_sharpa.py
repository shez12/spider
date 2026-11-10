"""Validate the supplied KUKA iiwa14 + left Sharpa URDF on the frozen demo."""

import argparse
import copy
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from examples.arm_hand.run_sharpa_complex import ROOT, fit, validate
from examples.arm_hand.try_sharpa import optimize


DEFAULT_DESCRIPTION = Path(
    "/home/szc/work/play2perfect/assets/urdf/kuka_sharpa_description"
)
URDF_NAME = "iiwa14_left_sharpa_adjusted_restricted.urdf"
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
PLAY2PERFECT_ARM_POSE = np.array([-1.571, 1.571, 0.0, 1.376, 0.0, 1.485, 1.308])
PLAY2PERFECT_ARM_KP = [600.0, 600.0, 500.0, 400.0, 200.0, 200.0, 200.0]
PLAY2PERFECT_ARM_KV = [
    27.027026473513512,
    27.027026473513512,
    24.672186769721083,
    22.067474708266914,
    9.752538131173853,
    9.147747263670984,
    9.147747263670984,
]
JOINT_ALIASES = {
    "left_thumb_CMC_FE": "left_1_thumb_CMC_FE",
    "left_index_MCP_FE": "left_2_index_MCP_FE",
    "left_middle_MCP_FE": "left_3_middle_MCP_FE",
    "left_ring_MCP_FE": "left_4_ring_MCP_FE",
    "left_pinky_CMC": "left_5_pinky_CMC",
}


def _canonical_urdf(urdf: Path, destination: Path) -> ET.Element:
    """Use MuJoCo's URDF importer so fixed links are merged exactly as simulated."""
    model = mujoco.MjModel.from_xml_path(str(urdf))
    mujoco.mj_saveLastXML(str(destination), model)
    return ET.parse(destination).getroot()


def _decorate_robot(robot_root: ET.Element, old_root: ET.Element):
    """Add collision names, native joint dynamics, sites, and servo definitions."""
    old_hand = ET.parse(ROOT / "spider/assets/robots/sharpa_ha4/left.xml").getroot()
    old_joints = {
        joint.get("name"): joint
        for joint in old_hand.findall(".//joint")
        if joint.get("name")
    }
    reverse_alias = {new: old for old, new in JOINT_ALIASES.items()}
    for joint in robot_root.findall(".//joint"):
        old_name = reverse_alias.get(joint.get("name"), joint.get("name"))
        native = old_joints.get(old_name)
        if native is not None and native.get("class"):
            joint.set("class", native.get("class"))

    count = {"arm": 0, "hand": 0}

    def name_geoms(body: ET.Element, hand=False):
        hand = hand or body.get("name", "").startswith("left_")
        for geom in body.findall("geom"):
            mesh = geom.get("mesh", "").removeprefix("kuka_sharpa_")
            is_hand = hand or mesh.startswith("left_") or mesh in {
                "MCP_VL",
                "elastomer",
                "thumb_elastomer",
            }
            kind = "hand" if is_hand else "arm"
            prefix = "collision_sharpa" if is_hand else "collision_kuka"
            geom.set("name", f"{prefix}_{count[kind]:02d}")
            count[kind] += 1
            if not is_hand:
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
        for child in body.findall("body"):
            name_geoms(child, hand)

    base = robot_root.find("worldbody/body[@name='kuka_base']")
    name_geoms(base)

    link7 = base.find(".//body[@name='iiwa14_link_7']")
    ET.SubElement(
        link7,
        "site",
        name="sharpa_palm",
        pos="0 -0.02 0.16",
        # The frozen dataset wrist quaternion uses the Sharpa palm frame.
        # Play2Perfect observes link_7 rotation, so retain its palm position
        # offset while applying the fixed link_7 -> Sharpa mounting rotation.
        quat="0.793352 0 0 -0.608763",
        size=".004",
        group="3",
    )
    tip_positions = {
        "thumb": [0.0174661904, 0.0063871292, 0.0006762835],
        "index": [0.0177802734, 0.0057631074, 0.0],
        "middle": [0.0177802734, 0.0057631074, 0.0],
        "ring": [0.0177802682, 0.0057631431, 0.0],
        "pinky": [0.0177802734, 0.0057631074, 0.0],
    }
    for finger, position in tip_positions.items():
        body = base.find(f".//body[@name='left_{finger}_DP']")
        ET.SubElement(
            body,
            "site",
            name=f"sharpa_{finger}_tip",
            pos=" ".join(map(str, position)),
            size=".003",
            group="3",
        )

    actuator = old_root.find("actuator")
    actuator.clear()
    arm_limits = [300.0] * 7
    for index, (force, kp, kv) in enumerate(
        zip(arm_limits, PLAY2PERFECT_ARM_KP, PLAY2PERFECT_ARM_KV, strict=True), 1
    ):
        ET.SubElement(
            actuator,
            "position",
            name=f"iiwa14_target_{index}",
            joint=f"iiwa14_joint_{index}",
            kp=f"{kp:g}",
            kv=f"{kv:.12g}",
            inheritrange="1",
            forcelimited="true",
            forcerange=f"{-force:g} {force:g}",
        )
    for native in old_hand.find("actuator"):
        servo = copy.deepcopy(native)
        servo.set("joint", JOINT_ALIASES.get(servo.get("joint"), servo.get("joint")))
        actuator.append(servo)
    return count


def build(description: Path, output: Path, reference: dict, base_position=None, source_scene=None):
    """Replace the generic robot while preserving the frozen task and object."""
    source_scene = source_scene or ROOT / "outputs/mano_to_sharpa/scene.xml"
    tree = ET.parse(source_scene)
    root = tree.getroot()
    root.set("model", "kuka_iiwa14_left_sharpa_validation")
    source_text = ET.tostring(root, encoding="unicode")
    dataset = ROOT / "example_datasets/retarget_full"
    snapshot = ROOT / "reproduction/snapshot-retarget_full"
    if not dataset.exists() and snapshot.exists():
        root = ET.fromstring(source_text.replace(str(dataset), str(snapshot)))
        tree = ET.ElementTree(root)

    converted_path = output / "kuka_sharpa_import.xml"
    imported = _canonical_urdf(description / URDF_NAME, converted_path)
    imported_assets = imported.find("asset")
    for mesh in imported_assets.findall("mesh"):
        mesh.set("name", "kuka_sharpa_" + mesh.get("name"))
        mesh.set("file", str((description / mesh.get("file")).resolve()))
    for geom in imported.findall(".//geom[@mesh]"):
        geom.set("mesh", "kuka_sharpa_" + geom.get("mesh"))
    root.find("asset").extend(copy.deepcopy(list(imported_assets)))

    imported_world = imported.find("worldbody")
    base = ET.Element("body", name="kuka_base", pos="0 0 0")
    base.extend(copy.deepcopy(list(imported_world)))
    robot_root = ET.Element("mujoco")
    robot_world = ET.SubElement(robot_root, "worldbody")
    robot_world.append(base)
    counts = _decorate_robot(robot_root, root)

    world = root.find("worldbody")
    floor = world.find("geom[@name='floor']")
    floor.set("type", "box")
    floor.set("size", "0.2375 0.2 0.15")
    floor.set("pos", "0 0 -0.15")
    old_arm = world.find("body[@name='arm_mount']")
    insert_at = list(world).index(old_arm)
    world.remove(old_arm)
    world.insert(insert_at, base)

    contact = root.find("contact")
    for pair in list(contact):
        if pair.tag == "exclude" or "collision_sharpa_" in str(pair.attrib):
            contact.remove(pair)

    if base_position is None:
        # Backward-compatible placement: align the established seed with the
        # first wrist pose. Explicit placement is preferable for fixture tests.
        temporary = output / "placement.xml"
        probe_root = copy.deepcopy(root)
        probe_root.find("worldbody/body[@name='right_object']").set("pos", "0 0 2")
        ET.indent(probe_root)
        ET.ElementTree(probe_root).write(temporary, encoding="unicode")
        probe_model = mujoco.MjModel.from_xml_path(str(temporary))
        probe_data = mujoco.MjData(probe_model)
        seed = np.r_[[0.5, 0.2, -0.4, 2.0, 0.0, -2.0, 0.3], np.zeros(22)]
        probe_data.qpos[:29] = np.clip(
            seed,
            probe_model.jnt_range[:29, 0] + 1e-4,
            probe_model.jnt_range[:29, 1] - 1e-4,
        )
        mujoco.mj_forward(probe_model, probe_data)
        offset = reference["wrist_pos"][0] - probe_data.site("sharpa_palm").xpos
    else:
        offset = np.asarray(base_position, dtype=float)
    base.set("pos", " ".join(map(str, offset)))

    object_geoms = [
        geom.get("name")
        for geom in world.find("body[@name='right_object']").findall("geom")
        if geom.get("name") != "right_object_visual"
    ]
    for geom in base.findall(".//geom"):
        # Explicit pairs remain active even when the imported arm uses zero
        # masks to suppress adjacent-link self contacts.
        ET.SubElement(contact, "pair", geom1=geom.get("name"), geom2="floor",
                      condim="4", friction="1 1 0.1 0 0")
        if geom.get("name", "").startswith("collision_sharpa_"):
            for object_geom in object_geoms:
                ET.SubElement(
                    contact,
                    "pair",
                    geom1=geom.get("name"),
                    geom2=object_geom,
                    condim="4",
                    friction="1 1 0.1 0 0",
                )

    sensor = root.find("sensor")
    sensor.clear()
    for tag, attrs in [
        ("framepos", {"objtype": "site", "objname": "sharpa_palm"}),
        ("framequat", {"objtype": "site", "objname": "sharpa_palm"}),
        ("framepos", {"objtype": "xbody", "objname": "right_object"}),
        ("framequat", {"objtype": "xbody", "objname": "right_object"}),
    ]:
        ET.SubElement(sensor, tag, **attrs)
    for finger in FINGERS[:4]:
        ET.SubElement(
            sensor, "framepos", objtype="site", objname=f"sharpa_{finger}_tip"
        )

    ET.indent(root)
    scene = output / "scene.xml"
    tree.write(scene, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(scene))
    assert (model.nq, model.nv, model.nu) == (36, 35, 29)
    joint_names = [model.joint(i).name for i in range(29)]
    assert joint_names[:7] == [f"iiwa14_joint_{i}" for i in range(1, 8)]
    return model, offset, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--description", type=Path, default=DEFAULT_DESCRIPTION)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/kuka_sharpa_validation"
    )
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--ik-stride", type=int, default=2)
    parser.add_argument(
        "--base-position",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Set the fixed KUKA base position in world coordinates (metres).",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    urdf = args.description / URDF_NAME
    if not urdf.is_file():
        parser.error(f"missing {urdf}")

    with np.load(ROOT / "outputs/mano_to_sharpa/source_reference.npz") as archive:
        reference = {key: archive[key] for key in archive.files}
    reference_sha256 = hashlib.sha256(
        (ROOT / "outputs/mano_to_sharpa/source_reference.npz").read_bytes()
    ).hexdigest()
    frozen_reference = {key: value.copy() for key, value in reference.items()}
    model, base_position, counts = build(
        args.description, args.output_dir, reference, args.base_position
    )
    reference = fit(
        model,
        reference,
        args.output_dir,
        args.ik_stride,
        initial_arm_pose=PLAY2PERFECT_ARM_POSE,
    )
    for key, expected in frozen_reference.items():
        np.testing.assert_array_equal(reference[key], expected)
    optimize(args, model, reference)
    old = json.loads((ROOT / "outputs/mano_to_sharpa/report.json").read_text())
    report = validate(args.output_dir, model, old["case"], extra_hold=False)
    report.update(
        robot="KUKA iiwa14 + left Sharpa HA4",
        supplied_urdf=str(urdf.resolve()),
        supplied_urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest(),
        base_position_m=base_position.tolist(),
        imported_collision_meshes=counts,
        frozen_human_and_object_reference=True,
        source_reference_sha256=reference_sha256,
        dataset_trajectory_arrays_unchanged=True,
        table={
            "size_m": [0.475, 0.4, 0.3],
            "center_z_m": -0.15,
            "top_z_m": 0.0,
        },
        play2perfect_arm_default_joint_pos_rad=PLAY2PERFECT_ARM_POSE.tolist(),
        play2perfect_arm_stiffness=PLAY2PERFECT_ARM_KP,
        play2perfect_arm_damping=PLAY2PERFECT_ARM_KV,
    )
    report["limitations"] = (
        "Simulation validation of the supplied URDF; the base pose is calibrated "
        "to the frozen task and KUKA link collisions are disabled because the "
        "fixture/table geometry is not part of the supplied description."
    )
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
