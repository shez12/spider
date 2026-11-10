"""Prepare SPIDER tracking adapters without altering the supplied Yimu models."""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from spider.robot_model import actuator_qpos_indices, project_joint_equalities

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "spider/assets/robots/yimu_v2"
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
WRIST_POS = np.array([-0.03156, -0.0154077, 0.0126013])
WRIST_QUAT = np.array([0.7071067811865476, 0, 0, 0.7071067811865476])


def build(side):
    """Prefix names and add virtual base controls and standard tracking sites."""
    source = ASSETS / "original" / f"yimu_v2_{side}" / f"yimu_v2_{side}.xml"
    tree = ET.parse(source)
    root = tree.getroot()
    names = {
        node.get("name"): f"{side}_{node.get('name')}"
        for node in root.iter()
        if node.get("name")
    }
    names["yimu_v2_palm"] = f"{side}_palm"
    reference_attributes = [
        "name",
        "joint",
        "joint1",
        "joint2",
        "body1",
        "body2",
        "mesh",
        "site",
    ]
    for node in root.iter():
        for attribute in reference_attributes:
            if node.get(attribute) in names:
                node.set(attribute, names[node.get(attribute)])
    for mesh in root.findall("asset/mesh"):
        mesh.set("file", str(Path("original") / f"yimu_v2_{side}" / mesh.get("file")))
        old_name = mesh.get("name")
        mesh.set("name", f"yimu_{old_name}")
        for geom in root.findall(".//geom"):
            if geom.get("mesh") == old_name:
                geom.set("mesh", f"yimu_{old_name}")
    root.find("compiler").set("meshdir", ".")
    option = ET.SubElement(
        root, "option", timestep="0.002", integrator="implicitfast", iterations="40"
    )
    del option
    palm = root.find("worldbody/body")
    palm.set("gravcomp", "1")
    for index, axis in enumerate(["1 0 0", "0 1 0", "0 0 1"] * 2):
        name = f"{side}_{'pos' if index < 3 else 'rot'}_{'xyz'[index % 3]}"
        joint = ET.Element(
            "joint",
            name=name,
            type="slide" if index < 3 else "hinge",
            axis=axis,
            range="-3 3" if index < 3 else "-6.28 6.28",
            damping="1",
            armature="0.01",
        )
        palm.insert(index, joint)
        actuator = ET.Element(
            "position",
            name=name,
            joint=name,
            kp="1000",
            kv="50",
            ctrlrange=joint.get("range"),
            forcerange="-50 50" if index < 3 else "-5 5",
        )
        root.find("actuator").insert(index, actuator)
    wrist = WRIST_POS.copy()
    if side == "left":
        wrist[1] *= -1
    ET.SubElement(
        palm,
        "site",
        name=f"{side}_palm",
        pos=" ".join(map(str, wrist)),
        quat=" ".join(map(str, WRIST_QUAT)),
        size="0.005",
        group="3",
    )
    ET.SubElement(
        palm,
        "camera",
        name=f"{side}_track",
        pos="0.45 -0.45 0.3",
        xyaxes="0.707 0.707 0 -0.3 0.3 0.9",
    )
    for finger_index, finger in enumerate(FINGERS, 1):
        terminal = 5 if finger_index == 1 else 4
        for link_index in range(1, terminal + 1):
            body = palm.find(f".//body[@name='{side}_{finger_index}-{link_index}']")
            for geom in body.findall("geom"):
                if geom.get("group") == "3":
                    geom.set(
                        "name",
                        f"collision_hand_{side}_{finger}_{terminal - link_index}",
                    )
        body = palm.find(f".//body[@name='{side}_{finger_index}-{terminal}']")
        native_tip = body.find(
            f"site[@name='{side}_{finger_index}-{terminal}_fingertip']"
        )
        for name in [
            f"{side}_{finger}_tip",
            f"track_hand_{side}_{finger}_tip",
            f"trace_hand_{side}_{finger}_tip",
        ]:
            ET.SubElement(
                body,
                "site",
                name=name,
                pos=native_tip.get("pos"),
                size="0.003",
                group="3",
            )
    for index, geom in enumerate(palm.findall("geom")):
        if geom.get("group") == "3":
            geom.set("name", f"collision_hand_{side}_palm_{index}")
    asset = root.find("asset")
    ET.SubElement(
        asset,
        "texture",
        name=f"{side}_groundplane",
        type="2d",
        builtin="checker",
        rgb1="0.95 0.95 0.95",
        rgb2="0.85 0.85 0.85",
        width="128",
        height="128",
    )
    ET.SubElement(
        asset,
        "material",
        name=f"{side}_groundplane",
        texture=f"{side}_groundplane",
        texrepeat="5 5",
    )
    for key in root.findall("keyframe/key"):
        key.set("qpos", "0 0 0 0 0 0 " + key.get("qpos"))
        key.set("ctrl", "0 0 0 0 0 0 " + key.get("ctrl"))
    ET.indent(root)
    path = ASSETS / f"{side}.xml"
    tree.write(path, encoding="unicode")
    return path


def audit(side, adapter):
    """Measure native stability, linkage error, mass and motor force limits."""
    native = ASSETS / "original" / f"yimu_v2_{side}" / f"yimu_v2_{side}.xml"
    model = mujoco.MjModel.from_xml_path(str(native))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    max_linkage_error, max_force = 0.0, 0.0
    for step in range(1500):
        controls = model.key_ctrl[0].copy()
        if step > 500:
            controls += 0.12 * np.sin((step - 500) * model.opt.timestep * 2)
        data.ctrl[:] = np.clip(
            controls, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
        )
        mujoco.mj_step(model, data)
        ideal = project_joint_equalities(model, data.qpos.copy())
        max_linkage_error = max(
            max_linkage_error, float(np.max(np.abs(ideal - data.qpos)))
        )
        max_force = max(max_force, float(np.max(np.abs(data.actuator_force))))
    adapted = mujoco.MjModel.from_xml_path(str(adapter))
    assert adapted.nq == 27 and adapted.nu == 23 and adapted.neq == 4
    np.testing.assert_allclose(
        adapted.actuator_forcerange[6:], [-0.16, 0.16] * np.ones((17, 2))
    )
    return {
        "side": side,
        "native_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
        "native_nq": model.nq,
        "motors": model.nu,
        "linkages": model.neq,
        "mass_kg": float(model.body_mass.sum()),
        "hold_and_sweep_finite": bool(np.isfinite(data.qpos).all()),
        "max_linkage_error_rad": max_linkage_error,
        "max_motor_torque_nm": max_force,
        "rated_speed_rad_s": 3.456,
        "speed_limit_enforced": False,
        "adapter_nq": adapted.nq,
        "adapter_nu": adapted.nu,
        "actuator_qpos_indices": actuator_qpos_indices(adapted).tolist(),
    }


def main():
    """Create both side adapters and save step-one/two integration evidence."""
    report = {side: audit(side, build(side)) for side in ["right", "left"]}
    right = ET.parse(ASSETS / "right.xml").getroot()
    left = ET.parse(ASSETS / "left.xml").getroot()
    right.remove(right.find("keyframe"))
    right.remove(right.find("size"))
    for section in ["asset", "worldbody", "contact", "equality", "actuator", "sensor"]:
        for node in list(left.find(section)):
            right.find(section).append(node)
    ET.indent(right)
    ET.ElementTree(right).write(ASSETS / "bimanual.xml", encoding="unicode")
    bimanual = mujoco.MjModel.from_xml_path(str(ASSETS / "bimanual.xml"))
    report["bimanual"] = {
        "nq": bimanual.nq,
        "nu": bimanual.nu,
        "linkages": bimanual.neq,
    }
    output = ROOT / "outputs/yimu_v2"
    output.mkdir(parents=True, exist_ok=True)
    (output / "model_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
