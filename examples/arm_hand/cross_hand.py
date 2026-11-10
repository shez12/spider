"""Transfer source Allegro task-space landmarks to arm-mounted Xhand joints.

Only the source Allegro dataset trajectory is read. No target-hand trajectory,
target reference controls or pretrained policy is loaded.
"""

import copy
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from spider.robot_model import actuator_qpos_indices, project_joint_equalities

FINGERS = ["thumb", "index", "middle", "ring"]
ROOT = Path(__file__).resolve().parents[2]


def retarget(
    args, unused_model, source_model, reference, source_qpos, quaternion_error
):
    """Build another hand, retarget landmarks by IK, and initialize controls."""
    del unused_model
    output = args.output_dir.resolve()
    is_yimu = args.target_hand == "yimu_v2"
    native_path = ROOT / "spider/assets/robots" / args.target_hand / "right.xml"
    native = ET.parse(native_path).getroot()
    tree = ET.parse(output / "arm_hand_scene.xml")
    root = tree.getroot()
    world, actuators = root.find("worldbody"), root.find("actuator")
    end = world.find(".//body[@name='arm_link_6']")
    end.remove(end.find("body[@name='right_palm']"))
    palm = copy.deepcopy(
        native.find(
            ".//body[@name='right_palm']"
            if is_yimu
            else ".//body[@name='right_hand_link']"
        )
    )
    for joint in list(palm.findall("joint")):
        palm.remove(joint)
    palm.set("name", "right_palm")
    if not is_yimu:
        palm.set("childclass", "target_xhand")
    palm.set("gravcomp", "1")
    # right_palm is the repository's canonical cross-embodiment wrist frame.
    # Xhand body coordinates differ from Allegro; preserve the site's transform.
    site_quat = np.fromstring(
        palm.find("site[@name='right_palm']").get("quat"), sep=" "
    )
    frame = Rotation.from_quat(site_quat[[1, 2, 3, 0]])
    mount_rotation = Rotation.from_euler("y", np.pi / 2) * frame.inv()
    palm.set("quat", " ".join(map(str, mount_rotation.as_quat()[[3, 0, 1, 2]])))
    site_position = np.fromstring(
        palm.find("site[@name='right_palm']").get("pos", "0 0 0"), sep=" "
    )
    body_position = (
        np.array([0.22, 0, 0]) - mount_rotation.apply(site_position)
        if is_yimu
        else np.array([0.125, 0, 0])
    )
    palm.set("pos", " ".join(map(str, body_position)))
    end.append(palm)
    if not is_yimu:
        target_default = ET.SubElement(
            root.find("default"), "default", {"class": "target_xhand"}
        )
        for element in native.find("default"):
            target_default.append(copy.deepcopy(element))
        position = target_default.find("position")
        position.attrib.pop("dampratio", None)
        position.set("kp", "1000")
        position.set("kv", "60")
    for mesh in native.findall("asset/mesh"):
        mesh = copy.deepcopy(mesh)
        mesh.set(
            "file",
            str(
                (
                    native_path.parent
                    / ("" if is_yimu else "assets")
                    / mesh.get("file")
                ).resolve()
            ),
        )
        root.find("asset").append(mesh)
    for actuator in list(actuators)[7:]:
        actuators.remove(actuator)
    for actuator in list(native.find("actuator"))[6:]:
        actuator = copy.deepcopy(actuator)
        if not is_yimu:
            actuator.set("class", "target_xhand")
        actuators.append(actuator)
    contact = root.find("contact")
    if is_yimu:
        for element in native.findall("contact/exclude"):
            contact.append(copy.deepcopy(element))
        equality = ET.SubElement(root, "equality")
        for element in native.find("equality"):
            equality.append(copy.deepcopy(element))
    for pair in list(contact.findall("pair")):
        if "collision_hand_" in pair.get("geom1", "") or "collision_hand_" in pair.get(
            "geom2", ""
        ):
            contact.remove(pair)
    object_geoms = [
        geom.get("name")
        for geom in world.findall("body[@name='right_object']/geom")
        if geom.get("name") != "right_object_visual"
    ]
    for geom in palm.findall(".//geom"):
        if geom.get("name", "").startswith("collision_hand_"):
            for obj in object_geoms:
                ET.SubElement(
                    contact,
                    "pair",
                    geom1=geom.get("name"),
                    geom2=obj,
                    condim="4",
                    friction="1 1 0.1 0 0",
                )
    sensors = root.find("sensor")
    sensors[0].set("objtype", "site")
    sensors[0].set("objname", "right_palm")
    sensors[1].set("objtype", "site")
    sensors[1].set("objname", "right_palm")
    for finger in FINGERS:
        ET.SubElement(
            sensors,
            "framepos",
            objtype="site",
            objname=f"track_hand_right_{finger}_tip",
        )
    if is_yimu:
        for actuator in list(actuators)[7:]:
            ET.SubElement(
                sensors,
                "actuatorfrc",
                name=f"yimu_motor_force_{actuator.get('name')}",
                actuator=actuator.get("name"),
            )
    ET.indent(root)
    tree.write(output / "arm_hand_scene.xml", encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(output / "arm_hand_scene.xml"))
    assert (model.nu, model.nq) == ((24, 35) if is_yimu else (19, 26))
    control_qpos = actuator_qpos_indices(model)
    robot_nq = model.nq - 7
    source_data, target_data = mujoco.MjData(source_model), mujoco.MjData(model)
    tip_ids = [source_model.site(f"track_hand_right_{f}_tip").id for f in FINGERS]
    wrist_id = source_model.site("right_palm").id
    positions, quaternions, tips = [], [], []
    for qpos in source_qpos:
        source_data.qpos[:] = qpos
        mujoco.mj_forward(source_model, source_data)
        positions.append(source_data.site_xpos[wrist_id].copy())
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, source_data.site_xmat[wrist_id])
        quaternions.append(quaternion)
        tips.append(source_data.site_xpos[tip_ids].copy())
    positions, quaternions, tips = (
        np.array(positions),
        np.array(quaternions),
        np.array(tips),
    )
    np.savez(
        output / "source_landmarks.npz",
        wrist_pos=positions,
        wrist_quat=quaternions,
        fingertips=tips,
        finger_names=FINGERS,
        object_qpos=source_qpos[:, -7:],
    )
    low, high = model.actuator_ctrlrange.T
    hand_seed = (
        np.fromstring(native.find("keyframe/key").get("ctrl"), sep=" ")[6:]
        if is_yimu
        else np.full(12, 0.2)
    )
    seed = np.r_[[-0.6, -0.5, 0.3, 1.6, -0.3, -0.6, -0.3], hand_seed]
    seed = np.clip(seed, low + 0.001, high - 0.001)
    target_joints, fitted_wrists, fitted_quats, errors = [], [], [], []
    hand_geom_ids = [
        g
        for g in range(model.ngeom)
        if model.geom(g).name.startswith("collision_hand_")
    ]
    hand_index = {g: i for i, g in enumerate(hand_geom_ids)}
    object_geom_ids = {model.geom(n).id for n in object_geoms}
    started = time.perf_counter()
    ik_stride = max(1, int(args.time_scale))
    ik_indices = np.unique(
        np.r_[np.arange(0, len(positions), ik_stride), len(positions) - 1]
    )
    for index in ik_indices:
        p, q, tip = positions[index], quaternions[index], tips[index]

        def residual(
            joints,
            desired_p=p,
            desired_q=q,
            desired_tips=tip,
            previous=seed,
            index=index,
        ):
            target_data.qpos[control_qpos] = joints
            project_joint_equalities(model, target_data.qpos)
            target_data.qpos[-7:] = source_qpos[index, -7:]
            mujoco.mj_forward(model, target_data)
            penetration = np.zeros(len(hand_geom_ids))
            for contact_pair in target_data.contact:
                g1, g2 = int(contact_pair.geom1), int(contact_pair.geom2)
                if g1 in hand_index and g2 in object_geom_ids:
                    penetration[hand_index[g1]] = max(
                        penetration[hand_index[g1]], -contact_pair.dist - 0.001
                    )
                if g2 in hand_index and g1 in object_geom_ids:
                    penetration[hand_index[g2]] = max(
                        penetration[hand_index[g2]], -contact_pair.dist - 0.001
                    )
            return np.r_[
                5 * (target_data.sensordata[:3] - desired_p),
                1.0 * quaternion_error(target_data.sensordata[3:7], desired_q),
                40
                * (target_data.sensordata[14:26].reshape(4, 3) - desired_tips).ravel(),
                180 * penetration,
                0.015 * (joints - previous),
            ]

        delta = (
            ik_stride
            * np.r_[np.full(7, 0.04), np.full(model.nu - 7, 0.03 if is_yimu else 0.08)]
        )
        lower = np.maximum(low + 0.0001, seed - delta) if index else low + 0.0001
        upper = np.minimum(high - 0.0001, seed + delta) if index else high - 0.0001
        fitted = least_squares(residual, seed, bounds=(lower, upper), max_nfev=70)
        seed = fitted.x
        residual(seed)
        target_joints.append(target_data.qpos[:robot_nq].copy())
        fitted_wrists.append(target_data.sensordata[:3].copy())
        fitted_quats.append(target_data.sensordata[3:7].copy())
        errors.append(
            np.linalg.norm(target_data.sensordata[14:26].reshape(4, 3) - tip, axis=1)
        )
    target_joints = np.array(target_joints)
    if ik_stride > 1:
        target_joints = np.column_stack(
            [
                np.interp(np.arange(len(positions)), ik_indices, column)
                for column in target_joints.T
            ]
        )
        fitted_wrists = np.column_stack(
            [
                np.interp(np.arange(len(positions)), ik_indices, column)
                for column in np.array(fitted_wrists).T
            ]
        )
        from scipy.spatial.transform import Slerp

        fitted_quats = Slerp(
            ik_indices, Rotation.from_quat(np.array(fitted_quats)[:, [1, 2, 3, 0]])
        )(np.arange(len(positions))).as_quat()[:, [3, 0, 1, 2]]
    # Target commands are constructed from target IK/dynamics; never copied
    # from the source's 16 finger commands or from a target-hand dataset.
    target_joints = gaussian_filter1d(target_joints, sigma=3, axis=0, mode="nearest")
    padded = np.zeros((len(target_joints), model.nq))
    padded[:, :robot_nq] = target_joints
    project_joint_equalities(model, padded)
    target_joints = padded[:, :robot_nq]
    velocities = np.gradient(target_joints, 0.01, axis=0)
    kp = model.actuator_gainprm[:, 0]
    kv = -model.actuator_biasprm[:, 2]
    commands = []
    for index, joints in enumerate(target_joints):
        command = joints[control_qpos] + np.clip(
            kv * velocities[index, control_qpos] / kp, -0.15, 0.15
        )
        command[:7] = joints[:7] + 0.05 * velocities[index, :7]
        commands.append(command)
    ref = dict(reference)
    ref.update(
        target_joint_ik=target_joints,
        target_ctrl_nominal=np.array(commands)[1:],
        wrist_pos=np.array(fitted_wrists)[1:],
        wrist_quat=np.array(fitted_quats)[1:],
        initial_wrist_pos=fitted_wrists[0],
        initial_wrist_quat=fitted_quats[0],
        finger_qpos=target_joints[1:, 7:],
        finger_ctrl=np.array(commands)[1:, 7:],
        initial_qpos=np.r_[np.zeros(6), target_joints[0, 7:], source_qpos[0, -7:]],
        initial_qvel=np.r_[
            np.zeros(6), velocities[0, 7:], reference["initial_qvel"][-6:]
        ],
    )
    np.savez(output / "target_ik_reference.npz", **ref)
    mapping = {
        "source_hand": "allegro",
        "source_finger_dofs": 16,
        "target_hand": args.target_hand,
        "target_finger_dofs": model.nu - 7,
        "target_finger_joints": robot_nq - 7,
        "preserved_linkages": model.neq,
        "target_arm_dofs": 7,
        "matched_fingers": FINGERS,
        "extra_target_finger": "pinky: smooth posture regularization, no source pinky",
        "mapping": "canonical wrist and Cartesian fingertips; no joint-angle correspondence",
        "target_trajectory_loaded": False,
        "ik_mean_fingertip_error_m": float(np.mean(errors)),
        "ik_seconds": time.perf_counter() - started,
    }
    (output / "cross_hand_mapping.json").write_text(
        json.dumps(mapping, indent=2) + "\n"
    )
    print(json.dumps(mapping, indent=2), flush=True)
    return model, ref
