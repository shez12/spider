"""Robot-specific training constants for SimToolReal.

Default Isaac Sim stack is KUKA iiwa14 + SharPa (29 DoF). KUKA + Yimu-left
adds a 28-DoF variant with a 21-DoF hand.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

# Shared Yimu hand PD (thumb 5 + four fingers × 4).
_YIMU_HAND_JOINTS: tuple[str, ...] = (
    "yimu_1-1", "yimu_1-2", "yimu_1-3", "yimu_1-4", "yimu_1-5",
    "yimu_2-1", "yimu_2-2", "yimu_2-3", "yimu_2-4",
    "yimu_3-1", "yimu_3-2", "yimu_3-3", "yimu_3-4",
    "yimu_4-1", "yimu_4-2", "yimu_4-3", "yimu_4-4",
    "yimu_5-1", "yimu_5-2", "yimu_5-3", "yimu_5-4",
)

_YIMU_HAND_STIFFNESS: dict[str, float] = {
    "yimu_1-1": 6.95, "yimu_1-2": 13.2, "yimu_1-3": 4.76, "yimu_1-4": 6.62, "yimu_1-5": 0.9,
    "yimu_2-1": 4.76, "yimu_2-2": 6.62, "yimu_2-3": 0.9, "yimu_2-4": 0.9,
    "yimu_3-1": 4.76, "yimu_3-2": 6.62, "yimu_3-3": 0.9, "yimu_3-4": 0.9,
    "yimu_4-1": 4.76, "yimu_4-2": 6.62, "yimu_4-3": 0.9, "yimu_4-4": 0.9,
    "yimu_5-1": 4.76, "yimu_5-2": 6.62, "yimu_5-3": 0.9, "yimu_5-4": 0.9,
}
_YIMU_HAND_DAMPING: dict[str, float] = {
    "yimu_1-1": 0.28676845, "yimu_1-2": 0.40845109, "yimu_1-3": 0.20394083,
    "yimu_1-4": 0.24044435, "yimu_1-5": 0.04190723,
    "yimu_2-1": 0.20859232, "yimu_2-2": 0.24595532, "yimu_2-3": 0.04243185, "yimu_2-4": 0.03504461,
    "yimu_3-1": 0.2085923, "yimu_3-2": 0.24595532, "yimu_3-3": 0.04243185, "yimu_3-4": 0.03504461,
    "yimu_4-1": 0.20859226, "yimu_4-2": 0.24595528, "yimu_4-3": 0.04243183, "yimu_4-4": 0.0350446,
    "yimu_5-1": 0.20859229, "yimu_5-2": 0.24595528, "yimu_5-3": 0.04243183, "yimu_5-4": 0.0350446,
}
_YIMU_HAND_ARMATURE: dict[str, float] = {
    "yimu_1-1": 0.0032, "yimu_1-2": 0.0032, "yimu_1-3": 0.00265, "yimu_1-4": 0.00265, "yimu_1-5": 0.0006,
    "yimu_2-1": 0.00265, "yimu_2-2": 0.00265, "yimu_2-3": 0.0006, "yimu_2-4": 0.00042,
    "yimu_3-1": 0.00265, "yimu_3-2": 0.00265, "yimu_3-3": 0.0006, "yimu_3-4": 0.00042,
    "yimu_4-1": 0.00265, "yimu_4-2": 0.00265, "yimu_4-3": 0.0006, "yimu_4-4": 0.00042,
    "yimu_5-1": 0.00265, "yimu_5-2": 0.00265, "yimu_5-3": 0.0006, "yimu_5-4": 0.00042,
}
_YIMU_HAND_EFFORT: dict[str, float] = {
    "yimu_1-1": 3.3, "yimu_1-2": 3.3, "yimu_1-3": 1.864, "yimu_1-4": 0.638, "yimu_1-5": 0.189369,
    "yimu_2-1": 1.864, "yimu_2-2": 1.864, "yimu_2-3": 0.638, "yimu_2-4": 0.189369,
    "yimu_3-1": 1.864, "yimu_3-2": 1.864, "yimu_3-3": 0.638, "yimu_3-4": 0.189369,
    "yimu_4-1": 1.864, "yimu_4-2": 1.864, "yimu_4-3": 0.638, "yimu_4-4": 0.189369,
    "yimu_5-1": 1.864, "yimu_5-2": 1.864, "yimu_5-3": 0.638, "yimu_5-4": 0.189369,
}
_YIMU_HAND_FRICTION: dict[str, float] = {
    "yimu_1-1": 0.132, "yimu_1-2": 0.132, "yimu_1-3": 0.07456, "yimu_1-4": 0.07456, "yimu_1-5": 0.01276,
    "yimu_2-1": 0.07456, "yimu_2-2": 0.07456, "yimu_2-3": 0.01276, "yimu_2-4": 0.00378738,
    "yimu_3-1": 0.07456, "yimu_3-2": 0.07456, "yimu_3-3": 0.01276, "yimu_3-4": 0.00378738,
    "yimu_4-1": 0.07456, "yimu_4-2": 0.07456, "yimu_4-3": 0.01276, "yimu_4-4": 0.00378738,
    "yimu_5-1": 0.07456, "yimu_5-2": 0.07456, "yimu_5-3": 0.01276, "yimu_5-4": 0.00378738,
}

_SHARPA_HAND_FRICTION: dict[str, float] = {
    "left_1_thumb_CMC_FE": 0.132, "left_thumb_CMC_AA": 0.132,
    "left_thumb_MCP_FE": 0.07456, "left_thumb_MCP_AA": 0.07456, "left_thumb_IP": 0.01276,
    "left_2_index_MCP_FE": 0.07456, "left_index_MCP_AA": 0.07456,
    "left_index_PIP": 0.01276, "left_index_DIP": 0.00378738,
    "left_3_middle_MCP_FE": 0.07456, "left_middle_MCP_AA": 0.07456,
    "left_middle_PIP": 0.01276, "left_middle_DIP": 0.00378738,
    "left_4_ring_MCP_FE": 0.07456, "left_ring_MCP_AA": 0.07456,
    "left_ring_PIP": 0.01276, "left_ring_DIP": 0.00378738,
    "left_5_pinky_CMC": 0.012, "left_pinky_MCP_FE": 0.07456,
    "left_pinky_MCP_AA": 0.07456, "left_pinky_PIP": 0.01276, "left_pinky_DIP": 0.00378738,
}


@dataclass(frozen=True)
class RobotSpec:
    """Per-robot names, PD, default pose, and palm/fingertip geometry."""

    name: str
    num_arm_joints: int
    num_hand_joints: int
    arm_joint_regex: str
    hand_joint_regex: str
    joint_names_canonical: tuple[str, ...]
    palm_body_name: str
    fingertip_link_names: tuple[str, ...]
    palm_center_offset: tuple[float, float, float]
    fingertip_offsets: tuple[tuple[float, float, float], ...]
    arm_joint_stiffness: dict[str, float]
    arm_joint_damping: dict[str, float]
    hand_joint_stiffness: dict[str, float]
    hand_joint_damping: dict[str, float]
    hand_joint_armature: dict[str, float]
    hand_joint_friction: dict[str, float]
    hand_joint_effort: dict[str, float]
    arm_default_joint_pos: dict[str, float]
    arm_higher_joint_deltas: dict[str, float]
    robot_base_pos: tuple[float, float, float]
    robot_base_rot_wxyz: tuple[float, float, float, float]
    adjacent_link_maps: tuple[str, ...]

    @property
    def num_joints(self) -> int:
        return self.num_arm_joints + self.num_hand_joints

    @property
    def num_fingertips(self) -> int:
        return len(self.fingertip_link_names)


KUKA_SHARPA_SPEC = RobotSpec(
    name="kuka_sharpa",
    num_arm_joints=7,
    num_hand_joints=22,
    arm_joint_regex="iiwa14_joint_.*",
    hand_joint_regex="left_.*",
    joint_names_canonical=(
        "iiwa14_joint_1", "iiwa14_joint_2", "iiwa14_joint_3", "iiwa14_joint_4",
        "iiwa14_joint_5", "iiwa14_joint_6", "iiwa14_joint_7",
        "left_1_thumb_CMC_FE", "left_thumb_CMC_AA", "left_thumb_MCP_FE",
        "left_thumb_MCP_AA", "left_thumb_IP",
        "left_2_index_MCP_FE", "left_index_MCP_AA", "left_index_PIP", "left_index_DIP",
        "left_3_middle_MCP_FE", "left_middle_MCP_AA", "left_middle_PIP", "left_middle_DIP",
        "left_4_ring_MCP_FE", "left_ring_MCP_AA", "left_ring_PIP", "left_ring_DIP",
        "left_5_pinky_CMC", "left_pinky_MCP_FE", "left_pinky_MCP_AA",
        "left_pinky_PIP", "left_pinky_DIP",
    ),
    palm_body_name="iiwa14_link_7",
    fingertip_link_names=(
        "left_index_DP", "left_middle_DP", "left_ring_DP",
        "left_thumb_DP", "left_pinky_DP",
    ),
    palm_center_offset=(-0.0, -0.02, 0.16),
    fingertip_offsets=(
        (0.02, 0.002, 0.0),
        (0.02, 0.002, 0.0),
        (0.02, 0.002, 0.0),
        (0.02, 0.002, 0.0),
        (0.02, 0.002, 0.0),
    ),
    arm_joint_stiffness={
        "iiwa14_joint_1": 600.0, "iiwa14_joint_2": 600.0, "iiwa14_joint_3": 500.0,
        "iiwa14_joint_4": 400.0, "iiwa14_joint_5": 200.0, "iiwa14_joint_6": 200.0,
        "iiwa14_joint_7": 200.0,
    },
    arm_joint_damping={
        "iiwa14_joint_1": 27.027026473513512, "iiwa14_joint_2": 27.027026473513512,
        "iiwa14_joint_3": 24.672186769721083, "iiwa14_joint_4": 22.067474708266914,
        "iiwa14_joint_5": 9.752538131173853, "iiwa14_joint_6": 9.147747263670984,
        "iiwa14_joint_7": 9.147747263670984,
    },
    hand_joint_stiffness={
        "left_1_thumb_CMC_FE": 6.95, "left_thumb_CMC_AA": 13.2, "left_thumb_MCP_FE": 4.76,
        "left_thumb_MCP_AA": 6.62, "left_thumb_IP": 0.9,
        "left_2_index_MCP_FE": 4.76, "left_index_MCP_AA": 6.62,
        "left_index_PIP": 0.9, "left_index_DIP": 0.9,
        "left_3_middle_MCP_FE": 4.76, "left_middle_MCP_AA": 6.62,
        "left_middle_PIP": 0.9, "left_middle_DIP": 0.9,
        "left_4_ring_MCP_FE": 4.76, "left_ring_MCP_AA": 6.62,
        "left_ring_PIP": 0.9, "left_ring_DIP": 0.9,
        "left_5_pinky_CMC": 1.38, "left_pinky_MCP_FE": 4.76, "left_pinky_MCP_AA": 6.62,
        "left_pinky_PIP": 0.9, "left_pinky_DIP": 0.9,
    },
    hand_joint_damping={
        "left_1_thumb_CMC_FE": 0.28676845, "left_thumb_CMC_AA": 0.40845109,
        "left_thumb_MCP_FE": 0.20394083, "left_thumb_MCP_AA": 0.24044435,
        "left_thumb_IP": 0.04190723,
        "left_2_index_MCP_FE": 0.20859232, "left_index_MCP_AA": 0.24595532,
        "left_index_PIP": 0.04243185, "left_index_DIP": 0.03504461,
        "left_3_middle_MCP_FE": 0.2085923, "left_middle_MCP_AA": 0.24595532,
        "left_middle_PIP": 0.04243185, "left_middle_DIP": 0.03504461,
        "left_4_ring_MCP_FE": 0.20859226, "left_ring_MCP_AA": 0.24595528,
        "left_ring_PIP": 0.04243183, "left_ring_DIP": 0.0350446,
        "left_5_pinky_CMC": 0.02782345, "left_pinky_MCP_FE": 0.20859229,
        "left_pinky_MCP_AA": 0.24595528, "left_pinky_PIP": 0.04243183,
        "left_pinky_DIP": 0.0350446,
    },
    hand_joint_armature={
        "left_1_thumb_CMC_FE": 0.0032, "left_thumb_CMC_AA": 0.0032,
        "left_thumb_MCP_FE": 0.00265, "left_thumb_MCP_AA": 0.00265, "left_thumb_IP": 0.0006,
        "left_2_index_MCP_FE": 0.00265, "left_index_MCP_AA": 0.00265,
        "left_index_PIP": 0.0006, "left_index_DIP": 0.00042,
        "left_3_middle_MCP_FE": 0.00265, "left_middle_MCP_AA": 0.00265,
        "left_middle_PIP": 0.0006, "left_middle_DIP": 0.00042,
        "left_4_ring_MCP_FE": 0.00265, "left_ring_MCP_AA": 0.00265,
        "left_ring_PIP": 0.0006, "left_ring_DIP": 0.00042,
        "left_5_pinky_CMC": 0.00012, "left_pinky_MCP_FE": 0.00265,
        "left_pinky_MCP_AA": 0.00265, "left_pinky_PIP": 0.0006, "left_pinky_DIP": 0.00042,
    },
    hand_joint_friction=_SHARPA_HAND_FRICTION,
    hand_joint_effort={},
    arm_default_joint_pos={
        "iiwa14_joint_1": -1.571, "iiwa14_joint_2": 1.571, "iiwa14_joint_3": 0.0,
        "iiwa14_joint_4": 1.376, "iiwa14_joint_5": 0.0, "iiwa14_joint_6": 1.485,
        "iiwa14_joint_7": 1.308,
    },
    arm_higher_joint_deltas={
        "iiwa14_joint_2": -math.radians(10.0),
        "iiwa14_joint_4": math.radians(10.0),
    },
    robot_base_pos=(0.0, 0.8, 0.0),
    robot_base_rot_wxyz=(1.0, 0.0, 0.0, 0.0),
    adjacent_link_maps=(
        "LEFT_SHARPA_KUKA_LINK_TO_ADJACENT_LINKS",
        "RIGHT_SHARPA_KUKA_LINK_TO_ADJACENT_LINKS",
    ),
)

KUKA_YIMU_SPEC = RobotSpec(
    name="kuka_yimu",
    num_arm_joints=7,
    num_hand_joints=21,
    arm_joint_regex="iiwa14_joint_.*",
    hand_joint_regex="yimu_.*",
    joint_names_canonical=(
        "iiwa14_joint_1", "iiwa14_joint_2", "iiwa14_joint_3", "iiwa14_joint_4",
        "iiwa14_joint_5", "iiwa14_joint_6", "iiwa14_joint_7",
        *_YIMU_HAND_JOINTS,
    ),
    palm_body_name="iiwa14_link_7",
    fingertip_link_names=(
        "yimu_2-4", "yimu_3-4", "yimu_4-4", "yimu_1-5", "yimu_5-4",
    ),
    palm_center_offset=(-0.0, -0.02, 0.16),
    fingertip_offsets=(
        (0.0112, -0.0006, 0.0212),
        (-0.0112, -0.0006, -0.0212),
        (-0.0112, -0.0006, -0.0212),
        (-0.0195, -0.0001, 0.0039),
        (-0.0112, -0.0006, -0.0212),
    ),
    arm_joint_stiffness=KUKA_SHARPA_SPEC.arm_joint_stiffness,
    arm_joint_damping=KUKA_SHARPA_SPEC.arm_joint_damping,
    hand_joint_stiffness={k.replace("-", "_"): v for k, v in _YIMU_HAND_STIFFNESS.items()},
    hand_joint_damping={k.replace("-", "_"): v for k, v in _YIMU_HAND_DAMPING.items()},
    hand_joint_armature={k.replace("-", "_"): v for k, v in _YIMU_HAND_ARMATURE.items()},
    hand_joint_friction={k.replace("-", "_"): v for k, v in _YIMU_HAND_FRICTION.items()},
    hand_joint_effort={k.replace("-", "_"): v for k, v in _YIMU_HAND_EFFORT.items()},
    arm_default_joint_pos=KUKA_SHARPA_SPEC.arm_default_joint_pos,
    arm_higher_joint_deltas=KUKA_SHARPA_SPEC.arm_higher_joint_deltas,
    robot_base_pos=(0.0, 0.8, 0.0),
    robot_base_rot_wxyz=(1.0, 0.0, 0.0, 0.0),
    adjacent_link_maps=("YIMU_LEFT_KUKA_LINK_TO_ADJACENT_LINKS",),
)

assert len(KUKA_SHARPA_SPEC.joint_names_canonical) == 29
assert len(KUKA_YIMU_SPEC.joint_names_canonical) == 28
assert len(KUKA_YIMU_SPEC.fingertip_offsets) == KUKA_YIMU_SPEC.num_fingertips


def hand_actuator_extras(spec: RobotSpec) -> dict:
    """Optional ImplicitActuatorCfg kwargs (friction, effort limits) from RobotSpec."""
    extras: dict = {}
    if spec.hand_joint_friction:
        extras["friction"] = spec.hand_joint_friction
    if spec.hand_joint_effort:
        extras["effort_limit_sim"] = spec.hand_joint_effort
    return extras


def normalize_robot_name(name: str) -> str:
    """USD may rewrite URDF hyphens as underscores; compare on the underscore form."""
    return name.replace("-", "_")


def match_robot_names(requested: tuple[str, ...] | list[str], available: list[str]) -> list[str]:
    """Map requested URDF names onto live Lab/USD names."""
    by_norm = {normalize_robot_name(name): name for name in available}
    resolved: list[str] = []
    missing: list[str] = []
    for name in requested:
        hit = by_norm.get(normalize_robot_name(name))
        if hit is None:
            missing.append(name)
        else:
            resolved.append(hit)
    if missing:
        raise RuntimeError(
            f"Robot names not found: {missing}. Available: {available}"
        )
    return resolved


def resolve_robot_spec(robot_urdf: str | Path) -> RobotSpec:
    """Pick the training spec from the configured robot URDF path."""
    path_str = str(robot_urdf).replace("\\", "/").lower()
    if "yimu" in path_str:
        return KUKA_YIMU_SPEC
    return KUKA_SHARPA_SPEC
