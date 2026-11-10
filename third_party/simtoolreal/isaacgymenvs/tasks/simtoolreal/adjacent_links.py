RIGHT_SHARPA_KUKA_LINK_TO_ADJACENT_LINKS = {
    "iiwa14_link_0": ["iiwa14_link_1"],
    "iiwa14_link_1": ["iiwa14_link_0", "iiwa14_link_2"],
    "iiwa14_link_2": ["iiwa14_link_1", "iiwa14_link_3"],
    "iiwa14_link_3": ["iiwa14_link_2", "iiwa14_link_4"],
    "iiwa14_link_4": ["iiwa14_link_3", "iiwa14_link_5"],
    "iiwa14_link_5": ["iiwa14_link_4", "iiwa14_link_6"],
    "iiwa14_link_6": ["iiwa14_link_5", "iiwa14_link_7"],
    "iiwa14_link_7": [
        "iiwa14_link_6",
        "right_index_MCP_VL",
        "right_middle_MCP_VL",
        "right_pinky_MC",
        "right_ring_MCP_VL",
        "right_thumb_CMC_VL",
    ],
    "right_index_MCP_VL": ["iiwa14_link_7", "right_index_PP"],
    "right_index_PP": ["right_index_MCP_VL", "right_index_MP"],
    "right_index_MP": ["right_index_PP", "right_index_DP"],
    "right_index_DP": ["right_index_MP"],
    "right_middle_MCP_VL": ["iiwa14_link_7", "right_middle_PP"],
    "right_middle_PP": ["right_middle_MCP_VL", "right_middle_MP"],
    "right_middle_MP": ["right_middle_PP", "right_middle_DP"],
    "right_middle_DP": ["right_middle_MP"],
    "right_pinky_MC": ["iiwa14_link_7", "right_pinky_MCP_VL"],
    "right_pinky_MCP_VL": ["right_pinky_MC", "right_pinky_PP"],
    "right_pinky_PP": ["right_pinky_MCP_VL", "right_pinky_MP"],
    "right_pinky_MP": ["right_pinky_PP", "right_pinky_DP"],
    "right_pinky_DP": ["right_pinky_MP"],
    "right_ring_MCP_VL": ["iiwa14_link_7", "right_ring_PP"],
    "right_ring_PP": ["right_ring_MCP_VL", "right_ring_MP"],
    "right_ring_MP": ["right_ring_PP", "right_ring_DP"],
    "right_ring_DP": ["right_ring_MP"],
    "right_thumb_CMC_VL": ["iiwa14_link_7", "right_thumb_MC"],
    "right_thumb_MC": ["right_thumb_CMC_VL", "right_thumb_MCP_VL"],
    "right_thumb_MCP_VL": ["right_thumb_MC", "right_thumb_PP"],
    "right_thumb_PP": ["right_thumb_MCP_VL", "right_thumb_DP"],
    "right_thumb_DP": ["right_thumb_PP"],
}

LEFT_SHARPA_KUKA_LINK_TO_ADJACENT_LINKS = {
    "iiwa14_link_0": ["iiwa14_link_1"],
    "iiwa14_link_1": ["iiwa14_link_0", "iiwa14_link_2"],
    "iiwa14_link_2": ["iiwa14_link_1", "iiwa14_link_3"],
    "iiwa14_link_3": ["iiwa14_link_2", "iiwa14_link_4"],
    "iiwa14_link_4": ["iiwa14_link_3", "iiwa14_link_5"],
    "iiwa14_link_5": ["iiwa14_link_4", "iiwa14_link_6"],
    "iiwa14_link_6": ["iiwa14_link_5", "iiwa14_link_7"],
    "iiwa14_link_7": [
        "iiwa14_link_6",
        "left_thumb_CMC_VL",
        "left_thumb_MC",
        "left_index_MCP_VL",
        "left_index_PP",
        "left_middle_MCP_VL",
        "left_middle_PP",
        "left_ring_MCP_VL",
        "left_ring_PP",
        "left_pinky_MC",
    ],
    "left_index_MCP_VL": ["iiwa14_link_7", "left_index_PP"],
    "left_index_PP": ["iiwa14_link_7", "left_index_MCP_VL", "left_index_MP"],
    "left_index_MP": ["left_index_PP", "left_index_DP"],
    "left_index_DP": ["left_index_MP"],
    "left_middle_MCP_VL": ["iiwa14_link_7", "left_middle_PP"],
    "left_middle_PP": ["iiwa14_link_7", "left_middle_MCP_VL", "left_middle_MP"],
    "left_middle_MP": ["left_middle_PP", "left_middle_DP"],
    "left_middle_DP": ["left_middle_MP"],
    "left_pinky_MC": ["iiwa14_link_7", "left_pinky_MCP_VL", "left_pinky_PP"],
    "left_pinky_MCP_VL": ["left_pinky_MC", "left_pinky_PP"],
    "left_pinky_PP": ["left_pinky_MC", "left_pinky_MCP_VL", "left_pinky_MP"],
    "left_pinky_MP": ["left_pinky_PP", "left_pinky_DP"],
    "left_pinky_DP": ["left_pinky_MP"],
    "left_ring_MCP_VL": ["iiwa14_link_7", "left_ring_PP"],
    "left_ring_PP": ["iiwa14_link_7", "left_ring_MCP_VL", "left_ring_MP"],
    "left_ring_MP": ["left_ring_PP", "left_ring_DP"],
    "left_ring_DP": ["left_ring_MP"],
    "left_thumb_CMC_VL": ["iiwa14_link_7", "left_thumb_MC"],
    "left_thumb_MC": [
        "iiwa14_link_7",
        "left_thumb_CMC_VL",
        "left_thumb_MCP_VL",
        "left_thumb_PP",
    ],
    "left_thumb_MCP_VL": ["left_thumb_MC", "left_thumb_PP"],
    "left_thumb_PP": ["left_thumb_MC", "left_thumb_MCP_VL", "left_thumb_DP"],
    "left_thumb_DP": ["left_thumb_PP"],
}


# KUKA iiwa14 + Yimu left hand.  The generated URDF prefixes every Yimu link
# with ``yimu_``; each finger is a serial chain rooted at yimu_base_link.
_YIMU_FINGER_CHAINS = (
    tuple(f"yimu_1-{j}" for j in range(1, 6)),
    tuple(f"yimu_2-{j}" for j in range(1, 5)),
    tuple(f"yimu_3-{j}" for j in range(1, 5)),
    tuple(f"yimu_4-{j}" for j in range(1, 5)),
    tuple(f"yimu_5-{j}" for j in range(1, 5)),
)
_YIMU_ALL_FINGER_LINKS = tuple(link for chain in _YIMU_FINGER_CHAINS for link in chain)
_YIMU_FINGER_ROOTS = tuple(chain[0] for chain in _YIMU_FINGER_CHAINS)
YIMU_LEFT_KUKA_LINK_TO_ADJACENT_LINKS = {
    "iiwa14_link_0": ["iiwa14_link_1"],
    "iiwa14_link_1": ["iiwa14_link_0", "iiwa14_link_2"],
    "iiwa14_link_2": ["iiwa14_link_1", "iiwa14_link_3"],
    "iiwa14_link_3": ["iiwa14_link_2", "iiwa14_link_4"],
    "iiwa14_link_4": ["iiwa14_link_3", "iiwa14_link_5"],
    "iiwa14_link_5": ["iiwa14_link_4", "iiwa14_link_6"],
    "iiwa14_link_6": ["iiwa14_link_5", "iiwa14_link_7"],
    "iiwa14_link_7": ["iiwa14_link_6", *_YIMU_ALL_FINGER_LINKS],
}
for _chain in _YIMU_FINGER_CHAINS:
    _chain_set = set(_chain)
    for _link in _chain:
        _neighbors = (_chain_set - {_link}) | {"iiwa14_link_7"}
        if _link in _YIMU_FINGER_ROOTS:
            _neighbors.update(_YIMU_FINGER_ROOTS)
            _neighbors.discard(_link)
        YIMU_LEFT_KUKA_LINK_TO_ADJACENT_LINKS[_link] = sorted(_neighbors)
