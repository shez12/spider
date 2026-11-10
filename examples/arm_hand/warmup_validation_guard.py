"""Reject numerical GPU replay failures without aborting the search.

Finite-trajectory validation delegates unchanged to frozen warmup_grasp_v2.
This module changes failure handling, never acceptance thresholds or physics.
"""
import numpy as np
from examples.arm_hand.warmup_acceptance import audit_for_optimizer as original_audit,VERSION


def audit_for_optimizer(model,reference,trajectory,backend='mujoco_warp'):
    bad=[name for name in ('qpos','qvel','ctrl') if name in trajectory and not np.isfinite(trajectory[name]).all()]
    if not bad:return original_audit(model,reference,trajectory,backend)
    inf=float('inf')
    return dict(acceptance_version=VERSION,invalid_trajectory=True,invalid_fields=bad,
                rejection_reason='Numerically nonfinite GPU physics; geometry not evaluated; not an infrastructure/acceptance success',
                trajectory_backend=f'GPU {backend}; nonfinite saved/replayed states rejected',
                initial_palm_error_m=inf,initial_fingertip_errors_m=[inf]*4,object_lift_m=inf,
                final_object_position_error_m=inf,final_object_orientation_error_deg=inf,
                object_rmse_m=inf,palm_rmse_m=inf,max_arm_table_penetration_m=inf,
                max_hand_table_penetration_m=inf,max_hand_self_penetration_m=inf,
                arm_table_frames_over_1mm=[],arm_table_clearance_pass=False,extra_hold=False,
                grasp_retention=dict(retained=False,invalid_trajectory=True,invalid_fields=bad))
