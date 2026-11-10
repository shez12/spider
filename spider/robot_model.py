"""Joint-space helpers for position-controlled robots with passive linkages."""

import mujoco
import numpy as np


def actuator_qpos_indices(model):
    """Return scalar joint qpos addresses in actuator order, excluding passives."""
    indices = []
    for transmission, pair in zip(
        model.actuator_trntype, model.actuator_trnid, strict=True
    ):
        joint = int(pair[0])
        if (
            transmission
            not in (mujoco.mjtTrn.mjTRN_JOINT, mujoco.mjtTrn.mjTRN_JOINTINPARENT)
            or joint < 0
        ):
            raise ValueError(
                "Position-reference extraction requires joint transmissions"
            )
        if model.jnt_type[joint] not in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            raise ValueError("Position-reference extraction requires scalar joints")
        indices.append(int(model.jnt_qposadr[joint]))
    return np.array(indices, dtype=int)


def project_joint_equalities(model, qpos):
    """Project active joint1=f(joint2) polynomial linkages in-place, including batches.

    This projects a kinematic reference only. Physics simulation still solves the
    original MuJoCo equality constraints; it does not overwrite executed states.
    """
    for index in range(model.neq):
        if (
            model.eq_type[index] != mujoco.mjtEq.mjEQ_JOINT
            or not model.eq_active0[index]
        ):
            continue
        dependent, driver = int(model.eq_obj1id[index]), int(model.eq_obj2id[index])
        if driver < 0:
            continue
        dependent_address = int(model.jnt_qposadr[dependent])
        driver_address = int(model.jnt_qposadr[driver])
        # MuJoCo's polynomial acts on displacements from the model's qpos0.
        x = qpos[..., driver_address] - model.qpos0[driver_address]
        qpos[..., dependent_address] = model.qpos0[
            dependent_address
        ] + np.polynomial.polynomial.polyval(x, model.eq_data[index, :5])
    return qpos
