"""Pure-torch core of the speed-commanded (continuous) goal.

Kept free of Isaac imports so the gating rules can be unit-tested without a
simulator (see ``isaacsimenvs/tests/test_continuous_goal_gate.py``).

Model
-----
The goal slides along a stored subtarget chain at ``goal_speed`` m/s.  Two rules:

* **Start gate** — until the object has reached subtarget 0 (judged with
  ``first_tolerance``), the goal stays parked there.  Otherwise it would start
  moving before the object ever picked it up.
* **Catch-up gate** — once released, the goal advances freely at the commanded
  speed, but parks at the next subtarget until the object is inside the running
  tolerance.  So a slow command throttles the object and a fast one is capped by
  how well the policy can keep up.
"""

from __future__ import annotations

import torch


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate ``v`` by quaternion ``q`` (wxyz)."""
    qvec = q[..., 1:]
    w = q[..., 0:1]
    uv = torch.cross(qvec, v, dim=-1)
    uuv = torch.cross(qvec, uv, dim=-1)
    return v + 2.0 * (w * uv + uuv)


def slerp_quat(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Shortest-path quaternion slerp (wxyz), ``t`` broadcastable to (..., 1)."""
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    theta = torch.acos(dot.clamp(-1.0, 1.0))
    sin_theta = torch.sin(theta)
    small = sin_theta.abs() < 1e-6
    lerp = q0 + (q1 - q0) * t
    slerp = (torch.sin((1.0 - t) * theta) * q0 + torch.sin(t * theta) * q1) / sin_theta.clamp(min=1e-9)
    out = torch.where(small, lerp, slerp)
    return out / out.norm(dim=-1, keepdim=True).clamp(min=1e-9)


def goal_pose_at_s(
    traj_pos: torch.Tensor,   # (T, K, 3)
    traj_quat: torch.Tensor,  # (T, K, 4) wxyz
    traj_id: torch.Tensor,    # (N,) long
    s: torch.Tensor,          # (N,) float in [0, K-1]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interpolated goal pose at fractional position ``s`` on the chain."""
    K = traj_pos.shape[1]
    i0 = torch.clamp(torch.floor(s).long(), max=K - 2)
    frac = (s - i0.float()).clamp(0.0, 1.0).unsqueeze(-1)
    p0 = traj_pos[traj_id, i0]
    p1 = traj_pos[traj_id, i0 + 1]
    q0 = traj_quat[traj_id, i0]
    q1 = traj_quat[traj_id, i0 + 1]
    return p0 + (p1 - p0) * frac, slerp_quat(q0, q1, frac)


def hammering_offset(
    phase: torch.Tensor,          # (N,) in [0, 1)
    amplitude: float,
    down_fraction: float,
    prev_phase: torch.Tensor | None = None,
    down_exponent: float = 2.0,
    up_exponent: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stroke offset for a pounding motion — accelerating on the way down.

    Returns ``(offset, at_bottom)``: ``offset`` goes 0 -> -amplitude -> 0 within
    one cycle (negative = downward).

    The down stroke is ``u ** down_exponent`` (u = progress through the down
    stroke), so the hammer *speeds up* as it falls and is fastest at the moment
    of impact — ``down_exponent = 1`` is the old constant-speed triangle, 2 is a
    constant-acceleration drop, larger is a sharper whip.  The lift uses
    ``up_exponent`` (1 = linear).  ``down_fraction < 0.5`` additionally makes the
    down stroke shorter in time than the lift.

    The bottom of the stroke is detected by *crossing* ``down_fraction``, not by
    thresholding the wave: with a discrete phase step the peak value is easily
    skipped, so a "value >= 0.999" test would never fire.
    """
    down = min(max(down_fraction, 0.05), 0.95)
    u_down = (phase / down).clamp(0.0, 1.0)
    u_up = ((1.0 - phase) / (1.0 - down)).clamp(0.0, 1.0)
    tri = torch.where(phase < down, u_down ** float(down_exponent), u_up ** float(up_exponent))
    tri = tri.clamp(0.0, 1.0)
    if prev_phase is None:
        at_bottom = tri >= 0.999
    else:
        at_bottom = (prev_phase < down) & (phase >= down)
    return -float(amplitude) * tri, at_bottom


def quat_from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Quaternion (wxyz) from a unit axis and an angle in radians.

    ``axis`` may be a single (3,) vector or one per env, (N, 3).
    """
    half = angle * 0.5
    w = torch.cos(half).reshape(-1, 1)
    s = torch.sin(half).reshape(-1, 1)
    if axis.dim() == 1:
        axis = axis.reshape(1, 3).expand(angle.shape[0], 3)
    return torch.cat([w, axis * s], dim=-1)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two (..., 4) wxyz quaternions."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def hammering_pose(
    base_pos: torch.Tensor,    # (N, 3) env-local, already offset by the stroke
    base_quat: torch.Tensor,   # (N, 4) wxyz
    tri: torch.Tensor,         # (N,) 0 at the top of the stroke, 1 at the bottom
    max_deg: float,
    local_axis: torch.Tensor,  # (3,) unit axis in the OBJECT frame (e.g. +y)
    pivot_local: torch.Tensor,  # (N, 3) pivot in the object frame (handle root)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate the goal *about a pivot on the object* (the handle root).

    The hammer's URDF puts the handle along +x with the head at +x and the link
    origin at the handle's centre, so simply spinning the quaternion turns the
    hammer about its middle — which looks wrong, and looks different every
    rollout because the grasp orientation changes.  Rotating about the handle
    root instead means both the orientation and the *position* swing, exactly
    like a real wrist-driven hammer strike.  ``max_deg = 0`` disables it.
    """
    if abs(float(max_deg)) < 1e-6:
        return base_pos, base_quat
    angle = tri * (float(max_deg) * 3.141592653589793 / 180.0)
    # Rotation axis expressed in the world frame by the current grasp.
    axis_world = quat_rotate(base_quat, local_axis.expand(base_quat.shape[0], 3))
    axis_world = axis_world / axis_world.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    dq = quat_from_axis_angle(axis_world, angle)

    pivot_world = base_pos + quat_rotate(base_quat, pivot_local)
    pos = pivot_world + quat_rotate(dq, base_pos - pivot_world)
    quat = quat_mul(dq, base_quat)
    return pos, quat


def _keypoints_world(center_pos, center_quat, offsets):
    """(N, P, 3) object-frame keypoints rotated into the world frame."""
    n, k, _ = offsets.shape
    q = center_quat.unsqueeze(1).expand(-1, k, -1).reshape(-1, 4)
    offs = offsets.reshape(-1, 3)
    rot = quat_rotate(q, offs).reshape(n, k, 3)
    return center_pos.unsqueeze(1) + rot


def continuous_goal_update(
    traj_pos: torch.Tensor,      # (T, K, 3)
    traj_quat: torch.Tensor,     # (T, K, 4)
    traj_id: torch.Tensor,       # (N,)
    s: torch.Tensor,             # (N,) current fractional position
    released: torch.Tensor,      # (N,) bool — object has taken subtarget 0
    goal_speed: torch.Tensor,    # (N,) commanded m/s
    obj_pos: torch.Tensor,       # (N, 3) env-local
    obj_quat: torch.Tensor,      # (N, 4) wxyz
    kp_offsets: torch.Tensor,    # (N, P, 3) already scaled
    dt: float,
    keypoint_scale: float,
    first_tolerance: float | None,   # tolerance param for subtarget 0
    running_tolerance: torch.Tensor,  # (N,) tolerance param for the rest
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One step of the leader-follower goal.

    Returns ``(s_new, released_new, pulse)``; ``pulse`` is one step high on the
    step a subtarget is cleared (that is what counts as a success).
    """
    K = traj_pos.shape[1]
    pulse = torch.zeros_like(released)

    def _dist_to_subtarget(ids, j):
        obj_kp = _keypoints_world(obj_pos[ids], obj_quat[ids], kp_offsets[ids])
        sub_kp = _keypoints_world(traj_pos[traj_id[ids], j], traj_quat[traj_id[ids], j], kp_offsets[ids])
        return torch.norm(obj_kp - sub_kp, dim=-1).max(dim=-1).values

    # --- 1) start gate: park on subtarget 0 until the object takes it -------
    pending = ~released
    released_new = released.clone()
    if bool(pending.any()):
        ids = pending.nonzero(as_tuple=False).squeeze(-1)
        dist = _dist_to_subtarget(ids, torch.zeros_like(ids))
        tol = (
            float(first_tolerance) * keypoint_scale
            if first_tolerance is not None
            else running_tolerance[ids] * keypoint_scale
        )
        ok = dist <= tol
        released_new[ids] = ok
        pulse[ids] = ok

    # --- 2) after release: slide on its own at the commanded speed ----------
    # The only constraint left is the usual one — it cannot jump *past* the next
    # subtarget unless the object is already there (otherwise it would skip
    # subtargets the object never visited).
    i0 = torch.clamp(torch.floor(s).long(), max=K - 2)
    seg = torch.norm(traj_pos[traj_id, i0 + 1] - traj_pos[traj_id, i0], dim=-1).clamp(min=1e-4)
    s_new = s + goal_speed * float(dt) / seg

    j_next = torch.clamp(i0 + 1, max=K - 1)
    crosses = released_new & (s_new >= j_next.float()) & (i0 < K - 1)
    if bool(crosses.any()):
        ids = crosses.nonzero(as_tuple=False).squeeze(-1)
        j = j_next[ids]
        dist = _dist_to_subtarget(ids, j)
        # Cap the gate at half the spacing, else reaching subtarget j already
        # satisfies j+1 and the subtargets stop meaning anything.
        base_tol = running_tolerance[ids] * keypoint_scale
        prev_pos = traj_pos[traj_id[ids], j - 1]
        seg_len = torch.norm(traj_pos[traj_id[ids], j] - prev_pos, dim=-1)
        ok = dist <= torch.minimum(base_tol, seg_len * 0.5)
        s_new[ids] = torch.where(ok, s_new[ids], j.float())
        pulse[ids] = ok

    s_new = torch.clamp(s_new, max=float(K - 1))

    # Not released yet -> the goal never leaves subtarget 0.
    s_new = torch.where(released_new, torch.clamp(s_new, max=float(K - 1)), torch.zeros_like(s_new))
    return s_new, released_new, pulse
