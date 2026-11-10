"""State allocation and reset helpers for SimToolReal."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from isaaclab.utils.math import random_orientation

from .action_utils import sample_log_uniform
from .goal_sampling import sample_absolute_goal_pose, sample_delta_goal_pose
from .obs_utils import KEYPOINT_CORNERS
from ..robot_spec import KUKA_SHARPA_SPEC, match_robot_names, normalize_robot_name

def allocate_state_buffers(env) -> None:
    """Populate every per-env buffer + index cache used by the hooks.

    Called once from ``__init__`` after ``super().__init__`` (which runs
    ``_setup_scene`` and makes ``env.robot``/``env.object``/``env.goal_viz``
    available).
    """
    dr = env.cfg.domain_randomization
    rew = env.cfg.reward

    spec = getattr(env, "_robot_spec", KUKA_SHARPA_SPEC)
    env._robot_spec = spec
    num_fingertips = spec.num_fingertips

    # --- Joint/body id caches ---
    env._arm_joint_ids = env.robot.find_joints(spec.arm_joint_regex)[0]
    env._hand_joint_ids = env.robot.find_joints(spec.hand_joint_regex)[0]
    env._palm_body_id = env.robot.find_bodies(spec.palm_body_name)[0][0]
    fingertip_names = match_robot_names(
        spec.fingertip_link_names, list(env.robot.data.body_names)
    )
    env._fingertip_body_ids = env.robot.find_bodies(
        fingertip_names, preserve_order=True
    )[0]
    assert len(env._arm_joint_ids) == spec.num_arm_joints
    assert len(env._hand_joint_ids) == spec.num_hand_joints
    assert len(env._fingertip_body_ids) == num_fingertips

    # Convert between Lab parser order and canonical policy order.
    lab_names = list(env.robot.data.joint_names)
    canonical = spec.joint_names_canonical
    lab_norm = [normalize_robot_name(n) for n in lab_names]
    canon_norm = [normalize_robot_name(n) for n in canonical]
    assert set(lab_norm) == set(canon_norm), (
        f"Lab joints {lab_names} != canonical {list(canonical)}"
    )
    env._perm_canon_to_lab = torch.tensor(
        [canon_norm.index(normalize_robot_name(n)) for n in lab_names],
        device=env.device, dtype=torch.long,
    )
    env._perm_lab_to_canon = torch.tensor(
        [lab_norm.index(normalize_robot_name(n)) for n in canonical],
        device=env.device, dtype=torch.long,
    )

    limits = env.robot.data.joint_pos_limits  # (N, num_joints, 2), Lab order

    # Canonical-order limits for normalizing joint_pos observations.
    env._joint_lower_canon = limits[0, :, 0][env._perm_lab_to_canon]  # (29,)
    env._joint_upper_canon = limits[0, :, 1][env._perm_lab_to_canon]  # (29,)

    # Lab-order limits for action target clamping.
    env._arm_lower = limits[:, env._arm_joint_ids, 0]
    env._arm_upper = limits[:, env._arm_joint_ids, 1]
    env._hand_lower = limits[:, env._hand_joint_ids, 0]
    env._hand_upper = limits[:, env._hand_joint_ids, 1]

    # --- Action target buffers  ---
    action_space = env.cfg.action_space
    env._cur_targets = torch.zeros(env.num_envs, action_space, device=env.device)
    env._prev_targets = torch.zeros(env.num_envs, action_space, device=env.device)

    # --- Keypoint offsets in object local frame ---
    corners = torch.tensor(
        KEYPOINT_CORNERS, device=env.device, dtype=torch.float32
    )  # (4, 3)

    env._keypoint_offsets = (
        corners.unsqueeze(0)
        * (env._object_scale_per_env * rew.object_base_size * rew.keypoint_scale * 0.5).unsqueeze(1)
    )  # (N, 4, 3)

    # Fixed-size reward keypoints follow legacy: fixed_size * keypoint_scale / 2.
    env._keypoint_offsets_fixed = (
        corners
        * (0.5 * rew.keypoint_scale * torch.tensor(rew.fixed_size, device=env.device)).unsqueeze(0)
    ).unsqueeze(0).expand(env.num_envs, -1, -1).contiguous()

    # --- Per-env DR priors (re-sampled on reset; seeded once here) ---
    lo, hi = dr.object_scale_noise_multiplier_range
    env._object_scale_multiplier = torch.empty(
        env.num_envs, 3, device=env.device
    ).uniform_(lo, hi)

    # --- Reward / termination trackers  ---
    # whether the object is lifted
    env._lifted_object = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    # the maximum distance between the object and the goal since the last goal reset
    env._closest_keypoint_max_dist = torch.full(
        (env.num_envs,), -1.0, device=env.device
    )
    # the minimum distance between a fingertip and the object since the last goal reset
    env._closest_fingertip_dist = torch.full(
        (env.num_envs, num_fingertips), -1.0, device=env.device
    )
    # the number of succeesses in the current episode
    env._successes = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.device
    )
    # the number of consecutive steps that the object is near the goal
    env._near_goal_steps = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.device
    )

    # --- Tolerance curriculum state ---
    env._current_success_tolerance: float = env.cfg.termination.success_tolerance
    env._prev_episode_successes = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.device
    )
    env._frame_counter: int = 0
    env._last_curriculum_update: int = 0

    # --- Object lifted-reward reference z (updated on each _reset_object_pose) ---
    init_z = env.cfg.reset.table_reset_z + env.cfg.reset.table_object_z_offset
    env._object_init_z = torch.full(
        (env.num_envs,), init_z, device=env.device
    )

    # --- Per-env table surface z (randomized in _reset_table_pose) ---
    env._table_z_per_env = torch.full(
        (env.num_envs,), env.cfg.reset.table_reset_z, device=env.device
    )

    # --- DR rolling buffers ---
    env._object_state_queue = torch.zeros(
        env.num_envs,
        max(1, dr.object_state_delay_max),
        13,  # pos(3) + quat(4) + lin_vel(3) + ang_vel(3)
        device=env.device,
    )
    env._obs_queue = torch.zeros(
        env.num_envs,
        max(1, dr.obs_delay_max),
        env.cfg.observation_space,
        device=env.device,
    )
    env._action_queue = torch.zeros(
        env.num_envs,
        max(1, dr.action_delay_max),
        action_space,
        device=env.device,
    )

    # --- Wrench DR state (Phase C/D) ---
    env._random_force_prob = sample_log_uniform(
        dr.force_prob_range, env.num_envs
    ).to(env.device)
    env._random_torque_prob = sample_log_uniform(
        dr.torque_prob_range, env.num_envs
    ).to(env.device)
    env._object_forces = torch.zeros(env.num_envs, 1, 3, device=env.device)
    env._object_torques = torch.zeros(env.num_envs, 1, 3, device=env.device)
    env._object_mass = env.object.data.default_mass[:, 0:1].to(env.device)  # (N, 1)

    # --- Step-shared caches populated by compute_intermediate_values (Phase F) ---
    env._keypoints_max_dist = torch.zeros(env.num_envs, device=env.device)
    env._curr_fingertip_distances = torch.zeros(
        env.num_envs, num_fingertips, device=env.device
    )
    env._near_goal = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    env._is_success = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )

    # --- Fixed-trajectory pool (trajectory_count ablation) ---
    # Loaded once at init from the JSON file. Different runs share the same
    # file and pick the first N via cfg.reset.fixed_trajectory_count
    # (0 = take all). Empty filename = ablation disabled (baseline path).
    if env.cfg.reset.fixed_trajectory_file:
        path = Path(env.cfg.reset.fixed_trajectory_file)
        with open(path) as f:
            payload = json.load(f)
        all_pos = torch.tensor(payload["pos"], device=env.device, dtype=torch.float32)
        all_quat = torch.tensor(payload["quat_wxyz"], device=env.device, dtype=torch.float32)
        n_total = all_pos.shape[0]
        n_take = env.cfg.reset.fixed_trajectory_count or n_total
        if n_take > n_total:
            raise ValueError(
                f"fixed_trajectory_count={n_take} exceeds pool size {n_total} in {path}"
            )
        env._fixed_traj_pos = all_pos[:n_take].contiguous()    # (N, K, 3)
        env._fixed_traj_quat = all_quat[:n_take].contiguous()  # (N, K, 4)
        env._traj_id = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._traj_step = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        # Continuous-goal state: fractional position along the subtarget chain,
        # plus whether the object has taken subtarget 0 yet (the goal stays
        # parked on it until then).
        env._traj_s = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        env._goal_released = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._goal_speed = torch.full(
            (env.num_envs,), float(env.cfg.reset.goal_speed_mps), device=env.device
        )
        env._crossed_pulse = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # Continuous goal only makes sense on top of a stored trajectory.
    env._continuous_goal_mode = bool(
        env.cfg.reset.continuous_goal and env.cfg.reset.fixed_trajectory_file
    )
    # Hammering: periodic up/down goal around the object's start pose.
    env._hammering_mode = bool(env.cfg.reset.hammering_enabled)
    if env._hammering_mode:
        env._hammer_phase = torch.zeros(env.num_envs, device=env.device)
        env._hammer_at_bottom = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._hammer_strikes = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._hammer_done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._hammer_base_pos = torch.zeros(env.num_envs, 3, device=env.device)
        env._hammer_base_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(
            env.num_envs, 1
        )
        # Hammering does not use subtarget successes at all.
        env._crossed_pulse = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # set the reward buffer to 0
    env.reward_buf = torch.zeros(env.num_envs, device=env.device)


def _randomize_robot_dof_state(env, env_ids: torch.Tensor) -> None:
    """Reset DOF state and seed previous targets from the reset pose."""
    cfg = env.cfg.reset
    default_pos = env.robot.data.default_joint_pos[env_ids]  # (n, num_dofs)
    lower = env.robot.data.joint_pos_limits[env_ids, :, 0]
    upper = env.robot.data.joint_pos_limits[env_ids, :, 1]

    reset_scale = torch.zeros_like(default_pos)
    reset_scale[:, env._arm_joint_ids] = cfg.reset_dof_pos_random_interval_arm
    reset_scale[:, env._hand_joint_ids] = cfg.reset_dof_pos_random_interval_fingers

    sampled_pos = lower + (upper - lower) * torch.rand_like(default_pos)
    joint_pos = torch.lerp(default_pos, sampled_pos, reset_scale).clamp(lower, upper)
    joint_vel = torch.empty_like(default_pos).uniform_(
        -cfg.reset_dof_vel_random_interval,
        cfg.reset_dof_vel_random_interval,
    )

    env.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    env._prev_targets[env_ids] = joint_pos
    env._cur_targets[env_ids] = joint_pos


def _reset_table_pose(env, env_ids: torch.Tensor) -> None:
    """Randomize the table's pose per env and write the new transform.

    Z position is required (the policy uses `_table_z_per_env` to set the
    object init height). XY position and yaw are gated on
    `table_reset_xy_range_m` / `table_reset_yaw_range_deg` — default ranges
    are zero so this is a no-op for runs that don't opt in.
    """
    cfg = env.cfg.reset
    n = env_ids.numel()
    env_origins = env.scene.env_origins[env_ids]

    dz = torch.empty(n, device=env.device).uniform_(
        -cfg.table_reset_z_range, cfg.table_reset_z_range
    )
    table_z = cfg.table_reset_z + dz
    env._table_z_per_env[env_ids] = table_z

    pos_local = torch.zeros(n, 3, device=env.device)
    pos_local[:, 2] = table_z

    # XY position noise — per-env independent uniform half-widths.
    xy_range = tuple(float(v) for v in cfg.table_reset_xy_range_m)
    if xy_range[0] > 0.0 or xy_range[1] > 0.0:
        rx = torch.empty(n, device=env.device).uniform_(-xy_range[0], xy_range[0])
        ry = torch.empty(n, device=env.device).uniform_(-xy_range[1], xy_range[1])
        pos_local[:, 0] = rx
        pos_local[:, 1] = ry

    # Yaw noise — sample uniform [-r, r] degrees, build a z-axis rotation quat.
    yaw_range_deg = float(cfg.table_reset_yaw_range_deg)
    if yaw_range_deg > 0.0:
        yaw_rad = (
            torch.empty(n, device=env.device).uniform_(-1.0, 1.0)
            * yaw_range_deg * (torch.pi / 180.0)
        )
        half = yaw_rad * 0.5
        w = torch.cos(half)
        z = torch.sin(half)
        quat = torch.stack([w, torch.zeros_like(w), torch.zeros_like(w), z], dim=-1)
    else:
        quat = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=env.device, dtype=torch.float32
        ).unsqueeze(0).expand(n, -1)

    pose = torch.cat([pos_local + env_origins, quat], dim=-1)
    env.table.write_root_pose_to_sim(pose, env_ids=env_ids)

def _reset_object_pose(env, env_ids: torch.Tensor) -> None:
    """Reset object pose and lifted-reward reference height."""
    cfg = env.cfg.reset
    n = env_ids.numel()
    env_origins = env.scene.env_origins[env_ids]

    if cfg.fixed_start_pose is not None:
        fixed = torch.as_tensor(cfg.fixed_start_pose, device=env.device, dtype=torch.float32)
        pos_local = fixed[:3].unsqueeze(0).expand(n, -1)
        quat = fixed[3:].unsqueeze(0).expand(n, -1)
    else:
        noise = torch.empty(n, 3, device=env.device).uniform_(-1.0, 1.0)
        pos_local = torch.stack(
            (
                noise[:, 0] * cfg.reset_position_noise_x,
                noise[:, 1] * cfg.reset_position_noise_y,
                env._table_z_per_env[env_ids]
                + cfg.table_object_z_offset
                + noise[:, 2] * cfg.reset_position_noise_z,
            ),
            dim=-1,
        )
        quat = random_orientation(n, device=env.device)

    pose = torch.cat([pos_local + env_origins, quat], dim=-1)
    env.object.write_root_pose_to_sim(pose, env_ids=env_ids)
    env.object.write_root_velocity_to_sim(
        torch.zeros(n, 6, device=env.device), env_ids=env_ids
    )

    env._object_init_z[env_ids] = pos_local[:, 2]


def _reset_goal_pose(env, env_ids: torch.Tensor, mode: str) -> None:
    """Resample the goal pose and write it to GoalViz."""
    cfg = env.cfg.reset
    n = env_ids.numel()
    env_origins = env.scene.env_origins[env_ids]

    if cfg.fixed_goal_pose is not None:
        fixed = torch.as_tensor(cfg.fixed_goal_pose, device=env.device, dtype=torch.float32)
        new_pos_local = fixed[:3].unsqueeze(0).expand(n, -1)
        new_quat = fixed[3:].unsqueeze(0).expand(n, -1)
        pose = torch.cat([new_pos_local + env_origins, new_quat], dim=-1)
        env.goal_viz.write_root_pose_to_sim(pose, env_ids=env_ids)
        return

    # Fixed-trajectory ablation: ignore goal_sampling_type, draw from the
    # pre-loaded pool. mode acts as the "hard reset vs intra-episode" signal:
    #   "absolute" → fresh episode → pick new traj_id, reset step to 0
    #   anything else → intra-episode goal-hit → advance step
    # The K-th success briefly calls this with step=K (out of bounds) before
    # max_consecutive_successes terminates the episode; the clamp keeps the
    # lookup safe — the goal we write is then immediately overwritten by the
    # full reset.
    if cfg.fixed_trajectory_file:
        K = env._fixed_traj_pos.shape[1]
        continuous = getattr(env, "_continuous_goal_mode", False)
        if mode == "absolute":
            n_traj = env._fixed_traj_pos.shape[0]
            env._traj_id[env_ids] = torch.randint(
                0, n_traj, (n,), device=env.device, dtype=torch.long,
            )
            env._traj_step[env_ids] = 0
            if continuous:
                env._traj_s[env_ids] = 0.0
                env._goal_released[env_ids] = False
        else:
            if continuous:
                # advance_continuous_goal owns the goal position in this mode.
                return
            env._traj_step[env_ids] = env._traj_step[env_ids] + 1

        if continuous:
            _write_goal_at_s(env, env_ids)
            return

        traj_id = env._traj_id[env_ids]
        step_clamped = env._traj_step[env_ids].clamp(max=K - 1)
        new_pos_local = env._fixed_traj_pos[traj_id, step_clamped]
        new_quat = env._fixed_traj_quat[traj_id, step_clamped]
        pose = torch.cat([new_pos_local + env_origins, new_quat], dim=-1)
        env.goal_viz.write_root_pose_to_sim(pose, env_ids=env_ids)
        return

    if mode == "delta":
        prev_pos_local = env.goal_viz.data.root_pos_w[env_ids] - env_origins
        prev_quat = env.goal_viz.data.root_quat_w[env_ids]
        new_pos_local, new_quat = sample_delta_goal_pose(
            prev_pos=prev_pos_local,
            prev_quat_wxyz=prev_quat,
            delta_distance=cfg.delta_goal_distance,
            delta_rotation_degrees=cfg.delta_rotation_degrees,
            mins=cfg.target_volume_mins,
            maxs=cfg.target_volume_maxs,
            scale=cfg.target_volume_region_scale,
        )
    elif mode == "absolute":
        new_pos_local, new_quat = sample_absolute_goal_pose(
            mins=cfg.target_volume_mins,
            maxs=cfg.target_volume_maxs,
            scale=cfg.target_volume_region_scale,
            n_envs=n,
            device=env.device,
        )
    else:
        raise ValueError(f"unknown goal sampling mode: {mode}")

    pose = torch.cat([new_pos_local + env_origins, new_quat], dim=-1)
    env.goal_viz.write_root_pose_to_sim(pose, env_ids=env_ids)


def _clear_goal_trackers(env, env_ids: torch.Tensor) -> None:
    env._closest_keypoint_max_dist[env_ids] = -1.0
    env._closest_fingertip_dist[env_ids] = -1.0
    env._near_goal_steps[env_ids] = 0


def _slerp_quat(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Shortest-path quaternion slerp (wxyz), t broadcastable to (N, 1)."""
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


def _write_goal_at_s(env, env_ids: torch.Tensor) -> None:
    """Write GoalViz at the (fractional) position ``_traj_s`` on the trajectory."""
    K = env._fixed_traj_pos.shape[1]
    traj_id = env._traj_id[env_ids]
    s = env._traj_s[env_ids]
    i0 = torch.clamp(torch.floor(s).long(), max=K - 2)
    frac = (s - i0.float()).clamp(0.0, 1.0).unsqueeze(-1)
    p0 = env._fixed_traj_pos[traj_id, i0]
    p1 = env._fixed_traj_pos[traj_id, i0 + 1]
    q0 = env._fixed_traj_quat[traj_id, i0]
    q1 = env._fixed_traj_quat[traj_id, i0 + 1]
    pos_local = p0 + (p1 - p0) * frac
    quat = _slerp_quat(q0, q1, frac)
    pose = torch.cat([pos_local + env.scene.env_origins[env_ids], quat], dim=-1)
    env.goal_viz.write_root_pose_to_sim(pose, env_ids=env_ids)


def advance_continuous_goal(env) -> None:
    """Slide the goal along the trajectory at the commanded speed.

    Leader-follower: the goal advances by ``goal_speed * dt`` of arc length, but
    it never runs away — it parks at the next subtarget until the object is
    inside the success tolerance (keypoint distance, same criterion the task
    already uses).  A slow command throttles the object; a fast one is capped by
    how well the policy can keep up.

    Sets ``_crossed_pulse`` (one step high when a subtarget is cleared) which is
    what counts as a success in this mode.
    """
    if not getattr(env, "_continuous_goal_mode", False):
        return

    from .continuous_goal import continuous_goal_update  # pure-torch core

    device = env.device
    K = env._fixed_traj_pos.shape[1]
    traj_id = env._traj_id
    rew_cfg = env.cfg.reward

    kp_offsets = (
        env._keypoint_offsets_fixed if rew_cfg.fixed_size_keypoint_reward else env._keypoint_offsets
    )
    kp_offsets = kp_offsets * env._object_scale_multiplier.unsqueeze(1)
    if hasattr(env, "_keypoint_success_tolerance_m"):
        running_tol = env._keypoint_success_tolerance_m()
    else:
        running_tol = torch.full(
            (env.num_envs,), env._current_success_tolerance, device=device
        )

    s_new, released_new, pulse = continuous_goal_update(
        traj_pos=env._fixed_traj_pos,
        traj_quat=env._fixed_traj_quat,
        traj_id=traj_id,
        s=env._traj_s,
        released=env._goal_released,
        goal_speed=env._goal_speed,
        obj_pos=env.object.data.root_pos_w - env.scene.env_origins,
        obj_quat=env.object.data.root_quat_w,
        kp_offsets=kp_offsets,
        dt=float(getattr(env, "step_dt", 1.0 / 60.0)),
        keypoint_scale=float(rew_cfg.keypoint_scale),
        first_tolerance=env.cfg.termination.first_target_tolerance,
        running_tolerance=running_tol,
    )

    env._traj_s = s_new
    env._goal_released = released_new
    env._crossed_pulse = pulse
    # Keep the discrete index in sync for logging / visualisation.
    env._traj_step = torch.clamp(torch.floor(env._traj_s).long(), max=K - 1)
    _write_goal_at_s(env, torch.arange(env.num_envs, device=device))


def advance_hammering_goal(env) -> None:
    """Oscillate the goal along one axis so the object pounds up and down.

    Counts strikes (one per cycle, at the bottom of the stroke) in
    ``env._hammer_strikes`` and marks the episode done in
    ``env._hammer_done`` once ``hammering_cycles`` is reached.
    """
    from .continuous_goal import hammering_offset, hammering_pose

    cfg = env.cfg.reset
    device = env.device
    dt = float(getattr(env, "step_dt", 1.0 / 60.0))
    axes = {"x": 0, "y": 1, "z": 2}
    axis = axes.get(str(cfg.hammering_axis).lower(), 2)

    # Before the object is in hand, keep the goal at the base pose so the policy
    # can grab it instead of chasing a moving target.  Give up waiting after
    # ~10 s so a policy that never lifts still gets to move.
    active = env._lifted_object
    if cfg.hammering_wait_for_lift:
        timed_out = env.episode_length_buf > 600
        active = active | timed_out
        waiting = (~active).nonzero(as_tuple=False).squeeze(-1)
        if waiting.numel():
            env._hammer_phase[waiting] = 0.0

    prev_phase = env._hammer_phase.clone()
    step = dt * float(cfg.hammering_frequency_hz)
    phase = env._hammer_phase + torch.where(
        active, torch.full_like(env._hammer_phase, step), torch.zeros_like(env._hammer_phase)
    )
    wrapped = phase >= 1.0
    phase = torch.where(wrapped, phase - 1.0, phase)
    env._hammer_phase = phase

    offset, at_bottom = hammering_offset(
        phase,
        cfg.hammering_amplitude_m,
        cfg.hammering_down_fraction,
        prev_phase=prev_phase,
        down_exponent=float(cfg.hammering_down_exponent),
        up_exponent=float(cfg.hammering_up_exponent),
    )

    # Strike counter: rising edge of "at the bottom of the stroke".
    new_strike = at_bottom & ~env._hammer_at_bottom & active
    env._hammer_at_bottom = at_bottom
    env._hammer_strikes = env._hammer_strikes + new_strike.long()
    if cfg.hammering_cycles > 0:
        env._hammer_done = env._hammer_strikes >= int(cfg.hammering_cycles)

    # The stroke translation moves the pivot; a separate (usually smaller)
    # amplitude keeps the hand from travelling the whole stroke, letting the
    # wrist rotation supply the rest of the head's arc.
    pivot_amp = cfg.hammering_pivot_amplitude_m
    pivot_amp = float(cfg.hammering_amplitude_m) if pivot_amp is None else float(pivot_amp)
    tri_for_pivot = (-offset / max(float(cfg.hammering_amplitude_m), 1e-6)).clamp(0.0, 1.0)

    base = env._hammer_base_pos
    pos = base.clone()
    pos[:, axis] = pos[:, axis] - pivot_amp * tri_for_pivot

    # The twist rides along with the stroke (tri: 0 at the top, 1 at the bottom).
    tri = (-offset / max(float(cfg.hammering_amplitude_m), 1e-6)).clamp(0.0, 1.0)

    # Pivot = root of the handle.  The URDF lays the handle along +x with the
    # head at +x and the link origin at the handle's centre, so the root is the
    # -x end; half the local x extent of the keypoint box gives its distance.
    kp = (
        env._keypoint_offsets_fixed
        if env.cfg.reward.fixed_size_keypoint_reward
        else env._keypoint_offsets
    )
    kp = kp * env._object_scale_multiplier.unsqueeze(1)
    half_x = kp[:, :, 0].abs().max(dim=1).values
    pivot_local = torch.stack(
        [-half_x, torch.zeros_like(half_x), torch.zeros_like(half_x)], dim=-1
    )

    local_axes = {
        "x": torch.tensor([1.0, 0.0, 0.0], device=env.device),
        "y": torch.tensor([0.0, 1.0, 0.0], device=env.device),
        "z": torch.tensor([0.0, 0.0, 1.0], device=env.device),
    }
    local_axis = local_axes.get(str(cfg.hammering_rotation_axis).lower(), local_axes["y"])
    pos, quat = hammering_pose(
        pos,
        env._hammer_base_quat,
        tri,
        float(cfg.hammering_rotation_deg),
        local_axis,
        pivot_local,
    )

    pose = torch.cat([pos + env.scene.env_origins, quat], dim=-1)
    env.goal_viz.write_root_pose_to_sim(pose)


def reset_goal_trackers(env, env_ids: torch.Tensor) -> None:
    """Clear per-goal trackers and (discrete mode only) sample the next goal."""
    _clear_goal_trackers(env, env_ids)
    if getattr(env, "_continuous_goal_mode", False):
        # The goal keeps sliding along the trajectory — teleporting it to the
        # next subtarget would defeat the whole point.
        return
    _reset_goal_pose(env, env_ids, mode=env.cfg.reset.goal_sampling_type)


def reset_env_state(env, env_ids: torch.Tensor) -> None:
    """Full per-env reset after ``super()._reset_idx``."""
    n = env_ids.numel()

    _randomize_robot_dof_state(env, env_ids)
    _reset_table_pose(env, env_ids)
    _reset_object_pose(env, env_ids)
    # Draw a fresh speed command before the goal is (re)placed.
    if getattr(env, "_continuous_goal_mode", False) and env.cfg.reset.goal_speed_randomize:
        lo, hi = env.cfg.reset.goal_speed_range_mps
        if hi > lo:
            env._goal_speed[env_ids] = torch.empty(n, device=env.device).uniform_(lo, hi)
        else:
            env._goal_speed[env_ids] = float(lo)
    _reset_goal_pose(env, env_ids, mode="absolute")  # full reset → always absolute

    # Hammering oscillates around the pose the object starts at.
    if getattr(env, "_hammering_mode", False):
        base = env.object.data.root_pos_w[env_ids] - env.scene.env_origins[env_ids]
        axes = {"x": 0, "y": 1, "z": 2}
        axis = axes.get(str(env.cfg.reset.hammering_axis).lower(), 2)
        base[:, axis] = base[:, axis] + float(env.cfg.reset.hammering_lift_m)
        env._hammer_base_pos[env_ids] = base
        env._hammer_base_quat[env_ids] = env.object.data.root_quat_w[env_ids]
        env._hammer_phase[env_ids] = 0.0
        env._hammer_strikes[env_ids] = 0
        env._hammer_done[env_ids] = False
        env._hammer_at_bottom[env_ids] = False

    env._prev_episode_successes[env_ids] = env._successes[env_ids]

    _clear_goal_trackers(env, env_ids)
    env._lifted_object[env_ids] = False
    env._successes[env_ids] = 0

    env._action_queue[env_ids] = 0.0
    env._obs_queue[env_ids] = 0.0
    env._object_state_queue[env_ids] = 0.0
    for queue_name in ("_student_camera_queue", "_student_obs_queue"):
        if hasattr(env, queue_name):
            getattr(env, queue_name)[env_ids] = 0.0
    env._object_forces[env_ids] = 0.0
    env._object_torques[env_ids] = 0.0

    dr = env.cfg.domain_randomization
    env._random_force_prob[env_ids] = sample_log_uniform(
        dr.force_prob_range, n
    ).to(env.device)
    env._random_torque_prob[env_ids] = sample_log_uniform(
        dr.torque_prob_range, n
    ).to(env.device)
    lo, hi = dr.object_scale_noise_multiplier_range
    env._object_scale_multiplier[env_ids] = torch.empty(
        n, 3, device=env.device
    ).uniform_(lo, hi)

    # Camera pose randomization (no-op when cfg.student_obs.use_camera_pose_rand is False).
    from .scene_utils import _apply_camera_pose_rand_at_reset
    _apply_camera_pose_rand_at_reset(env, env_ids)


__all__ = [
    "allocate_state_buffers",
    "reset_env_state",
    "reset_goal_trackers",
]
