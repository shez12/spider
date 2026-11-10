"""Continuous multi-subtarget goal trajectories for SimToolReal evaluation.

Training samples a *new random* goal every time the hand reaches the current
one (``goal_sampling_type: delta`` — a bounded random walk inside
``reset.target_volume_*``).  For evaluation we instead want ONE continuous
trajectory that the hand has to follow: ``K`` subtargets sampled along a
single smooth curve, with segments that are deliberately **dense** (small
arc-length spacing) and segments that are **sparse** (large spacing), so the
same rollout exercises both fine-grained tracking and long jumps.

Pipeline
--------
    1. sample ``num_waypoints`` control points inside the target volume
       (random walk with momentum, reflected at the walls → smooth curve)
    2. fit a Catmull-Rom spline through them and build an arc-length table
    3. split the arc length into ``num_segments`` chunks of random relative
       length and tag them alternating dense / sparse
    4. distribute the ``K - 1`` inter-subtarget intervals across the chunks
       proportionally to ``length / desired_spacing`` → each chunk keeps its
       own spacing, and the total is exactly ``K`` subtargets
    5. walk the arc length at each chunk's spacing and read off positions;
       orientations come from slerping a bounded random quaternion chain
       defined on the same spline parameter

Output format matches ``ResetCfg.fixed_trajectory_file`` (see
``reset_utils.allocate_state_buffers``): a JSON dict with

    pos        : (N, K, 3) float  env-local goal positions
    quat_wxyz  : (N, K, 4) float  goal orientations, (w, x, y, z)

plus an optional ``meta`` block (segment lengths / spacings / kinds) that the
env ignores and the eval tooling uses for plotting.

Pure torch — no Isaac imports — so the generator can be unit-tested without
booting a Kit app.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrajectorySpec:
    """Knobs for :func:`sample_trajectory`.

    Defaults match ``ResetCfg``'s target volume
    (``target_volume_mins=(-0.35, -0.2, 0.6)``, ``maxs=(0.35, 0.2, 0.95)``).
    """

    mins: tuple[float, float, float] = (-0.35, -0.2, 0.6)
    maxs: tuple[float, float, float] = (0.35, 0.2, 0.95)
    # Shrink the sampling box by this much on every side so the spline (which
    # can overshoot its control polygon) still lands inside `mins`/`maxs`.
    margin: float = 0.05

    num_subtargets: int = 40
    num_waypoints: int = 5
    num_segments: int = 4  # alternating dense / sparse chunks

    # Desired arc-length spacing (meters) per segment kind.  When
    # `num_subtargets` is fixed, only the RATIO matters (the realized spacings
    # are a common rescaling that spends exactly `num_subtargets` subtargets on
    # the sampled curve).  Set `derive_num_subtargets=True` to honor the
    # absolute spacings and let the subtarget count follow from the curve.
    dense_spacing: float = 0.05
    sparse_spacing: float = 0.20
    derive_num_subtargets: bool = False
    max_subtargets: int = 256  # cap used when `derive_num_subtargets` is set

    # Max orientation change between two consecutive control points (degrees).
    max_rotation_deg: float = 45.0
    # Start with a dense chunk (False → start sparse).
    dense_first: bool = True

    arc_samples: int = 2000  # spline resolution for the arc-length table
    # Optional per-trajectory override of the dense/sparse pattern, e.g.
    # (True, False, True, False).  Length must equal `num_segments`.
    segment_kinds: tuple[bool, ...] | None = field(default=None)


# ---------------------------------------------------------------------------
# Small torch helpers (kept local so this module stays Isaac-free)
# ---------------------------------------------------------------------------


def _scale_bounds(
    mins: tuple[float, float, float],
    maxs: tuple[float, float, float],
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale workspace bounds about their center (mirrors goal_sampling)."""
    lo = torch.as_tensor(mins, dtype=torch.float32)
    hi = torch.as_tensor(maxs, dtype=torch.float32)
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo) * scale
    return center - half, center + half


def _random_quat(generator: torch.Generator) -> torch.Tensor:
    """Uniformly distributed unit quaternion (wxyz), Shoemake's method."""
    return F.normalize(torch.randn(4, generator=generator), dim=-1)


def _quat_from_angle_axis(angle: float, axis: torch.Tensor) -> torch.Tensor:
    half = 0.5 * angle
    return torch.cat([torch.cos(torch.as_tensor(half)).unsqueeze(0), torch.sin(torch.as_tensor(half)) * axis])


def _quat_mul(q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two wxyz quaternions."""
    w0, x0, y0, z0 = q0.unbind()
    w1, x1, y1, z1 = q1.unbind()
    return torch.stack(
        (
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        )
    )


def _quat_slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Slerp batches of wxyz quaternions; ``t`` is an (N,) tensor in [0, 1]."""
    t = t.unsqueeze(-1)
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    # Take the shorter arc.
    flip = dot < 0.0
    q1 = torch.where(flip, -q1, q1)
    dot = torch.where(flip, -dot, dot).clamp(-1.0, 1.0)

    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    degenerate = sin_theta.abs() < 1e-6
    w0 = torch.where(degenerate, 1.0 - t, torch.sin((1.0 - t) * theta) / sin_theta)
    w1 = torch.where(degenerate, t, torch.sin(t * theta) / sin_theta)
    return F.normalize(w0 * q0 + w1 * q1, dim=-1)


def _catmull_rom(ctrl: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Centripetal-free (uniform) Catmull-Rom through ``ctrl`` ((W, 3)).

    ``u`` is an (M,) tensor in [0, 1] spanning the whole spline.  Endpoints are
    duplicated so the curve starts/ends exactly at ``ctrl[0]`` / ``ctrl[-1]``.
    """
    num_ctrl = ctrl.shape[0]
    num_seg = num_ctrl - 1
    assert num_seg >= 1, "need at least 2 control points"

    x = u.clamp(0.0, 1.0) * num_seg
    idx = torch.clamp(x.floor().long(), 0, num_seg - 1)
    f = (x - idx.to(x.dtype)).unsqueeze(-1)

    p0 = ctrl[torch.clamp(idx - 1, 0, num_ctrl - 1)]
    p1 = ctrl[idx]
    p2 = ctrl[idx + 1]
    p3 = ctrl[torch.clamp(idx + 2, 0, num_ctrl - 1)]
    return 0.5 * (
        2.0 * p1
        + (p2 - p0) * f
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * f**2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * f**3
    )


def _arc_length_table(ctrl: torch.Tensor, num_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (u_grid, cumulative arc length) sampled uniformly in ``u``."""
    u_grid = torch.linspace(0.0, 1.0, num_samples)
    pts = _catmull_rom(ctrl, u_grid)
    seg = (pts[1:] - pts[:-1]).norm(dim=-1)
    cum = torch.cat([torch.zeros(1, dtype=seg.dtype), torch.cumsum(seg, dim=0)])
    return u_grid, cum


def _u_from_arc_length(u_grid: torch.Tensor, cum: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Invert the arc-length table: arc length ``s`` (N,) → spline param ``u``."""
    total = cum[-1].clamp(min=1e-8)
    s = s.clamp(0.0, total)
    idx = torch.searchsorted(cum, s, right=True).clamp(1, cum.numel() - 1)
    s0 = cum[idx - 1]
    s1 = cum[idx]
    w = ((s - s0) / (s1 - s0).clamp(min=1e-8)).clamp(0.0, 1.0)
    return u_grid[idx - 1] + w * (u_grid[idx] - u_grid[idx - 1])


def _allocate_counts(weights: torch.Tensor, total: int) -> torch.Tensor:
    """Integer counts ∝ ``weights``, each >= 1, summing exactly to ``total``."""
    n = weights.numel()
    if total < n:
        raise ValueError(f"cannot split {total} intervals across {n} segments")

    scaled = weights / weights.sum().clamp(min=1e-8) * float(total)
    counts = scaled.floor().long().clamp(min=1)
    remainder = scaled - counts.float()
    deficit = total - int(counts.sum())

    order = torch.argsort(remainder, descending=True).tolist()
    if deficit > 0:
        for k in range(deficit):
            counts[order[k % n]] += 1
    else:
        while deficit < 0:
            moved = False
            for i in order:
                if counts[i] > 1:
                    counts[i] -= 1
                    deficit += 1
                    moved = True
                    if deficit == 0:
                        break
            if not moved:  # pragma: no cover - guarded by the `total < n` check
                break
    return counts


# ---------------------------------------------------------------------------
# Trajectory sampling
# ---------------------------------------------------------------------------


def _sample_control_points(spec: TrajectorySpec, generator: torch.Generator) -> torch.Tensor:
    """Random walk with momentum inside the (shrunk) target volume."""
    lo, hi = _scale_bounds(spec.mins, spec.maxs, 1.0)
    lo = lo + spec.margin
    hi = hi - spec.margin
    if (hi <= lo).any():
        raise ValueError(f"margin={spec.margin} is too large for target volume {spec.mins}..{spec.maxs}")

    step = (hi - lo).norm() / max(1, spec.num_waypoints - 1)
    pts = [lo + (hi - lo) * torch.rand(3, generator=generator)]
    direction = F.normalize(torch.randn(3, generator=generator), dim=-1)
    for _ in range(spec.num_waypoints - 1):
        # Momentum keeps consecutive steps correlated → a smooth path instead
        # of a zig-zag between independent random points.
        direction = F.normalize(direction + 0.8 * torch.randn(3, generator=generator), dim=-1)
        scale = 0.6 + 0.8 * float(torch.rand((), generator=generator))
        nxt = pts[-1] + direction * step * scale
        clamped = torch.clamp(nxt, lo, hi)
        # Reflect the direction wherever we bumped into a wall.
        direction = torch.where((clamped != nxt).float() > 0.0, -direction, direction)
        pts.append(clamped)
    return torch.stack(pts)


def _sample_orientations(spec: TrajectorySpec, generator: torch.Generator) -> torch.Tensor:
    """A bounded random quaternion chain — one orientation per control point."""
    quats = [_random_quat(generator)]
    max_angle = math.radians(spec.max_rotation_deg)
    for _ in range(spec.num_waypoints - 1):
        axis = F.normalize(torch.randn(3, generator=generator), dim=-1)
        angle = max_angle * float(torch.rand((), generator=generator))
        quats.append(_quat_mul(_quat_from_angle_axis(angle, axis), quats[-1]))
    return torch.stack(quats)


def sample_trajectory(
    spec: TrajectorySpec,
    generator: torch.Generator | None = None,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Sample one continuous trajectory.

    Returns ``(pos, quat, meta)`` with ``pos`` (K, 3) and ``quat`` (K, 4)
    (wxyz), both env-local, plus a ``meta`` dict describing the segments.
    """
    generator = generator or torch.Generator()
    num_subtargets = int(spec.num_subtargets)
    if not spec.derive_num_subtargets and num_subtargets < 2:
        raise ValueError(f"num_subtargets must be >= 2, got {num_subtargets}")
    num_segments = int(spec.num_segments)
    if num_segments < 1:
        raise ValueError(f"num_segments must be >= 1, got {num_segments}")

    ctrl = _sample_control_points(spec, generator)
    orientations = _sample_orientations(spec, generator)

    u_grid, cum = _arc_length_table(ctrl, int(spec.arc_samples))
    total_length = float(cum[-1].item())
    if total_length < 1e-6:  # pragma: no cover - degenerate control points
        raise RuntimeError("sampled trajectory has zero arc length")

    # --- split the arc length into segments of random relative length ---
    frac = torch.rand(num_segments, generator=generator) + 0.5  # in [0.5, 1.5]
    frac = frac / frac.sum()
    seg_lengths = frac * total_length

    kinds = list(spec.segment_kinds) if spec.segment_kinds is not None else [
        (spec.dense_first == (i % 2 == 0)) for i in range(num_segments)
    ]
    if len(kinds) != num_segments:
        raise ValueError(f"segment_kinds has {len(kinds)} entries, expected {num_segments}")
    desired = torch.tensor(
        [spec.dense_spacing if dense else spec.sparse_spacing for dense in kinds],
        dtype=torch.float32,
    )

    if spec.derive_num_subtargets:
        # Honor the absolute spacings: the curve length decides how many
        # subtargets it takes, capped at `max_subtargets`.
        natural = torch.clamp((seg_lengths / desired).round().long(), min=1)
        derived = min(int(natural.sum().item()) + 1, int(spec.max_subtargets))
        num_subtargets = max(derived, num_segments + 1)
    counts = _allocate_counts(seg_lengths / desired, num_subtargets - 1)
    num_subtargets = int(counts.sum().item()) + 1

    # --- walk the arc length, one spacing per segment ---
    starts = torch.cat([torch.zeros(1), torch.cumsum(seg_lengths, dim=0)[:-1]])
    arc_lengths = [torch.zeros(1)]
    for start, length, count in zip(starts.tolist(), seg_lengths.tolist(), counts.tolist()):
        if count <= 0:
            continue
        spacing = length / count
        arc_lengths.append(start + torch.arange(1, count + 1, dtype=torch.float32) * spacing)

    s = torch.cat(arc_lengths)
    if s.numel() != num_subtargets:  # pragma: no cover - allocation invariant
        raise RuntimeError(f"produced {s.numel()} subtargets, expected {num_subtargets}")
    s[-1] = min(s[-1].item(), total_length)

    u = _u_from_arc_length(u_grid, cum, s)
    pos = _catmull_rom(ctrl, u)
    # Safety net: keep the trajectory inside the configured target volume.
    lo, hi = _scale_bounds(spec.mins, spec.maxs, 1.0)
    pos = torch.clamp(pos, lo, hi)

    # --- orientation: slerp the quaternion chain along the same parameter ---
    t = u.clamp(0.0, 1.0) * (spec.num_waypoints - 1)
    idx = torch.clamp(t.floor().long(), 0, spec.num_waypoints - 2)
    frac_t = t - idx.to(t.dtype)
    quat = _quat_slerp(orientations[idx], orientations[idx + 1], frac_t)
    quat = F.normalize(quat, dim=-1)

    meta = {
        "num_subtargets": num_subtargets,
        "total_length_m": total_length,
        "segment_kinds": ["dense" if dense else "sparse" for dense in kinds],
        "segment_lengths_m": [float(v) for v in seg_lengths.tolist()],
        "segment_subtarget_counts": [int(v) for v in counts.tolist()],
        "segment_spacing_m": [
            float(length / count) if count > 0 else 0.0
            for length, count in zip(seg_lengths.tolist(), counts.tolist())
        ],
        "control_points": ctrl.tolist(),
        "volume_mins": list(spec.mins),
        "volume_maxs": list(spec.maxs),
    }
    return pos.to(device), quat.to(device), meta


def generate_trajectory_pool(
    spec: TrajectorySpec,
    num_trajectories: int = 1,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    """Sample ``num_trajectories`` independent trajectories.

    Returns ``(pos, quat, metas)`` shaped (N, K, 3) and (N, K, 4).  ``K`` is
    identical across the pool, which is what the ``fixed_trajectory_file``
    loader in ``reset_utils`` requires — so with
    ``spec.derive_num_subtargets=True`` the first trajectory decides ``K`` and
    the remaining ones reuse it (their spacings get rescaled accordingly).
    """
    if num_trajectories < 1:
        raise ValueError(f"num_trajectories must be >= 1, got {num_trajectories}")
    positions, quaternions, metas = [], [], []
    for i in range(num_trajectories):
        generator = torch.Generator()
        generator.manual_seed(int(seed) + i)
        pos, quat, meta = sample_trajectory(spec, generator=generator, device=device)
        if spec.derive_num_subtargets and i == 0:
            spec = replace(spec, num_subtargets=pos.shape[0], derive_num_subtargets=False)
        positions.append(pos)
        quaternions.append(quat)
        metas.append(meta)
    return torch.stack(positions), torch.stack(quaternions), metas


# ---------------------------------------------------------------------------
# Serialization / inspection helpers
# ---------------------------------------------------------------------------


def write_trajectory_json(
    path: str | Path,
    pos: torch.Tensor,
    quat_wxyz: torch.Tensor,
    metas: list[dict] | None = None,
) -> Path:
    """Write the ``fixed_trajectory_file`` payload (extra keys are ignored)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "pos": [[p.tolist() for p in traj] for traj in pos],
        "quat_wxyz": [[q.tolist() for q in traj] for traj in quat_wxyz],
    }
    if metas is not None:
        payload["meta"] = metas
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def subtarget_spacings(pos: torch.Tensor) -> torch.Tensor:
    """Consecutive subtarget distances, (K - 1,) meters."""
    return (pos[1:] - pos[:-1]).norm(dim=-1)


def subtarget_kinds(meta: dict, num_subtargets: int) -> list[str]:
    """Per-subtarget "dense"/"sparse" label (interval i is labeled by its end)."""
    kinds: list[str] = ["dense"]
    for kind, count in zip(meta["segment_kinds"], meta["segment_subtarget_counts"]):
        kinds.extend([kind] * int(count))
    kinds = kinds[:num_subtargets]
    if len(kinds) < num_subtargets:  # pragma: no cover - defensive pad
        kinds.extend([kinds[-1]] * (num_subtargets - len(kinds)))
    return kinds


__all__ = [
    "TrajectorySpec",
    "sample_trajectory",
    "generate_trajectory_pool",
    "write_trajectory_json",
    "subtarget_spacings",
    "subtarget_kinds",
]
