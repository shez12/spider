"""Versioned warmup acceptance: retain grasp while allowing object orientation diversity.

This is sampled geometric evidence of retention, not a force-closure certificate.
The original strict benchmark and its reports remain unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from examples.arm_hand.audit_right_trajectory import audit_states
from examples.arm_hand.mustard_valid100 import valid

VERSION = 'warmup_grasp_v2'


def grasp_retention(model, trajectory):
    dt = float(trajectory['control_dt'])
    poses = np.asarray(trajectory['qpos'])
    if not np.isclose(dt, .01) or len(poses) < 20 or not np.isfinite(poses).all():
        raise ValueError('Expected finite saved 10 ms frames and at least 200 ms of trajectory')
    data = mujoco.MjData(model)
    object_body = model.body('right_object').id
    explicit = {tuple(sorted((int(a), int(b)))) for a, b in zip(model.pair_geom1, model.pair_geom2)}
    def enabled(a, b):
        return tuple(sorted((a, b))) in explicit or bool(
            model.geom_contype[a] & model.geom_conaffinity[b] or
            model.geom_contype[b] & model.geom_conaffinity[a])
    # Visual geoms have zero masks and no explicit pairs. Never use their convex hulls.
    objects = [g for g in range(model.ngeom) if model.geom_bodyid[g] == object_body
               and (model.geom_contype[g] or model.geom_conaffinity[g] or
                    any(g in pair for pair in explicit))]
    parts = {}
    for g in range(model.ngeom):
        if not (model.geom(g).name or '').startswith('collision_sharpa_'):
            continue
        body = model.body(int(model.geom_bodyid[g])).name or ''
        part = next((p for p in ('thumb', 'index', 'middle', 'ring', 'pinky') if p in body), 'palm')
        if any(enabled(g, o) for o in objects):
            parts.setdefault(part, []).append(g)
    if not objects or 'thumb' not in parts:
        raise ValueError('Expected native right Sharpa hand and object collision geometry')
    floor = model.geom('floor').id
    rows, relative = [], []
    for pose in poses[-20:]:
        data.qpos[:] = pose
        mujoco.mj_fwdPosition(model, data)
        distances = {p: min(mujoco.mj_geomDistance(model, data, h, o, .02, None)
                           for h in geoms for o in objects if enabled(h, o)) for p, geoms in parts.items()}
        near = [p for p, dist in distances.items() if dist <= .001]
        # Thumb opposed by another digit; resting on the palm alone is insufficient.
        supported = 'thumb' in near and any(p in near for p in ('index', 'middle', 'ring', 'pinky'))
        table_clearance = min(mujoco.mj_geomDistance(model, data, o, floor, .02, None) for o in objects)
        palm = data.site('sharpa_palm')
        relative.append(palm.xmat.reshape(3, 3).T @ (data.xpos[object_body] - palm.xpos))
        rows.append(dict(near_parts=near, distances_m=distances,
                         opposed_digit_support=bool(supported), object_table_clearance_m=float(table_clearance)))
    relative = np.asarray(relative)
    span = float(np.max(np.linalg.norm(relative[:, None] - relative[None, :], axis=-1)))
    fraction = float(np.mean([r['opposed_digit_support'] for r in rows]))
    clear = all(r['object_table_clearance_m'] > .001 for r in rows)
    retained = rows[-1]['opposed_digit_support'] and fraction >= .9 and span < .03 and clear
    return dict(retained=bool(retained), evidence='saved-state convex-geometry distances; no force-closure or continuous-time guarantee',
                tail_frames=20, frame_dt_s=dt, digit_distance_tolerance_m=.001,
                minimum_supported_fraction=.9, maximum_relative_span_m=.03,
                supported_fraction=fraction, relative_span_m=span, clear_of_table=bool(clear), frames=rows)


def assess(model, reference, trajectory):
    metrics = audit_states(model, reference, trajectory)
    grasp = grasp_retention(model, trajectory)
    position = metrics['object_rmse_m'] < .03 and metrics['final_object_position_error_m'] < .03
    collision = max(metrics[k] for k in ('max_arm_table_penetration_m', 'max_hand_table_penetration_m',
                                        'max_hand_self_penetration_m')) < .001
    return dict(strict_accepted=valid(metrics), warmup_accepted=bool(position and collision and grasp['retained']),
                metrics=metrics, grasp=grasp)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', required=True, type=Path)
    p.add_argument('--gpu-dir', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    model = mujoco.MjModel.from_xml_path(str(a.case / 'scene.xml'))
    paths = [a.gpu_dir / 'trajectory.npz', a.gpu_dir / 'replay_trajectory.npz']
    with np.load(a.case / 'reference.npz') as ref, np.load(paths[0]) as actual, np.load(paths[1]) as replay:
        for tr in (actual, replay):
            if len(tr['qpos']) != len(ref['object_qpos']) - 1:
                raise ValueError('Reference timeline differs')
            np.testing.assert_array_equal(tr['initial_state'], ref['initial_state'])
        np.testing.assert_array_equal(actual['ctrl'], replay['ctrl'])
        selected = assess(model, ref, actual)
        independent = assess(model, ref, replay)
    inputs = [a.case / 'scene.xml', a.case / 'reference.npz', *paths]
    report = dict(acceptance_version=VERSION, final_orientation_gated=False,
                  strict_accepted=selected['strict_accepted'] and independent['strict_accepted'],
                  warmup_accepted=selected['warmup_accepted'] and independent['warmup_accepted'],
                  admitted=False, selected=selected, replay=independent,
                  input_sha256={str(f.resolve()): hashlib.sha256(f.read_bytes()).hexdigest() for f in inputs})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('acceptance_version', 'strict_accepted', 'warmup_accepted', 'admitted')}))


if __name__ == '__main__':
    main()


def audit_for_optimizer(model, reference, trajectory, backend='mujoco_warp'):
    metrics = audit_states(model, reference, trajectory, backend)
    states = dict(qpos=trajectory['qpos'], control_dt=trajectory.get('control_dt', np.array(.01)))
    metrics['acceptance_version'] = VERSION
    metrics['grasp_retention'] = grasp_retention(model, states)
    return metrics


def valid_warmup(report):
    if report.get('acceptance_version') != VERSION:
        raise ValueError('Warmup acceptance requires explicit versioned grasp evidence')
    return bool(report['object_rmse_m'] < .03 and report['final_object_position_error_m'] < .03
                and max(report[k] for k in ('max_arm_table_penetration_m','max_hand_table_penetration_m',
                                             'max_hand_self_penetration_m')) < .001
                and report['grasp_retention']['retained'])
