"""Read-only consistency and optional fresh geometry audit of admitted trajectories."""
import argparse
from collections import Counter
import datetime
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch
from examples.arm_hand.audit_right_trajectory import audit_states
from examples.arm_hand.mustard_valid100 import valid


def inspect(case,recompute):
    state=json.loads((case/'state.json').read_text());result=json.loads((case/'gpu/result.json').read_text())
    required=['source_reference.npz','reference.npz','scene.xml','geometry_report.json','gpu/checkpoint.pt','gpu/history.json','gpu/trajectory.npz','gpu/replay_trajectory.npz','best_gpu.mp4']
    for name in required:
        if not (case/name).is_file() or not (case/name).stat().st_size:raise ValueError('Missing/empty '+name)
    if not state.get('validated_success') or not valid(state['selected_metrics']) or not valid(state['replay_metrics']):raise ValueError('Stored dual acceptance differs')
    if not json.loads((case/'geometry_report.json').read_text()).get('ready_for_gpu'):raise ValueError('IK geometry gate differs')
    model=mujoco.MjModel.from_xml_path(str(case/'scene.xml'));floor=model.geom('floor').id;base=model.body('kuka_base').id
    if model.geom_type[floor]!=mujoco.mjtGeom.mjGEOM_BOX:raise ValueError('Table is not the required box')
    if not np.isclose(model.geom_pos[floor,2]+model.geom_size[floor,2]-model.body_pos[base,2],.53,atol=1e-10):raise ValueError('Table/base height differs')
    for body in range(1,model.nbody):
        ancestor=body
        while ancestor and ancestor!=base:ancestor=int(model.body_parentid[ancestor])
        if ancestor==base and model.body_gravcomp[body]!=1:raise ValueError('Robot gravity compensation differs')
    if model.body_gravcomp[model.body('right_object').id]!=0:raise ValueError('Object gravity compensation differs')
    args=result['arguments']
    if args['samples']!=512 or result['iterations']>200 or args['max_iterations']>200 or not args['disable_target_stop'] or result['extra_hold']:raise ValueError('Original sampling budget/stop/no-hold settings differ')
    if result.get('export_policy')!='actual_selected_rollout_v1':raise ValueError('Actual-rollout provenance missing')
    checkpoint=torch.load(case/'gpu/checkpoint.pt',map_location='cpu',weights_only=False)
    import hashlib
    signature={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [case/'scene.xml',case/'reference.npz']}
    if signature!=checkpoint['signature'] or signature!=result['signature']:raise ValueError('Scene/reference signature differs')
    with np.load(case/'reference.npz') as ref,np.load(case/'gpu/trajectory.npz') as selected,np.load(case/'gpu/replay_trajectory.npz') as replay:
        np.testing.assert_array_equal(selected['ctrl'],replay['ctrl'])
        for trace in [selected,replay]:
            np.testing.assert_array_equal(trace['initial_state'],ref['initial_state'])
            if not np.isclose(float(trace['control_dt']),.01) or not np.isclose(float(trace['physics_dt']),.002):raise ValueError('Timebase differs')
            if len(trace['qpos'])!=len(ref['ctrl_nominal']) or not np.isfinite(trace['qpos']).all():raise ValueError('Invalid state timeline')
        for key in ['qpos','qvel','ctrl']:
            np.testing.assert_array_equal(checkpoint['best_rollout'][key].numpy(),selected[key])
        if recompute:
            reports=[audit_states(model,ref,trace) for trace in [selected,replay]]
            if not all(valid(report) for report in reports):raise ValueError('Fresh saved-state dual audit fails')
        else:reports=None
    return dict(folder=str(case),source_index=state.get('source_index'),passed=True,fresh_dual_reports=reports)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--recompute',action='store_true');a=p.parse_args();started=time.monotonic()
    groups=json.loads((a.campaign/'selected_sources.json').read_text());accepted=json.loads((a.campaign/'accepted.json').read_text())
    if len(groups)!=22 or len({g['task'] for g in groups})!=22:raise ValueError('Original source deduplication differs')
    if len({r['folder'] for r in accepted})!=len(accepted):raise ValueError('Duplicate admitted folder')
    counts=Counter(r['source_index'] for r in accepted)
    if any(n>100 for n in counts.values()):raise ValueError('Source count exceeds target')
    rows=[]
    for i,row in enumerate(accepted):
        try:record=inspect(Path(row['folder']),a.recompute)
        except Exception as error:record=dict(folder=row['folder'],source_index=row['source_index'],passed=False,error=f'{type(error).__name__}: {error}')
        rows.append(record)
        if (i+1)%20==0:print(json.dumps(dict(checked=i+1,total=len(accepted),failed=sum(not r['passed'] for r in rows))),flush=True)
    report=dict(generated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),snapshot_admitted_count=len(accepted),source_count=len(groups),passed=sum(r['passed'] for r in rows),failed=sum(not r['passed'] for r in rows),recomputed_saved_state_geometry=a.recompute,elapsed_s=time.monotonic()-started,cases=rows,
        limitations='Read-only audit of the accepted-manifest snapshot. Recomputes saved 10 ms states, not a new dynamics replay or continuous-time collision guarantee; no production counts are changed.')
    temporary=a.output.with_suffix('.tmp');temporary.write_text(json.dumps(report,indent=2));temporary.replace(a.output)
    print(json.dumps({k:v for k,v in report.items() if k!='cases'}),flush=True)


if __name__=='__main__':main()
