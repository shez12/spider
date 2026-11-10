"""Recover rotation-diverse warmup trajectories with two saved GPU traces.

Keeps original strict campaign/benchmark results immutable. Manifests reference
original full artifacts; a warmup-ready entry also requires a saved-state video.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import mujoco
import numpy as np
from examples.arm_hand.warmup_acceptance import VERSION, assess
from examples.arm_hand.resource_slots import resource_slot


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def eligible(metrics):
    return metrics and metrics['object_rmse_m'] < .03 and metrics['final_object_position_error_m'] < .03 and max(
        metrics[k] for k in ('max_arm_table_penetration_m','max_hand_table_penetration_m','max_hand_self_penetration_m')) < .001


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,default=Path('outputs/mustard_multisource_valid100'))
    p.add_argument('--experiments',type=Path,default=Path('outputs/world_model_pilot'))
    p.add_argument('--output',type=Path,default=Path('outputs/mustard_warmup_grasp_v1'))
    p.add_argument('--render',action='store_true');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    candidates=[]
    for path in sorted(a.campaign.glob('groups/*/candidate_*/state.json')):
        state=json.loads(path.read_text())
        if state.get('validated_success') or not eligible(state.get('selected_metrics')) or not eligible(state.get('replay_metrics')):continue
        candidates.append((path.parent,path.parent,state))
    for path in sorted(a.experiments.rglob('execution.json')):
        try:state=json.loads(path.read_text())
        except (OSError,json.JSONDecodeError):continue
        if state.get('status')!='complete' or not eligible(state.get('selected')) or not eligible(state.get('replay')):continue
        candidates.append((Path(state['case']),path.parent,state))
    rows=[];start=time.monotonic()
    for case,execution,state in candidates:
        key=hashlib.sha256(str(execution.resolve()).encode()).hexdigest()[:16]
        destination=a.output/key;destination.mkdir(exist_ok=True)
        try:
            gpu=execution/'gpu';result=json.loads((gpu/'result.json').read_text())
            if result['signature']!={n:digest(case/n) for n in ('scene.xml','reference.npz')}:raise ValueError('Scene/reference signature differs')
            args=result['arguments']
            if result['extra_hold'] or args['samples']!=512 or args['max_iterations']>200 or not args['disable_target_stop']:raise ValueError('Sampling protocol differs')
            if not json.loads((case/'geometry_report.json').read_text())['ready_for_gpu']:raise ValueError('IK/geometric gate failed')
            with np.load(case/'reference.npz') as ref,np.load(gpu/'trajectory.npz') as actual,np.load(gpu/'replay_trajectory.npz') as replay:
                np.testing.assert_array_equal(actual['ctrl'],replay['ctrl'])
                for tr in (actual,replay):
                    np.testing.assert_array_equal(tr['initial_state'],ref['initial_state'])
                    if len(tr['qpos'])!=len(ref['object_qpos'])-1 or not np.isclose(float(tr['physics_dt']),.002):raise ValueError('Timebase differs')
                selected=assess(mujoco.MjModel.from_xml_path(str(case/'scene.xml')),ref,actual)
                independent=assess(mujoco.MjModel.from_xml_path(str(case/'scene.xml')),ref,replay)
            passed=selected['warmup_accepted'] and independent['warmup_accepted']
            video=execution/'best_gpu.mp4'
            if passed and a.render and not video.is_file():
                video=destination/'best_gpu.mp4'
                if not video.is_file():
                    with resource_slot(a.campaign/'resource_slots/gpu',10,block=True):
                        with (destination/'render.log').open('w') as log:
                            subprocess.run([sys.executable,'-m','examples.arm_hand.record_gpu_best','--input-dir',str(gpu.resolve()),'--scene',str((case/'scene.xml').resolve()),'--output',str(video.resolve())],check=True,stdout=log,stderr=subprocess.STDOUT)
            files=[case/'scene.xml',case/'reference.npz',gpu/'trajectory.npz',gpu/'replay_trajectory.npz',gpu/'checkpoint.pt',gpu/'result.json']
            if not all(f.is_file() for f in files):raise ValueError('Missing original physics artifacts')
            report=dict(acceptance_version=VERSION,case=str(case.resolve()),execution=str(execution.resolve()),
                warmup_accepted=bool(passed),warmup_ready=bool(passed and video.is_file()),strict_accepted=bool(selected['strict_accepted'] and independent['strict_accepted']),
                selected=selected,replay=independent,video=str(video.resolve()) if video.is_file() else None,
                input_sha256={str(f.resolve()):digest(f) for f in files},source_state=state,
                note='Additional warmup acceptance; original strict results/counts unchanged; sampled geometric grasp evidence only')
            (destination/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
            row={k:report[k] for k in ('case','execution','warmup_accepted','warmup_ready','video')};row['report']=str((destination/'validation.json').resolve());rows.append(row)
        except Exception as exc:
            rows.append(dict(case=str(case),execution=str(execution),error=str(exc),warmup_accepted=False,warmup_ready=False))
        print(json.dumps(rows[-1]),flush=True)
    # Count augmentations, never repeated controller runs for the same reference.
    unique={digest(Path(row['case'])/'source_reference.npz') for row in rows if row['warmup_ready']}
    manifest=dict(acceptance_version=VERSION,updated_at_utc=datetime.now(timezone.utc).isoformat(),
                  rotation_only_candidates=len(candidates),unique_ready_augmentations=len(unique),
                  ready_runs=sum(row['warmup_ready'] for row in rows),wall_s=time.monotonic()-start,rows=rows)
    temporary=a.output/'manifest.tmp';temporary.write_text(json.dumps(manifest,indent=2)+'\n');temporary.replace(a.output/'manifest.json')
    print(json.dumps({k:v for k,v in manifest.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
