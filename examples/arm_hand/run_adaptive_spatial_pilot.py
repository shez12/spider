"""Isolated equal-budget uniform/adaptive pilot with unchanged production gates."""
import argparse
import concurrent.futures
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
from examples.arm_hand.batch_kuka_gpu import ROOT,save,digest
from examples.arm_hand.adaptive_spatial_sampling import proposals,coverage
from examples.arm_hand.spatial_augmentation import augment,solve_ik_candidate,check_ik
from examples.arm_hand.mustard_valid100 import physical_candidate_impl
from examples.arm_hand.resource_slots import resource_slot


def feedback_snapshot(folder):
    result=[]
    for path in sorted(folder.glob('candidate_*/state.json')):
        state=json.loads(path.read_text());metadata=path.parent/'augmentation.json'
        if not metadata.exists():continue
        if 'ik_pass' not in state:continue
        a=json.loads(metadata.read_text());report=path.parent/'geometry_report.json'
        g=json.loads(report.read_text()) if report.exists() else {}
        score=None
        if 'initial_palm_error_m' in g:
            score=max(0,g['initial_palm_error_m']/.01-1)
            score+=sum(max(0,x/.03-1)for x in g.get('initial_fingertip_errors_m',[]))
            score+=max(0,g.get('mean_wrist_error_m',0)/.08-1)
            score+=1000*max(g.get('initial_robot_table_penetration_m',0),g.get('max_fitted_robot_table_penetration_m',0))
            score+=1000*max(g.get('initial_hand_self_penetration_m',0),g.get('max_hand_self_penetration_m',0))
        stage='physical_success' if state.get('validated_success') else ('ik_success' if state.get('ik_pass') else 'geometry_failure')
        result.append(dict(candidate_id=state['candidate_id'],point=[a['yaw_deg'],*a['sampled_translation_xy_m']],
            stage=stage,geometry_score=score,physical_evaluated='validated_success'in state,
            physical_success=state.get('validated_success',False)))
    return result


def ik_job(state):
    folder=Path(state['folder']);control=json.loads((folder.parent/'parallelism.json').read_text())
    queued=time.monotonic()
    with resource_slot(control['global_ik_slots_dir'],control['global_ik_slots'],block=True):
        wait=time.monotonic()-queued;started=time.monotonic();result=solve_ik_candidate(state)
        result.update(ik_elapsed_s=time.monotonic()-started,ik_slot_wait_s=wait)
        save(folder/'state.json',result)
    return result


def physics_job(state):
    folder=Path(state['folder']);control=json.loads((folder.parent/'parallelism.json').read_text())
    queued=time.monotonic()
    with resource_slot(control['global_gpu_slots_dir'],control['global_gpu_slots'],block=True):
        state=dict(state,gpu_slot_wait_s=time.monotonic()-queued)
        return physical_candidate_impl(state)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/mustard_adaptive_sampling_pilot_v1')
    p.add_argument('--count-per-arm',type=int,default=16)
    p.add_argument('--seed',type=int,default=20261030)
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    lock=(out/'pilot.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    production=ROOT/'outputs/mustard_multisource_valid100'
    selections=json.loads((production/'selected_sources.json').read_text())
    groups=[g for g in selections if g['source_index']in [2,10,18,8]]
    settings=dict(source_indices=[g['source_index']for g in groups],count_per_arm=args.count_per_arm,seed=args.seed,
        global_fraction=.25,adaptive_fraction=.75,min_normalized_separation=.1,
        proposal_bandwidth_normalized=[.25,.15,.15],samples=512,max_iterations=200,
        strict_check_every=10,frozen_historical_feedback=True,
        source_reference_unchanged=True,production_sampler_modified=False)
    if (out/'settings.json').exists():
        if json.loads((out/'settings.json').read_text())!=settings:raise ValueError('Pilot settings differ')
    else:save(out/'settings.json',settings)
    timing_path=out/'timing.json'
    timing=json.loads(timing_path.read_text())if timing_path.exists()else dict(started_at_utc=datetime.now(timezone.utc).isoformat())
    save(timing_path,timing)
    all_states=[]
    for group in groups:
        index=group['source_index'];source=Path(group['source_dir']);parent=out/f'group_{index:03d}'
        parent.mkdir(exist_ok=True)
        frozen_path=parent/'feedback.json'
        if not frozen_path.exists():save(frozen_path,feedback_snapshot(Path(group['result_dir'])))
        feedback=json.loads(frozen_path.read_text())
        # Diagnose the unaugmented source once; never overwrite source artifacts.
        diagnostic=parent/'source_diagnostic';diagnostic.mkdir(exist_ok=True)
        if not (diagnostic/'geometry_report.json').exists():
            import shutil
            for name in ['scene.xml','reference.npz','retargeting.json']:
                shutil.copy2(source/name,diagnostic/name)
            check_ik(diagnostic)
        production_control=json.loads((Path(group['result_dir'])/'parallelism.json').read_text())
        for arm in ['uniform','adaptive']:
            arm_dir=parent/arm;arm_dir.mkdir(exist_ok=True)
            control={key:production_control[key]for key in ['global_gpu_slots_dir','global_gpu_slots','global_ik_slots_dir','global_ik_slots','mps_pipe_dir']}
            save(arm_dir/'parallelism.json',control)
            save(arm_dir/'throughput_policy.json',dict(strict_check_every=10))
            plan=proposals(feedback,args.count_per_arm,args.seed+100*index+(arm=='adaptive'),arm=='adaptive')
            save(arm_dir/'sampling_plan.json',dict(proposals=plan,feedback_sha256=digest(frozen_path),
                candidate_coverage=coverage([r['point']for r in plan])))
            for row in plan:
                folder=arm_dir/f'candidate_{row["index"]:05d}';folder.mkdir(exist_ok=True)
                if not (folder/'state.json').exists():
                    ref=dict(np.load(source/'source_reference.npz'))
                    tree=ET.parse(source/'scene.xml');base=tree.getroot().find("worldbody/body[@name='kuka_base']")
                    base_pos=np.fromstring(base.get('pos'),sep=' ')
                    for body in [base,*base.findall('.//body')]:body.set('gravcomp','1')
                    tree.write(folder/'scene.xml',encoding='unicode')
                    point=np.asarray(row['point']);transformed,metadata=augment(ref,point[0],point[1:],base_pos)
                    np.savez(folder/'source_reference.npz',**transformed)
                    metadata.update(row,source_task=group['task'],source_reference_sha256=digest(source/'source_reference.npz'))
                    save(folder/'augmentation.json',metadata)
                    save(folder/'state.json',dict(candidate_id=row['index'],source_index=index,task=group['task'],
                        arm=arm,folder=str(folder),status='generated',pilot=True))
                all_states.append(json.loads((folder/'state.json').read_text()))
    save(out/'manifest.json',all_states)

    def update(phase):
        states=[json.loads((Path(s['folder'])/'state.json').read_text())for s in all_states]
        rows=[]
        for group in groups:
            for arm in ['uniform','adaptive']:
                cases=[s for s in states if s['source_index']==group['source_index']and s['arm']==arm]
                accepted=[s for s in cases if s.get('validated_success')]
                points=[json.loads((Path(s['folder'])/'augmentation.json').read_text())['point']for s in accepted]
                compute=sum(s.get('ik_elapsed_s',0)+s.get('physics_elapsed_s',0)+s.get('validation_elapsed_s',0)for s in cases)
                physical=sum('validated_success'in s for s in cases)
                rows.append(dict(source_index=group['source_index'],arm=arm,candidates=len(cases),
                    ik_screened=sum('ik_pass'in s for s in cases),ik_passed=sum(s.get('ik_pass',False)for s in cases),
                    physical_evaluated=physical,physical_valid=len(accepted),
                    physics_failed=sum(s.get('status')=='physics_failed'for s in cases),
                    compute_service_s=compute,valid_per_service_hour=3600*len(accepted)/compute if compute else None,
                    valid_coverage=coverage(points)))
        wall=(datetime.now(timezone.utc)-datetime.fromisoformat(timing['started_at_utc'])).total_seconds()
        summary=dict(phase=phase,updated_at_utc=datetime.now(timezone.utc).isoformat(),elapsed_wall_s=wall,
            candidate_count=len(states),valid_count=sum(s.get('validated_success',False)for s in states),rows=rows,
            limitations=['Service-hour rates sum CPU/GPU/audit worker time; they are not isolated end-to-end wall-clock rates.',
                'Shared production slots and other training remain active; queue wait is recorded separately.',
                '16 proposals per source/arm is a pilot with substantial statistical uncertainty.',
                'Historical feedback is frozen before evaluation; no future outcomes leak into proposals.',
                'Uniform control uses the same diversity guard as adaptive; raw unconstrained iid is not this control.'])
        save(out/'summary.json',summary)
        save(out/'accepted.json',[s for s in states if s.get('validated_success')])
        timing.update(elapsed_wall_s=wall);save(timing_path,timing)
    update('prepared')
    if args.prepare_only:return
    # Interleave paired arms and source groups to avoid running one arm wholly
    # before the other under a different background GPU load.
    all_states.sort(key=lambda s:(s['candidate_id'],s['source_index'],s['arm']!='uniform'))
    with concurrent.futures.ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'))as cpu,concurrent.futures.ThreadPoolExecutor(max_workers=2)as gpu:
        ik_pending={cpu.submit(ik_job,s)for s in all_states if 'ik_pass'not in s}
        physics_pending={gpu.submit(physics_job,s)for s in all_states if s.get('ik_pass')and 'validated_success'not in s}
        while ik_pending or physics_pending:
            done,_=concurrent.futures.wait(ik_pending|physics_pending,timeout=30,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                result=future.result()
                if future in ik_pending:
                    ik_pending.remove(future)
                    if result.get('ik_pass'):physics_pending.add(gpu.submit(physics_job,result))
                else:physics_pending.remove(future)
            update('running')
    timing['completed_at_utc']=datetime.now(timezone.utc).isoformat();update('complete')


if __name__=='__main__':main()
