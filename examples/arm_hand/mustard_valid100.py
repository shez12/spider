"""Accumulate 100 spatially augmented mustard trajectories passing two GPU rollouts."""
import argparse
import concurrent.futures
from datetime import datetime, timezone
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import json

import numpy as np
from examples.arm_hand.batch_kuka_gpu import ROOT,save,digest
from examples.arm_hand.spatial_augmentation import augment,solve_ik_candidate
from examples.arm_hand.resource_slots import resource_slot
from zoneinfo import ZoneInfo

TASK='dexycb_01_right_006_mustard_bottle_20200709_143211_00'


def valid(report):
    return bool(report['object_rmse_m']<.03 and report['final_object_position_error_m']<.03
        and report['final_object_orientation_error_deg']<20
        and max(report['max_arm_table_penetration_m'],report['max_hand_table_penetration_m'],
                report['max_hand_self_penetration_m'])<.001)


def audit_trajectory(folder,trajectory,logname):
    target=folder/'gpu' if trajectory=='trajectory.npz' else folder/'replay_audit'
    target.mkdir(exist_ok=True)
    if trajectory!='trajectory.npz':
        link=target/'trajectory.npz'
        if not link.exists():link.symlink_to(folder/'gpu'/trajectory)
    with (folder/logname).open('w') as log:
        subprocess.run([sys.executable,'-m','examples.arm_hand.audit_right_trajectory',
            '--case-dir',str(folder),'--gpu-dir',str(target)],cwd=ROOT,
            stdout=log,stderr=subprocess.STDOUT,check=True)
    return json.loads((target/'tracking_collision_report.json').read_text())


def physical_candidate(state):
    folder=Path(state['folder'])
    control=json.loads((folder.parent/'parallelism.json').read_text())
    with resource_slot(control.get('global_gpu_slots_dir'),control.get('global_gpu_slots',10)):
        return physical_candidate_impl(state)


def physical_candidate_impl(state):
    """Each thread launches its own isolated GPU optimizer process and files."""
    state=dict(state);folder=Path(state['folder']);gpu=folder/'gpu';gpu.mkdir(exist_ok=True)
    adopted=state.pop('adopt_optimizer_pid',None)
    previous_start=state.get('physics_started_at_utc')
    state.update(status='physics_running',physics_started_at_utc=previous_start if adopted else datetime.now(timezone.utc).isoformat())
    save(folder/'state.json',state)
    start=None
    try:
        if adopted:
            while Path(f'/proc/{adopted}/stat').exists():
                try:
                    if Path(f'/proc/{adopted}/stat').read_text().split(') ',1)[1].startswith('Z'):break
                except FileNotFoundError:break
                time.sleep(2)
            state['physics_elapsed_s']=state.get('physics_elapsed_s',0)+(datetime.now(timezone.utc)-datetime.fromisoformat(previous_start)).total_seconds()
            state['physics_started_at_utc']=datetime.now(timezone.utc).isoformat()
            save(folder/'state.json',state)
        if not (gpu/'result.json').exists():
            execution_env=os.environ.copy()
            control=json.loads((folder.parent/'parallelism.json').read_text())
            pipe=control.get('mps_pipe_dir')
            if pipe:
                if not (Path(pipe)/'nvidia-cuda-mps-control.pid').exists():
                    raise RuntimeError(f'MPS control daemon is unavailable: {pipe}')
                execution_env['CUDA_MPS_PIPE_DIRECTORY']=pipe
                state['gpu_execution']='cuda_mps'
                state['mps_pipe_dir']=pipe
            else:state['gpu_execution']='independent_cuda_context'
            save(folder/'state.json',state)
            command=[sys.executable,'-u','-m','examples.arm_hand.optimize_kuka_gpu',
                '--input-dir',str(folder),'--output-dir',str(gpu),'--samples','512','--seed','0',
                '--max-iterations','200','--patience','30','--relative-tol','.001','--absolute-tol','.01',
                '--disable-target-stop','--active','all','--table-collision-weight','0']
            if (gpu/'checkpoint.pt').exists():command.append('--resume')
            start=time.monotonic()
            with (folder/'training.log').open('a') as log:
                subprocess.run(command,cwd=ROOT,env=execution_env,stdout=log,stderr=subprocess.STDOUT,check=True)
            state['physics_elapsed_s']=state.get('physics_elapsed_s',0)+time.monotonic()-start
            start=None
        started_validation=time.monotonic()
        best=audit_trajectory(folder,'trajectory.npz','audit.log')
        replay=audit_trajectory(folder,'replay_trajectory.npz','replay_audit.log')
        passed=valid(best) and valid(replay)
        if passed:
            with (folder/'render.log').open('w') as log:
                subprocess.run([sys.executable,'-m','examples.arm_hand.record_gpu_best',
                    '--input-dir',str(gpu),'--scene',str(folder/'scene.xml'),
                    '--output',str(folder/'best_gpu.mp4')],cwd=ROOT,
                    stdout=log,stderr=subprocess.STDOUT,check=True)
        state.update(status='accepted' if passed else 'physics_rejected',validated_success=passed,
            selected_metrics=best,replay_metrics=replay,
            validation_elapsed_s=time.monotonic()-started_validation)
    except Exception as error:
        if start is not None:state['physics_elapsed_s']=state.get('physics_elapsed_s',0)+time.monotonic()-start
        state.update(status='physics_failed',validated_success=False,error=str(error))
    save(folder/'state.json',state)
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/mustard_spatial_valid100')
    parser.add_argument('--target',type=int,default=100)
    parser.add_argument('--ik-workers',type=int,default=8)
    parser.add_argument('--physics-workers',type=int,default=2)
    parser.add_argument('--mps-pipe-dir',type=Path)
    parser.add_argument('--task',default=TASK)
    parser.add_argument('--source-dir',type=Path)
    parser.add_argument('--source-index',type=int,default=5)
    parser.add_argument('--augmentation-seed',type=int,default=20261026)
    parser.add_argument('--request-started-at')
    args=parser.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    if args.physics_workers<1 or args.ik_workers<1:parser.error('Worker counts must be positive')
    import fcntl
    run_lock=(out/'coordinator.lock').open('a+')
    try:fcntl.flock(run_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another coordinator already owns this output directory')
    control=json.loads((out/'parallelism.json').read_text()) if (out/'parallelism.json').exists() else {}
    control.update(ik_workers=args.ik_workers,physics_workers=args.physics_workers,
        updated_at_utc=datetime.now(timezone.utc).isoformat(),samples_per_gpu_task=512)
    if args.mps_pipe_dir:control['mps_pipe_dir']=str(args.mps_pipe_dir.resolve())
    save(out/'parallelism.json',control)
    task=args.task
    source=args.source_dir.resolve() if args.source_dir else ROOT/'outputs/kuka_sharpa_right_batch20_ik'/f'05_{TASK}'
    settings=dict(target=args.target,task=task,seed=args.augmentation_seed,translation_bound_m=.05,yaw_bound_deg=10,
        max_iterations=200,samples=512,patience=30,robot_gravcomp=1,object_gravcomp=0,
        independent_gpu_replay_required=True,extra_hold=False,version=1)
    if args.source_dir:
        settings.update(source_directory=str(source),source_reference_sha256=digest(source/'source_reference.npz'))
    if (out/'settings.json').exists() and json.loads((out/'settings.json').read_text())!=settings:
        raise ValueError('Run settings differ')
    save(out/'settings.json',settings)
    if not (out/'timing.json').exists():
        started=datetime.fromisoformat(args.request_started_at) if args.request_started_at else datetime.now(timezone.utc)
        save(out/'timing.json',dict(request_started_at_utc=started.astimezone(timezone.utc).isoformat(),
            request_started_at_local=started.astimezone(ZoneInfo('Asia/Shanghai')).isoformat(),
            pipeline_started_at_utc=datetime.now(timezone.utc).isoformat(),runs=[]))
    timing=json.loads((out/'timing.json').read_text());run=dict(started_at_utc=datetime.now(timezone.utc).isoformat())
    timing['runs'].append(run);run_start=time.monotonic()

    state_cache={}
    def states():
        result=[]
        for path in sorted(out.glob('candidate_*/state.json')):
            stamp=path.stat().st_mtime_ns
            cached=state_cache.get(path)
            if cached is None or cached[0]!=stamp:
                cached=(stamp,json.loads(path.read_text()));state_cache[path]=cached
            result.append(cached[1])
        return result

    def update(phase):
        cases=states();accepted=[s for s in cases if s.get('validated_success')]
        elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(timing['request_started_at_utc'])).total_seconds()
        run['active_elapsed_s']=time.monotonic()-run_start
        timing.update(elapsed_wall_s=elapsed,elapsed_wall_hours=elapsed/3600,
            ik_compute_sum_s=sum(s.get('ik_elapsed_s',0) for s in cases),
            physics_optimization_sum_s=sum(s.get('physics_elapsed_s',0) for s in cases),
            audit_and_render_sum_s=sum(s.get('validation_elapsed_s',0) for s in cases))
        save(out/'timing.json',timing)
        summary=dict(phase=phase,target=args.target,valid_count=len(accepted),
            physics_workers=args.physics_workers,
            generated=len(cases),ik_screened=sum('ik_pass' in s for s in cases),
            ik_passed=sum(s.get('ik_pass',False) for s in cases),
            physical_evaluated=sum('validated_success' in s for s in cases),
            elapsed_wall_s=elapsed,timing_file=str(out/'timing.json'),cases=cases)
        save(out/'summary.json',summary)
        save(out/'accepted.json',[dict(candidate_id=s['candidate_id'],folder=s['folder'],
            trajectory=str(Path(s['folder'])/'gpu/trajectory.npz'),
            replay_trajectory=str(Path(s['folder'])/'gpu/replay_trajectory.npz'),
            video=str(Path(s['folder'])/'best_gpu.mp4')) for s in accepted])
        return cases,accepted

    def generate(index):
        folder=out/f'candidate_{index:05d}';folder.mkdir(exist_ok=True)
        if (folder/'state.json').exists():return
        previous=ROOT/'outputs/kuka_sharpa_spatial100'/f'05_{TASK}'/f'variant_{index:02d}'
        reused=task==TASK and not args.source_dir and index<=5 and (previous/'geometry_report.json').exists()
        if reused:
            for name in ['scene.xml','reference.npz','source_reference.npz','retargeting.json','geometry_report.json','augmentation.json']:
                shutil.copy2(previous/name,folder/name)
            report=json.loads((folder/'geometry_report.json').read_text())
            state=dict(ik_pass=report['ik_pass'],status='ik_passed' if report['ik_pass'] else 'ik_rejected',
                       reused_ik_from=str(previous),ik_elapsed_s=0)
        else:
            rng=np.random.default_rng(np.random.SeedSequence([settings['seed'],index]))
            yaw=rng.uniform(-10,10);delta=rng.uniform(-.05,.05,2)
            with np.load(source/'source_reference.npz') as a:ref={k:a[k] for k in a.files}
            tree=ET.parse(source/'scene.xml');base=tree.getroot().find("worldbody/body[@name='kuka_base']")
            pos=np.fromstring(base.get('pos'),sep=' ')
            for body in [base,*base.findall('.//body')]:body.set('gravcomp','1')
            tree.write(folder/'scene.xml',encoding='unicode')
            ref,metadata=augment(ref,yaw,delta,pos)
            np.savez(folder/'source_reference.npz',**ref)
            metadata.update(source_task=task,source_reference_sha256=digest(source/'source_reference.npz'),
                            seed=settings['seed'],candidate_id=index)
            save(folder/'augmentation.json',metadata);state=dict(status='generated')
        state.update(candidate_id=index,source_index=args.source_index,variant=index,task=task,folder=str(folder))
        save(folder/'state.json',state)

    try:
        while True:
            cases,accepted=update('running')
            if len(accepted)>=args.target:break
            pending=[s for s in cases if s.get('ik_pass') and 'validated_success' not in s]
            unscreened=[s for s in cases if 'ik_pass' not in s]
            if not pending and not unscreened:
                count=max(32,args.target-len(accepted));start=len(cases)+1
                for index in range(start,start+count):generate(index)
                cases,_=update('generating');unscreened=[s for s in cases if 'ik_pass' not in s]
            if unscreened and not pending:
                update('parallel_ik')
                # Feed eligible candidates to GPU without waiting for every
                # candidate in a 100-case generation batch to finish IK.
                unscreened=unscreened[:max(4,2*args.ik_workers)]
                with concurrent.futures.ProcessPoolExecutor(max_workers=args.ik_workers,
                        mp_context=multiprocessing.get_context('spawn')) as pool:
                    futures={pool.submit(timed_ik,s):s for s in unscreened}
                    for future in concurrent.futures.as_completed(futures):
                        state=future.result();save(Path(state['folder'])/'state.json',state)
                        update('parallel_ik');print(json.dumps(state),flush=True)
            queued=[s for s in states() if s.get('ik_pass') and 'validated_success' not in s]
            queued.sort(key=lambda s:(not bool(s.get('adopt_optimizer_pid')),s['candidate_id']))
            queue=iter(queued)
            with concurrent.futures.ThreadPoolExecutor(max_workers=15) as pool:
                active=set()
                while True:
                    control=json.loads((out/'parallelism.json').read_text())
                    args.physics_workers=max(1,min(15,int(control['physics_workers'])))
                    _,accepted=update('parallel_physics')
                    slots=min(args.physics_workers-len(active),args.target-len(accepted)-len(active))
                    for _ in range(max(0,slots)):
                        state=next(queue,None)
                        if state is None:break
                        active.add(pool.submit(physical_candidate,state))
                    if not active:break
                    done,_=concurrent.futures.wait(active,timeout=30,
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in done:
                        active.remove(future);print(json.dumps(future.result()),flush=True)
        run['finished_at_utc']=datetime.now(timezone.utc).isoformat();update('complete')
    except BaseException:
        update('interrupted_or_error');raise


def timed_ik(state):
    control_path=Path(state['folder']).parent/'parallelism.json'
    control=json.loads(control_path.read_text()) if control_path.exists() else {}
    with resource_slot(control.get('global_ik_slots_dir'),control.get('global_ik_slots',8)):
        start=time.monotonic();result=solve_ik_candidate(state)
        result['ik_elapsed_s']=time.monotonic()-start
    return result


if __name__=='__main__':main()
