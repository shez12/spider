"""Resume independent 100-valid campaigns for every available mustard demonstration."""
import argparse
from collections import Counter
import concurrent.futures
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from examples.arm_hand.batch_kuka_gpu import ROOT, save, prepare
from examples.arm_hand.mustard_valid100 import TASK, valid


class AdoptedCoordinator:
    """Observe an existing coordinator without resetting it or starting a duplicate."""
    def __init__(self,pid,folder):
        self.pid=int(pid);self.folder=str(folder)
        self.identity=self._identity()

    def _identity(self):
        try:
            raw=Path(f'/proc/{self.pid}/stat').read_text().split(') ',1)[1].split()
            command=Path(f'/proc/{self.pid}/cmdline').read_bytes().split(bytes([0]))
            if raw[0]=='Z' or b'examples.arm_hand.mustard_valid100' not in command or self.folder.encode() not in command:return None
            return raw[19]
        except (FileNotFoundError,ProcessLookupError):return None

    def poll(self):return None if self.identity is not None and self._identity()==self.identity else 1


def select(dataset, limit):
    priority={'allegro':0,'sharpa':1,'xhand':2,'inspire':3}
    available={}
    for row in json.loads((dataset/'manifest.json').read_text())['runs']:
        if row['dataset']!='dexycb_lifted' or row['embodiment_type']!='right' or '_right_006_mustard_bottle_' not in row['task']:
            continue
        if not (dataset/row['scene']).is_file() or not (dataset/row['path']/'trajectory_mjwp.npz').is_file():
            continue
        old=available.get(row['task'])
        if old is None or priority.get(row['robot_type'],99)<priority.get(old['robot_type'],99):
            available[row['task']]=row
    rows=sorted(available.values(),key=lambda r:(r['task']!=TASK,r['task']))
    return rows[:limit],len(rows)


def prepare_source(group, dataset):
    folder=Path(group['source_dir']);folder.mkdir(parents=True,exist_ok=True)
    try:
        with (folder/'prepare.log').open('a') as log:
            # A subprocess isolates native MuJoCo/BLAS initialization and logs.
            command=[sys.executable,'-u','-m',__name__ if __name__!='__main__' else 'examples.arm_hand.mustard_multisource',
                     '--prepare-one',str(folder/'source.json'),'--dataset-dir',str(dataset)]
            save(folder/'source.json',group['source'])
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        return None
    except Exception as error:return str(error)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/mustard_multisource_valid100')
    parser.add_argument('--dataset-dir',type=Path,default=ROOT/'example_datasets/retarget_full')
    parser.add_argument('--max-sources',type=int,default=100)
    parser.add_argument('--gpu-workers',type=int,default=10)
    parser.add_argument('--reserved-gpu-slots',type=int,default=0,help='Keep slots available for model learning; total GPU slots remain gpu-workers.')
    parser.add_argument('--adopt-existing',action='store_true')
    parser.add_argument('--ik-workers',type=int,default=8)
    parser.add_argument('--mps-pipe-dir',type=Path,default=Path('/tmp/spider-mps-mustard-1005'))
    parser.add_argument('--prepare-one',type=Path)
    args=parser.parse_args()
    if args.prepare_one:
        prepare(json.loads(args.prepare_one.read_text()),args.dataset_dir,args.prepare_one.parent,2,allow_geometry_failure=True)
        return
    if args.gpu_workers<1 or not 0<=args.reserved_gpu_slots<args.gpu_workers:parser.error('Reserve fewer than total positive GPU slots')
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    lock=(out/'campaign.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not (args.mps_pipe_dir/'nvidia-cuda-mps-control.pid').exists():
        raise RuntimeError('Private CUDA MPS daemon must be running before launching')
    selection=out/'selected_sources.json'
    if selection.exists():groups=json.loads(selection.read_text())
    else:
        rows,total=select(args.dataset_dir,args.max_sources)
        if not rows:raise RuntimeError('No complete mustard source demonstrations found')
        groups=[]
        for index,row in enumerate(rows,1):
            name=f'{index:03d}_{row["task"]}'
            reused=row['task']==TASK
            groups.append(dict(source_index=index,task=row['task'],source=row,
                source_dir=str(ROOT/'outputs/kuka_sharpa_right_batch20_ik'/f'05_{TASK}') if reused else str(out/'sources'/name),
                result_dir=str(ROOT/'outputs/mustard_spatial_valid100') if reused else str(out/'groups'/name),
                reused_completed_group=reused))
        save(selection,groups)
        save(out/'selection_inventory.json',dict(available_distinct_sources=total,requested_sources=args.max_sources,
            selected_sources=len(groups),target_per_source=100,target_total=100*len(groups),
            scope='DexYCB right 006_mustard_bottle; deduplicated by original task, across published robot embodiments'))
    timing_path=out/'timing.json'
    timing=json.loads(timing_path.read_text()) if timing_path.exists() else dict(started_at_utc=datetime.now(timezone.utc).isoformat())
    save(timing_path,timing)
    children={};retries={};errors={}
    if args.adopt_existing:
        for group in groups:
            folder=Path(group['result_dir'])
            for filename in ['launcher.json','repair_launcher.json']:
                path=folder/filename
                if not path.exists():continue
                candidate=AdoptedCoordinator(json.loads(path.read_text())['pid'],folder)
                if candidate.poll() is None:
                    children[group['source_index']]=candidate;break

    def aggregate():
        accepted=[];statuses=[];candidate_counts=Counter();ik_seconds=0.;physics_seconds=0.
        for group in groups:
            folder=Path(group['result_dir']);summary_path=folder/'summary.json'
            summary=json.loads(summary_path.read_text()) if summary_path.exists() else {}
            if not group['reused_completed_group']:
                for state in summary.get('cases',[]):
                    candidate_counts[state.get('status','unknown')]+=1
                    ik_seconds+=state.get('ik_elapsed_s',0)
                    physics_seconds+=state.get('physics_elapsed_s',0)
            manifest=folder/'accepted.json'
            items=json.loads(manifest.read_text()) if manifest.exists() else []
            if len(items)>100:raise RuntimeError(f'Group exceeds target: {folder}')
            for item in items:
                accepted.append(dict(item,source_index=group['source_index'],source_task=group['task']))
            statuses.append(dict(source_index=group['source_index'],task=group['task'],valid_count=len(items),
                phase='needs_attention' if group['source_index'] in errors else summary.get('phase','preparing'),
                result_dir=str(folder),error=errors.get(group['source_index'])))
        elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(timing['started_at_utc'])).total_seconds()
        complete=all(s['valid_count']==100 for s in statuses)
        save(out/'accepted.json',accepted)
        save(out/'summary.json',dict(phase='complete' if complete else 'running',source_count=len(groups),
            target_per_source=100,target_total=len(groups)*100,valid_count=len(accepted),
            completed_sources=sum(s['valid_count']==100 for s in statuses),gpu_workers=args.gpu_workers,reserved_gpu_slots=args.reserved_gpu_slots,
            updated_at_utc=datetime.now(timezone.utc).isoformat(),elapsed_wall_s=elapsed,groups=statuses))
        timing.update(elapsed_wall_s=elapsed,updated_at_utc=datetime.now(timezone.utc).isoformat())
        if complete:timing['completed_at_utc']=datetime.now(timezone.utc).isoformat()
        save(timing_path,timing)
        # Keep a small progress history for measured throughput rather than
        # extrapolating from the previously completed single-source campaign.
        now=datetime.now(timezone.utc).timestamp()
        history_path=out/'throughput_history.json'
        samples=json.loads(history_path.read_text()) if history_path.exists() else []
        if not samples or now-samples[-1]['timestamp']>=300:
            samples.append(dict(timestamp=now,valid_count=len(accepted),candidate_counts=dict(candidate_counts)))
            save(history_path,samples)
        recent=[sample for sample in samples if now-sample['timestamp']<=7200]
        anchor=recent[0] if recent else samples[-1]
        measured_hours=(now-anchor['timestamp'])/3600
        save(out/'throughput.json',dict(updated_at_utc=datetime.now(timezone.utc).isoformat(),
            candidate_counts=dict(candidate_counts),ik_compute_sum_s=ik_seconds,
            physics_compute_sum_s=physics_seconds,window_hours=measured_hours,
            new_valid_in_window=len(accepted)-anchor['valid_count'],
            valid_per_hour=(len(accepted)-anchor['valid_count'])/measured_hours if measured_hours>=.25 else None,
            note='Recent observed wall-clock production throughput; existing reused group excluded from candidate counts.'))
        return statuses,complete
    # Verify reused artifacts and both physical reports before counting them.
    for group in groups:
        if not group['reused_completed_group']:continue
        folder=Path(group['result_dir']);items=json.loads((folder/'accepted.json').read_text())
        if len(items)!=100:raise RuntimeError('Reused group is incomplete')
        for item in items:
            state=json.loads((Path(item['folder'])/'state.json').read_text())
            if not (valid(state['selected_metrics']) and valid(state['replay_metrics'])):
                raise RuntimeError('Reused validation no longer satisfies acceptance criteria')
            for key in ['trajectory','replay_trajectory','video']:
                if not Path(item[key]).is_file():raise FileNotFoundError(item[key])
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.ik_workers) as pool:
        preparing={pool.submit(prepare_source,g,args.dataset_dir):g for g in groups if not g['reused_completed_group']}
        while True:
            for future in list(preparing):
                if not future.done():continue
                group=preparing.pop(future);error=future.result()
                if error:errors[group['source_index']]=error
            statuses,complete=aggregate()
            if complete:break
            eligible=[g for g,s in zip(groups,statuses) if s['valid_count']<100 and g['source_index'] not in errors
                      and (Path(g['source_dir'])/'preparation.json').exists()]
            budget=min(args.gpu_workers,max(1,math.ceil(args.gpu_workers/max(1,len(eligible)))))
            for group in eligible:
                index=group['source_index'];folder=Path(group['result_dir']);folder.mkdir(parents=True,exist_ok=True)
                control=dict(physics_workers=budget,ik_workers=1,global_gpu_slots=args.gpu_workers-args.reserved_gpu_slots,
                    global_gpu_slots_dir=str(out/'resource_slots/gpu'),global_ik_slots=args.ik_workers,
                    global_ik_slots_dir=str(out/'resource_slots/ik'),mps_pipe_dir=str(args.mps_pipe_dir))
                # Only the supervisor writes this file after startup; child writes once before doing work.
                child=children.get(index)
                if child is not None and child.poll() is None:
                    save(folder/'parallelism.json',control);continue
                if child is not None:
                    if not isinstance(child,AdoptedCoordinator):
                        try:os.killpg(child.pid,signal.SIGTERM)
                        except ProcessLookupError:pass
                    retries[index]=retries.get(index,0)+1
                    if retries[index]>3:
                        errors[index]=f'Coordinator exited repeatedly, last exit {child.returncode}';continue
                save(folder/'parallelism.json',control)
                command=[sys.executable,'-u','-m','examples.arm_hand.mustard_valid100','--output-dir',str(folder),
                    '--source-dir',group['source_dir'],'--task',group['task'],'--source-index',str(index),
                    '--augmentation-seed',str(20261029+index),'--ik-workers','1','--physics-workers',str(budget)]
                with (folder/'coordinator.log').open('a') as log:
                    child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                children[index]=child
                save(folder/'launcher.json',dict(pid=child.pid,command=command))
            if not preparing and not any(c.poll() is None for c in children.values()) and errors:
                aggregate();raise RuntimeError(f'Campaign requires attention: {errors}')
            time.sleep(30)


if __name__=='__main__':main()
