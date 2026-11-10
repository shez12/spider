"""Prepare and optimize twenty published DexYCB tasks with the current KUKA setup.

Each task gets independent IK, GPU optimization, history, checkpoint and video.
Restart this command to skip completed cases and resume incomplete GPU cases.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
import mujoco

from examples.arm_hand.run_sharpa_complex import fit, ROOT
from examples.arm_hand.prepare_kuka_right import build as build_right, reference
from examples.arm_hand.validate_kuka_sharpa import PLAY2PERFECT_ARM_POSE


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(dataset, count):
    runs = json.loads((dataset/'manifest.json').read_text())['runs']
    groups = defaultdict(list)
    seen = set()
    for row in sorted(runs, key=lambda r:(r['task'],r['path'])):
        if row['dataset'] != 'dexycb_lifted' or row['robot_type'] != 'allegro' or row['embodiment_type'] != 'right':
            continue
        if row['task'] in seen:
            continue
        if not (dataset/row['scene']).is_file() or not (dataset/row['path']/'trajectory_mjwp.npz').is_file():
            continue
        seen.add(row['task'])
        groups[row['task'].split('_right_')[-1][:3]].append(row)
    selected = []
    while len(selected) < count and any(groups.values()):
        for key in sorted(groups):
            if groups[key]:
                selected.append(groups[key].pop(0))
                if len(selected) == count:
                    break
    if len(selected) != count:
        raise RuntimeError(f'Only {len(selected)} complete distinct tasks available, requested {count}')
    return selected


def prepare(row, dataset, output, stride, allow_geometry_failure=False):
    output.mkdir(parents=True, exist_ok=True)
    if (output/'preparation.json').exists():
        return
    scene = dataset/row['scene']
    trajectory = dataset/row['path']/'trajectory_mjwp.npz'
    for path, expected in [(scene,row.get('source_sha256',{}).get('scene.xml')),
                           (trajectory,row.get('source_sha256',{}).get('trajectory_mjwp.npz'))]:
        if expected and digest(path) != expected:
            raise RuntimeError(f'Dataset checksum mismatch: {path}')
    ref = reference(scene, trajectory, output)
    frames = len(ref['wrist_pos'])
    frozen = {key:value.copy() for key,value in ref.items()}
    base = np.array([0,0.8,-0.53])
    model = build_right(scene, output, base)
    fit(model,ref,output,stride,initial_arm_pose=PLAY2PERFECT_ARM_POSE,first_frame_restarts=8)
    audit = json.loads((output/'retargeting.json').read_text())
    audit.update(ready_for_gpu=audit['initial_alignment_pass'] and audit['table_clearance_pass'],
                 reference_mirrored=False, hand='native right Sharpa Wave')
    save(output/'geometry_report.json',audit)
    if not allow_geometry_failure and (not audit['initial_alignment_pass'] or not audit['table_clearance_pass']):
        raise RuntimeError('IK geometry audit failed: initial alignment or robot-table clearance; see retargeting.json. GPU optimization was not started.')
    for key,value in frozen.items():
        np.testing.assert_array_equal(ref[key],value)
    save(output/'preparation.json',dict(
        source=row,reference_source=f"published retarget_full {row['robot_type']} trajectory; not raw human MANO",
        source_scene_sha256=digest(scene),source_trajectory_sha256=digest(trajectory),
        reference_sha256=digest(output/'source_reference.npz'),frames=frames,
        base_position_m=base.tolist(),extra_hold=False,
        alignment='native right hand; no reflection, translation, scaling or time extension',
        reference_arrays_unchanged_after_ik=True,
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir',type=Path,default=ROOT/'example_datasets/retarget_full')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/kuka_sharpa_right_batch20_spider_300')
    parser.add_argument('--selection-file',type=Path,default=ROOT/'outputs/kuka_sharpa_batch20/selected.json')
    parser.add_argument('--count',type=int,default=20)
    parser.add_argument('--samples',type=int,default=512)
    parser.add_argument('--ik-stride',type=int,default=2)
    parser.add_argument('--max-iterations',type=int,default=300)
    parser.add_argument('--target-cost',type=float,default=100)
    parser.add_argument('--plateau-stop',action='store_true')
    parser.add_argument('--patience',type=int,default=30)
    parser.add_argument('--relative-tol',type=float,default=.001)
    parser.add_argument('--absolute-tol',type=float,default=.01)
    parser.add_argument('--ik-cache-dir',type=Path)
    parser.add_argument('--allow-ik-geometry-failure',action='store_true')
    parser.add_argument('--robot-gravity-compensation',action='store_true',
                        help='Set gravcomp=1 for KUKA and hand bodies only, matching simtoolreal.')
    args=parser.parse_args()
    if args.max_iterations < 1:
        parser.error('batch iteration limit must be positive')
    dataset=args.dataset_dir.resolve(); out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    manifest=out/'selected.json'
    if manifest.exists():
        rows=json.loads(manifest.read_text())
        if len(rows)!=args.count:
            raise ValueError('Existing selection count differs; choose a new output directory')
    else:
        rows=json.loads(args.selection_file.read_text()) if args.selection_file.exists() else select(dataset,args.count)
        if len(rows)!=args.count:raise ValueError('Selection count differs')
        save(manifest,rows)
    statuses=[]
    settings=dict(count=args.count,samples=args.samples,max_iterations=args.max_iterations,
                                 min_iterations=1,target_cost=args.target_cost,plateau_stopping=args.plateau_stop,
                                 patience=args.patience,relative_tol=args.relative_tol,absolute_tol=args.absolute_tol,
                                 ik_cache_dir=str(args.ik_cache_dir),allow_ik_geometry_failure=args.allow_ik_geometry_failure,
                                 robot_gravity_compensation=args.robot_gravity_compensation,
                                 target_stopping=False,optimizer='spider_make_optimize_once_v1',
                                 hand='native right Sharpa Wave',extra_hold=False,dataset=str(dataset))
    if (out/'settings.json').exists() and json.loads((out/'settings.json').read_text())!=settings:
        raise ValueError('Existing batch settings differ; choose a new output directory')
    save(out/'settings.json',settings)
    for index,row in enumerate(rows,1):
        folder=out/f'{index:02d}_{row["task"]}'
        state=dict(index=index,task=row['task'],folder=str(folder),status='running')
        statuses.append(state); save(out/'status.json',statuses)
        print(f'CASE {index}/{len(rows)} {row["task"]}',flush=True)
        start=time.monotonic()
        try:
            folder.mkdir(exist_ok=True)
            cache=args.ik_cache_dir/folder.name if args.ik_cache_dir else None
            if cache and (cache/'ik_report.json').exists() and not (folder/'reference.npz').exists():
                for name in ['scene.xml','reference.npz','source_reference.npz','retargeting.json','geometry_report.json']:
                    if (cache/name).exists():shutil.copy2(cache/name,folder/name)
                save(folder/'preparation.json',dict(source=row,ik_cache=str(cache.resolve()),
                    reference_sha256=digest(folder/'reference.npz'),extra_hold=False))
            prepare(row,dataset,folder,args.ik_stride,args.allow_ik_geometry_failure)
            if args.robot_gravity_compensation:
                tree=ET.parse(folder/'scene.xml')
                base=tree.getroot().find("worldbody/body[@name='kuka_base']")
                if base is None:raise ValueError('KUKA base not found')
                for body in [base,*base.findall('.//body')]:body.set('gravcomp','1')
                tree.write(folder/'scene.xml',encoding='unicode')
                model=mujoco.MjModel.from_xml_path(str(folder/'scene.xml'))
                assert model.ngravcomp>0 and model.body_gravcomp[model.body('right_object').id]==0
                save(folder/'gravity_compensation.json',dict(robot_gravcomp=1,object_gravcomp=0,
                    initial_velocity_unchanged=True,reference_sha256=digest(folder/'reference.npz')))
            geometry=json.loads((folder/'geometry_report.json').read_text())
            state['ik_initial_alignment_pass']=geometry['initial_alignment_pass']
            state['ik_table_clearance_pass']=geometry['table_clearance_pass']
            old=ROOT/'outputs/kuka_sharpa_batch20'/folder.name
            for name in ['source_trajectory.mp4','source_visualization.json']:
                if (old/name).exists() and not (folder/name).exists():shutil.copy2(old/name,folder/name)
            gpu=folder/'gpu'; gpu.mkdir(exist_ok=True)
            result_path=gpu/'result.json'
            done=False
            if result_path.exists():
                prior=json.loads(result_path.read_text())
                done=prior['iterations']>=args.max_iterations or (args.plateau_stop and prior['stop_reason']=='cost_plateau')
            if not done:
                command=[sys.executable,'-u','-m','examples.arm_hand.optimize_kuka_gpu',
                         '--input-dir',str(folder),'--output-dir',str(gpu),
                         '--samples',str(args.samples),'--max-iterations',str(args.max_iterations),
                         '--min-iterations','1','--patience',str(args.patience),
                         '--relative-tol',str(args.relative_tol),'--absolute-tol',str(args.absolute_tol),
                         '--target-cost',str(args.target_cost),'--disable-target-stop',
                         '--active','all','--table-collision-weight','0']
                if not args.plateau_stop:command.append('--disable-plateau-stop')
                if args.allow_ik_geometry_failure:command.append('--allow-ik-geometry-failure')
                if (gpu/'checkpoint.pt').exists(): command.append('--resume')
                with (folder/'gpu.log').open('a') as log:
                    subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            result=json.loads(result_path.read_text())
            assert result['iterations']==args.max_iterations or (args.plateau_stop and result['stop_reason']=='cost_plateau')
            state.update(status='complete',best_cost=result['best_cost'],
                         export_rollout_cost=result['export_rollout_cost'],iterations=result['iterations'],
                         stop_reason=result['stop_reason'],below_target=result['best_cost']<args.target_cost)
            with (folder/'audit.log').open('w') as log:
                subprocess.run([sys.executable,'-m','examples.arm_hand.audit_right_trajectory',
                    '--case-dir',str(folder),'--gpu-dir',str(gpu)],cwd=ROOT,
                    stdout=log,stderr=subprocess.STDOUT,check=True)
            video=folder/'best_gpu.mp4'
            if not video.exists():
                with (folder/'render.log').open('a') as log:
                    subprocess.run([sys.executable,'-m','examples.arm_hand.record_gpu_best',
                                    '--input-dir',str(gpu),'--scene',str(folder/'scene.xml'),
                                    '--output',str(video)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        except Exception as error:
            state.update(status='failed',error=f'{type(error).__name__}: {error}')
        state['elapsed_s']=time.monotonic()-start
        save(out/'status.json',statuses)
        print(json.dumps(state),flush=True)
    save(out/'summary.json',dict(completed=sum(s['status']=='complete' for s in statuses),
         failed=sum(s['status']=='failed' for s in statuses),cases=statuses))
    if any(s['status']=='failed' for s in statuses):
        raise SystemExit(1)


if __name__=='__main__':
    main()
