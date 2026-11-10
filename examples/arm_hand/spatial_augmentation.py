"""Generate 20x5 shared planar/yaw variants, screen IK, then validate GPU physics."""
import argparse
import concurrent.futures
import contextlib
import multiprocessing
import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from examples.arm_hand.batch_kuka_gpu import ROOT, save, digest
from examples.arm_hand.run_sharpa_complex import fit,InitialGeometryRejected
from examples.arm_hand.validate_kuka_sharpa import PLAY2PERFECT_ARM_POSE

KEYS=['wrist_pos','wrist_quat','fingertips','object_qpos']


def augment(ref, yaw, delta, base):
    rotation=Rotation.from_euler('z',yaw,degrees=True)
    center=np.array([*base[:2],0.])
    rotated_object=rotation.apply(ref['object_qpos'][0,:3]-center)+center
    # Use each source object's original initial XY as its workspace anchor.
    shift=np.r_[ref['object_qpos'][0,:2]+delta-rotated_object[:2],0.]
    def points(x):return rotation.apply((x-center).reshape(-1,3)).reshape(x.shape)+center+shift
    def quats(q):return (rotation*Rotation.from_quat(q[:,[1,2,3,0]])).as_quat()[:,[3,0,1,2]]
    result=dict(wrist_pos=points(ref['wrist_pos']),wrist_quat=quats(ref['wrist_quat']),
        fingertips=points(ref['fingertips']),object_qpos=np.c_[points(ref['object_qpos'][:,:3]),quats(ref['object_qpos'][:,3:])])
    for key in ['wrist_pos','fingertips']:
        relative=ref[key]-ref['object_qpos'][:,:3].reshape((-1,)+(1,)*(ref[key].ndim-2)+(3,))
        transformed=result[key]-result['object_qpos'][:,:3].reshape((-1,)+(1,)*(ref[key].ndim-2)+(3,))
        np.testing.assert_allclose(transformed,rotation.apply(relative.reshape(-1,3)).reshape(relative.shape),atol=1e-12)
    for key in ['wrist_pos','fingertips']:
        np.testing.assert_allclose(result[key][...,2],ref[key][...,2],atol=1e-12)
    return result,dict(yaw_deg=float(yaw),sampled_translation_xy_m=delta.tolist(),
        anchor_xy_m=ref['object_qpos'][0,:2].tolist(),shared_translation_xyz_m=shift.tolist(),
        rotation_center_xyz_m=center.tolist(),relative_interaction_geometry_preserved=True)


def object_footprint_prefilter(model,object_qpos):
    """Conservative early rejection; the original full IK audit still decides survivors."""
    floor=model.geom('floor').id;body=model.body('right_object').id
    if model.geom_bodyid[floor]!=0:return dict(definitely_outside=False,unsupported=True)
    geometry=[g for g in range(model.ngeom) if (model.geom(g).name or '').startswith('right_object_') and model.geom(g).name!='right_object_visual']
    if any(model.geom_bodyid[g]!=body for g in geometry):return dict(definitely_outside=False,unsupported=True)
    poses=np.asarray(object_qpos);rot=Rotation.from_quat(poses[:,[4,5,6,3]]).as_matrix()
    outside=np.zeros(len(poses),dtype=bool);worst=-float('inf')
    for g in geometry:
        local=Rotation.from_quat(model.geom_quat[g][[1,2,3,0]]).as_matrix();world_rot=rot@local
        center=poses[:,:3]+(rot@model.geom_pos[g][:,None]).squeeze(-1)+(world_rot@model.geom_aabb[g,:3,None]).squeeze(-1)
        extent=(np.abs(world_rot)@model.geom_aabb[g,3:,None]).squeeze(-1)
        violation=np.abs(center[:,:2]-model.geom_pos[floor,:2])+extent[:,:2]-(model.geom_size[floor,:2]-.001)
        outside|=(violation>0).any(-1);worst=max(worst,float(violation.max()))
    return dict(definitely_outside=worst>1e-7,table_footprint_pass=not bool(outside.any()),outside_table_frames=np.flatnonzero(outside).tolist(),max_footprint_violation_m=worst)


def check_ik(folder):
    model=mujoco.MjModel.from_xml_path(str(folder/'scene.xml'));data=mujoco.MjData(model)
    with np.load(folder/'reference.npz') as a:ref={k:a[k] for k in a.files}
    raw=json.loads((folder/'retargeting.json').read_text())
    wrist=[];tips=[];worst_self=0.;outside=[]
    floor=model.geom('floor').id;floor_pos=model.geom_pos[floor];floor_size=model.geom_size[floor]
    object_geoms=[g for g in range(model.ngeom) if (model.geom(g).name or '').startswith('right_object_') and model.geom(g).name!='right_object_visual']
    for i,joints in enumerate(ref['joint_ik']):
        data.qpos[:]=np.r_[joints,ref['object_qpos'][i]];mujoco.mj_forward(model,data)
        wrist.append(np.linalg.norm(data.sensordata[:3]-ref['wrist_pos'][i]))
        tips.append(np.linalg.norm(data.sensordata[14:26].reshape(4,3)-ref['fingertips'][i],axis=1))
        for c in data.contact:
            if all((model.geom(int(g)).name or '').startswith('collision_sharpa_') for g in [c.geom1,c.geom2]):
                worst_self=max(worst_self,-float(c.dist))
        for g in object_geoms:
            rot=data.geom_xmat[g].reshape(3,3);aabb=model.geom_aabb[g]
            center=data.geom_xpos[g]+rot@aabb[:3];extent=np.abs(rot)@aabb[3:]
            if np.any(np.abs(center[:2]-floor_pos[:2])+extent[:2]>floor_size[:2]-.001):
                outside.append(i);break
    low,high=model.actuator_ctrlrange.T
    limits=bool(np.all(ref['joint_ik']>=low-1e-8) and np.all(ref['joint_ik']<=high+1e-8))
    passed=bool(raw['initial_alignment_pass'] and raw['table_clearance_pass'] and
        np.mean(wrist)<=.08 and worst_self<.001 and limits and not outside)
    raw.update(ready_for_gpu=passed,ik_pass=passed,mean_wrist_error_m=float(np.mean(wrist)),
        max_wrist_error_m=float(max(wrist)),mean_fingertip_error_m=float(np.mean(tips)),
        max_hand_self_penetration_m=worst_self,joint_limits_pass=limits,
        table_footprint_pass=not outside,outside_table_frames=outside,
        thresholds=dict(mean_wrist_error_m=.08,robot_table_penetration_m=.001,
                        hand_self_penetration_m=.001,table_edge_clearance_m=.001),
        note='8 cm mean wrist threshold follows paper; additional collision, initial alignment and table-footprint filters are project checks.')
    save(folder/'geometry_report.json',raw)
    return raw


def summarize(out,states):
    screened=sum(s['status'] not in ['generated','ik_running'] for s in states)
    passed=sum(s.get('ik_pass',False) for s in states)
    evaluated=sum('physical_success' in s for s in states)
    success=sum(s.get('physical_success',False) for s in states)
    summary=dict(total_candidates=len(states),ik_screened=screened,ik_passed=passed,
        ik_pass_rate_among_screened=passed/screened if screened else None,
        physical_evaluated=evaluated,physical_successes=success,
        physical_errors=sum(s['status']=='physics_failed' for s in states),
        physical_success_rate_among_evaluated=success/evaluated if evaluated else None,
        validated_success_fraction_of_all_candidates=success/len(states),
        pending_physics=sum(s.get('ik_pass',False) and 'physical_success' not in s and s['status']!='physics_failed' for s in states),
        cases=states)
    summary['per_source']=[dict(task=task,total=5,
        ik_screened=sum(s['status'] not in ['generated','ik_running'] for s in states if s['task']==task),
        ik_passed=sum(s.get('ik_pass',False) for s in states if s['task']==task),
        physical_evaluated=sum('physical_success' in s for s in states if s['task']==task),
        physical_successes=sum(s.get('physical_success',False) for s in states if s['task']==task))
        for task in dict.fromkeys(s['task'] for s in states)]
    save(out/'summary.json',summary);save(out/'status.json',states)


def solve_ik_candidate(state):
    """Independent CPU worker: preserve the solver and per-case random seeds."""
    state=dict(state);folder=Path(state['folder'])
    try:
        with (folder/'ik.log').open('a',buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            if not (folder/'retargeting.json').exists():
                with np.load(folder/'source_reference.npz') as a:ref={k:a[k] for k in a.files}
                frozen={k:v.copy() for k,v in ref.items()}
                model=mujoco.MjModel.from_xml_path(str(folder/'scene.xml'))
                footprint=object_footprint_prefilter(model,ref['object_qpos'])
                if footprint['definitely_outside']:
                    save(folder/'geometry_report.json',dict(footprint,ready_for_gpu=False,ik_pass=False,
                        ik_not_solved=True,reason='Object reference definitely outside original table footprint before IK'))
                    state.update(ik_pass=False,status='ik_rejected',ik_skipped_reason='object_table_footprint')
                    save(folder/'state.json',state)
                    return state
                fit(model,ref,folder,2,initial_arm_pose=PLAY2PERFECT_ARM_POSE,first_frame_restarts=8,reject_bad_initial=True)
                for k,v in frozen.items():np.testing.assert_array_equal(ref[k],v)
            report=check_ik(folder)
        state.update(status='ik_passed' if report['ik_pass'] else 'ik_rejected',ik_pass=report['ik_pass'])
    except InitialGeometryRejected as error:
        save(folder/'geometry_report.json',error.report)
        state.update(status='ik_rejected',ik_pass=False,ik_skipped_reason='initial_geometry')
    except Exception as e:state.update(status='ik_failed',ik_pass=False,error=str(e))
    save(folder/'state.json',state)
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/kuka_sharpa_spatial100')
    parser.add_argument('--generate-only',action='store_true')
    parser.add_argument('--ik-workers',type=int,default=8)
    args=parser.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    rows=json.loads((ROOT/'outputs/kuka_sharpa_batch20/selected.json').read_text())
    settings=dict(seed=0,variants_per_source=5,translation_bound_m=.05,yaw_bound_deg=10,
        max_iterations=200,samples=512,patience=30,robot_gravcomp=1,object_gravcomp=0,
        anchor='original source initial object XY',extra_hold=False,version=1)
    if (out/'settings.json').exists() and json.loads((out/'settings.json').read_text())!=settings:
        raise ValueError('Run settings differ')
    save(out/'settings.json',settings);rng=np.random.default_rng(0);states=[]
    for i,row in enumerate(rows,1):
        source=ROOT/'outputs/kuka_sharpa_right_batch20_ik'/f'{i:02d}_{row["task"]}'
        for j in range(1,6):
            yaw=rng.uniform(-10,10);delta=rng.uniform(-.05,.05,2)
            folder=out/f'{i:02d}_{row["task"]}'/f'variant_{j:02d}';folder.mkdir(parents=True,exist_ok=True)
            if not (folder/'augmentation.json').exists():
                with np.load(source/'source_reference.npz') as a:ref={k:a[k] for k in KEYS}
                tree=ET.parse(source/'scene.xml');base=tree.getroot().find("worldbody/body[@name='kuka_base']")
                base_pos=np.fromstring(base.get('pos'),sep=' ')
                for body in [base,*base.findall('.//body')]:body.set('gravcomp','1')
                tree.write(folder/'scene.xml',encoding='unicode')
                transformed,metadata=augment(ref,yaw,delta,base_pos)
                np.savez(folder/'source_reference.npz',**transformed)
                metadata.update(source_task=row['task'],source_reference_sha256=digest(source/'source_reference.npz'),
                    original_reference_path=str(source/'source_reference.npz'),seed=0,variant=j)
                save(folder/'augmentation.json',metadata)
            state_path=folder/'state.json'
            state=json.loads(state_path.read_text()) if state_path.exists() else dict(source_index=i,variant=j,task=row['task'],folder=str(folder),status='generated')
            states.append(state)
    summarize(out,states);print('Generated 100 candidates',flush=True)
    if args.generate_only:return
    # Finish all geometric screening before starting expensive physical optimization.
    pending=[i for i,s in enumerate(states) if 'ik_pass' not in s]
    save(out/'phase.json',dict(phase='parallel_cpu_ik',workers=args.ik_workers,pending=len(pending)))
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.ik_workers,
            mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(solve_ik_candidate,states[i]):i for i in pending}
        for future in concurrent.futures.as_completed(futures):
            i=futures[future];states[i]=future.result();summarize(out,states)
            print(json.dumps(states[i]),flush=True)
    # Existing original-20 training keeps its GPU until it finishes.
    old_batch=ROOT/'outputs/kuka_sharpa_right_batch20_gravcomp_plateau200'
    while not (old_batch/'summary.json').exists():
        save(out/'phase.json',dict(phase='waiting_for_original_20_gpu_batch',batch=str(old_batch)))
        time.sleep(30)
    for state in states:
        if not state['ik_pass'] or state['status'] in ['complete','physics_failed']:continue
        folder=Path(state['folder']);gpu=folder/'gpu';gpu.mkdir(exist_ok=True)
        state['status']='physics_running';save(folder/'state.json',state);summarize(out,states)
        save(out/'phase.json',dict(phase='physics',folder=str(folder)))
        try:
            if not (gpu/'result.json').exists():
                command=[sys.executable,'-u','-m','examples.arm_hand.optimize_kuka_gpu',
                    '--input-dir',str(folder),'--output-dir',str(gpu),'--samples','512','--seed','0',
                    '--max-iterations','200','--patience','30','--relative-tol','.001','--absolute-tol','.01',
                    '--disable-target-stop','--active','all','--table-collision-weight','0']
                if (gpu/'checkpoint.pt').exists():command.append('--resume')
                with (folder/'training.log').open('a') as log:
                    subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            with (folder/'audit.log').open('w') as log:
                subprocess.run([sys.executable,'-m','examples.arm_hand.audit_right_trajectory',
                    '--case-dir',str(folder),'--gpu-dir',str(gpu)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            report=json.loads((gpu/'tracking_collision_report.json').read_text())
            success=bool(report['object_rmse_m']<.03 and report['final_object_position_error_m']<.03
                and report['final_object_orientation_error_deg']<20
                and max(report['max_arm_table_penetration_m'],report['max_hand_table_penetration_m'],report['max_hand_self_penetration_m'])<.001)
            if not (folder/'best_gpu.mp4').exists():
                with (folder/'render.log').open('w') as log:
                    subprocess.run([sys.executable,'-m','examples.arm_hand.record_gpu_best',
                        '--input-dir',str(gpu),'--scene',str(folder/'scene.xml'),
                        '--output',str(folder/'best_gpu.mp4')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            state.update(status='complete',physical_success=success,metrics=report)
        except Exception as e:state.update(status='physics_failed',physical_success=False,error=str(e))
        save(folder/'state.json',state);summarize(out,states);print(json.dumps(state),flush=True)
    save(out/'phase.json',dict(phase='complete'))


if __name__=='__main__':main()
