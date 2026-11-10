"""Three CUDA stages: prescribed-wrist hand, fixed-hand arm, joint refinement.

python -u -m examples.arm_hand.staged_kuka_gpu
Each stage runs exactly 100 rounds by default, and writes a video and report.
Re-run to skip completed stages and resume interrupted ones.
"""

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_KEYS = ['wrist_pos','wrist_quat','fingertips','object_qpos']


def signature(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def load_reference(path):
    with np.load(path) as archive:return {k:archive[k] for k in archive.files}


def initial_state(model, original, qpos, qvel, controls):
    data=mujoco.MjData(model)
    data.qpos[:]=qpos;data.qvel[:]=qvel;data.ctrl[:]=controls
    mujoco.mj_forward(model,data)
    state=np.empty(mujoco.mj_stateSize(model,mujoco.mjtState.mjSTATE_FULLPHYSICS))
    mujoco.mj_getState(model,data,state,mujoco.mjtState.mjSTATE_FULLPHYSICS)
    return state


def prepare_hand(source, out):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'preparation.json').exists():return
    ref=load_reference(source/'reference.npz')
    tree=ET.parse(source/'scene.xml');root=tree.getroot();world=root.find('worldbody')
    base=world.find("body[@name='kuka_base']")
    palm=copy.deepcopy(base.find(".//body[@name='right_hand_C_MC']"))
    removed_geoms={g.get('name') for g in base.findall('.//geom') if (g.get('name') or '').startswith('collision_kuka_')}
    world.remove(base)
    # world_palm_site = world_hand_root * local_palm_site.
    # Invert that fixed transform; never move the dataset reference.
    site=palm.find("site[@name='sharpa_palm']")
    local_pos=np.fromstring(site.get('pos','0 0 0'),sep=' ')
    local_quat=np.fromstring(site.get('quat','1 0 0 0'),sep=' ')
    world_rotation=Rotation.from_quat(ref['wrist_quat'][:,[1,2,3,0]])
    root_rotation=world_rotation*Rotation.from_quat(local_quat[[1,2,3,0]]).inv()
    mocap_pos=ref['wrist_pos']-root_rotation.apply(np.broadcast_to(local_pos,ref['wrist_pos'].shape))
    mocap_quat=root_rotation.as_quat()[:,[3,0,1,2]]
    palm.set('mocap','true')
    palm.set('pos',' '.join(map(str,mocap_pos[0])))
    palm.set('quat',' '.join(map(str,mocap_quat[0])))
    world.insert(0,palm)
    actuators=root.find('actuator')
    for node in list(actuators)[:7]:actuators.remove(node)
    contact=root.find('contact')
    for node in list(contact):
        if any(value in removed_geoms for value in node.attrib.values()):contact.remove(node)
    ET.indent(root);tree.write(out/'scene.xml',encoding='unicode')
    model=mujoco.MjModel.from_xml_path(str(out/'scene.xml'))
    assert (model.nq,model.nu,model.nmocap)==(29,22,1)
    full=mujoco.MjModel.from_xml_path(str(source/'scene.xml'));data=mujoco.MjData(full)
    mujoco.mj_setState(full,data,ref['initial_state'],mujoco.mjtState.mjSTATE_FULLPHYSICS)
    ref['joint_ik']=ref['joint_ik'][:,7:].copy()
    ref['ctrl_nominal']=ref['ctrl_nominal'][:,7:].copy()
    ref['initial_state']=initial_state(model,None,np.r_[data.qpos[7:29],data.qpos[-7:]],
                                      np.r_[data.qvel[7:29],data.qvel[-6:]],ref['ctrl_nominal'][0])
    ref.update(mocap_pos=mocap_pos,mocap_quat=mocap_quat)
    np.savez(out/'reference.npz',**ref)
    verify=load_reference(source/'reference.npz')
    for key in REFERENCE_KEYS:np.testing.assert_array_equal(ref[key],verify[key])
    save(out/'geometry_report.json',dict(ready_for_gpu=True,stage='prescribed_wrist_hand',
         note='Original combined IK passed geometry checks; mocap wrist is ideal support. Dynamic contact is audited after optimization.'))
    save(out/'preparation.json',dict(stage='prescribed_wrist_hand',source=str(source),
         source_reference_sha256=signature(source/'reference.npz'),reference_arrays_unchanged=True,
         optimized_actuators=22,extra_hold=False,ideal_support=True))


def prepare_combined(source, out, previous, stage):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'preparation.json').exists():return
    shutil.copy2(source/'scene.xml',out/'scene.xml')
    shutil.copy2(source/'geometry_report.json',out/'geometry_report.json')
    ref=load_reference(source/'reference.npz')
    prev=load_reference(previous/'trajectory.npz')
    if stage=='arm':
        # The first-stage optimized finger commands are immutable in stage 2.
        assert prev['ctrl'].shape==ref['ctrl_nominal'][:,7:].shape
        ref['ctrl_nominal'][:,7:]=prev['ctrl']
        ref['joint_ik'][1:,7:]=prev['qpos'][:,:22]
    else:
        assert prev['ctrl'].shape==ref['ctrl_nominal'].shape
        ref['ctrl_nominal']=prev['ctrl'].copy()
    np.savez(out/'reference.npz',**ref)
    original=load_reference(source/'reference.npz')
    for key in REFERENCE_KEYS:np.testing.assert_array_equal(ref[key],original[key])
    save(out/'preparation.json',dict(stage=stage,source=str(source),
         inherited_trajectory_sha256=signature(previous/'trajectory.npz'),
         reference_arrays_unchanged=True,extra_hold=False))


def audit(folder, stage, baseline):
    model=mujoco.MjModel.from_xml_path(str(folder/'scene.xml'));data=mujoco.MjData(model)
    ref=load_reference(folder/'reference.npz');traj=load_reference(folder/'gpu/trajectory.npz')
    arm=model.nu-22;floor=model.geom('floor').id
    worst_arm=0.;worst_hand=0.;worst_self=0.;palm=[];rot=[];tips=[]
    for i,pose in enumerate(traj['qpos']):
        data.qpos[:]=pose
        if 'mocap_pos' in traj:
            data.mocap_pos[:]=traj['mocap_pos'][i];data.mocap_quat[:]=traj['mocap_quat'][i]
        mujoco.mj_forward(model,data)
        palm.append(data.sensordata[:3].copy());rot.append(data.sensordata[3:7].copy())
        tips.append(data.sensordata[14:26].reshape(4,3).copy())
        for c in data.contact:
            a,b=int(c.geom1),int(c.geom2);na=model.geom(a).name or '';nb=model.geom(b).name or ''
            if floor in [a,b]:
                name=nb if a==floor else na
                if name.startswith('collision_kuka_'):worst_arm=max(worst_arm,-float(c.dist))
                if name.startswith('collision_sharpa_'):worst_hand=max(worst_hand,-float(c.dist))
            if na.startswith('collision_sharpa_') and nb.startswith('collision_sharpa_'):
                worst_self=max(worst_self,-float(c.dist))
    q=traj['qpos'][:,-7:];target=ref['object_qpos'][1:]
    dot=np.abs(np.sum(q[:,3:]*target[:,3:],axis=1))
    angle=2*np.arccos(np.clip(dot/(np.linalg.norm(q[:,3:],axis=1)*np.linalg.norm(target[:,3:],axis=1)),0,1))
    baseline_rmse=None
    if baseline is not None and baseline.exists():
        prev=load_reference(baseline)
        baseline_rmse=float(np.sqrt(np.mean(np.sum((prev['qpos'][:,-7:-4]-target[:,:3])**2,axis=1))))
    report=dict(stage=stage,ideal_wrist_support=bool(model.nmocap),backend='mujoco_warp',
        object_lift_m=float(q[-1,2]-ref['object_qpos'][0,2]),
        object_rmse_m=float(np.sqrt(np.mean(np.sum((q[:,:3]-target[:,:3])**2,axis=1)))),
        final_object_error_m=float(np.linalg.norm(q[-1,:3]-target[-1,:3])),
        final_object_orientation_error_deg=float(np.degrees(angle[-1])),
        wrist_rmse_m=float(np.sqrt(np.mean(np.sum((np.array(palm)-ref['wrist_pos'][1:])**2,axis=1)))),
        fingertip_rmse_m=float(np.sqrt(np.mean(np.sum((np.array(tips)-ref['fingertips'][1:])**2,axis=2)))),
        max_arm_table_penetration_m=worst_arm,max_hand_table_penetration_m=worst_hand,
        max_hand_self_penetration_m=worst_self,extra_hold=False,
        previous_stage_object_rmse_m=baseline_rmse,
        previous_stage_regression=bool(baseline_rmse is not None and np.sqrt(np.mean(np.sum((q[:,:3]-target[:,:3])**2,axis=1)))>baseline_rmse))
    if stage=='arm':
        difference=float(np.max(np.abs(traj['ctrl'][:,7:]-ref['ctrl_nominal'][:,7:])))
        assert difference<1e-6, f'Frozen hand controls changed: {difference}'
        report['frozen_hand_control_max_error']=difference
    report['pose_tracking_pass']=report['final_object_error_m']<.03 and report['final_object_orientation_error_deg']<20
    report['table_clearance_pass']=max(worst_arm,worst_hand)<.001
    save(folder/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=ROOT/'outputs/kuka_sharpa_right_foam_clearance')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/kuka_sharpa_staged_spider_foam')
    parser.add_argument('--iterations',type=int,default=100)
    parser.add_argument('--samples',type=int,default=512)
    args=parser.parse_args();source=args.input_dir.resolve();out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    source_report=json.loads((source/'geometry_report.json').read_text())
    if not source_report['ready_for_gpu']:raise RuntimeError('Combined initial geometry audit must pass first')
    settings=dict(iterations_per_stage=args.iterations,samples=args.samples,pipeline_version='original_spider_sampling_v1',
                  source_scene_sha256=signature(source/'scene.xml'),source_reference_sha256=signature(source/'reference.npz'))
    if (out/'settings.json').exists() and json.loads((out/'settings.json').read_text())!=settings:
        raise ValueError('Existing run settings differ; choose a new output directory')
    save(out/'settings.json',settings)
    stages=[('hand',out/'01_hand'),('arm',out/'02_arm'),('joint',out/'03_joint')]
    reports=[];previous=None
    for stage,folder in stages:
        save(out/'status.json',dict(current_stage=stage,status='running',completed_stages=len(reports)))
        print(f'STAGE {stage}',flush=True)
        if stage=='hand':prepare_hand(source,folder)
        else:prepare_combined(source,folder,previous,stage)
        gpu=folder/'gpu';gpu.mkdir(exist_ok=True)
        result_path=gpu/'result.json';done=False
        if result_path.exists():
            done=json.loads(result_path.read_text())['iterations']>=args.iterations
        if not done:
            command=[sys.executable,'-u','-m','examples.arm_hand.optimize_kuka_gpu',
                '--input-dir',str(folder),'--output-dir',str(gpu),'--samples',str(args.samples),
                '--max-iterations',str(args.iterations),'--disable-target-stop','--disable-plateau-stop',
                '--active','all' if stage=='joint' else stage,
                '--table-collision-weight','0']
            if (gpu/'checkpoint.pt').exists():command+=['--resume']
            with (folder/'training.log').open('a') as log:
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        report=audit(folder,stage,previous/'trajectory.npz' if previous else None)
        result=json.loads(result_path.read_text())
        report.update(best_cost=result['best_cost'],export_rollout_cost=result['export_rollout_cost'],iterations=result['iterations'])
        save(folder/'report.json',report)
        video=folder/'best_gpu.mp4'
        if not video.exists():
            with (folder/'render.log').open('w') as log:
                subprocess.run([sys.executable,'-m','examples.arm_hand.record_gpu_best','--input-dir',str(gpu),
                    '--scene',str(folder/'scene.xml'),'--output',str(video)],cwd=ROOT,
                    stdout=log,stderr=subprocess.STDOUT,check=True)
        reports.append(report);previous=gpu
        save(out/'comparison.json',reports)
        print(json.dumps(report),flush=True)
    save(out/'status.json',dict(status='complete',completed_stages=3))


if __name__=='__main__':main()
