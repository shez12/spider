"""Retrain one existing GPU case with robot-only gravity compensation."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from examples.arm_hand.batch_kuka_gpu import ROOT, save


def startup(folder, optimized=False):
    model=mujoco.MjModel.from_xml_path(str(folder/'scene.xml'))
    data=mujoco.MjData(model)
    with np.load(folder/'reference.npz') as ref:
        mujoco.mj_setState(model,data,ref['initial_state'],mujoco.mjtState.mjSTATE_FULLPHYSICS)
        target=ref['wrist_pos'].copy();commands=ref['ctrl_nominal'].copy()
    mujoco.mj_forward(model,data);z0=float(data.site('sharpa_palm').xpos[2])
    zs=[]
    if optimized:
        with np.load(folder/'gpu/trajectory.npz') as traj:
            for pose in traj['qpos'][:20]:
                data.qpos[:]=pose;mujoco.mj_forward(model,data)
                zs.append(float(data.site('sharpa_palm').xpos[2]))
    else:
        for ctrl in commands[:20]:
            data.ctrl[:]=ctrl
            for _ in range(5):mujoco.mj_step(model,data)
            mujoco.mj_forward(model,data);zs.append(float(data.site('sharpa_palm').xpos[2]))
    return dict(backend='saved_GPU_states' if optimized else 'CPU_diagnostic_only',
        times_ms=[10,50,100,200],
        palm_z_delta_mm=((np.array(zs)[[0,4,9,19]]-z0)*1000).tolist(),
        source_palm_z_delta_mm=((target[[1,5,10,20],2]-target[0,2])*1000).tolist())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'outputs/kuka_sharpa_right_batch20_spider_plateau300/01_dexycb_01_right_002_master_chef_can_20200709_141754_00')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/kuka_sharpa_gravcomp_master_chef_can')
    args=parser.parse_args();source=args.source.resolve();out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    for name in ['reference.npz','source_reference.npz','retargeting.json','geometry_report.json']:
        shutil.copy2(source/name,out/name)
    tree=ET.parse(source/'scene.xml');base=tree.getroot().find("worldbody/body[@name='kuka_base']")
    assert base is not None
    bodies=[base,*base.findall('.//body')]
    for body in bodies:body.set('gravcomp','1')
    tree.write(out/'scene.xml',encoding='unicode')
    model=mujoco.MjModel.from_xml_path(str(out/'scene.xml'))
    assert model.body_gravcomp[model.body('right_object').id]==0
    assert model.ngravcomp>0
    assert (out/'reference.npz').read_bytes()==(source/'reference.npz').read_bytes()
    save(out/'experiment.json',dict(source=str(source),changed='robot body gravcomp=1 only',
        robot_bodies=len(bodies),object_gravcomp=0,initial_velocity_unchanged=True,
        reference_sha256=hashlib.sha256((out/'reference.npz').read_bytes()).hexdigest(),
        max_iterations=300,patience=30,seed=0,samples=512,extra_hold=False))
    save(out/'startup_nominal_comparison.json',dict(without_gravcomp=startup(source),with_gravcomp=startup(out)))
    save(out/'status.json',dict(status='training'))
    gpu=out/'gpu';gpu.mkdir(exist_ok=True)
    command=[sys.executable,'-u','-m','examples.arm_hand.optimize_kuka_gpu',
        '--input-dir',str(out),'--output-dir',str(gpu),'--samples','512','--seed','0',
        '--max-iterations','300','--patience','30','--relative-tol','.001','--absolute-tol','.01',
        '--disable-target-stop','--active','all','--table-collision-weight','0','--allow-ik-geometry-failure']
    if (gpu/'checkpoint.pt').exists():command.append('--resume')
    with (out/'training.log').open('a') as log:
        subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    for module,arguments in [
        ('audit_right_trajectory',['--case-dir',str(out),'--gpu-dir',str(gpu)]),
        ('record_gpu_best',['--input-dir',str(gpu),'--scene',str(out/'scene.xml'),'--output',str(out/'best_gpu.mp4')])]:
        with (out/f'{module}.log').open('w') as log:
            subprocess.run([sys.executable,'-m',f'examples.arm_hand.{module}',*arguments],cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT,check=True)
    report={}
    for label,folder in [('without_gravcomp',source),('with_gravcomp',out)]:
        report[label]=dict(startup=startup(folder,True),
            metrics=json.loads((folder/'gpu/tracking_collision_report.json').read_text()),
            optimization=json.loads((folder/'gpu/result.json').read_text()))
    save(out/'comparison.json',report);save(out/'status.json',dict(status='complete'))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
