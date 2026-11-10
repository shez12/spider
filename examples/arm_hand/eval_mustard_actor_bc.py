"""Evaluate the absolute-target BC actor on source-simulator closed-loop rollouts."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import mujoco
import torch

from examples.arm_hand.train_mustard_actor_bc import build_model,make_obs,forward,zero_state
from examples.arm_hand.test_mustard_simtoolreal_bridge import save,angle


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,default=Path('outputs/mustard_actor_bc20_v2'))
    ap.add_argument('--frozen',type=Path,default=Path('outputs/mustard_sapg_bridge20_v1'))
    ap.add_argument('--simtoolreal',type=Path,default=Path('/home/szc/work/simtoolreal_origin/simtoolreal'))
    args=ap.parse_args();torch.set_num_threads(2);started=time.monotonic()
    ckpt=torch.load(args.run/'actor_best.pt',map_location='cpu',weights_only=False)
    model,_=build_model(args.simtoolreal,1,torch.device('cpu'));model.load_state_dict(ckpt['model']);model.eval()
    schema=ckpt['schema'];manifest=json.loads((args.frozen/'manifest.json').read_text())
    model.bc_residual=schema.get('residual_decoder',False)
    if model.bc_residual:model.bc_limits=torch.tensor(schema['joint_limits_rad'])
    dataset=np.load(args.run/'dataset_60hz.npz');rows=[]
    for index,item in enumerate(manifest):
        folder=args.frozen/f"candidate_{item['candidate_id']:05d}"
        m=mujoco.MjModel.from_xml_path(str(folder/'scene.xml'));d=np.load(folder/'gpu/trajectory.npz');ref=np.load(folder/'reference.npz')
        limits=m.jnt_range[m.actuator_trnid[:,0]]
        data=mujoco.MjData(m);mujoco.mj_setState(m,data,d['initial_state'],mujoco.mjtState.mjSTATE_FULLPHYSICS);mujoco.mj_forward(m,data)
        previous=data.qpos[m.jnt_qposadr[m.actuator_trnid[:,0]]].copy()
        h=zero_state(model,1);targets=[];poses=[]
        # Evaluate task policy at SAPG coefficient ID 0 (the low-exploration group).
        for t in range(len(dataset['time_s'])):
            obs=make_obs(m,data,previous,ref['object_qpos'][-1],schema['object_calibration'])
            with torch.no_grad():
                pred,h=forward(model,torch.tensor(obs)[None,None],h,torch.zeros(1))
            mu=pred.numpy()[0,0];target=limits[:,0]+.5*(np.clip(mu,-1,1)+1)*(limits[:,1]-limits[:,0])
            data.ctrl[:]=target;previous=target.copy();targets.append(target)
            # ZOH policy scheduling on the source engine's fixed 2 ms physics grid.
            end_time=min((t+1)/60,len(d['ctrl'])*float(d['control_dt']))
            while data.time < end_time-1e-10:
                mujoco.mj_step(m,data)
            poses.append(data.qpos.copy())
        goal=ref['object_qpos'][-1]
        row=dict(candidate_id=item['candidate_id'],split=item['split'],
                 final_position_error_m=float(np.linalg.norm(data.qpos[-7:-4]-goal[:3])),
                 final_orientation_error_deg=angle(data.qpos[-4:],goal[3:]),
                 target_rmse_vs_demo_rad=float(np.sqrt(np.mean((np.array(targets)-dataset['targets_rad'][index])**2))),
                 final_object_z_m=float(data.qpos[-5]),finite=bool(np.isfinite(data.qpos).all()))
        row['pose_only_pass']=row['final_position_error_m']<.03 and row['final_orientation_error_deg']<20
        rows.append(row)
        np.savez_compressed(args.run/f"closed_loop_{item['candidate_id']:05d}.npz",ctrl=np.array(targets),qpos=np.array(poses),control_dt=1/60,physics_dt=m.opt.timestep)
        print(json.dumps(row),flush=True)
    groups={split:dict(count=sum(r['split']==split for r in rows),pose_only_pass=sum(r['pose_only_pass'] and r['split']==split for r in rows)) for split in ['train','validation']}
    report=dict(backend='CPU source MuJoCo, absolute-target BC actor, coefficient ID 0, no action sampling',rows=rows,groups=groups,
                elapsed_s=time.monotonic()-started,limitations=['Not PhysX or native SAPG action pipeline.','Pose threshold only, no hold/collision acceptance.','Same source episode across spatial-variant split.'])
    save(args.run/'closed_loop_report.json',report)


if __name__=='__main__':main()
