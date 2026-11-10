"""Audit saved right-hand GPU states for tracking and table penetration."""

import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from examples.run_arm_hand import quaternion_error


def audit_states(model,ref,traj,backend='mujoco_warp'):
    data=mujoco.MjData(model)
    floor=model.geom('floor').id
    worst_arm=0.;worst_hand=0.;worst_self=0.;arm_frames=[];palm=[]
    for frame,pose in enumerate(traj['qpos']):
        data.qpos[:]=pose;mujoco.mj_forward(model,data);palm.append(data.site('sharpa_palm').xpos.copy())
        touched=False
        for c in data.contact:
            a,b=int(c.geom1),int(c.geom2);na=model.geom(a).name or '';nb=model.geom(b).name or ''
            if floor in [a,b]:
                other=nb if a==floor else na
                if other.startswith('collision_kuka_'):
                    worst_arm=max(worst_arm,-float(c.dist));touched=touched or c.dist<-.001
                elif other.startswith('collision_sharpa_'):
                    worst_hand=max(worst_hand,-float(c.dist))
            if na.startswith('collision_sharpa_') and nb.startswith('collision_sharpa_'):
                worst_self=max(worst_self,-float(c.dist))
        if touched:arm_frames.append(frame)
    mujoco.mj_setState(model,data,traj['initial_state'],mujoco.mjtState.mjSTATE_FULLPHYSICS)
    mujoco.mj_forward(model,data)
    return dict(
        trajectory_backend=f'GPU {backend}; CPU source-model forward kinematics/contact geometry audit only',
        initial_palm_error_m=float(np.linalg.norm(data.site('sharpa_palm').xpos-ref['wrist_pos'][0])),
        initial_fingertip_errors_m=np.linalg.norm(data.sensordata[14:26].reshape(4,3)-ref['fingertips'][0],axis=1).tolist(),
        object_lift_m=float(traj['qpos'][-1,-5]-ref['object_qpos'][0,2]),
        final_object_position_error_m=float(np.linalg.norm(traj['qpos'][-1,-7:-4]-ref['object_qpos'][-1,:3])),
        final_object_orientation_error_deg=float(np.degrees(np.linalg.norm(quaternion_error(traj['qpos'][-1,-4:],ref['object_qpos'][-1,3:])))),
        object_rmse_m=float(np.sqrt(np.mean(np.sum((traj['qpos'][:,-7:-4]-ref['object_qpos'][1:,:3])**2,axis=1)))),
        palm_rmse_m=float(np.sqrt(np.mean(np.sum((np.array(palm)-ref['wrist_pos'][1:])**2,axis=1)))),
        max_arm_table_penetration_m=worst_arm,max_hand_table_penetration_m=worst_hand,
        max_hand_self_penetration_m=worst_self,arm_table_frames_over_1mm=arm_frames,
        arm_table_clearance_pass=worst_arm<.001,extra_hold=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-dir',type=Path,default=Path('outputs/kuka_sharpa_right_foam'))
    parser.add_argument('--gpu-dir',type=Path)
    args=parser.parse_args();p=args.case_dir;gpu=args.gpu_dir or p/'gpu'
    model=mujoco.MjModel.from_xml_path(str(p/'scene.xml'));data=mujoco.MjData(model)
    ref=np.load(p/'reference.npz');traj=np.load(gpu/'trajectory.npz')
    result_path=next((q for q in [gpu/'result.json',gpu.parent/'result.json'] if q.exists()),None)
    backend=json.loads(result_path.read_text()).get('backend','mujoco_warp') if result_path else 'mujoco_warp'
    report=audit_states(model,ref,traj,backend)
    (gpu/'tracking_collision_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
