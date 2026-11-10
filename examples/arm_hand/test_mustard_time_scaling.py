"""CPU diagnosis: can time dilation repair the default SimToolReal control bridge?"""
from pathlib import Path
import argparse
import importlib.util
import json
import numpy as np
import mujoco
import torch
from examples.arm_hand.test_mustard_simtoolreal_bridge import action_env, inverse, rollout, angle, save


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,default=Path('outputs/mustard_sapg_bridge20_v1'))
    args=ap.parse_args();torch.set_num_threads(1)
    path='/home/szc/work/simtoolreal_origin/simtoolreal/isaacsimenvs/tasks/simtoolreal/utils/action_utils.py'
    spec=importlib.util.spec_from_file_location('action_pipeline',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    manifest=json.loads((args.run/'manifest.json').read_text())
    results=[]
    for item in manifest[:5]:
        folder=args.run/f"candidate_{item['candidate_id']:05d}"
        m=mujoco.MjModel.from_xml_path(str(folder/'scene.xml'))
        d=np.load(folder/'gpu/trajectory.npz');ref=np.load(folder/'reference.npz')
        data=mujoco.MjData(m);mujoco.mj_setState(m,data,d['initial_state'],mujoco.mjtState.mjSTATE_FULLPHYSICS)
        q0=data.qpos[m.jnt_qposadr[m.actuator_trnid[:,0]]].copy()
        for factor in [1,2,4,8]:
            # Initial transition is included in the slower timing, not bypassed at reset.
            knots=np.vstack([q0,d['ctrl']]);knot_t=np.arange(len(knots))*.01*factor
            times=np.arange(int(np.ceil(knot_t[-1]*60)))/60
            desired=np.stack([np.interp(times,knot_t,knots[:,j]) for j in range(m.nu)],axis=1)
            env=action_env(m,q0);commands=[];raw=[]
            for target in desired:
                a=inverse(env,target);raw.append(a)
                mod.apply_action_pipeline(env,torch.tensor(np.clip(a,-1,1)[None],dtype=torch.float64))
                commands.append(env._cur_targets.numpy()[0].copy())
            final=rollout(m,d['initial_state'],np.array(commands),1/60);goal=ref['object_qpos'][-1]
            row=dict(candidate_id=item['candidate_id'],time_scale=factor,duration_s=len(times)/60,
                     position_error_m=float(np.linalg.norm(final[:3]-goal[:3])),orientation_error_deg=angle(final[3:],goal[3:]),
                     feedback_action_saturation_fraction=float((np.abs(raw)>1).mean()))
            row['pose_only_pass']=row['position_error_m']<.03 and row['orientation_error_deg']<20
            results.append(row);print(json.dumps(row),flush=True)
    save(args.run/'time_scaling_report.json',dict(backend='CPU MuJoCo + actual SimToolReal action function, NOT PhysX',rows=results))


if __name__=='__main__':main()
