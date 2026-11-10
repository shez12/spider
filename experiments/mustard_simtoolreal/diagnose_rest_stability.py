"""Is a source's initial object pose a stable rest in Isaac? Hand parked away, 1 s of physics.

Object pose follows train_bc.collect (MuJoCo initial_state + calibration center offset, z+0.53).
Variants: extra start height (m) and object/table friction.
"""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parents[1]
sys.path[:0]=[str(PROJECT/'third_party/simtoolreal'),str(PROJECT/'third_party/simtoolreal/rl_games'),str(ROOT)]


def main():
    from isaaclab.app import AppLauncher
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',type=int,nargs='+',default=[5,13,16,9]);p.add_argument('--per-source',type=int,default=8)
    p.add_argument('--heights',type=float,nargs='+',default=[0.,.002,.005]);p.add_argument('--frictions',type=float,nargs='+',default=[.5,1.])
    p.add_argument('--output',type=Path,required=True)
    AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
    try:
        import numpy as np,torch
        from scipy.spatial.transform import Rotation
        from environment import create_env
        man=json.loads((PROJECT/'outputs/mustard_classified_native_20261102_inputs/source_manifest.json').read_text())
        items=[x for s in a.sources for x in [m for m in man if m['source_index']==s][:a.per_source]]
        cal=json.loads((ROOT/'assets/object/calibration.json').read_text());center=np.array(cal['source_to_new_root_center_in_source_m'])
        poses=np.stack([np.load(m['trajectory'])['initial_state'][30:37] for m in items])
        poses[:,:3]+=Rotation.from_quat(poses[:,[4,5,6,3]]).apply(np.broadcast_to(center,(len(items),3)));poses[:,2]+=.53
        results=[]
        for fr in a.frictions:
            variants=[(h,i) for h in a.heights for i in range(len(items))];n=len(variants)
            def configure(c):
                dr=c.domain_randomization;dr.use_action_delay=False;dr.use_obs_delay=False;dr.use_object_state_delay_noise=False
                dr.joint_velocity_obs_noise_std=0.;dr.force_scale=0.;dr.torque_scale=0.;c.reset.table_reset_z_range=0.
                c.assets.robot_friction=fr
            env=create_env(n,configure=configure);env.reset();dev=env.device
            # Park the arm at its default pose (far from the table centre) and hold it there.
            pose=torch.tensor(np.stack([poses[i] for _,i in variants]),device=dev,dtype=torch.float32)
            pose[:,2]+=torch.tensor([h for h,_ in variants],device=dev)
            st=env.object.data.default_root_state.clone();st[:,:7]=pose;st[:,:3]+=env.scene.env_origins;st[:,7:]=0
            env.object.write_root_state_to_sim(st);env.scene.write_data_to_sim();env.sim.forward();env.scene.update(0.)
            from diverse_resets import hold_actions
            from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import compute_intermediate_values
            from isaacsimenvs.tasks.simtoolreal.utils.reward_utils import compute_rewards
            def no_done():
                compute_intermediate_values(env);z=torch.zeros(n,device=dev,dtype=torch.bool);return z,z.clone()
            env._get_dones=no_done;env._get_rewards=lambda:compute_rewards(env)
            for _ in range(60):env.step(hold_actions(env))
            q0=Rotation.from_quat(pose[:,[4,5,6,3]].cpu().numpy());q1=env.object.data.root_quat_w.cpu().numpy()
            ang=np.degrees((q0.inv()*Rotation.from_quat(q1[:,[1,2,3,0]])).magnitude())
            pos=(env.object.data.root_pos_w-env.scene.env_origins).cpu().numpy();disp=np.linalg.norm(pos-pose[:,:3].cpu().numpy(),axis=1)
            for k,(h,i) in enumerate(variants):results.append(dict(friction=fr,height=h,source=items[i]['source_index'],rotation_deg=float(ang[k]),displacement_m=float(disp[k])))
            env.close()
        a.output.write_text(json.dumps(results,indent=1))
        for fr in a.frictions:
            for h in a.heights:
                line=[]
                for s in a.sources:
                    r=[x for x in results if x['friction']==fr and x['height']==h and x['source']==s]
                    line.append(f"src{s}: rot {np.median([x['rotation_deg'] for x in r]):.0f}° disp {np.median([x['displacement_m'] for x in r])*100:.1f}cm")
                print(f'REST friction {fr} height +{h*1000:.0f}mm | '+' | '.join(line),flush=True)
    except BaseException:
        import traceback;traceback.print_exc();sys.stdout.flush();raise
    finally:app.close()


if __name__=='__main__':main()
