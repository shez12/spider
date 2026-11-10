"""Restore left_reset_bank states in the unmodified env and hold the restored controller targets: does the object
stay where it was (per kind)? Distinguishes a broken restore from a policy that has not learned yet."""
import argparse,json,sys
from pathlib import Path
PRISTINE=Path('/home/szc/work/newsimtoolreal/simtoolreal');HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(PRISTINE),str(PRISTINE/'rl_games')];sys.path.append(str(HERE))


def main():
    from isaaclab.app import AppLauncher
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bank',type=Path,required=True);p.add_argument('--num-envs',type=int,default=512)
    p.add_argument('--dr',action='store_true',help='keep the task DR (delays/noise/wrench impulses) as in training')
    AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app;code=0
    try:run(a)
    except BaseException:
        import traceback;traceback.print_exc();code=1
    finally:sys.stdout.flush();import os;os._exit(code)


def run(a):
    import numpy as np,torch,yaml
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env import SimToolRealEnv
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg
    from diverse_resets import restore_states,hold_actions
    from left_reset_bank import FIELDS
    cal=json.loads((HERE/'assets/object/calibration.json').read_text())
    cfg=SimToolRealEnvCfg();cfg.from_dict(yaml.safe_load((PRISTINE/'isaacsimenvs/cfg/task/SimToolReal.yaml').read_text()));cfg.scene.num_envs=a.num_envs
    cfg.assets.object_urdf=str(HERE/'assets/object/mustard_bottle.urdf');cfg.assets.object_scale=tuple(cal['policy_object_scale']);cfg.reward.fixed_size=tuple(cal['bbox_dims_m'])
    if not a.dr:
        dr=cfg.domain_randomization;dr.use_action_delay=False;dr.use_obs_delay=False;dr.use_object_state_delay_noise=False;dr.force_scale=0.;dr.torque_scale=0.
    env=SimToolRealEnv(cfg);dev=env.device;env.reset();n=a.num_envs
    d=np.load(a.bank);kind=d['kind'];rng=np.random.default_rng(0);idx=rng.integers(0,len(kind),n)
    ids=torch.arange(n,device=dev);restore_states(env,ids,{f:torch.tensor(d[f][idx],device=dev) for f in FIELDS})
    env.scene.write_data_to_sim();env.sim.forward();env.scene.update(0.)
    z0=torch.tensor(d['root'][idx,2],device=dev);out={}
    from isaacsimenvs.tasks.simtoolreal.utils.reset_utils import _reset_goal_pose
    from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import compute_intermediate_values
    # Same first goal as StateResetBank(delta_first_goal=True); lift reference/latch as in the bank.
    root=torch.tensor(d['root'][idx],device=dev);env.goal_viz.write_root_pose_to_sim(torch.cat([root[:,:3]+env.scene.env_origins,root[:,3:7]],-1),env_ids=ids)
    _reset_goal_pose(env,ids,'delta');env._object_init_z[:]=env._table_z_per_env+env.cfg.reset.table_object_z_offset
    env._lifted_object[:]=(.05+root[:,2]-env._object_init_z)>env.cfg.reward.lifting_bonus_threshold
    env.scene.write_data_to_sim();env.sim.forward();env.scene.update(0.);compute_intermediate_values(env)
    tol=env._current_success_tolerance*env.cfg.reward.keypoint_scale;kp0=env._keypoints_max_dist.clone()
    zs=[];succ=torch.zeros(n,dtype=torch.bool,device=dev)
    for t in range(90):
        env.step(hold_actions(env));zs.append(env.object.data.root_pos_w[:,2]-env.scene.env_origins[:,2]);succ|=env._successes>0
    dz=(torch.stack(zs)[-1]-z0).cpu().numpy();ep=env.episode_length_buf.cpu().numpy()
    for k in (1,2,3):
        m=kind[idx]==k
        mt=torch.tensor(m,device=dev)
        out[str(k)]=dict(n=int(m.sum()),mean_dz_after_90_steps=float(dz[m].mean()),dropped_over_3cm=float((dz[m]<-.03).mean()),reset_during=float((ep[m]<90).mean()),
            initial_kp_over_threshold_median=float((kp0[mt]/tol).median()),initial_within_threshold=float((kp0[mt]<=tol).float().mean()),
            hold_only_goal_reached=float(succ[mt].float().mean()))
    print('HOLD',json.dumps(dict(dr=a.dr,**out)),flush=True)


if __name__=='__main__':main()
