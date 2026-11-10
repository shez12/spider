"""Roll out a train_pristine_mustard.py checkpoint deterministically and summarize where episodes stall.

Per episode: lifted at all, steps lifted, closest keypoint max-distance to the goal reached while lifted (vs the
success threshold), object height, and how the episode ended (goal, timeout, drop/fall, hand far).
"""
import argparse,json,sys
from pathlib import Path
PRISTINE=Path('/home/szc/work/newsimtoolreal/simtoolreal');HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(PRISTINE),str(PRISTINE/'rl_games')];sys.path.append(str(HERE))


def main():
    from isaaclab.app import AppLauncher
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--num-envs',type=int,default=256)
    p.add_argument('--steps',type=int,default=1200);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--state-bank',type=Path);p.add_argument('--bank-probabilities',type=float,nargs=4,default=(.25,.25,.25,.25))
    AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
    try:run(a)
    finally:sys.stdout.flush();import os;os._exit(0)


def run(a):
    import numpy as np,torch,yaml,tempfile
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env import SimToolRealEnv
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg
    from deployment.rl_player import RlPlayer
    ra=json.loads((a.run_dir/'run_arguments.json').read_text());cal=json.loads((HERE/'assets/object/calibration.json').read_text())
    cfg=SimToolRealEnvCfg();cfg.from_dict(yaml.safe_load((PRISTINE/'isaacsimenvs/cfg/task/SimToolReal.yaml').read_text()))
    cfg.scene.num_envs=a.num_envs;cfg.assets.object_urdf=str(HERE/'assets/object/mustard_bottle.urdf')
    cfg.assets.object_scale=tuple(cal['policy_object_scale']);cfg.reward.fixed_size=tuple(cal['bbox_dims_m'])
    repr_=ra.get('object_repr','bbox');tmp=Path(tempfile.mkdtemp())
    if repr_=='adept':
        from adept_obs import install_adept_obs,ADEPT_OBS,ADEPT_STATE
        install_adept_obs(ra['pc_points'],tmp);cfg.obs.obs_list=list(ADEPT_OBS);cfg.obs.state_list=list(ADEPT_STATE)
    elif repr_=='pointcloud':
        from train_pristine_mustard import install_pointcloud_obs
        install_pointcloud_obs(ra['pc_points'],tmp)
    env=SimToolRealEnv(cfg);bank=None
    if a.state_bank:
        from left_reset_bank import StateResetBank
        bank=StateResetBank(env,a.state_bank,tuple(a.bank_probabilities),delta_first_goal=True)
    from isaaclab.utils.math import quat_mul,quat_conjugate
    train=json.loads((a.run_dir/'sapg_config.json').read_text())
    train['params']['config']['num_actors']=a.num_envs;(tmp/'config.yaml').write_text(yaml.safe_dump(dict(train=train)))
    pl=RlPlayer(num_observations=env.cfg.observation_space,num_actions=29,config_path=str(tmp/'config.yaml'),checkpoint_path=None,device='cuda:0',num_envs=a.num_envs)
    ck=torch.load(a.run_dir/'sapg_latest.pth',map_location='cuda:0',weights_only=False);ck=ck.get(0,ck) if isinstance(ck,dict) else ck[0]
    pl.player.model.load_state_dict(ck['model']);pl.player.model.eval()
    tol=float(env._current_success_tolerance)*cfg.reward.keypoint_scale
    obs,_=env.reset();n=a.num_envs;dev=env.device
    lifted_steps=torch.zeros(n,device=dev);best=torch.full((n,),9.,device=dev);maxz=torch.zeros(n,device=dev);rows=[]
    bpos=torch.full((n,),9.,device=dev);brot=torch.full((n,),999.,device=dev);kind=bank.current_kind.clone() if bank else torch.zeros(n,dtype=torch.long,device=dev)
    for step in range(a.steps):
        act=pl.get_normalized_action(obs['policy'],deterministic_actions=True)
        prev_succ=env._successes.clone();lifted=env._lifted_object.clone()
        obs,_,term,trunc,_=env.step(act)
        lifted_steps+=env._lifted_object.float();z=env.object.data.root_pos_w[:,2]-env.scene.env_origins[:,2];maxz=torch.maximum(maxz,z)
        best=torch.where(env._lifted_object,torch.minimum(best,env._keypoints_max_dist),best)
        o=env.scene.env_origins;pe=(env.object.data.root_pos_w-env.goal_viz.data.root_pos_w).norm(dim=-1)
        q=quat_mul(quat_conjugate(env.goal_viz.data.root_quat_w),env.object.data.root_quat_w);re=torch.rad2deg(2*torch.acos(q[:,0].abs().clamp(max=1.)))
        bpos=torch.minimum(bpos,pe);brot=torch.where(pe<=bpos+1e-6,re,brot)
        for i in (term|trunc).nonzero().flatten().tolist():
            rows.append(dict(goals=int(env._prev_episode_successes[i]),lifted_steps=int(lifted_steps[i]),best_kp=float(best[i]),
                max_z=float(maxz[i]),timeout=bool(trunc[i]),lifted_at_end=bool(lifted[i]),
                reason=[k for k,v in env._termination_reasons.items() if bool(v[i])],kind=int(kind[i]),best_pos_err=float(bpos[i]),rot_err_at_best_pos=float(brot[i])))
        d=(term|trunc);lifted_steps[d]=0;best[d]=9.;maxz[d]=0.;bpos[d]=9.;brot[d]=999.
        if bank is not None:kind=torch.where(d,bank.current_kind,kind)
        for s in pl.player.states:s[:,d]=0
    g=np.array([r['goals'] for r in rows]);ls=np.array([r['lifted_steps'] for r in rows]);b=np.array([r['best_kp'] for r in rows])
    lifted=ls>0
    out=dict(run=str(a.run_dir),epoch=ck.get('epoch'),episodes=len(rows),success_threshold_m=tol,
        lifted_rate=float(lifted.mean()),any_goal_rate=float((g>=1).mean()),timeout_rate=float(np.mean([r['timeout'] for r in rows])),
        lifted_steps_median=float(np.median(ls[lifted])) if lifted.any() else 0.,
        best_kp_while_lifted_quantiles_m=np.quantile(b[lifted],[.1,.25,.5,.75]).round(4).tolist() if lifted.any() else None,
        within_2x_threshold=float((b[lifted]<2*tol).mean()) if lifted.any() else 0.,
        max_object_z_median=float(np.median([r['max_z'] for r in rows])),
        end_reasons={k:float(np.mean([k in r['reason'] for r in rows])) for k in ('fall','hand_far','timeout','max_successes')},
        lifted_end_reasons={k:float(np.mean([k in r['reason'] for r in rows if r['lifted_steps']>0])) for k in ('fall','hand_far','timeout')})
    per={}
    for k in sorted({r['kind'] for r in rows}):
        r=[x for x in rows if x['kind']==k]
        per[str(k)]=dict(episodes=len(r),any_goal=float(np.mean([x['goals']>=1 for x in r])),lifted=float(np.mean([x['lifted_steps']>0 for x in r])),
            reasons={q:float(np.mean([q in x['reason'] for x in r])) for q in ('fall','hand_far','timeout')},
            best_pos_err_median_m=float(np.median([x['best_pos_err'] for x in r])),rot_err_at_best_pos_median_deg=float(np.median([x['rot_err_at_best_pos'] for x in r])),
            lifted_steps_median=float(np.median([x['lifted_steps'] for x in r])))
    out['per_kind']=per
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(out,episodes_detail=rows[:400]),indent=1))
    print('DIAG',json.dumps(out),flush=True)


if __name__=='__main__':main()
