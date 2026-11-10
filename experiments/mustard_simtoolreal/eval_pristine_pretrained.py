"""Evaluate the official SimToolReal pretrained policy in the unmodified environment.

--object default keeps the procedural tool pool; --object mustard swaps in only the mustard bottle (same
overrides as train_pristine_mustard.py). Reports per-episode lift rate and completed goals.
"""
import argparse,json,sys
from pathlib import Path
PRISTINE=Path('/home/szc/work/newsimtoolreal/simtoolreal');HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(PRISTINE),str(PRISTINE/'rl_games')]


def main():
    from isaaclab.app import AppLauncher
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--object',choices=('default','mustard','pool'),required=True);p.add_argument('--num-envs',type=int,default=512)
    p.add_argument('--pool',type=Path,help='build_object_pool.py output; envs are assigned objects round-robin')
    p.add_argument('--episodes-per-env',type=int,default=2);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stochastic',action='store_true');p.add_argument('--no-dr',action='store_true')
    p.add_argument('--max-steps',type=int,default=1800,help='stop after this many policy steps; unfinished episodes are scored as they stand (0 = no cap)')
    AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
    try:run(a)
    finally:sys.stdout.flush();import os;os._exit(0)


def run(a):
    import numpy as np,torch,yaml
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env import SimToolRealEnv
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg
    from deployment.rl_player import RlPlayer
    cfg=SimToolRealEnvCfg();cfg.from_dict(yaml.safe_load((PRISTINE/'isaacsimenvs/cfg/task/SimToolReal.yaml').read_text()));cfg.scene.num_envs=a.num_envs
    if a.object=='mustard':
        cal=json.loads((HERE/'assets/object/calibration.json').read_text())
        cfg.assets.object_urdf=str(HERE/'assets/object/mustard_bottle.urdf');cfg.assets.object_scale=tuple(cal['policy_object_scale']);cfg.reward.fixed_size=tuple(cal['bbox_dims_m'])
    names=None
    if a.object=='pool':
        from isaacsimenvs.tasks.simtoolreal.utils import scene_utils
        objs=json.loads((a.pool/'manifest.json').read_text())['objects'];names=[o['name'] for o in objs]
        scene_utils.generate_handle_head_urdfs=lambda **_:([a.pool/n/'object.urdf' for n in names],[tuple(o['object_scale']) for o in objs])
    if a.no_dr:
        d=cfg.domain_randomization;d.force_scale=0.;d.torque_scale=0.
    env=SimToolRealEnv(cfg);pp=PRISTINE/'pretrained_policy'
    pl=RlPlayer(num_observations=140,num_actions=29,config_path=str(pp/'config.yaml'),checkpoint_path=str(pp/'model.pth'),device='cuda:0',num_envs=a.num_envs)
    obs,_=env.reset();lifted_ever=torch.zeros(a.num_envs,dtype=torch.bool,device='cuda:0');rows=[];done_count=torch.zeros(a.num_envs,dtype=torch.long,device='cuda:0')
    step=0
    while bool((done_count<a.episodes_per_env).any()):
        step+=1
        if a.max_steps and step>a.max_steps:
            for i in (done_count<a.episodes_per_env).nonzero().flatten().tolist():
                rows.append(dict(goals=int(env._successes[i]),lifted=bool(lifted_ever[i]),asset=int(env._object_asset_index_per_env[i]),capped=True))
            break
        act=pl.get_normalized_action(obs['policy'],deterministic_actions=not a.stochastic)
        obs,_,term,trunc,_=env.step(act);lifted_ever|=env._lifted_object;done=(term|trunc)
        ids=done.nonzero().flatten()
        for i in ids.tolist():
            if done_count[i]<a.episodes_per_env:rows.append(dict(goals=int(env._prev_episode_successes[i]),lifted=bool(lifted_ever[i]),
                asset=int(env._object_asset_index_per_env[i])))
        if len(ids):
            lifted_ever[ids]=False;done_count[ids]+=1
            for s in pl.player.states:s[:,ids]=0
    g=np.array([r['goals'] for r in rows]);l=np.array([r['lifted'] for r in rows])
    out=dict(object=a.object,episodes=len(rows),lift_rate=float(l.mean()),any_goal_rate=float((g>=1).mean()),mean_goals=float(g.mean()),
        median_goals=float(np.median(g)),goals_ge10=float((g>=10).mean()),tolerance=float(env._current_success_tolerance),deterministic=not a.stochastic,no_dr=a.no_dr,max_steps=a.max_steps,capped_episodes=sum(1 for r in rows if r.get('capped')))
    if names:
        per={}
        for k,n in enumerate(names):
            r=[x for x in rows if x['asset']==k];gg=np.array([x['goals'] for x in r])
            per[n]=dict(episodes=len(r),lift_rate=float(np.mean([x['lifted'] for x in r])),any_goal_rate=float((gg>=1).mean()),mean_goals=float(gg.mean()))
        out['per_object']=per
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print('RESULT',json.dumps(out))


if __name__=='__main__':main()
