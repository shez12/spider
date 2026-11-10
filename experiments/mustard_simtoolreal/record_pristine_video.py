"""Record a rollout of a train_pristine_mustard.py SAPG checkpoint in the same pristine environment.

Rebuilds the environment exactly as training (original YAML, full domain randomization, native resets,
mustard object, same --object-repr), loads the rl_games checkpoint and runs the leader block (coefficient
50, as deployment/rl_player.py) with mean actions. The viewer follows the last env; per-episode lift and
goal counts are reported over all envs.
"""
import argparse,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))


def main():
    from isaaclab.app import AppLauncher
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True,help='train_pristine_mustard.py output (run_arguments.json, sapg_config.json)')
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--num-envs',type=int,default=16);p.add_argument('--frames',type=int,default=1800)
    p.add_argument('--seed',type=int,default=7);p.add_argument('--stochastic',action='store_true')
    p.add_argument('--eye',type=float,nargs=3,default=(0.6,-0.55,0.95));p.add_argument('--lookat',type=float,nargs=3,default=(0.,0.1,0.55))
    p.add_argument('--resolution',type=int,nargs=2,default=(960,720))
    AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.enable_cameras=True
    a.output.mkdir(parents=True,exist_ok=False);app=AppLauncher(a).app
    try:run(a)
    except BaseException:
        import traceback;traceback.print_exc();raise
    finally:sys.stdout.flush();sys.stderr.flush();app.close()


def run(a):
    import cv2,imageio.v2 as imageio,torch,yaml
    import train_pristine_mustard as tp
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env import SimToolRealEnv
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg
    from rl_games.algos_torch import model_builder
    ra=json.loads((a.run_dir/'run_arguments.json').read_text());tc=json.loads((a.run_dir/'sapg_config.json').read_text())
    cal=json.loads((HERE/'assets/object/calibration.json').read_text())
    cfg=SimToolRealEnvCfg();cfg.from_dict(yaml.safe_load((tp.PRISTINE/'isaacsimenvs/cfg/task/SimToolReal.yaml').read_text()))
    cfg.seed=a.seed;cfg.scene.num_envs=a.num_envs
    cfg.assets.object_urdf=str(HERE/'assets/object/mustard_bottle.urdf');cfg.assets.object_scale=tuple(cal['policy_object_scale'])
    cfg.reward.fixed_size=tuple(cal['bbox_dims_m'])
    if ra['object_repr']=='pointcloud':tp.install_pointcloud_obs(ra['pc_points'],a.output)
    if ra['object_repr']=='adept':
        from adept_obs import install_adept_obs,ADEPT_OBS,ADEPT_STATE
        install_adept_obs(ra['pc_points'],a.output);cfg.obs.obs_list=list(ADEPT_OBS);cfg.obs.state_list=list(ADEPT_STATE)
    cfg.viewer.origin_type='env';cfg.viewer.env_index=a.num_envs-1
    cfg.viewer.eye=tuple(a.eye);cfg.viewer.lookat=tuple(a.lookat);cfg.viewer.resolution=tuple(a.resolution)
    env=SimToolRealEnv(cfg,render_mode='rgb_array');dev=env.device;n=a.num_envs;view=n-1
    ck=torch.load(a.checkpoint,map_location=dev,weights_only=False)[0]
    nobs=ck['model']['running_mean_std.running_mean'].shape[0];assert nobs==env.cfg.observation_space,(nobs,env.cfg.observation_space)
    params=tc['params'];blocks=6
    model=model_builder.ModelBuilder().load(dict(network=params['network'],model=params['model'])).build(dict(
        actions_num=29,input_shape=(nobs+1,),num_seqs=n,value_size=1,normalize_input=True,normalize_value=True,
        type='extra_param',coef_ids=torch.linspace(50.,0.,blocks,device=dev),coef_id_idx=nobs)).to(dev)
    model.load_state_dict(ck['model']);model.eval()
    states=[s.to(dev).zero_() for s in model.get_default_rnn_state()]
    obs,_=env.reset()
    for _ in range(20):
        f=env.render(recompute=True)
        if f is not None and f.size and f.max()>0:break
    lifted=torch.zeros(n,dtype=torch.bool,device=dev);episodes=[];leader=torch.full((n,1),50.,device=dev);ep_view=0
    with imageio.get_writer(str(a.output/'rollout.mp4'),fps=60,codec='libx264',macro_block_size=1) as w,torch.no_grad():
        for step in range(a.frames):
            inp=torch.cat([obs['policy'].clamp(-10,10),leader],-1)
            mu,logstd,_,h=model.a2c_network(dict(obs=model.norm_obs(inp),rnn_states=states,seq_length=1));states=list(h)
            act=mu+torch.randn_like(mu)*logstd.exp() if a.stochastic else mu
            obs,_,term,trunc,_=env.step(act.clamp(-1,1));lifted|=env._lifted_object;done=term|trunc
            frame=env.render()
            if frame is not None and frame.size:
                frame=frame[...,:3].copy()
                txt=f'ep {ep_view} t {int(env.episode_length_buf[view])}  lifted {bool(lifted[view])}  goals {int(env._successes[view])}'
                cv2.putText(frame,txt,(10,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2,cv2.LINE_AA);w.append_data(frame)
            for i in done.nonzero().flatten().tolist():
                episodes.append(dict(env=i,step=step,goals=int(env._prev_episode_successes[i]),lifted=bool(lifted[i])))
                if i==view:ep_view+=1
            if done.any():
                lifted[done]=False
                for s in states:s[:,done]=0
            if step%300==0:print('VIDEO_STEP',step,'episodes',len(episodes),flush=True)
    g=[e['goals'] for e in episodes];l=[e['lifted'] for e in episodes]
    out=dict(checkpoint=str(a.checkpoint),epoch=int(ck['epoch']),frames_trained=int(ck['frame']),object_repr=ra['object_repr'],
        num_envs=n,steps=a.frames,deterministic=not a.stochastic,leader_coef=50.,viewed_env=view,completed_episodes=len(episodes),
        lift_rate=sum(l)/len(l) if l else None,any_goal_rate=sum(x>=1 for x in g)/len(g) if g else None,
        mean_goals=sum(g)/len(g) if g else None,viewed_env_episodes=[e for e in episodes if e['env']==view],episodes=episodes)
    (a.output/'metrics.json').write_text(json.dumps(out,indent=2)+'\n')
    print('RESULT',json.dumps({k:v for k,v in out.items() if k!='episodes'}),flush=True)
    env.close()


if __name__=='__main__':main()
