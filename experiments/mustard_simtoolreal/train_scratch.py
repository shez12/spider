"""Matched from-scratch SAPG baseline: same mustard environment/config as the warm-start runs.

Environment settings match train_critic.configure (no delays/noise/wrench impulses, 16 s since
last success, 50 native delta goals, native tolerance curriculum) with 100% unmodified native
resets. Learner uses the unmodified vendor SimToolRealSAPG.yaml (log-std 0, adaptive lr 1e-4,
clip 0.1, online input normalization). Writes sapg_latest.pth/training_state.json every
--checkpoint-every epochs for the shared stage-evaluation monitor.
"""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parents[1];VENDOR=PROJECT/'third_party/simtoolreal'
sys.path[:0]=[str(VENDOR),str(VENDOR/'rl_games'),str(ROOT)]


def main():
    from isaaclab.app import AppLauncher
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--num-envs',type=int,default=8184);p.add_argument('--block-size',type=int,default=1364)
    p.add_argument('--epochs',type=int,default=100000);p.add_argument('--episode-seconds',type=float,default=16.)
    p.add_argument('--checkpoint-every',type=int,default=500);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--wandb-project',default='simtoolreal-warmstart')
    p.add_argument('--pristine-settings',action='store_true',
        help='0c: keep the original task YAML (delays/noise/wrench impulses incl. torque_scale=2, 10 s); only robot/object assets differ')
    p.add_argument('--diverse-reset-bank',type=Path,help='Arm S: mix native resets with validated bank states (scratch init).')
    p.add_argument('--bank-probabilities',type=float,nargs=4,default=(.4,.25,.2,.15),help='native, near_object, stable_grasp, near_goal')
    p.add_argument('--obs',choices=('native','adept'),default='native',help='adept: ADEPT-style object observations (adept_obs.py, 64 points)')
    p.add_argument('--actor-init',type=Path,help='Arm D: BC actor checkpoint (actor_init.pt); critic starts fresh.')
    p.add_argument('--init-sigma',type=float,default=.4,help='Arm D: initial per-block action std after loading the BC actor.')
    p.add_argument('--critic-epochs',type=int,default=0,help='Arm D: actor (incl. sigma) frozen for these epochs; only the critic learns.')
    p.add_argument('--align-left-physics',action='store_true',help='physics_align.py: depenetration 1000 and hand joint friction 0, as the original left hand')
    p.add_argument('--restrict-abduction',action='store_true',help='with --align-left-physics: finger MCP_AA +-2 deg, thumb CMC_AA upper 0.131 (original left limits)')
    p.add_argument('--online-near-prob',type=float,default=0.,help='with --diverse-reset-bank: fraction of native resets restarted from '
        'online-captured native states with a fingertip within 6 cm of the resting object (online_resets.py)')
    p.add_argument('--self-collision-filter',nargs='*',default=[],choices=('palm-thumb','thumb-index','palm-fingers','pinky-ring','adjacent-fingers'),
        help='Extra right-hand self-collision filtered pairs (self_collision_filters.py); pinky-ring removes the rest-pose overlap')
    p.add_argument('--palm-hull',action='store_true',help='palm_hull.py: right palm collision = one convex hull (as PhysX treats the original left palm mesh)')
    p.add_argument('--official-hand',action='store_true',help='official_hand.py: official Sharpa Wave right hand URDF (same kinematics, official meshes/pads/inertials)')
    p.add_argument('--resume',type=Path,help='Full rl_games SAPG checkpoint (sapg_latest.pth): weights, optimizers, epoch and frame count continue.')
    AppLauncher.add_app_launcher_args(p);a=p.parse_args()
    assert a.num_envs==6*a.block_size and a.checkpoint_every>0
    a.output=a.output.resolve();assert not a.output.exists(),'Choose a fresh output directory.'
    a.output.mkdir(parents=True);app=None
    try:
        app=AppLauncher(a).app;train(a)
    except BaseException as error:
        import traceback;traceback.print_exc();sys.stdout.flush()
        (a.output/'failed.json').write_text(json.dumps(dict(error=type(error).__name__,message=str(error)))+'\n');raise
    finally:
        if app is not None:app.close()


def train(a):
    import torch,yaml
    from environment import create_env
    from rl_games.torch_runner import Runner
    from rl_games.common.algo_observer import AlgoObserver
    from isaacsimenvs.utils.rlgames_utils import register_rlgames_env,EnvStatsAlgoObserver,MultiObserver
    def save(name,value):
        path=a.output/name;tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,default=str)+'\n');tmp.replace(path)
    save('run_arguments.json',vars(a))
    def configure(cfg):
        dr=cfg.domain_randomization
        if a.pristine_settings:
            dr.torque_scale=2.;return
        dr.use_action_delay=False;dr.use_obs_delay=False;dr.use_object_state_delay_noise=False
        dr.joint_velocity_obs_noise_std=0.;dr.force_scale=0.;dr.torque_scale=0.
        cfg.episode_length_s=a.episode_seconds;cfg.termination.episode_length=round(a.episode_seconds*60)
        cfg.termination.tolerance_curriculum_interval=3000
    assert a.align_left_physics or not a.restrict_abduction,'--restrict-abduction requires --align-left-physics'
    if a.align_left_physics:
        import physics_align;save('physics_align.json',physics_align.install(restrict_abduction=a.restrict_abduction))
    if a.palm_hull:
        assert not a.restrict_abduction,'palm_hull and restrict_abduction both rewrite the URDF'
        import palm_hull;hull_info=palm_hull.install()
    if a.self_collision_filter:
        import self_collision_filters;save('self_collision_filters.json',self_collision_filters.install(a.self_collision_filter))
    if a.actor_init is not None:
        layout=torch.load(a.actor_init,map_location='cpu',weights_only=False).get('obs_layout','native')
        assert layout==a.obs,f'BC actor was trained on {layout} observations, run uses {a.obs}'
    if a.official_hand:
        from official_hand import create_env
    if a.obs=='native':env=create_env(a.num_envs,configure=configure)
    else:
        from adept_obs import install_adept_obs,ADEPT_OBS,ADEPT_STATE
        import isaacsimenvs.tasks.simtoolreal.simtoolreal_env as module
        dims=install_adept_obs(64,a.output);built=[];cls=module.SimToolRealEnv
        def configure_adept(cfg):
            configure(cfg);cfg.obs.obs_list=list(ADEPT_OBS);cfg.obs.state_list=list(ADEPT_STATE)
        module.SimToolRealEnv=lambda *x,**kw:built.append(cls(*x,**kw)) or built[-1]
        try:env=create_env(a.num_envs,configure=configure_adept)
        except AssertionError:
            # environment.py (hashed into the reset-bank signature) asserts the native 140/162 dims.
            env=built[-1];assert (env.cfg.observation_space,env.cfg.state_space)==dims,(env.cfg.observation_space,env.cfg.state_space,dims)
        finally:module.SimToolRealEnv=cls
    assert env.cfg.termination.max_consecutive_successes==50 and env.cfg.reset.goal_sampling_type=='delta'
    save('environment_config.json',env.cfg.to_dict());bank=None
    if a.palm_hull:save('palm_hull.json',hull_info)
    if a.diverse_reset_bank:
        from diverse_resets import DiverseResetBank,audit_bank_resets
        bank=DiverseResetBank(env,a.diverse_reset_bank,tuple(a.bank_probabilities));save('diverse_reset_check.json',audit_bank_resets(env,bank))
    online=None
    if a.online_near_prob:
        assert bank is not None,'--online-near-prob requires --diverse-reset-bank'
        from online_resets import OnlineNativeNearBuffer
        online=OnlineNativeNearBuffer(env,bank,probability=a.online_near_prob)
    register_rlgames_env(env,rl_device='cuda:0',clip_obs=10.,clip_actions=1.)
    cfg=yaml.safe_load((VENDOR/'isaacsimenvs/cfg/train/SimToolRealSAPG.yaml').read_text())
    cfg['params']['seed']=a.seed;config=cfg['params']['config']
    config.update(name='0_mustard_scratch_sapg',full_experiment_name='0_mustard_scratch_sapg',train_dir=str(a.output),
        num_actors=a.num_envs,expl_coef_block_size=a.block_size,device='cuda:0',device_name='cuda:0',max_epochs=a.epochs,save_frequency=0)
    save('sapg_config.json',cfg)
    save('sapg_contract.json',dict(baseline='matched from-scratch SAPG',initialization='random (vendor)',resets='100% native',
        environment='original SimToolReal.yaml settings (DR on, torque_scale=2, 10 s); right-hand robot + mustard assets' if a.pristine_settings else
            'train_critic.configure equivalent: no delays/noise/impulses, 16 s since last success, 50 delta goals, native tolerance curriculum',
        learner='unmodified vendor SimToolRealSAPG.yaml except actor count/block size/device/epochs'))
    sidecar=None
    if a.wandb_project:
        with (a.output/'wandb_sync.log').open('a') as log:
            sidecar=subprocess.Popen([sys.executable,'-u',str(PROJECT/'examples/arm_hand/sync_mustard_sapg_wandb.py'),'--run-dir',str(a.output),
                '--project',a.wandb_project,'--run-id',a.output.name],stdout=log,stderr=log,stdin=subprocess.DEVNULL,start_new_session=True)

    class BaselineObserver(AlgoObserver):
        def after_init(self,agent):
            self.agent=agent;self.started=time.monotonic()
            if a.actor_init is None:return
            import math
            old=torch.load(a.actor_init,map_location=agent.ppo_device,weights_only=False)
            assert old.get('actor_obs_dim',140)==env.cfg.observation_space,(old.get('actor_obs_dim'),env.cfg.observation_space)
            agent.model.load_state_dict(old['model'],strict=True)
            with torch.no_grad():agent.model.a2c_network.sigma.fill_(math.log(a.init_sigma))
            # Keep the BC observation scale (as in v5): the normalizer never updates.
            train_mode=agent.model.running_mean_std.train
            agent.model.running_mean_std.train=lambda mode=True:train_mode(False)
            self.lr0=float(agent.last_lr);step=agent.optimizer.step
            agent.optimizer.step=lambda *x,**kw:None if agent.epoch_num<=a.critic_epochs else step(*x,**kw)
            save('actor_init_check.json',dict(actor_init=str(a.actor_init),bc_epochs=old.get('bc_epochs'),obs_layout=old.get('obs_layout','native'),init_sigma=a.init_sigma,
                critic_epochs=a.critic_epochs,normalizer_frozen=True,lr0=self.lr0))
        def after_print_stats(self,frame,epoch_num,total_time):
            g=self.agent;goals=env._prev_episode_successes.float();extra={}
            if a.actor_init is not None and epoch_num<=a.critic_epochs:g.last_lr=self.lr0;g.update_lr(self.lr0)
            if bank is not None:
                kinds=bank.last_completed_kind
                extra['kind_any_goal_rate']={str(k):float((goals[kinds==k]>=1).float().mean()) for k in range(5) if bool((kinds==k).any())}
                goals=goals[kinds==0]
            if online is not None:extra['online_buffer']=online.report()
            if not len(goals):goals=torch.zeros(1,device=env.device)
            row=dict(epoch=epoch_num,frames=frame,elapsed_s=time.monotonic()-self.started,
                mean_reward=float(g.game_rewards.get_mean()[0]) if g.game_rewards.current_size else None,
                curriculum_mean_completed_goals=float(goals.mean()),native_any_goal_rate=float((goals>=1).float().mean()),
                success_tolerance=float(env._current_success_tolerance),actor_lr=float(g.last_lr),
                policy_sigma_mean=float(g.model.a2c_network.sigma.exp().mean()),lifted_fraction=float(env._lifted_object.float().mean()),**extra)
            with (a.output/'sapg_progress.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print('SAPG_PROGRESS',json.dumps(row),flush=True)
            if epoch_num%a.checkpoint_every==0:
                g.save(str(a.output/'sapg_latest'))
                if bank is not None:save('diverse_reset_coverage.json',bank.report())
                save('training_state.json',dict(epoch=epoch_num,frames=frame,success_tolerance=float(env._current_success_tolerance)))
    runner=Runner(MultiObserver([EnvStatsAlgoObserver(),BaselineObserver()]))
    runner.load(cfg);runner.reset()
    runner.run(dict(train=True,play=False,checkpoint=str(a.resume) if a.resume else None))
    save('completed.json',dict(stage='scratch SAPG',epochs=a.epochs))
    if sidecar is not None:sidecar.terminate()
    env.close()


if __name__=='__main__':main()
