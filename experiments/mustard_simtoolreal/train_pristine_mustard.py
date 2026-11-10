"""Ablation: unmodified SimToolReal (newsimtoolreal) with only the object replaced by the mustard bottle.

Left KUKA+Sharpa, original task YAML (10 s, full domain randomization incl. wrench impulses), original
SAPG YAML. Only changes: assets.object_urdf/object_scale -> mustard, reward.fixed_size -> mustard bbox,
actor count/block size/minibatch for 8184 envs (4 minibatches per mini-epoch as in the 24576-env default).
"""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
PRISTINE=Path(os.environ.get('SIMTOOLREAL_PRISTINE','/home/szc/work/newsimtoolreal/simtoolreal'));HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[1]
sys.path[:0]=[str(PRISTINE),str(PRISTINE/'rl_games')]
sys.path.append(str(HERE))


def main():
    from isaaclab.app import AppLauncher
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--num-envs',type=int,default=8184)
    p.add_argument('--epochs',type=int,default=8000);p.add_argument('--checkpoint-every',type=int,default=500)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--wandb-project',default='simtoolreal-warmstart')
    p.add_argument('--object-repr',choices=('bbox','pointcloud','adept'),default='bbox',
        help='pointcloud: observation keypoints_rel_palm/goal use K mesh-surface points instead of 4 bbox corners; '
        'adept: ADEPT-style fields (object pos/rot, goal pos/quat, K-point object cloud in env frame, no keypoints); reward unchanged')
    p.add_argument('--pc-points',type=int,default=64)
    p.add_argument('--resume',type=Path,help='full rl_games checkpoint (sapg_latest.pth): weights, optimizer, epoch and frames continue')
    p.add_argument('--rot-curriculum',action='store_true',help='goal_curriculum.py: goal re-orientation alpha 0->1 gated on native any-goal rate')
    p.add_argument('--rot-threshold',type=float,default=.2);p.add_argument('--rot-step',type=float,default=.1)
    p.add_argument('--state-bank',type=Path,help='left_reset_bank.py bank (MPC grasp states); mixed with vendor native resets')
    p.add_argument('--bank-delta-first-goal',action='store_true',help='bank episodes: first goal from the vendor delta sampler around the restored object')
    p.add_argument('--bank-probabilities',type=float,nargs=4,default=(.5,.2,.15,.15),help='native, near_object, grasp, lifted')
    p.add_argument('--align-left-physics',action='store_true',help='vendored code only: physics_align.install() (depenetration 1000, hand joint friction 0)')
    p.add_argument('--stop-at-tolerance-floor',action='store_true',
        help='Stop (and checkpoint) once the native tolerance curriculum reaches termination.target_success_tolerance; '
        'every tolerance step also saves sapg_tol_<tol>_ep<epoch>.pth and a line in tolerance_changes.jsonl')
    AppLauncher.add_app_launcher_args(p);a=p.parse_args()
    assert a.num_envs%6==0;a.output=a.output.resolve();assert not a.output.exists();a.output.mkdir(parents=True);app=None
    try:app=AppLauncher(a).app;train(a)
    except BaseException as e:
        import traceback;traceback.print_exc();sys.stdout.flush()
        (a.output/'failed.json').write_text(json.dumps(dict(error=type(e).__name__,message=str(e)))+'\n');raise
    finally:
        if app is not None:app.close()


def train(a):
    import torch,yaml
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env import SimToolRealEnv
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg
    from rl_games.torch_runner import Runner
    from rl_games.common.algo_observer import AlgoObserver
    from isaacsimenvs.utils.rlgames_utils import register_rlgames_env,EnvStatsAlgoObserver,MultiObserver
    def save(name,v):(a.output/name).write_text(json.dumps(v,indent=2,default=str)+'\n')
    save('run_arguments.json',vars(a))
    cal=json.loads((HERE/'assets/object/calibration.json').read_text())
    cfg=SimToolRealEnvCfg();cfg.from_dict(yaml.safe_load((PRISTINE/'isaacsimenvs/cfg/task/SimToolReal.yaml').read_text()))
    cfg.seed=a.seed;cfg.scene.num_envs=a.num_envs
    cfg.assets.object_urdf=str(HERE/'assets/object/mustard_bottle.urdf');cfg.assets.object_scale=tuple(cal['policy_object_scale'])
    cfg.reward.fixed_size=tuple(cal['bbox_dims_m'])
    if a.object_repr=='pointcloud':install_pointcloud_obs(a.pc_points,a.output)
    if a.object_repr=='adept':
        from adept_obs import install_adept_obs,ADEPT_OBS,ADEPT_STATE
        install_adept_obs(a.pc_points,a.output);cfg.obs.obs_list=list(ADEPT_OBS);cfg.obs.state_list=list(ADEPT_STATE)
    if a.align_left_physics:
        import physics_align;save('physics_align.json',physics_align.install())
    env=SimToolRealEnv(cfg);save('environment_config.json',env.cfg.to_dict())
    bank=None
    if a.state_bank:
        from left_reset_bank import StateResetBank,signature
        sig=signature(PRISTINE/cfg.assets.robot_urdf,cfg.assets.object_urdf,PRISTINE/'isaacsimenvs/cfg/task/SimToolReal.yaml')
        bank=StateResetBank(env,a.state_bank,tuple(a.bank_probabilities),expected_signature=sig,delta_first_goal=a.bank_delta_first_goal);save('state_bank.json',bank.report())
    curriculum=None
    if a.rot_curriculum:
        from goal_curriculum import RotationCurriculum
        if bank is not None:env._curriculum_kind=lambda:bank.current_kind
        curriculum=RotationCurriculum(env,step=a.rot_step,threshold=a.rot_threshold)
    register_rlgames_env(env,rl_device='cuda:0',clip_obs=10.,clip_actions=1.)
    tc=yaml.safe_load((PRISTINE/'isaacsimenvs/cfg/train/SimToolRealSAPG.yaml').read_text());tc['params']['seed']=a.seed;c=tc['params']['config']
    batch=c['horizon_length']*a.num_envs
    c.update(name='0_pristine_mustard_sapg',full_experiment_name='0_pristine_mustard_sapg',train_dir=str(a.output),num_actors=a.num_envs,
        expl_coef_block_size=a.num_envs//6,device='cuda:0',device_name='cuda:0',max_epochs=a.epochs,save_frequency=0,minibatch_size=batch//4)
    c['central_value_config']['minibatch_size']=batch//4
    save('sapg_config.json',tc)
    if a.wandb_project:
        subprocess.Popen([sys.executable,'-u',str(PROJECT/'examples/arm_hand/sync_mustard_sapg_wandb.py'),'--run-dir',str(a.output),'--project',a.wandb_project,
            '--run-id',a.output.name],stdout=(a.output/'wandb_sync.log').open('a'),stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    class Obs(AlgoObserver):
        def after_init(self,agent):self.agent=agent;self.t0=time.monotonic();self.tol=float(env._current_success_tolerance)
        def after_print_stats(self,frame,epoch,total_time):
            g=self.agent;goals=env._prev_episode_successes.float();tol=float(env._current_success_tolerance)
            extra={}
            if bank is not None:
                kinds=bank.last_completed_kind
                extra['kind_any_goal_rate']={str(k):float((goals[kinds==k]>=1).float().mean()) for k in range(4) if bool((kinds==k).any())}
                goals=goals[kinds==0] if bool((kinds==0).any()) else torch.zeros(1,device=env.device)
            if tol<self.tol-1e-12:
                g.save(str(a.output/f'sapg_tol_{tol:.4f}_ep{epoch}'))
                with (a.output/'tolerance_changes.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=epoch,frames=frame,old=self.tol,new=tol))+'\n')
                self.tol=tol
            floor=float(env.cfg.termination.target_success_tolerance)
            if a.stop_at_tolerance_floor and tol<=floor+1e-9 and g.max_epochs!=epoch:
                # rl_games checks max_epochs right after this callback, saves its own last_* checkpoint and exits.
                g.save(str(a.output/'sapg_latest'));save('training_state.json',dict(epoch=epoch,frames=frame))
                save('tolerance_floor_reached.json',dict(epoch=epoch,frames=frame,tolerance=tol,floor=floor));g.max_epochs=epoch
            if curriculum is not None:
                change=curriculum.update(epoch,float((goals>=1).float().mean()))
                if change:
                    with (a.output/'curriculum_changes.jsonl').open('a') as f:f.write(json.dumps(change)+'\n')
                extra['rot_alpha']=curriculum.alpha
            row=dict(epoch=epoch,frames=frame,elapsed_s=time.monotonic()-self.t0,mean_reward=float(g.game_rewards.get_mean()[0]) if g.game_rewards.current_size else None,
                curriculum_mean_completed_goals=float(goals.mean()),native_any_goal_rate=float((goals>=1).float().mean()),
                success_tolerance=float(env._current_success_tolerance),actor_lr=float(g.last_lr),policy_sigma_mean=float(g.model.a2c_network.sigma.exp().mean()),
                lifted_fraction=float(env._lifted_object.float().mean()),**extra)
            with (a.output/'sapg_progress.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if epoch%a.checkpoint_every==0:g.save(str(a.output/'sapg_latest'));save('training_state.json',dict(epoch=epoch,frames=frame))
    r=Runner(MultiObserver([EnvStatsAlgoObserver(),Obs()]));r.load(tc);r.reset();r.run(dict(train=True,play=False,checkpoint=str(a.resume) if a.resume else None))
    save('completed.json',dict(epochs=a.epochs));env.close()


def install_pointcloud_obs(k,output):
    """Swap the observation keypoint offsets for surface points; reward keeps _keypoint_offsets_fixed (bbox)."""
    import numpy as np,torch
    from adept_obs import mustard_surface_points
    from isaacsimenvs.tasks.simtoolreal.utils import obs_utils
    import isaacsimenvs.tasks.simtoolreal.simtoolreal_env as env_mod
    pts=mustard_surface_points(k);np.save(output/'pointcloud_object_frame.npy',pts)
    obs_utils.OBS_FIELD_SIZES['keypoints_rel_palm']=3*k;obs_utils.OBS_FIELD_SIZES['keypoints_rel_goal']=3*k
    orig=env_mod.allocate_state_buffers
    def allocate(env):
        orig(env);env._keypoint_offsets=torch.as_tensor(pts,device=env.device).unsqueeze(0).expand(env.num_envs,-1,-1).contiguous()
    env_mod.allocate_state_buffers=allocate


if __name__=='__main__':main()
