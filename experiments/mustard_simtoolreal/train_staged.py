"""Waypoint-target SAPG warmup -> random-target critic calibration -> joint SAPG."""
import argparse
import json
import sys
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parents[1];VENDOR=PROJECT/'third_party/simtoolreal'
sys.path[:0]=[str(VENDOR),str(VENDOR/'rl_games'),str(ROOT)]


def main():
    from isaaclab.app import AppLauncher
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--num-envs',type=int,default=120)
    parser.add_argument('--waypoint-epochs',type=int,default=10000)
    parser.add_argument('--critic-epochs',type=int,default=683)
    parser.add_argument('--joint-epochs',type=int,default=200000)
    parser.add_argument('--waypoints',type=Path,default=ROOT/'waypoints.json')
    parser.add_argument('--capture-video',action='store_true')
    parser.add_argument('--video-interval',type=int,default=6000)
    parser.add_argument('--video-length',type=int,default=300)
    parser.add_argument('--verify-goal-transitions',action='store_true',help='Controlled simulator checks for smoke runs only.')
    parser.add_argument('--wandb-project',default=None,help='Attach the metrics-only W&B sidecar; videos stay local.')
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args()
    assert args.num_envs%6==0 and min(args.waypoint_epochs,args.critic_epochs,args.joint_epochs)>0
    args.output=args.output.resolve();assert not args.output.exists(),'Choose a fresh output directory.'
    assert args.waypoints.exists(),'Run prepare_waypoints.py first.'
    if args.capture_video:args.enable_cameras=True
    app=AppLauncher(args).app
    try:train(args)
    except BaseException as error:
        if args.output.exists():
            (args.output/'failed.json').write_text(json.dumps(dict(error=type(error).__name__,message=str(error)))+'\n')
        raise
    finally:app.close()


def train(args):
    import torch
    import yaml
    from environment import create_env
    from waypoint_goals import install_waypoint_goals
    from rl_games.torch_runner import Runner
    from rl_games.common.algo_observer import AlgoObserver
    from rl_games.common.schedulers import IdentityScheduler
    from isaacsimenvs.utils.rlgames_utils import register_rlgames_env,EnvStatsAlgoObserver,MultiObserver
    args.output.mkdir(parents=True)
    def save(name,data):
        (args.output/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str)+'\n')
    save('run_arguments.json',vars(args))
    env=create_env(args.num_envs,render_mode='rgb_array' if args.capture_video else None)
    goals=install_waypoint_goals(env,args.waypoints)
    if args.verify_goal_transitions:
        from waypoint_goals import verify_transitions
        save('goal_transition_checks.json',verify_transitions(env,goals,args.waypoints))
    save('environment_config.json',env.cfg.to_dict())
    save('waypoint_pool.json',json.loads(args.waypoints.read_text()))
    cfg=yaml.safe_load((VENDOR/'isaacsimenvs/cfg/train/SimToolRealSAPG.yaml').read_text())
    config=cfg['params']['config'];total=args.waypoint_epochs+args.critic_epochs+args.joint_epochs
    config.update(name='0_mustard_waypoint_sapg',full_experiment_name='0_mustard_waypoint_sapg',
        train_dir=str(args.output),device='cuda:0',device_name='cuda:0',num_actors=args.num_envs,
        expl_coef_block_size=args.num_envs//6,max_epochs=total,save_frequency=10000,
        learning_rate=1e-4,save_best_after=10)
    save('sapg_config.json',cfg)
    save('stage_contract.json',dict(action_contract='simtoolreal_native_v1',actor_observation_dim=140,
        actor_warmup='SAPG actor+critic on object waypoint targets; no BC/action labels',
        critic_calibration='switch to native random targets; reset critic weights+optimizer; freeze actor and actor input normalization',
        joint='native random targets; unfreeze actor and input normalization; actor lr 1e-5 linear decay, clip .05, critic lr 5e-5',
        waypoint_epoch_end=args.waypoint_epochs,critic_epoch_end=args.waypoint_epochs+args.critic_epochs,
        total_epochs=total,demonstration_resets=False,trajectory_tracking_reward=False,
        waypoint_end_behavior='native random targets for remainder of episode; never repeat final point',
        old_residual_checkpoint_loaded=False,phase_transition='hard reset env/recurrent state/episode meters; policy copied in memory'))
    if args.wandb_project:
        import subprocess
        with (args.output/'wandb_sync.log').open('a') as log:
            subprocess.Popen([sys.executable,'-u',str(PROJECT/'examples/arm_hand/sync_mustard_sapg_wandb.py'),
                '--run-dir',str(args.output),'--project',args.wandb_project],stdout=log,stderr=log,
                stdin=subprocess.DEVNULL,start_new_session=True)
    wrapped=env
    if args.capture_video:
        import gymnasium as gym
        wrapped=gym.wrappers.RecordVideo(env,video_folder=str(args.output/'videos'),
            step_trigger=lambda step:step%args.video_interval==0,video_length=args.video_length,
            name_prefix='target_pose_reaching',disable_logger=True)
    register_rlgames_env(wrapped,rl_device='cuda:0',clip_obs=10.,clip_actions=1.)
    class Stages(AlgoObserver):
        def after_init(self,a):
            self.a=a;self.started=time.monotonic();self.phase='waypoint_actor';self.steps=0
            self.critic_initial={k:v.detach().cpu().clone() for k,v in a.central_value_net.model.state_dict().items()}
            original_step=a.optimizer.step
            def step(*x,**kw):
                if self.phase=='critic_only':return None
                self.steps+=1;return original_step(*x,**kw)
            a.optimizer.step=step
            normalizer=a.model.running_mean_std;original_normalizer_train=normalizer.train
            normalizer.train=lambda mode=True:original_normalizer_train(False if self.phase=='critic_only' else mode)
            epoch_original=a.train_epoch
            def epoch():
                n=a.epoch_num
                if n==args.waypoint_epochs+1:self.enter_critic()
                if n==args.waypoint_epochs+args.critic_epochs+1:self.enter_joint()
                if self.phase=='joint_sapg':
                    progress=(n-args.waypoint_epochs-args.critic_epochs-1)/args.joint_epochs
                    a.last_lr=1e-5*(1-progress);a.update_lr(a.last_lr)
                return epoch_original()
            a.train_epoch=epoch

        def enter_critic(self):
            a=self.a;self.phase='critic_only';goals['enabled']=False
            a.central_value_net.model.load_state_dict(self.critic_initial)
            a.central_value_net.optimizer.state.clear();a.central_value_net.lr=5e-5
            for group in a.central_value_net.optimizer.param_groups:group['lr']=5e-5
            with torch.no_grad():a.model.a2c_network.sigma.fill_(-2.)
            a.model.running_mean_std.eval()
            self.actor_frozen={k:v.detach().cpu().clone() for k,v in a.model.state_dict().items() if not k.startswith('value_mean_std.')}
            self.critic_before={k:v.detach().cpu().clone() for k,v in a.central_value_net.model.state_dict().items()}
            self.steps_before=self.steps
            a.scheduler=IdentityScheduler();a.e_clip=.05
            a.mini_epochs_num=4;a.central_value_net.mini_epoch=4
            a.obs=a.env_reset()
            for state in a.rnn_states:state.zero_()
            a.dones.fill_(1)
            for name in ('current_rewards','current_shaped_rewards','current_lengths'):getattr(a,name).zero_()
            for name in ('game_rewards','game_shaped_rewards','game_lengths'):getattr(a,name).clear()
            a.algo_observer.after_clear_stats()
            save('critic_reset_check.json',dict(fresh_critic=True,optimizer_state_entries=len(a.central_value_net.optimizer.state),
                random_targets=True,demonstration_resets=False,epoch=a.epoch_num))

        def enter_joint(self):
            a=self.a
            changed=[k for k,v in self.actor_frozen.items() if not torch.equal(v,a.model.state_dict()[k].detach().cpu())]
            critic_changed=any(not torch.equal(v,a.central_value_net.model.state_dict()[k].detach().cpu()) for k,v in self.critic_before.items())
            save('critic_warmup_check.json',dict(actor_changed_keys=changed,critic_changed=critic_changed,
                actor_optimizer_steps_during_freeze=self.steps-self.steps_before))
            assert not changed and critic_changed and self.steps==self.steps_before
            self.phase='joint_sapg';a.model.running_mean_std.train()

        def after_print_stats(self,frame,epoch_num,total_time):
            a=self.a
            row=dict(epoch=epoch_num,frames=frame,phase=self.phase,actor_optimizer_steps=self.steps,
                elapsed_s=time.monotonic()-self.started,waypoint_enabled=goals['enabled'],
                waypoints_assigned=goals['assigned'],waypoints_skipped=goals['skipped'],
                chains_completed=goals['completed'],mean_reward=float(a.game_rewards.get_mean()[0]) if a.game_rewards.current_size else None)
            with (args.output/'sapg_progress.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print('SAPG_PROGRESS',json.dumps(row),flush=True)
            if epoch_num==args.waypoint_epochs:a.save(str(args.output/'waypoint_actor'))
            if epoch_num==args.waypoint_epochs+args.critic_epochs:a.save(str(args.output/'critic_calibrated'))
            if epoch_num%1000==0 or epoch_num==total:a.save(str(args.output/'sapg_latest'))
    observer=Stages();runner=Runner(MultiObserver([EnvStatsAlgoObserver(),observer]))
    runner.load(cfg);runner.reset();runner.run(dict(train=True,play=False,checkpoint=None))
    save('completed.json',dict(stages=['waypoint_actor','critic_only','joint_sapg'],epochs=total,
        actor_optimizer_steps=observer.steps,action_contract='simtoolreal_native_v1'))
    goals['restore']();wrapped.close()


if __name__=='__main__':main()
