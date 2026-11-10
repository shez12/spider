"""Historical reward-based SAPG warmup; current action imitation uses train_bc.py."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parents[1];VENDOR=PROJECT/'third_party/simtoolreal'
sys.path[:0]=[str(VENDOR),str(VENDOR/'rl_games'),str(ROOT)]


def main():
    from isaaclab.app import AppLauncher
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--num-envs',type=int,default=8184)
    parser.add_argument('--block-size',type=int,default=1364)
    parser.add_argument('--epochs',type=int,default=10000)
    parser.add_argument('--waypoints',type=Path,default=ROOT/'waypoints100.json')
    parser.add_argument('--reset-xy-noise-m',type=float,default=.005)
    parser.add_argument('--reset-yaw-noise-deg',type=float,default=2.)
    parser.add_argument('--checkpoint-every',type=int,default=100)
    parser.add_argument('--video-every',type=int,default=1000,help='Separate six-env evaluator; 0 disables it.')
    parser.add_argument('--video-length',type=int,default=300)
    parser.add_argument('--wandb-project',default=None)
    parser.add_argument('--verify-reset',action='store_true')
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args();args.output=args.output.resolve()
    assert args.num_envs==6*args.block_size and args.num_envs%4==0
    assert args.epochs>0 and args.checkpoint_every>0 and args.video_every>=0
    assert 0<=args.reset_xy_noise_m<=.02 and 0<=args.reset_yaw_noise_deg<=10
    assert not args.output.exists(),'Use a new output directory.'
    pool=json.loads(args.waypoints.read_text())
    assert len(pool['trajectories'])==100 and {len(t['pos']) for t in pool['trajectories']}=={2}
    assert all('initial_pose' in t and t['final_pose_included'] for t in pool['trajectories'])
    app=AppLauncher(args).app
    try:train(args)
    except BaseException as error:
        if args.output.exists():(args.output/'failed.json').write_text(json.dumps(dict(error=type(error).__name__,message=str(error)))+'\n')
        raise
    finally:app.close()


def train(args):
    import subprocess
    import torch
    import yaml
    from environment import create_env
    from trajectory_resets import PairedTrajectoryResets
    from rl_games.torch_runner import Runner
    from rl_games.common.algo_observer import AlgoObserver
    from isaacsimenvs.utils.rlgames_utils import register_rlgames_env,EnvStatsAlgoObserver,MultiObserver
    args.output.mkdir(parents=True)
    def save(name,data):
        (args.output/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str)+'\n')
    save('run_arguments.json',vars(args))
    def configure(cfg):
        cfg.termination.max_consecutive_successes=2
        # Original threshold 3 is unreachable when a chain contains two targets.
        cfg.termination.tolerance_curriculum_success_threshold=1.
    env=create_env(args.num_envs,configure=configure)
    reset=PairedTrajectoryResets(env,args.waypoints,args.reset_xy_noise_m,args.reset_yaw_noise_deg)
    save('waypoint_pool.json',json.loads(args.waypoints.read_text()))
    save('environment_config.json',env.cfg.to_dict())
    save('stage_contract.json',dict(stage='actor_waypoint_warmup_only',later_critic_calibration=False,later_joint_training=False,
        actor_and_auxiliary_critic_train=True,action_contract='simtoolreal_native_v1',residual_checkpoint_loaded=False,
        trajectory_count=100,waypoints_per_trajectory=2,blocks=6,block_size=args.block_size,
        object_reset='same selected trajectory initial pose, zero velocity',robot_reset='original randomized robot reset',
        perturbations='shared XY/yaw transform applied to object start and every goal; tabletop height variation shared',
        chain_end='terminate and sample new trajectory; no random-goal fallback',
        reset_xy_noise_m=args.reset_xy_noise_m,reset_yaw_noise_deg=args.reset_yaw_noise_deg,
        max_consecutive_successes=2,curriculum_threshold=1.,success_steps=10,success_steps_consecutive=False,
        actor_obs_dim=140,critic_obs_dim=162,num_actions=29,sim_hz=120,control_hz=60,
        nominal_samples_per_epoch=args.num_envs*16,max_transitions=args.epochs*args.num_envs*16,
        waypoint_sha256=hashlib.sha256(args.waypoints.read_bytes()).hexdigest(),
        video='asynchronous six-env process; training process has no camera/render wrapper'))
    if args.verify_reset:
        from verify_paired_resets import verify
        save('paired_reset_checks.json',verify(env,reset))
    cfg=yaml.safe_load((VENDOR/'isaacsimenvs/cfg/train/SimToolRealSAPG.yaml').read_text())
    config=cfg['params']['config']
    config.update(name='0_mustard_actor_warmup',full_experiment_name='0_mustard_actor_warmup',
        train_dir=str(args.output),device='cuda:0',device_name='cuda:0',num_actors=args.num_envs,
        expl_coef_block_size=args.block_size,max_epochs=args.epochs,save_frequency=1000)
    # Preserve original SAPG LR=1e-4/adaptive, clip=.1, mini_epochs=2,
    # minibatch_per_env=4, AMP, sigma initialization, CV and normalizers.
    assert config['minibatch_size_per_env']==4 and args.num_envs*4%16==0
    assert config['use_others_experience']=='lf' and config['expl_type']=='mixed_expl_learn_param'
    save('sapg_config.json',cfg)
    register_rlgames_env(env,rl_device='cuda:0',clip_obs=10.,clip_actions=1.)
    if args.wandb_project:
        with (args.output/'wandb_sync.log').open('a') as log:
            subprocess.Popen([sys.executable,'-u',str(PROJECT/'examples/arm_hand/sync_mustard_sapg_wandb.py'),
                '--run-dir',str(args.output),'--project',args.wandb_project],stdout=log,stderr=log,
                stdin=subprocess.DEVNULL,start_new_session=True)
    class Warmup(AlgoObserver):
        def after_init(self,a):
            self.a=a;self.start=time.monotonic();self.video_process=None
            ids=a.intr_reward_coef_embd[::a.intr_coef_block_size,0].cpu().tolist()
            assert ids==[50.,40.,30.,20.,10.,0.]
            save('initialization_check.json',dict(blocks=len(ids),block_size=a.intr_coef_block_size,
                num_envs=a.num_actors,coef_ids=ids,minibatch_size=a.minibatch_size,
                nominal_batch_size=a.batch_size,actor_obs_dim=env.cfg.observation_space,
                critic_state_dim=env.cfg.state_space,action_dim=env.cfg.action_space,
                imported_environment=sys.modules['isaacsimenvs'].__file__,imported_sapg=sys.modules['rl_games'].__file__))

        def after_print_stats(self,frame,epoch_num,total_time):
            a=self.a
            row=dict(epoch=epoch_num,frames=a.frame,phase='actor_waypoint_warmup',elapsed_s=time.monotonic()-self.start,
                num_envs=args.num_envs,blocks=6,block_size=args.block_size,chains_ended=reset.completed,
                chains_successful=reset.successful,waypoints_skipped=reset.skipped,
                mean_reward=float(a.game_rewards.get_mean()[0]) if a.game_rewards.current_size else None,
                gpu_allocated_gib=torch.cuda.memory_allocated()/2**30,gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30)
            with (args.output/'sapg_progress.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print('ACTOR_WARMUP',json.dumps(row),flush=True)
            if epoch_num%args.checkpoint_every==0 or epoch_num==args.epochs:
                a.save(str(args.output/'sapg_latest'))
                save('trajectory_sampling_counts.json',dict(candidate_ids=reset.candidate_ids.cpu().tolist(),
                    reset_counts_per_block=reset.reset_counts.cpu().tolist()))
            if args.video_every and (epoch_num%args.video_every==0 or epoch_num==args.epochs):
                if self.video_process is None or self.video_process.poll() is not None:
                    checkpoint=args.output/f'video_actor_{epoch_num:06d}.pt'
                    torch.save(dict(model=a.model.state_dict(),network=cfg['params']['network'],action_contract='simtoolreal_native_v1'),checkpoint)
                    with (args.output/'video_evaluator.log').open('a') as log:
                        self.video_process=subprocess.Popen([sys.executable,str(ROOT/'evaluate_actor_video.py'),
                            '--headless','--checkpoint',str(checkpoint),'--output',str(args.output/'videos'/f'epoch_{epoch_num:06d}.mp4'),
                            '--waypoints',str(args.waypoints),'--frames',str(args.video_length)],stdout=log,stderr=log,
                            stdin=subprocess.DEVNULL,start_new_session=True)
    observer=Warmup();runner=Runner(MultiObserver([EnvStatsAlgoObserver(),observer]))
    runner.load(cfg);runner.reset();runner.run(dict(train=True,play=False,checkpoint=None))
    a=observer.a
    a.save(str(args.output/'actor_warmup'))
    torch.save(dict(model=a.model.state_dict(),network=cfg['params']['network'],action_contract='simtoolreal_native_v1',
        actor_obs_dim=140,coef_ids=[50,40,30,20,10,0],actor_only_warmup_finished=True),args.output/'actor_init.pt')
    a.writer.flush()
    save('completed.json',dict(stage='actor_waypoint_warmup_only',epochs=args.epochs,
        transitions=a.frame,later_stages_run=False,actor_checkpoint='actor_init.pt'))
    reset.restore();env.close()


if __name__=='__main__':main()
