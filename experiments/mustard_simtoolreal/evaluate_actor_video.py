"""Record a small, separate native-action rollout without cameras in training."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parents[1];VENDOR=PROJECT/'third_party/simtoolreal'
sys.path[:0]=[str(VENDOR),str(VENDOR/'rl_games'),str(ROOT)]


def main():
    from isaaclab.app import AppLauncher
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--waypoints',type=Path,required=True)
    parser.add_argument('--frames',type=int,default=300)
    parser.add_argument('--reset-mode',choices=['native','demo'],default='native')
    parser.add_argument('--dataset',type=Path,help='Native BC cache required for demo reset.')
    parser.add_argument('--episode-seconds',type=float,help='BC diagnostic horizon only; formal SAPG is unchanged.')
    parser.add_argument('--final-goal-only',action='store_true',help='Evaluate the same final target used by fixed-goal BC.')
    parser.add_argument('--reset-xy-noise',type=float,default=0.)
    parser.add_argument('--reset-yaw-noise',type=float,default=0.)
    parser.add_argument('--reset-joint-noise',type=float,default=0.)
    parser.add_argument('--success-tolerance',type=float)
    parser.add_argument('--multi-target-level',type=int,choices=range(5))
    parser.add_argument('--native-object-reset',action='store_true')
    parser.add_argument('--native-robot-reset',action='store_true')
    parser.add_argument('--reset-z-noise',type=float,default=0.)
    parser.add_argument('--native-position-mix',type=float,default=0.)
    parser.add_argument('--coef',type=float,default=50.,choices=(50.,40.,30.,20.,10.,0.),help='SAPG block coefficient; 50 = leader')
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args();args.enable_cameras=True
    app=AppLauncher(args).app
    try:
        import torch
        import imageio.v2 as imageio
        from environment import create_env
        from trajectory_resets import PairedTrajectoryResets
        from rl_games.algos_torch import model_builder
        if args.reset_mode=='demo':assert args.dataset is not None and args.final_goal_only
        def configure(cfg):
            if args.reset_mode=='demo':
                dr=cfg.domain_randomization
                dr.use_action_delay=False;dr.use_obs_delay=False;dr.use_object_state_delay_noise=False
                dr.joint_velocity_obs_noise_std=0.;dr.force_scale=0.;dr.torque_scale=0.
                if not args.native_object_reset:cfg.reset.table_reset_z_range=0.
            if args.episode_seconds is not None:
                assert args.episode_seconds>0
                cfg.episode_length_s=args.episode_seconds
                cfg.termination.episode_length=round(args.episode_seconds*60)
            cfg.termination.max_consecutive_successes=1 if args.final_goal_only else 2
            cfg.termination.tolerance_curriculum_success_threshold=1.
            if args.success_tolerance is not None:
                assert args.success_tolerance>0
                cfg.termination.eval_success_tolerance=args.success_tolerance
        env=create_env(6,render_mode='rgb_array',configure=configure)
        assert min(args.reset_xy_noise,args.reset_yaw_noise,args.reset_joint_noise)>=0
        paired=PairedTrajectoryResets(env,args.waypoints,xy_noise=args.reset_xy_noise,yaw_noise_deg=args.reset_yaw_noise)
        assert args.reset_z_noise>=0 and 0<=args.native_position_mix<=1
        paired.z_noise=args.reset_z_noise;paired.native_position_mix=args.native_position_mix
        if args.final_goal_only:
            paired.positions=paired.positions[:,-1:].repeat(1,2,1)
            paired.rotations=paired.rotations[:,-1:].repeat(1,2,1)
        reference=None;restore_demo=None
        if args.reset_mode=='demo':
            from bc_demo_reset import attach_demo_resets
            reference,restore_demo=attach_demo_resets(env,paired,args.dataset,perturbations={'joint_rad':args.reset_joint_noise})
        multi=None
        if args.multi_target_level is not None:
            from multi_target_goals import MultiTargetGoals
            multi=MultiTargetGoals(env,paired,args.multi_target_level)
        if args.native_object_reset:
            from native_object_resets import attach_native_object_resets
            attach_native_object_resets(env,paired,native_robot=args.native_robot_reset)
        old=torch.load(args.checkpoint,map_location=env.device,weights_only=False)
        assert old['action_contract']=='simtoolreal_native_v1'
        params=dict(network=old['network'],model=dict(name='continuous_a2c_logstd'))
        model=model_builder.ModelBuilder().load(params).build(dict(actions_num=29,input_shape=(141,),
            num_seqs=6,value_size=1,normalize_input=True,normalize_value=True,type='extra_param',
            coef_ids=torch.linspace(50,0,6,device=env.device),coef_id_idx=140)).to(env.device)
        model.load_state_dict(old['model']);model.eval()
        states=[s.to(env.device) for s in model.get_default_rnn_state()]
        obs,_=env.reset();args.output.parent.mkdir(parents=True,exist_ok=True)
        # Initialize the render product before recording, without advancing
        # physics or consuming policy frames (initial frames can be black).
        for _ in range(20):
            frame=env.render(recompute=True)
            if frame.size and frame.max()>0:break
        initial_error=float((obs['policy'].clamp(-10,10)-reference()).abs().max()) if reference is not None else None
        if initial_error is not None and not args.native_object_reset and not any((args.reset_xy_noise,args.reset_yaw_noise,args.reset_joint_noise,args.reset_z_noise,args.native_position_mix)):
            assert initial_error<1e-4,f'Demo observation reset mismatch: {initial_error}'
        sums={};completed_episodes=0;successful_episodes=0
        def add(name,value):
            if isinstance(value,dict):
                for key,item in value.items():add(name+'/'+key,item)
            elif torch.is_tensor(value):
                if value.numel():sums[name]=sums.get(name,0.)+float(value.float().mean())
            elif isinstance(value,(int,float)):sums[name]=sums.get(name,0.)+float(value)
        with imageio.get_writer(str(args.output),fps=60,codec='libx264',macro_block_size=1) as writer:
            for _ in range(args.frames):
                inputs=torch.cat([obs['policy'],torch.full((6,1),args.coef,device=env.device)],-1)
                with torch.no_grad():result=model(dict(is_train=False,obs=inputs,rnn_states=states))
                states=result['rnn_states'];obs,reward,done,truncated,info=env.step(result['mus'].clamp(-1,1))
                ended=done|truncated
                completed_episodes+=int(ended.sum())
                final=info.get('episode_final',{})
                if 'successes' in final:
                    required=env.cfg.termination.max_consecutive_successes
                    successful_episodes+=int(((final['successes']>=required)&ended).sum())
                add('reward',reward)
                for key in ('episode_cumulative','episode_final','current_success_tolerance'):add(key,info.get(key,{}))
                for state in states:state[:,done|truncated]=0
                writer.append_data(env.render())
        metrics={key:value/args.frames for key,value in sums.items()}
        # Legacy episode_final/* values above are time averages of events,
        # not per-episode success rates. Report completed-episode rates explicitly.
        metrics['eval/completed_episodes']=completed_episodes
        metrics['eval/successful_episodes']=successful_episodes
        metrics['eval/success_rate']=successful_episodes/completed_episodes if completed_episodes else None
        metrics['demo_reset']=float(args.reset_mode=='demo' and not args.native_object_reset)
        metrics['native_object_reset']=float(args.native_object_reset)
        metrics['native_robot_reset']=float(args.native_robot_reset)
        if initial_error is not None:metrics['initial_observation_max_error']=initial_error
        metrics['evaluated_bc_epoch']=old.get('bc_epochs',-1)
        metrics['evaluated_bc_total_epochs']=old.get('bc_epochs',0)+(old.get('initial_checkpoint_epoch') or 0)
        metrics['evaluated_training_epoch']=old.get('training_epoch',0)
        metrics['reset_xy_noise']=args.reset_xy_noise
        metrics['reset_yaw_noise_deg']=args.reset_yaw_noise
        metrics['reset_joint_noise_rad']=args.reset_joint_noise
        metrics['reset_z_noise']=args.reset_z_noise
        metrics['native_position_mix']=args.native_position_mix
        metrics['max_targets']=env.cfg.termination.max_consecutive_successes
        metrics['target_transitions']=multi.transitions if multi is not None else 0
        metrics_path=args.output.with_suffix('.metrics.json');temporary=metrics_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(metrics,indent=2)+'\n');temporary.replace(metrics_path)
        if restore_demo is not None:restore_demo()
        if multi is not None:multi.restore()
        paired.restore();env.close()
    finally:app.close()


if __name__=='__main__':main()
