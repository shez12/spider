"""Frozen BC policy, fresh native asymmetric critic, paired demonstration resets.

Optional joint SAPG follows recalibration, with small exploration noise and a
success-gated initial-state curriculum. Default behavior remains critic-only.
"""
import argparse
from collections import deque
import math
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
VENDOR=PROJECT/'third_party/simtoolreal'
sys.path[:0]=[str(VENDOR),str(VENDOR/'rl_games'),str(ROOT)]


def main():
    from isaaclab.app import AppLauncher
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--waypoints',type=Path,required=True)
    p.add_argument('--num-envs',type=int,default=8184)
    p.add_argument('--block-size',type=int,default=1364)
    p.add_argument('--critic-epochs',type=int,default=683)
    p.add_argument('--critic-lr',type=float,default=5e-5)
    p.add_argument('--critic-initialization',type=Path)
    p.add_argument('--sigma-scale',type=float,default=1.)
    p.add_argument('--joint-epochs',type=int,default=0)
    p.add_argument('--actor-lr',type=float,default=1e-5)
    p.add_argument('--actor-clip',type=float,default=.05)
    p.add_argument('--curriculum-interval',type=int,default=500)
    p.add_argument('--curriculum-min-episodes',type=int,default=1000)
    p.add_argument('--curriculum-success',type=float,default=.70,help='Legacy option; monotonic curriculum now uses the native mean-goal gate.')
    p.add_argument('--verify-reset-curriculum',action='store_true')
    p.add_argument('--resume-training',type=Path,help='Full SAPG checkpoint; restore both optimizers and restart physical episodes.')
    p.add_argument('--multi-target',action='store_true')
    p.add_argument('--multi-target-level',type=int,default=0)
    p.add_argument('--native-object-reset',action='store_true',help='Vendor random object position/full orientation and first goal immediately; retain demo robot reset.')
    p.add_argument('--native-robot-reset',action='store_true',help='Use vendor joint position/velocity reset after demo wrapper.')
    p.add_argument('--native-tolerance',action='store_true',help='Enable vendor tolerance curriculum immediately.')
    p.add_argument('--start-unfrozen',action='store_true',help='Start joint epoch1 from the frozen actor/calibrated critic checkpoint, with fresh optimizers.')
    p.add_argument('--diverse-reset-bank',type=Path,help='Validated GPU physical-state cache, mixed with native resets.')
    p.add_argument('--fixed-difficulty',action='store_true',help='Hold timeout/tolerance fixed for a controlled reset ablation.')
    p.add_argument('--restore-native-curriculum',action='store_true',help='Expand object reset positions, then shorten 16s timeout to native 10s.')
    p.add_argument('--verify-multi-target',action='store_true')
    p.add_argument('--run-epochs-limit',type=int,default=0,help='Smoke-test limit without changing the LR schedule horizon.')
    p.add_argument('--preuploaded-history',action='store_true',help='Allow a prepared output only after remote epoch1-5000 history verification.')
    p.add_argument('--pose-curriculum',action='store_true',help='Blend native-kind resets from the demo start toward native samples (requires --native-object-reset).')
    p.add_argument('--pose-alpha',type=float,default=0.,help='Initial maximum demo-to-native mix.')
    p.add_argument('--pose-step',type=float,default=.1)
    p.add_argument('--pose-threshold',type=float,default=.5,help='Frontier any-goal rate needed to raise the mix.')
    p.add_argument('--pose-window',type=int,default=500,help='Completed leader frontier episodes per gate decision.')
    p.add_argument('--pose-frontier-probability',type=float,default=.5)
    p.add_argument('--pose-goal-curriculum',action='store_true',help='Also blend the first goal from the demo target toward the native absolute goal.')
    p.add_argument('--adaptive-kl',type=float,default=0.,help='Use rl_games adaptive actor LR with this KL threshold (0 keeps linear decay).')
    p.add_argument('--resume-logstd',type=float,help='Overwrite every block log-std after loading --resume-training.')
    p.add_argument('--episode-seconds',type=float,default=16.)
    p.add_argument('--checkpoint-every',type=int,default=100)
    p.add_argument('--video-every',type=int,default=200)
    p.add_argument('--video-length',type=int,default=960)
    p.add_argument('--wandb-project',default='simtoolreal-warmstart')
    AppLauncher.add_app_launcher_args(p)
    a=p.parse_args()
    assert a.num_envs==6*a.block_size and a.block_size>0
    assert a.critic_epochs>0 and a.critic_lr>0 and a.checkpoint_every>0
    assert a.episode_seconds>0 and a.sigma_scale>0 and a.joint_epochs>=0
    assert a.actor_lr>0 and 0<a.actor_clip<1
    if a.native_robot_reset:assert a.native_object_reset
    if a.pose_curriculum:assert a.native_object_reset
    if a.pose_goal_curriculum:assert a.pose_curriculum
    if a.resume_logstd is not None:assert a.resume_training is not None
    assert a.adaptive_kl>=0
    if a.native_tolerance:assert not a.fixed_difficulty
    if a.start_unfrozen:assert a.critic_initialization is not None and a.resume_training is None and a.joint_epochs>0
    assert a.curriculum_interval>0 and a.curriculum_min_episodes>0 and 0<a.curriculum_success<=1
    if a.critic_initialization is not None:
        a.critic_initialization=a.critic_initialization.resolve();assert a.critic_initialization.is_file()
    if a.resume_training is not None:
        a.resume_training=a.resume_training.resolve();assert a.resume_training.is_file()
        assert (a.resume_training.parent/'critic_optimizer.pt').is_file()
    assert 0<=a.multi_target_level<=4
    assert a.run_epochs_limit>=0
    assert a.video_every>=0 and a.video_length>0
    assert (a.num_envs*4)%16==0,'Native actor dataset requires full sequences.'
    for name in ('checkpoint','dataset','waypoints'):
        path=getattr(a,name).resolve();assert path.is_file(),path;setattr(a,name,path)
    a.output=a.output.resolve()
    if a.preuploaded_history:
        proof=json.loads((a.output/'history_remote_verification.json').read_text())
        assert proof['verified_rows']==5000 and proof['min_epoch']==1 and proof['max_epoch']==5000
        assert proof['run_id']==a.output.name and proof['complete'] is True
        assert not (a.output/'sapg_progress.jsonl').exists(),'Prepared output already contains training.'
    else:assert not a.output.exists(),'Choose a fresh output directory.'
    a.output.mkdir(parents=True,exist_ok=a.preuploaded_history)
    app=None
    try:
        app=AppLauncher(a).app
        train(a)
    except BaseException as error:
        import traceback
        traceback.print_exc()
        (a.output/'failed.json').write_text(json.dumps(dict(error=type(error).__name__,message=str(error)))+'\n')
        raise
    finally:
        if app is not None:app.close()


def train(args):
    import numpy as np
    import torch
    import yaml
    from environment import create_env
    from trajectory_resets import PairedTrajectoryResets
    from bc_demo_reset import attach_demo_resets
    from rl_games.torch_runner import Runner
    from rl_games.algos_torch import torch_ext
    from numerics import gaussian_policy_kl
    torch_ext.policy_kl=gaussian_policy_kl
    from isaacsimenvs.utils.rlgames_utils import register_rlgames_env,EnvStatsAlgoObserver

    def save(name,data):
        path=args.output/name;tmp=path.with_suffix(path.suffix+'.tmp')
        tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str)+'\n');tmp.replace(path)
    save('run_arguments.json',vars(args))
    def actor_scheduler():
        from rl_games.common.schedulers import IdentityScheduler,AdaptiveScheduler
        return AdaptiveScheduler(args.adaptive_kl) if args.adaptive_kl else IdentityScheduler()
    old=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    assert old['action_contract']=='simtoolreal_native_v1' and old['actor_obs_dim']==140
    with np.load(args.dataset) as data:
        assert str(data['action_contract'])=='simtoolreal_native_v1'
        assert 0<data['obs'].shape[0]<=args.num_envs and data['obs'].shape[-1]==140
        assert len(set(data['candidate_ids'].tolist()))==data['obs'].shape[0]
        assert not bool(data['smoke_test'])
    def configure(cfg):
        dr=cfg.domain_randomization
        dr.use_action_delay=False;dr.use_obs_delay=False;dr.use_object_state_delay_noise=False
        dr.joint_velocity_obs_noise_std=0.;dr.force_scale=0.;dr.torque_scale=0.
        if not args.native_object_reset:cfg.reset.table_reset_z_range=0.
        cfg.episode_length_s=args.episode_seconds
        cfg.termination.episode_length=round(args.episode_seconds*60)
        # Single final target; no changing tolerance while the policy is frozen.
        cfg.termination.max_consecutive_successes=1
        cfg.termination.tolerance_curriculum_interval=3000 if args.native_tolerance else 1000000000
    env=create_env(args.num_envs,configure=configure)
    assert env.cfg.termination.episode_length==round(args.episode_seconds*60)
    paired=PairedTrajectoryResets(env,args.waypoints,xy_noise=0.,yaw_noise_deg=0.)
    paired.positions=paired.positions[:,-1:].repeat(1,2,1)
    paired.rotations=paired.rotations[:,-1:].repeat(1,2,1)
    perturbations={'joint_rad':0.}
    reference,restore=attach_demo_resets(env,paired,args.dataset,perturbations=perturbations)
    multi=None;pose=None
    if args.multi_target:
        from multi_target_goals import MultiTargetGoals,LEVELS as TARGET_LEVELS
        multi=MultiTargetGoals(env,paired,args.multi_target_level)
        if args.verify_multi_target:
            from multi_target_goals import verify_lifecycle
            save('multi_target_lifecycle_check.json',verify_lifecycle(env,paired,multi))
            multi.set_level(args.multi_target_level)
    if args.native_object_reset:
        assert (args.resume_training is not None or args.start_unfrozen) and args.multi_target and args.multi_target_level==4
        from native_object_resets import attach_native_object_resets,audit_native_object_resets
        if args.pose_curriculum:
            from pose_curriculum import PoseCurriculum
            pose=PoseCurriculum(env,reference,alpha=args.pose_alpha,step=args.pose_step,
                frontier_probability=args.pose_frontier_probability,threshold=args.pose_threshold,window=args.pose_window,
                demo_goal=(lambda ids:paired.transformed(ids,torch.zeros_like(ids))) if args.pose_goal_curriculum else None)
            pose.enabled=False # native reset audits below must see the unmodified vendor distribution
        native_restore=attach_native_object_resets(env,paired,native_robot=args.native_robot_reset,
            after=pose.apply if args.pose_curriculum else None)
        object_check=audit_native_object_resets(env)
        if args.native_robot_reset:object_check['robot_reset']='unmodified vendor _randomize_robot_dof_state'
        save('native_reset_check.json',object_check)
        if args.native_robot_reset:
            from native_object_resets import audit_native_robot_resets
            save('native_robot_reset_check.json',audit_native_robot_resets(env))
    reset_bank=None
    if args.diverse_reset_bank:
        assert args.native_object_reset and (args.fixed_difficulty or args.native_tolerance)
        from diverse_resets import DiverseResetBank,audit_bank_resets
        reset_bank=DiverseResetBank(env,args.diverse_reset_bank)
        save('diverse_reset_check.json',audit_bank_resets(env,reset_bank))
    if pose is not None:
        pose.enabled=True
        kind_mask=reset_bank.curriculum_eligible_mask if reset_bank is not None else None
        env._curriculum_eligible_mask=lambda:(pose.eligible()&kind_mask()) if kind_mask else pose.eligible()
        from pose_curriculum import audit_pose_resets
        save('pose_curriculum_check.json',audit_pose_resets(env,pose,reset_bank))
    levels=[dict(xy_m=0.,yaw_deg=0.,joint_rad=0.),dict(xy_m=.001,yaw_deg=.5,joint_rad=.002),
        dict(xy_m=.003,yaw_deg=1.,joint_rad=.005),dict(xy_m=.005,yaw_deg=2.,joint_rad=.01)]
    if args.restore_native_curriculum:
        levels.extend([dict(xy_m=.01,yaw_deg=2.,joint_rad=.01,z_m=.002,native_mix=.25),
            dict(xy_m=.02,yaw_deg=2.,joint_rad=.01,z_m=.005,native_mix=.5),
            dict(xy_m=.05,yaw_deg=2.,joint_rad=.01,z_m=.01,native_mix=.75),
            dict(xy_m=.1,yaw_deg=2.,joint_rad=.01,z_m=.02,native_mix=1.)])
    if args.native_object_reset:
        levels=[dict(xy_m=.1,yaw_deg=180.,joint_rad=0.,z_m=.02,native_mix=1.)]
    def set_level(index):
        if args.native_object_reset:
            paired.xy_noise=0.;paired.yaw_noise_deg=0.;paired.z_noise=0.;paired.native_position_mix=0.;perturbations['joint_rad']=0.
            return
        level=levels[index];paired.xy_noise=level['xy_m'];paired.yaw_noise_deg=level['yaw_deg']
        perturbations['joint_rad']=level['joint_rad']
        paired.z_noise=level.get('z_m',0.);paired.native_position_mix=level.get('native_mix',0.)
    if args.verify_reset_curriculum:
        from isaaclab.utils.math import quat_apply,quat_mul
        checks=[]
        for index,level in enumerate(levels):
            set_level(index);obs,_=env.reset();ref=reference()
            lo=env._joint_lower_canon;hi=env._joint_upper_canon
            observed_q=lo+.5*(obs['policy'][:,:29]+1)*(hi-lo)
            expected_q=lo+.5*(ref[:,:29]+1)*(hi-lo)
            qerror=float((observed_q-expected_q).abs().max())
            assert qerror<=level['joint_rad']+1e-4
            center=ref[:,113:125].reshape(-1,4,3).mean(1)+ref[:,87:90]
            actual=env.object.data.root_pos_w-env.scene.env_origins
            assert float((actual-center-paired.translation).abs().max())<1e-4
            ids=torch.arange(env.num_envs,device=env.device)
            goalp,goalq=paired.transformed(ids,torch.zeros_like(ids))
            assert float((env.goal_viz.data.root_pos_w-env.scene.env_origins-goalp).abs().max())<1e-4
            actualq=env.object.data.root_quat_w
            expectedrot=quat_mul(paired.yaw,ref[:,[97,94,95,96]])
            assert float((1-(actualq*expectedrot).sum(-1).abs()).abs().max())<1e-4
            assert float((env._prev_targets-env.robot.data.joint_pos).abs().max())<.1
            if level.get('native_mix',0.)==1.:
                assert float(actual[:,:2].abs().max())<=level['xy_m']+1e-4
                base=env._table_z_per_env+env.cfg.reset.table_object_z_offset
                assert float((actual[:,2]-base).abs().max())<=level['z_m']+1e-4
            checks.append(dict(level=index,joint_max_delta=qerror,object_goal_transform_checked=True))
        if args.restore_native_curriculum:
            horizon_checks=[]
            for seconds in (16.,14.,12.,10.):
                env.cfg.episode_length_s=seconds;env.cfg.termination.episode_length=round(seconds*60)
                env.reset();env._near_goal_steps.zero_();env.episode_length_buf[:]=env.max_episode_length-1
                _,truncated=env._get_dones();assert not truncated.any()
                env.episode_length_buf[:]=env.max_episode_length
                _,truncated=env._get_dones();assert truncated.all()
                horizon_checks.append(dict(seconds=seconds,effective_steps=env.max_episode_length))
            save('horizon_checks.json',horizon_checks)
            env.cfg.episode_length_s=args.episode_seconds;env.cfg.termination.episode_length=round(args.episode_seconds*60)
        set_level(0);save('reset_curriculum_checks.json',checks)
    save('environment_config.json',env.cfg.to_dict())
    cfg=yaml.safe_load((VENDOR/'isaacsimenvs/cfg/train/SimToolRealSAPG.yaml').read_text())
    assert cfg['params']['network']==old['network'],'BC network must match native SAPG exactly.'
    config=cfg['params']['config']
    experiment_name='0_mustard_warmstart_sapg' if args.joint_epochs else '0_mustard_bc_critic'
    config.update(name=experiment_name,full_experiment_name=experiment_name,
        train_dir=str(args.output),num_actors=args.num_envs,expl_coef_block_size=args.block_size,
        device='cuda:0',device_name='cuda:0',max_epochs=args.critic_epochs+args.joint_epochs,
        save_frequency=0,lr_schedule=None,learning_rate=args.actor_lr,e_clip=args.actor_clip,mini_epochs=4)
    config['central_value_config'].update(learning_rate=args.critic_lr,mini_epochs=4,lr_schedule=None)
    save('sapg_config.json',cfg)
    save('stage_contract.json',dict(stage='joint_from_calibrated' if args.start_unfrozen else 'resumed_joint_sapg' if args.resume_training else 'critic_only',actor_checkpoint=str(args.checkpoint),
        actor_checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        bc_base_epochs=old.get('initial_checkpoint_epoch'),bc_extra_epochs=old['bc_epochs'],
        action_contract='simtoolreal_native_v1',actor_obs_dim=140,critic_obs_dim=162,
        blocks=6,block_size=args.block_size,demonstration_resets=not args.native_robot_reset,initial_state_noise=None if args.resume_training or args.native_robot_reset else 0,
        object_velocity='native zero velocity; cached resets preserve validated physical velocity' if reset_bank is not None else 'zero; actor-visible reset, not full PhysX snapshot',
        goals=('native absolute or cached validated first goal; native delta continuation' if reset_bank is not None else
            'native absolute first goal; native delta continuation' if args.native_object_reset else
            'paired first target; native delta continuation and native termination' if args.multi_target else 'paired final target; terminate on completion, no waypoint reward'),
        episode_seconds=args.episode_seconds,success_tolerance_curriculum='native from first joint epoch' if args.native_tolerance else 'held at native initial tolerance',
        policy='critic: scaled BC sigma and all actor buffers frozen; joint: native SAPG actor and sigma learnable',
        returns='native reward, reward scale .01, gamma .99, GAE .95, native per-block entropy bonus',
        critic_batches='on-policy rollout; no actor experience augmentation or actor minibatches',
        critic_optimizer='restored central Adam; lr 5e-5 default; four mini epochs' if args.resume_training else 'fresh Adam; lr 5e-5 default; four mini epochs',
        automatic_joint_transition=bool(args.joint_epochs),joint_epochs=args.joint_epochs,
        resume_checkpoint=args.resume_training,multi_target=args.multi_target,
        multi_target_levels=TARGET_LEVELS if multi is not None else None,
        native_object_reset=args.native_object_reset,robot_reset_mode='vendor_native_or_validated_bank' if args.native_robot_reset and reset_bank is not None else 'vendor_native' if args.native_robot_reset else 'demo_initial_or_validated_bank' if reset_bank is not None else 'cached_demo',first_goal_mode='vendor_absolute_or_validated_bank' if reset_bank is not None else 'vendor_absolute' if args.native_object_reset else 'demo_final',
        native_robot_reset=args.native_robot_reset,native_tolerance_immediate=args.native_tolerance,start_unfrozen=args.start_unfrozen,
        diverse_reset_bank=str(args.diverse_reset_bank) if args.diverse_reset_bank else None,fixed_difficulty=args.fixed_difficulty,
        initial_target_level=multi.level if multi is not None else None,
        target_count_fixed=bool(multi is not None and multi.level==len(TARGET_LEVELS)-1),
        initial_max_targets=env.cfg.termination.max_consecutive_successes,
        timeout='16s since last success; native timer reset' if multi is not None else 'single-target 16s',
        initial_logstd=args.resume_logstd if args.resume_training else -2.+math.log(args.sigma_scale),critic_initialization=args.critic_initialization,
        actor_lr=args.actor_lr,actor_lr_schedule=f'rl_games adaptive, KL threshold {args.adaptive_kl}' if args.adaptive_kl else 'linear decay over joint phase',actor_clip=args.actor_clip,
        pose_curriculum=dict(start_alpha=args.pose_alpha,step=args.pose_step,threshold=args.pose_threshold,window=args.pose_window,
            frontier_probability=args.pose_frontier_probability,scope='native-kind resets only; joints/velocities/targets/object XY lerp, orientation slerp; native z for alpha>0',
            first_goal='demo final target slerped/lerped toward native absolute goal with the same alpha' if args.pose_goal_curriculum else 'native absolute',
            tolerance_gate='fully native (alpha=1) native-kind episodes only') if args.pose_curriculum else None,
        actor_input_statistics='fixed BC statistics throughout both stages',
        kl_diagnostic='FP32 exact Gaussian KL; no additive variance epsilon bias at small sigma',
        restore_native_curriculum=args.restore_native_curriculum,timeout_curriculum=[16,14,12,10] if args.restore_native_curriculum else None,
        gate_definition='native mean goals over last completed episodes; >=3 for caps>=3, >=80% of cap for one/two-goal bridge; low performance holds difficulty',
        reset_curriculum_levels=levels,curriculum_min_joint_epochs=args.curriculum_interval,
        tolerance_curriculum='native update_tolerance_curriculum after target/reset/timeout restoration: >=3000 control steps, mean goals>=3, tolerance*=.9 down to .01; hold on poor performance',
        curriculum_gate=dict(min_completed_episodes=args.curriculum_min_episodes,native_mean_goals_threshold=3.,single_two_goal_bridge_fraction=.8,
            window_episodes=10000,rollback=False,low_performance='hold current difficulty'),video='separate 6-env deterministic rollout; native-only reset evaluation' if reset_bank is not None else 'separate 6-env deterministic current-policy rollout; matching reset, goal and timeout curriculum'))
    register_rlgames_env(env,rl_device='cuda:0',clip_obs=10.,clip_actions=1.)
    outcomes=deque(maxlen=10000);full_outcomes=deque(maxlen=10000)
    kind_episodes=[0]*4;kind_goals=[0.]*4;kind_reached=[0]*4
    class CalibrationObserver(EnvStatsAlgoObserver):
        def process_infos(self,infos,done_indices,**kwargs):
            super().process_infos(infos,done_indices,**kwargs)
            if not isinstance(infos,dict):return
            indices=done_indices.flatten()
            # Curriculum is gated by completed leader-block episodes only.
            indices=indices[indices>=env.num_envs-args.block_size]
            if len(indices) and 'successes' in infos.get('episode_final',{}):
                threshold=min(3,env.cfg.termination.max_consecutive_successes) if multi is not None else 1
                outcomes.extend((infos['episode_final']['successes'][indices]>=threshold).cpu().tolist())
                full_outcomes.extend((infos['episode_final']['all_goals_hit'][indices]>=1).cpu().tolist())
                if reset_bank is not None:
                    kinds=reset_bank.last_completed_kind[indices]
                    goals=infos['episode_final']['successes'][indices]
                    for kind in range(4):
                        chosen=goals[kinds==kind]
                        kind_episodes[kind]+=len(chosen);kind_goals[kind]+=float(chosen.sum())
                        kind_reached[kind]+=int((chosen>=1).sum())
                if pose is not None:
                    goals=infos['episode_final']['successes'][indices]
                    native=reset_bank.last_completed_kind[indices]==0 if reset_bank is not None else torch.ones_like(goals,dtype=torch.bool)
                    pose.record(pose.last_completed[indices][native],(goals[native]>=1))
    observer=CalibrationObserver();runner=Runner(observer);runner.load(cfg);runner.reset()
    agent=runner.algo_factory.create(runner.algo_name,base_name='run',params=runner.params)
    agent.model.load_state_dict(old['model'],strict=True)
    with torch.no_grad():agent.model.a2c_network.sigma.add_(math.log(args.sigma_scale))
    if args.critic_initialization is not None:
        state=torch.load(args.critic_initialization,map_location=agent.ppo_device,weights_only=False)
        state=state.get(0,state)
        agent.central_value_net.load_state_dict(state['assymetric_vf_nets'],strict=True)
    agent.model.eval()
    for parameter in agent.model.parameters():parameter.requires_grad_(False)
    assert len(agent.optimizer.state)==0 and len(agent.central_value_net.optimizer.state)==0
    def forbid_actor_step(*_args,**_kwargs):raise RuntimeError('Actor optimizer must never step in critic-only stage.')
    actor_original_step=agent.optimizer.step;actor_steps=0
    def actor_step(*x,**kw):
        nonlocal actor_steps
        result=actor_original_step(*x,**kw);actor_steps+=1;return result
    agent.optimizer.step=forbid_actor_step
    # Unfreezing network weights must not silently change the BC observation scale.
    normalizer_train=agent.model.running_mean_std.train
    agent.model.running_mean_std.train=lambda mode=True:normalizer_train(False)
    critic_steps=0;step_original=agent.central_value_net.optimizer.step
    def critic_step(*x,**kw):
        nonlocal critic_steps
        result=step_original(*x,**kw);critic_steps+=1;return result
    agent.central_value_net.optimizer.step=critic_step
    last_central_loss=None;central_train_original=agent.central_value_net.train_net
    def train_central():
        nonlocal last_central_loss
        last_central_loss=float(central_train_original());return last_central_loss
    agent.central_value_net.train_net=train_central
    agent.init_tensors();start_epoch=0;resume_row={};resume_level=0
    if args.start_unfrozen:
        checkpoint=torch.load(args.critic_initialization,map_location=agent.ppo_device,weights_only=False)
        checkpoint=checkpoint.get(0,checkpoint)
        assert int(checkpoint['epoch'])==args.critic_epochs and not checkpoint['optimizer']['state']
        agent.set_full_state_weights(checkpoint,set_epoch=True);start_epoch=args.critic_epochs
        assert not agent.optimizer.state and not agent.central_value_net.optimizer.state
        agent.central_value_net.frame=agent.frame;agent.central_value_net.epoch_num=start_epoch
        for state in agent.rnn_states:state.zero_()
        agent.dones.fill_(1)
        for name in ('current_rewards','current_shaped_rewards','current_lengths'):getattr(agent,name).zero_()
        for name in ('game_rewards','game_shaped_rewards','game_lengths'):getattr(agent,name).clear()
        observer.after_clear_stats();outcomes.clear();full_outcomes.clear()
        save('unfreeze_initialization_check.json',dict(source=str(args.critic_initialization),source_epoch=start_epoch,
            joint_epochs_already_trained=0,first_joint_epoch=1,actor_optimizer='fresh',critic_optimizer='fresh; historical epoch683 Adam unavailable',
            actor_matches_calibrated=all(torch.equal(agent.model.state_dict()[k],v) for k,v in checkpoint['model'].items()),
            critic_matches_calibrated=all(torch.equal(agent.central_value_net.state_dict()[k],v) for k,v in checkpoint['assymetric_vf_nets'].items())))
    if args.resume_training is not None:
        checkpoint=torch.load(args.resume_training,map_location=agent.ppo_device,weights_only=False)
        checkpoint=checkpoint.get(0,checkpoint)
        start_epoch=int(checkpoint['epoch']);assert start_epoch>args.critic_epochs
        agent.set_full_state_weights(checkpoint,set_epoch=True)
        central=torch.load(args.resume_training.parent/'critic_optimizer.pt',map_location=agent.ppo_device,weights_only=False)
        assert central['epoch']==start_epoch,'Central Adam must match checkpoint epoch.'
        agent.central_value_net.optimizer.load_state_dict(central['optimizer'])
        for group in agent.central_value_net.optimizer.param_groups:group['lr']=args.critic_lr
        agent.central_value_net.lr=args.critic_lr
        progress=args.resume_training.parent/'sapg_progress.jsonl'
        if progress.exists():
            resume_row=next(json.loads(line) for line in progress.read_text().splitlines() if json.loads(line)['epoch']==start_epoch)
            resume_level=0 if args.native_object_reset else resume_row['curriculum_level'];set_level(resume_level)
        curriculum=args.resume_training.parent/'curriculum_changes.jsonl'
        if curriculum.exists():
            for line in curriculum.read_text().splitlines():
                entry=json.loads(line)
                if entry['epoch']<=start_epoch and 'success_tolerance' in entry:
                    env._current_success_tolerance=entry['success_tolerance']
        for parameter in agent.model.parameters():parameter.requires_grad_(True)
        agent.optimizer.step=actor_step
        agent.scheduler=actor_scheduler()
        if args.resume_logstd is not None:
            with torch.no_grad():agent.model.a2c_network.sigma.fill_(args.resume_logstd)
        if args.adaptive_kl:agent.last_lr=args.actor_lr;agent.update_lr(args.actor_lr)
        actor_steps=resume_row.get('actor_optimizer_steps',0)
        critic_steps=resume_row.get('critic_optimizer_steps',0)
        agent.central_value_net.frame=agent.frame;agent.central_value_net.epoch_num=start_epoch
        # Saved observations/recurrent state are not a restorable PhysX snapshot.
        for state in agent.rnn_states:state.zero_()
        agent.dones.fill_(1)
        for name in ('current_rewards','current_shaped_rewards','current_lengths'):getattr(agent,name).zero_()
        for name in ('game_rewards','game_shaped_rewards','game_lengths'):getattr(agent,name).clear()
        observer.after_clear_stats();outcomes.clear();full_outcomes.clear()
        save('resume_check.json',dict(epoch=start_epoch,frames=agent.frame,
            actor_optimizer_entries=len(agent.optimizer.state),critic_optimizer_entries=len(agent.central_value_net.optimizer.state),
            amp_scaler_restored=True,physical_episodes_restarted=True,reset_level=resume_level,
            success_tolerance=env._current_success_tolerance,
            sigma_rescaled_again=False,actor_lr_schedule_restarted=False))
    frozen={k:v.detach().cpu().clone() for k,v in agent.model.state_dict().items()}
    initial_critic={k:v.detach().cpu().clone() for k,v in agent.central_value_net.model.state_dict().items()}
    agent.obs=agent.env_reset()
    error=float((agent.obs['obs'][:,:140]-reference().clamp(-10,10)).abs().max())
    if args.resume_training is None and not args.start_unfrozen:assert error<1e-4,f'BC initialization mismatch: {error}'
    save('initialization_check.json',dict(initial_observation_max_error=error,
        actor_loaded_with_sigma_scale=args.sigma_scale if args.resume_training is None and not args.start_unfrozen else None,
        critic_weights_initialized_from=args.critic_initialization,fresh_critic_optimizer=args.resume_training is None,actor_optimizer_steps=actor_steps,
        critic_optimizer_steps=0,coefficient_ids=agent.intr_reward_coef_embd[::args.block_size,0].tolist()))
    sidecar=None;video=None
    if args.wandb_project:
        with (args.output/'wandb_sync.log').open('a') as log:
            run_id=args.output.parent.name+'_'+args.output.name if args.fixed_difficulty and args.output.parent.name!='outputs' else args.output.name
            sidecar=subprocess.Popen([sys.executable,'-u',str(PROJECT/'examples/arm_hand/sync_mustard_sapg_wandb.py'),
                '--run-dir',str(args.output),'--project',args.wandb_project,'--run-id',run_id],stdout=log,stderr=log,
                stdin=subprocess.DEVNULL,start_new_session=True)
    def launch_video(epoch):
        nonlocal video
        if not args.video_every:return
        if video is not None:
            if video.poll() is None:return
            assert video.returncode==0,'Frozen actor video failed; see video_evaluator.log.'
        path=args.output/'videos'/f'epoch_{epoch:06d}.mp4'
        snapshot=args.output/'video_actor.pt';temporary=snapshot.with_suffix('.tmp')
        payload=dict(old,model={k:v.detach().cpu().clone() for k,v in agent.model.state_dict().items()},training_epoch=epoch)
        torch.save(payload,temporary);temporary.replace(snapshot)
        command=[sys.executable,str(ROOT/'evaluate_actor_video.py'),'--headless',
                '--checkpoint',str(snapshot),'--dataset',str(args.dataset),'--waypoints',str(args.waypoints),
                '--output',str(path),'--frames',str(args.video_length),'--reset-mode','demo',
                '--final-goal-only','--episode-seconds',str(env.cfg.episode_length_s),
                '--success-tolerance',str(env._current_success_tolerance),
                '--reset-xy-noise',str(paired.xy_noise),'--reset-yaw-noise',str(paired.yaw_noise_deg),
                '--reset-joint-noise',str(perturbations['joint_rad'])]
        command+=['--reset-z-noise',str(getattr(paired,'z_noise',0.)),
            '--native-position-mix',str(getattr(paired,'native_position_mix',0.))]
        if multi is not None:command+=['--multi-target-level',str(multi.level)]
        if args.native_object_reset:command+=['--native-object-reset']
        if args.native_robot_reset:command+=['--native-robot-reset']
        with (args.output/'video_evaluator.log').open('a') as log:
            video=subprocess.Popen(command,stdout=log,stderr=log,stdin=subprocess.DEVNULL)
    started=time.monotonic()
    level_index=resume_level;last_level_change=0;joint_initial=None;next_difficulty_dimension='target';last_difficulty=None;native_tolerance_enabled=args.native_tolerance
    total=args.critic_epochs+args.joint_epochs
    end_epoch=min(total,start_epoch+args.run_epochs_limit) if args.run_epochs_limit else total
    if args.resume_training is not None:
        state_file=args.resume_training.parent/'training_state.json'
        if state_file.exists():
            runtime=json.loads(state_file.read_text());assert runtime['epoch']==start_epoch
            level_index=0 if args.native_object_reset else runtime['reset_level'];set_level(level_index)
            if multi is not None and runtime['target_level'] is not None:multi.set_level(runtime['target_level'])
            env.cfg.episode_length_s=runtime['episode_seconds']
            env.cfg.termination.episode_length=round(env.cfg.episode_length_s*60)
            env._current_success_tolerance=runtime['success_tolerance']
            next_difficulty_dimension=runtime['next_difficulty'];last_difficulty=runtime['last_difficulty']
            native_tolerance_enabled=(args.native_tolerance or runtime.get('native_tolerance_enabled',False)) and not args.fixed_difficulty
            if pose is not None and runtime.get('pose_alpha_max') is not None:pose.alpha_max=max(args.pose_alpha,runtime['pose_alpha_max'])
            if native_tolerance_enabled:
                env.cfg.termination.tolerance_curriculum_interval=3000
                env.cfg.termination.tolerance_curriculum_success_threshold=3.
                env._last_curriculum_update=env._frame_counter
            agent.obs=agent.env_reset()
    for epoch in range(start_epoch+1,end_epoch+1):
        agent.epoch_num=epoch
        joint_epoch=max(0,epoch-args.critic_epochs)
        if joint_epoch==1:
            changed=[k for k,v in frozen.items() if not torch.equal(v,agent.model.state_dict()[k].detach().cpu())]
            assert not changed and actor_steps==0
            save('critic_freeze_final.json',dict(actor_changed_keys=changed,actor_optimizer_steps=actor_steps,
                critic_optimizer_steps=critic_steps,epoch=epoch-1))
            agent.save(str(args.output/'critic_calibrated'))
            joint_initial={k:v.detach().cpu().clone() for k,v in agent.model.state_dict().items()}
            for parameter in agent.model.parameters():parameter.requires_grad_(True)
            agent.optimizer.step=actor_step
            agent.scheduler=actor_scheduler();outcomes.clear();full_outcomes.clear()
            if args.adaptive_kl:agent.last_lr=args.actor_lr;agent.update_lr(args.actor_lr)
            # Do not reset physics or recurrent state at this transition: actor is unchanged.
        if joint_epoch:
            if not args.adaptive_kl:
                lr=args.actor_lr*max(0.,1-(joint_epoch-1)/args.joint_epochs)
                agent.last_lr=lr;agent.update_lr(lr)
            agent.e_clip=args.actor_clip
            change=pose.maybe_advance() if pose is not None else None
            if change is not None:
                with (args.output/'curriculum_changes.jsonl').open('a') as f:
                    f.write(json.dumps(dict(epoch=epoch,joint_epoch=joint_epoch,change='pose_alpha',
                        window=args.pose_window,threshold=args.pose_threshold,**change))+'\n')
            if not args.fixed_difficulty and reset_bank is None and joint_epoch%args.curriculum_interval==0 and len(outcomes)>=args.curriculum_min_episodes:
                # Match native tolerance: mean goals in each environment's last
                # completed episode. Low performance never reduces difficulty.
                mean_goals=float(env._prev_episode_successes.float().mean())
                from curriculum import next_difficulty,mean_goal_threshold
                change=next_difficulty(mean_goals=mean_goals,
                    max_targets=env.cfg.termination.max_consecutive_successes,
                    target_level=multi.level if multi is not None else None,
                    target_levels=len(TARGET_LEVELS) if multi is not None else 1,
                    position_level=level_index,position_levels=len(levels),
                    next_dimension=next_difficulty_dimension,seconds=env.cfg.episode_length_s,
                    restore_native=args.restore_native_curriculum,native_tolerance_enabled=native_tolerance_enabled)
                if change=='target':
                    multi.set_level(multi.level+1);next_difficulty_dimension='position'
                elif change=='position':
                    level_index+=1;set_level(level_index);next_difficulty_dimension='target'
                elif change=='timeout':
                    env.cfg.episode_length_s=max(10.,env.cfg.episode_length_s-2.)
                    env.cfg.termination.episode_length=round(env.cfg.episode_length_s*60)
                elif change=='enable_native_tolerance':
                    native_tolerance_enabled=True
                    env.cfg.termination.tolerance_curriculum_interval=3000
                    env.cfg.termination.tolerance_curriculum_success_threshold=3.
                    env._last_curriculum_update=env._frame_counter
                if change is not None:
                    event=dict(epoch=epoch,joint_epoch=joint_epoch,change=change,
                        mean_completed_goals=mean_goals,threshold=mean_goal_threshold(env.cfg.termination.max_consecutive_successes),
                        completed_leader_episodes=len(outcomes),level=level_index,**levels[level_index],
                        target_level=multi.level if multi is not None else None,
                        max_targets=env.cfg.termination.max_consecutive_successes,
                        episode_seconds=env.cfg.episode_length_s,success_tolerance=env._current_success_tolerance)
                    with (args.output/'curriculum_changes.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
                    last_difficulty=change
                    outcomes.clear();full_outcomes.clear()
            result=agent.train_epoch()
            step_s,play_s,update_s,elapsed,a_losses,c_losses,b_losses,entropies,kls,last_lr,lr_mul,extras=result
            agent.frame+=int(agent.curr_frames)
            from rl_games.algos_torch import torch_ext
            loss=last_central_loss
            actor_aux_value_loss=float(torch_ext.mean_list(c_losses))
            actor_policy_loss=float(torch_ext.mean_list(a_losses))
            policy_kl=float(torch_ext.mean_list(kls))
            agent.write_stats(time.monotonic()-started,epoch,step_s,play_s,update_s,a_losses,c_losses,
                entropies,kls,last_lr,lr_mul,agent.frame,elapsed,play_s,int(agent.curr_frames))
            agent.writer.add_scalar('losses/bounds_loss',float(torch_ext.mean_list(b_losses)),agent.frame)
            ev=None;rmse=None
            agent.dataset.update_values_dict(None)
            del result,a_losses,c_losses,b_losses,entropies,kls
        else:
            agent.model.eval();play_start=time.monotonic()
            with torch.no_grad():batch,extras=agent.play_steps()
            play_s=time.monotonic()-play_start
            frames=int(batch.pop('played_frames'));agent.frame+=frames
            target=batch['returns'].detach();prediction=batch['values'].detach()
            variance=target.var(unbiased=False)
            ev=float(1-(target-prediction).var(unbiased=False)/variance) if float(variance)>1e-8 else None
            rmse=float((target-prediction).square().mean().sqrt())
            agent.prepare_dataset(batch)
            update_start=time.monotonic();loss=float(agent.central_value_net.train_net())
            assert np.isfinite(loss),'Nonfinite critic loss'
            agent.dataset.update_values_dict(None)
            update_s=time.monotonic()-update_start
            del batch,target,prediction
            actor_aux_value_loss=None;actor_policy_loss=None;policy_kl=None
        assert np.isfinite(loss),'Nonfinite central critic loss'
        if joint_epoch:assert np.isfinite(actor_policy_loss),'Nonfinite actor policy loss'
        row=dict(epoch=epoch,frames=agent.frame,phase='joint_sapg' if joint_epoch else 'critic_only',joint_epoch=joint_epoch,critic_loss=loss,
            rollout_value_explained_variance=ev,rollout_value_rmse=rmse,
            actor_optimizer_steps=actor_steps,critic_optimizer_steps=critic_steps,
            mean_reward=float(agent.game_rewards.get_mean()[0]) if agent.game_rewards.current_size else None,
            play_s=play_s,update_s=update_s,elapsed_s=time.monotonic()-started,
            completed_target_chains=paired.completed,initial_position_noise=paired.xy_noise,
            reset_yaw_noise_deg=paired.yaw_noise_deg,reset_joint_noise_rad=perturbations['joint_rad'],
            curriculum_level=level_index,completed_leader_episodes=len(outcomes),
            completed_leader_success_rate=sum(full_outcomes)/len(full_outcomes) if full_outcomes else None,
            completed_leader_progress_rate=sum(outcomes)/len(outcomes) if outcomes else None,
            actor_lr=agent.last_lr if joint_epoch else 0.,actor_clip=agent.e_clip,
            policy_sigma_mean=float(agent.model.a2c_network.sigma.exp().mean()))
        row.update(curriculum_mean_completed_goals=float(env._prev_episode_successes.float().mean()),
            native_tolerance_enabled=native_tolerance_enabled,actor_policy_loss=actor_policy_loss,actor_aux_value_loss=actor_aux_value_loss,policy_kl=policy_kl,
            episode_seconds=env.cfg.episode_length_s,object_reset_z_noise=getattr(paired,'z_noise',0.),
            native_position_mix=getattr(paired,'native_position_mix',0.),success_tolerance=env._current_success_tolerance)
        if args.native_object_reset:row.update(native_object_reset=True,initial_position_noise=.1,object_reset_z_noise=.02,native_position_mix=1.,reset_yaw_noise_deg=180.,object_orientation_mode='uniform_3d')
        if args.native_robot_reset:row.update(native_robot_reset=True,robot_initial_position_scale=.1,robot_initial_velocity_bound=.5,robot_reset_mode='vendor_native_or_validated_bank' if reset_bank is not None else 'vendor_native')
        if reset_bank is not None:
            row.update(diverse_resets=True)
            for kind,name in enumerate(('native','near_object','stable_grasp','near_goal')):
                row['reset_count_'+name]=int(reset_bank.counts[:,kind].sum())
                row['completed_leader_episodes_'+name]=kind_episodes[kind]
                row['completed_leader_mean_goals_'+name]=kind_goals[kind]/kind_episodes[kind] if kind_episodes[kind] else None
                row['completed_leader_any_goal_rate_'+name]=kind_reached[kind]/kind_episodes[kind] if kind_episodes[kind] else None
        if pose is not None:row.update(pose_alpha_max=pose.alpha_max,pose_frontier_rate=pose.frontier_rate(),
            pose_frontier_episodes=len(pose.outcomes),pose_last_window_rate=pose.last_window_rate)
        if multi is not None:row.update(target_curriculum_level=multi.level,
            max_targets=env.cfg.termination.max_consecutive_successes,target_transitions=multi.transitions,
            target_delta_m=env.cfg.reset.delta_goal_distance,target_rotation_deg=env.cfg.reset.delta_rotation_degrees)
        for key,value in row.items():
            if isinstance(value,(int,float)):agent.writer.add_scalar('critic_calibration/'+key,value,agent.frame)
        observer.after_print_stats(agent.frame,epoch,row['elapsed_s'])
        agent.writer.flush()
        with (args.output/'sapg_progress.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print('SAPG_PROGRESS',json.dumps(row),flush=True)
        if epoch==start_epoch+1 or epoch%args.checkpoint_every==0 or epoch in (args.critic_epochs,end_epoch) or joint_epoch==1:
            changed=[k for k,v in frozen.items() if not torch.equal(v,agent.model.state_dict()[k].detach().cpu())]
            critic_changed=any(not torch.equal(v,agent.central_value_net.model.state_dict()[k].detach().cpu()) for k,v in initial_critic.items())
            if not joint_epoch:assert not changed and critic_changed and len(agent.optimizer.state)==0
            else:
                assert actor_steps>0 and changed,'Joint training must update actor weights.'
                unchanged_stats=[k for k in frozen if k.startswith('running_mean_std.')]
                assert all(torch.equal(frozen[k],agent.model.state_dict()[k].detach().cpu()) for k in unchanged_stats)
            assert all(torch.isfinite(v).all() for v in agent.model.state_dict().values() if v.is_floating_point())
            assert all(torch.isfinite(v).all() for v in agent.central_value_net.model.state_dict().values() if v.is_floating_point())
            save('joint_update_check.json' if joint_epoch else 'freeze_check.json',dict(epoch=epoch,actor_changed_keys=changed,
                actor_optimizer_steps=actor_steps,critic_changed=critic_changed,critic_optimizer_steps=critic_steps))
            save('reset_coverage.json',dict(candidate_ids=paired.candidate_ids.cpu().tolist(),
                counts_per_block=paired.reset_counts.cpu().tolist(),
                extra_xy_noise=paired.xy_noise,extra_yaw_noise_deg=paired.yaw_noise_deg))
            if reset_bank is not None:save('diverse_reset_coverage.json',reset_bank.report())
            agent.save(str(args.output/('sapg_latest' if joint_epoch else 'critic_latest')))
            # Vendor full checkpoints omit central Adam state: persist it explicitly for handoff.
            path=args.output/'critic_optimizer.pt';tmp=path.with_suffix('.tmp')
            torch.save(dict(epoch=epoch,optimizer=agent.central_value_net.optimizer.state_dict()),tmp);tmp.replace(path)
            save('training_state.json',dict(epoch=epoch,reset_level=level_index,
                target_level=multi.level if multi is not None else None,
                episode_seconds=env.cfg.episode_length_s,success_tolerance=env._current_success_tolerance,
                next_difficulty=next_difficulty_dimension,last_difficulty=last_difficulty,native_tolerance_enabled=native_tolerance_enabled,
                pose_alpha_max=pose.alpha_max if pose is not None else None))
        if args.video_every and (epoch==start_epoch+1 or epoch%args.video_every==0 or epoch==args.critic_epochs or joint_epoch==1 or epoch==end_epoch):launch_video(epoch)
        del extras
    agent.save(str(args.output/('sapg_final' if args.joint_epochs else 'critic_calibrated')))
    if video is not None:
        while video.poll() is None:time.sleep(1)
        assert video.returncode==0,'Frozen actor video failed; see video_evaluator.log.'
    save('completed.json',dict(stage='joint_sapg' if args.joint_epochs else 'critic_only',epochs=end_epoch,planned_epochs=total,frames=agent.frame,
        actor_optimizer_steps=actor_steps,critic_optimizer_steps=critic_steps,joint_started=bool(args.joint_epochs)))
    agent.writer.close()
    if reset_bank is not None:reset_bank.restore()
    if args.native_object_reset:native_restore()
    restore();paired.restore();env.close()


if __name__=='__main__':main()
