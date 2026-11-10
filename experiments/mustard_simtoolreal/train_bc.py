"""Pure supervised native-action warm start; never constructs a SAPG learner."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import subprocess

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
VENDOR=PROJECT/'third_party/simtoolreal'
sys.path[:0]=[str(VENDOR),str(VENDOR/'rl_games'),str(ROOT)]


def save(path, value):
    path.write_text(json.dumps(value,indent=2,default=str)+'\n')


def plan_commands(previous, commands, lo, hi, arm_alpha, hand_alpha, speed_dt):
    """Retiming of command segments, with exact inverse of native EMA/clamps.

    Shrink targets slightly inside limits so reaching a hand limit does not
    require infinitely many EMA steps. Preserve every source control waypoint.
    """
    import numpy as np
    margin=(hi-lo)*.001
    targets=np.clip(commands,lo+margin,hi-margin)
    p=np.clip(previous,lo+margin,hi-margin).copy()
    initial=p.copy();actions=[];expected=[];indices=[]
    for index in range(commands.shape[1]):
        goal=targets[:,index]
        for _ in range(10000):
            delta=goal-p
            if np.abs(delta).max()<1e-7:break
            # Limit the *raw* arm target as well as the post-EMA increment.
            up=np.concatenate([arm_alpha*np.minimum(speed_dt,hi[:7]-p[:,:7]),hand_alpha*(hi[7:]-p[:,7:])],axis=1)
            down=np.concatenate([arm_alpha*np.minimum(speed_dt,p[:,:7]-lo[:7]),hand_alpha*(p[:,7:]-lo[7:])],axis=1)
            available=np.where(delta>=0,up,down)
            fraction=np.minimum(1.,.95*np.min(available/np.maximum(np.abs(delta),1e-12),axis=1))
            # Each trajectory advances at its own feasible rate; all trajectories
            # wait at this source waypoint before advancing to the next one.
            nxt=p+fraction[:,None]*delta
            a=np.empty_like(p)
            a[:,:7]=(nxt[:,:7]-p[:,:7])/(arm_alpha*speed_dt)
            raw=(nxt[:,7:]-(1-hand_alpha)*p[:,7:])/hand_alpha
            a[:,7:]=2*(raw-lo[7:])/(hi[7:]-lo[7:])-1
            assert np.abs(a).max()<=1.00001
            decoded=np.empty_like(p)
            decoded[:,:7]=(1-arm_alpha)*p[:,:7]+arm_alpha*np.clip(p[:,:7]+speed_dt*a[:,:7],lo[:7],hi[:7])
            decoded[:,7:]=(1-hand_alpha)*p[:,7:]+hand_alpha*(lo[7:]+.5*(a[:,7:]+1)*(hi[7:]-lo[7:]))
            assert np.max(np.abs(decoded-nxt))<1e-6
            actions.append(a.astype('float32'));expected.append(nxt.astype('float32'));indices.append(index)
            p=nxt
        else:raise RuntimeError('Native retiming failed to converge')
    return initial,np.stack(actions),np.stack(expected),np.array(indices),dict(
        source_steps=commands.shape[1],retimed_steps=len(actions),action_max_abs=float(np.abs(actions).max()),
        target_projection_max_rad=float(np.abs(commands-targets).max()),
        projected_target_fraction=float(np.mean(np.abs(commands-targets)>1e-7)),
        initial_projection_max_rad=float(np.abs(previous-initial).max()),
        inverse_verified=True,terminal_margin_fraction=.001,
        warning='Retiming preserves command waypoints, not source contact dynamics or task success.')


def collect(args):
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation
    from environment import create_env
    from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import compute_intermediate_values
    items=json.loads(args.source.read_text())
    assert len(items)==args.expected_trajectories
    pool=json.loads(args.waypoints.read_text())['trajectories']
    by_id={item['candidate_id']:item for item in pool}
    assert len({item['candidate_id'] for item in items})==len(items)
    for item in items:
        assert hashlib.sha256(Path(item['trajectory']).read_bytes()).hexdigest()==by_id[item['candidate_id']]['source_sha256'],'Trajectory changed since goal extraction'
    save(args.output/'source_manifest.json',items)
    data=[np.load(item['trajectory']) for item in items]
    assert all(d['ctrl'].shape[1]==29 for d in data)
    def configure(cfg):
        # Clean teacher collection only. Native PD, geometry, action pipeline,
        # physics rate and observation fields remain untouched.
        dr=cfg.domain_randomization
        dr.use_action_delay=False;dr.use_obs_delay=False;dr.use_object_state_delay_noise=False
        dr.joint_velocity_obs_noise_std=0.;dr.force_scale=0.;dr.torque_scale=0.
        cfg.reset.table_reset_z_range=0.
        if args.teacher_video:
            cfg.viewer.env_index=0;cfg.sim.render_interval=12
    env=create_env(args.num_envs,render_mode='rgb_array' if args.teacher_video else None,configure=configure)
    globals()['_COLLECTION_ENV']=env
    save(args.output/'collection_environment.json',env.cfg.to_dict())
    device=env.device;n=len(items)
    lo=env._joint_lower_canon.cpu().numpy();hi=env._joint_upper_canon.cpu().numpy()
    q0=np.stack([d['initial_state'][1:30] for d in data])
    commands=np.stack([d['ctrl'] for d in data]) if len({d['ctrl'].shape for d in data})==1 else None
    action_cfg=env.cfg.action
    if args.retiming=='legacy':
        initial,actions,targets,source_indices,audit=plan_commands(q0,commands,lo,hi,
            action_cfg.arm_moving_average,action_cfg.hand_moving_average,action_cfg.dof_speed_scale*env.step_dt)
    else:
        from retiming import continuous_plan
        # Retiming each source independently also supports unequal source lengths.
        plans=[continuous_plan(q0[i:i+1],d['ctrl'][None],lo,hi,
            action_cfg.arm_moving_average,action_cfg.hand_moving_average,action_cfg.dof_speed_scale*env.step_dt,
            realized_arm=d['qpos'][None,:,:7] if args.path_reference=='realized-arm' else None,
            source_dt=float(d['control_dt']),policy_dt=env.step_dt,smoothing_sigma=args.arm_smoothing,hold_steps=30)
            for i,d in enumerate(data)]
        initial=np.concatenate([p[0] for p in plans]);lengths=[len(p[1]) for p in plans];total=max(lengths)
        def padded(value,size):
            return np.concatenate([value,np.repeat(value[-1:],size-len(value),axis=0)],axis=0)
        targets=np.concatenate([padded(p[2],total) for p in plans],axis=1)
        # Padding holds the final target; a repeated final arm action would drift.
        actions=np.concatenate([padded(p[1],total) for p in plans],axis=1)
        for i,length in enumerate(lengths):actions[length:,i,:7]=0
        source_indices=np.stack([padded(np.asarray(p[3]).reshape(-1),total) for p in plans],axis=1)
        audit=dict(valid_lengths=lengths,retimed_steps=total,independent_trajectories=True,
            per_trajectory=[p[4] for p in plans])
    if args.teacher_control=='absolute-diagnostic':
        sample_t=np.arange(int(np.ceil(commands.shape[1]*float(data[0]['control_dt'])/env.step_dt)))*env.step_dt
        source_t=np.arange(commands.shape[1])*float(data[0]['control_dt'])
        targets=np.stack([np.stack([np.interp(sample_t,source_t,c[:,j]) for j in range(29)],-1) for c in commands],1).astype('float32')
        targets=np.clip(targets,lo,hi)
        actions=np.zeros_like(targets);source_indices=np.minimum((sample_t/float(data[0]['control_dt'])).astype(int),commands.shape[1]-1)
    if args.smoke_steps:
        actions=actions[:args.smoke_steps];targets=targets[:args.smoke_steps];source_indices=source_indices[:args.smoke_steps]
    audit.update(teacher_control=args.teacher_control,smoke_test=bool(args.smoke_steps),collected_steps=len(actions),candidate_ids=[i['candidate_id'] for i in items],blocks=6,block_size=args.block_size,
        collection_envs=args.num_envs,cached_envs=n,control_hz=1/env.step_dt,
        source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest())
    save(args.output/'retiming_audit.json',audit)
    print('BC_RETIMING',json.dumps(audit),flush=True)
    env.reset()
    ids=torch.arange(env.num_envs,device=device)%n
    q=torch.tensor(initial,device=device,dtype=torch.float32)[ids][:,env._perm_canon_to_lab]
    env.robot.write_joint_state_to_sim(q,torch.zeros_like(q))
    env.robot.set_joint_position_target(q)
    env._prev_targets[:]=q;env._cur_targets[:]=q
    calibration=json.loads((ROOT/'assets/object/calibration.json').read_text())
    center=np.array(calibration['source_to_new_root_center_in_source_m'])
    poses=np.stack([d['initial_state'][30:37] for d in data])
    poses[:,:3]+=Rotation.from_quat(poses[:,[4,5,6,3]]).apply(np.broadcast_to(center,(n,3)))
    # MuJoCo soft-contact rest poses sit slightly inside the table; PhysX depenetration can kick them.
    poses[:,2]+=.53+getattr(args,'object_z_offset',0.)
    pose=torch.tensor(poses,device=device,dtype=torch.float32)[ids]
    state=env.object.data.default_root_state.clone();state[:,:7]=pose
    state[:,:3]+=env.scene.env_origins;state[:,7:]=0
    env.object.write_root_state_to_sim(state)
    env._object_init_z[:]=pose[:,2]
    goals=torch.tensor([by_id[item['candidate_id']]['pos'][-1]+by_id[item['candidate_id']]['quat_wxyz'][-1]
        for item in items],device=device,dtype=torch.float32)[ids]
    goals[:,:3]+=env.scene.env_origins
    env.goal_viz.write_root_pose_to_sim(goals)
    env.scene.write_data_to_sim();env.sim.step(render=False);env.scene.update(env.physics_dt)
    # Collector owns episode boundaries. Avoid reset/goal changes halfway through
    # a time-indexed demonstration, while keeping native geometry observations.
    def no_dones():
        compute_intermediate_values(env)
        return torch.zeros(env.num_envs,device=device,dtype=torch.bool),torch.zeros(env.num_envs,device=device,dtype=torch.bool)
    env._get_dones=no_dones
    env._get_rewards=lambda:torch.zeros(env.num_envs,device=device)
    observations=[];max_error=0.;started=time.monotonic();horizon_summary=None
    lift_run=torch.zeros(n,device=device,dtype=torch.long);hold_run=lift_run.clone();max_lift_run=lift_run.clone();max_hold_run=lift_run.clone()
    max_lift=torch.full((n,),-100.,device=device);min_corner=torch.full((n,),100.,device=device)
    near_count=torch.zeros(n,device=device,dtype=torch.long);strict_count=near_count.clone();tracking_sum=torch.zeros((n,29),device=device)
    video=None
    if args.teacher_video:
        import imageio.v2 as imageio
        video=imageio.get_writer(str(args.output/'teacher.mp4'),fps=10,codec='libx264',macro_block_size=1)
        globals()['_COLLECTION_VIDEO']=video
        for _ in range(20):
            frame=env.render(recompute=True)
            if frame.max()>0:break
    with torch.no_grad():
        for t,action in enumerate(actions):
            compute_intermediate_values(env)
            obs=env._get_observations()['policy'][:n].clamp(-10,10).clone()
            observations.append(obs.cpu().numpy())
            # Re-invert from the actual float32 accumulator to prevent a
            # small rounding error accumulating over thousands of arm steps.
            previous=env._prev_targets[:n,env._perm_lab_to_canon]
            desired=torch.tensor(targets[t],device=device)
            command=torch.empty_like(previous)
            command[:,:7]=(desired[:,:7]-previous[:,:7])/(action_cfg.arm_moving_average*action_cfg.dof_speed_scale*env.step_dt)
            raw=(desired[:,7:]-(1-action_cfg.hand_moving_average)*previous[:,7:])/action_cfg.hand_moving_average
            lower=env._joint_lower_canon[7:];upper=env._joint_upper_canon[7:]
            command[:,7:]=2*(raw-lower)/(upper-lower)-1
            if args.teacher_control=='absolute-diagnostic':
                env._replay_target_lab_order=desired[ids][:,env._perm_canon_to_lab]
                command.zero_()
            else:assert float(command.abs().max())<=1.0001, f'Native inverse out of bounds at step {t}: {float(command.abs().max())}'
            command=command.clamp(-1,1) # roundoff only, bounded above
            actions[t]=command.cpu().numpy()
            env.step(command[ids])
            actual=env._prev_targets[:n,env._perm_lab_to_canon]
            max_error=max(max_error,float((actual-torch.tensor(targets[t],device=device)).abs().max()))
            position=env.object.data.root_pos_w[:n]-env.scene.env_origins[:n]
            height=position[:,2]-pose[:n,2]
            active=torch.tensor(t<np.asarray(audit.get('valid_lengths',[len(actions)]*n)),device=device)
            lift=(height>=.05)&active
            palm=env._get_observations()['policy'][:n,87:90]
            close=torch.linalg.vector_norm(position-palm,dim=-1)<.18
            fingers=env._curr_fingertip_distances[:n].sort(-1).values[:,:3].mean(-1)<.10
            calm=torch.linalg.vector_norm(env.object.data.root_lin_vel_w[:n],dim=-1)<2.
            hold=lift&close&fingers&calm
            lift_run=torch.where(lift,lift_run+1,0);hold_run=torch.where(hold,hold_run+1,0)
            max_lift_run=torch.maximum(max_lift_run,lift_run);max_hold_run=torch.maximum(max_hold_run,hold_run)
            max_lift=torch.maximum(max_lift,height);min_corner=torch.minimum(min_corner,env._keypoints_max_dist[:n])
            near_count+=((env._keypoints_max_dist[:n]<=.075*env.cfg.reward.keypoint_scale)&active).long()
            strict_count+=((env._keypoints_max_dist[:n]<=.01*env.cfg.reward.keypoint_scale)&active).long()
            tracking_sum+=(env.robot.data.joint_pos[:n,env._perm_lab_to_canon]-desired).square()
            if t+1==env.cfg.termination.episode_length:
                horizon_summary=dict(policy_steps=t+1,lifted_5cm_for_30_steps=int((max_lift_run>=30).sum()),
                    held_lift_proxy_30_steps=int((max_hold_run>=30).sum()),
                    reached_initial_tolerance_10_steps=int((near_count>=10).sum()),
                    reached_final_tolerance_10_steps=int((strict_count>=10).sum()))
            if video is not None and t%6==0:video.append_data(env.render())
            if t%100==0:print('BC_COLLECT',t,len(actions),round(time.monotonic()-started,1),flush=True)
    if video is not None:video.close()
    assert max_error<2e-5,f'Native target decoder mismatch: {max_error}'
    compute_intermediate_values(env)
    final=env.object.data.root_pos_w[:n]-env.scene.env_origins[:n]
    goal=goals[:n,:3]-env.scene.env_origins[:n]
    summary=dict(original_horizon_result=horizon_summary,trajectories=n,teacher_control=args.teacher_control,policy_compatible=args.teacher_control=='native',
        lifted_5cm_for_30_steps=int((max_lift_run>=30).sum()),held_lift_proxy_30_steps=int((max_hold_run>=30).sum()),
        reached_initial_tolerance_10_cumulative_steps=int((near_count>=10).sum()),
        reached_final_tolerance_10_cumulative_steps=int((strict_count>=10).sum()),
        median_max_lift_m=float(max_lift.median()),max_lift_m=max_lift.cpu().tolist(),
        longest_lift_steps=max_lift_run.cpu().tolist(),longest_hold_proxy_steps=max_hold_run.cpu().tolist(),
        min_keypoint_error_m=min_corner.cpu().tolist(),
        arm_tracking_rmse_rad=torch.sqrt(tracking_sum[:,:7].mean(-1)/len(actions)).cpu().tolist(),
        hand_tracking_rmse_rad=torch.sqrt(tracking_sum[:,7:].mean(-1)/len(actions)).cpu().tolist(),
        candidate_ids=audit['candidate_ids'],smoke_test=bool(args.smoke_steps),
        holding_criterion='height rise >=5cm, palm within18cm, mean nearest3 fingertips within10cm, speed<2m/s, consecutive30 policy steps; geometric proxy, not contact force certificate')
    save(args.output/'grasp_validation.json',summary)
    print('GRASP_VALIDATION',json.dumps({k:v for k,v in summary.items() if not isinstance(v,list)}),flush=True)
    save(args.output/'teacher_rollout.json',dict(native_decoder_max_error_rad=max_error,
        final_keypoint_max_error_m=env._keypoints_max_dist[:n].cpu().tolist(),
        final_within_initial_tolerance=int(env._near_goal[:n].sum()),
        final_object_position_error_m=torch.linalg.vector_norm(final-goal,dim=-1).cpu().tolist(),
        note='Open-loop teacher execution diagnostic, not a reaching success certificate. Rewards disabled in collection.'))
    valid_lengths=np.minimum(audit.get('valid_lengths',[len(actions)]*n),len(actions))
    np.savez_compressed(args.output/'native_bc_dataset.npz',obs=np.stack(observations,axis=1),
        actions=actions.transpose(1,0,2),source_indices=source_indices,candidate_ids=audit['candidate_ids'],
        valid_lengths=valid_lengths,action_contract=np.array('simtoolreal_native_v1'),smoke_test=np.array(bool(args.smoke_steps)))
    env.close()


def learn(args):
    import numpy as np
    import torch
    import yaml
    from rl_games.algos_torch import model_builder
    torch.manual_seed(args.seed)
    device=torch.device(args.learner_device)
    d=np.load(args.output/'native_bc_dataset.npz')
    assert not bool(d['smoke_test']) or args.smoke_steps,'Do not use a truncated smoke cache for formal training'
    if (args.output/'selection.json').exists():
        selection=json.loads((args.output/'selection.json').read_text())
        assert hashlib.sha256((args.output/'native_bc_dataset.npz').read_bytes()).hexdigest()==selection['cache_sha256']
    assert str(d['action_contract'])=='simtoolreal_native_v1','Wrong BC action contract'
    x=torch.tensor(d['obs'],device=device);y=torch.tensor(d['actions'],device=device)
    valid_lengths=d['valid_lengths'] if 'valid_lengths' in d else np.full(len(x),x.shape[1])
    assert len(valid_lengths)==len(x) and np.all((valid_lengths>0)&(valid_lengths<=x.shape[1]))
    valid_lengths_t=torch.tensor(valid_lengths,device=device)
    frame_weights=torch.ones(len(x),device=device)
    if args.sample_weights:
        from bc_balance import load_weights
        weights=load_weights(args.sample_weights,d['candidate_ids'],valid_lengths,args.output/'native_bc_dataset.npz')
        frame_weights=torch.tensor(weights,device=device,dtype=torch.float32)
        import shutil
        shutil.copy2(args.sample_weights,args.output/'sample_weights.npz')
    assert len(x)==args.expected_trajectories
    if not args.smoke_steps:
        quality=json.loads((args.output/'grasp_validation.json').read_text())
        if args.qualification=='full-path-grasp':
            valid=(quality['teacher_control']=='native' and quality['held_lift_proxy_30_steps']==len(x)
                and int(valid_lengths.max())<=round(args.demo_max_seconds*60))
        else:
            within=quality.get('original_horizon_result')
            valid=(quality['teacher_control']=='native' and within is not None
                and within['held_lift_proxy_30_steps']==len(x)
                and within.get('reached_initial_tolerance_10_steps')==len(x))
        if not valid:raise RuntimeError('Demonstrations failed the requested qualification; inspect grasp_validation.json.')
    assert x.shape[:2]==y.shape[:2] and x.shape[-1]==140 and y.shape[-1]==29
    if args.obs=='adept':
        # Exact offline conversion of the clean cached observations (check_native_to_adept.py).
        assert not args.val_fraction and not args.goal_relabel_prob and not args.video_every,'native-layout tools only'
        from adept_obs import native_to_adept,mustard_surface_points
        x=native_to_adept(x,mustard_surface_points(64))
    D=x.shape[-1]
    assert torch.isfinite(x).all() and torch.isfinite(y).all() and y.abs().max()<=1.00001
    from goal_relabel import keypoint_offsets,object_keypoints,pregrasp_boundaries,quat_apply,relabel,sample_goal_keypoints,stratified_split
    candidate_ids=d['candidate_ids'];n_all=len(x)
    manifest_path=args.output/'source_manifest.json'
    sources={m['candidate_id']:m.get('source_index',0) for m in json.loads(manifest_path.read_text())} if manifest_path.exists() else {}
    groups=[sources.get(int(c),0) for c in candidate_ids]
    train_idx,val_idx=stratified_split(candidate_ids,groups,args.val_fraction,args.split_seed) if args.val_fraction else (np.arange(n_all),np.array([],dtype=int))
    relabeling=args.goal_relabel_prob>0 or len(val_idx)>0
    if relabeling:
        task=yaml.safe_load((VENDOR/'isaacsimenvs/cfg/task/SimToolReal.yaml').read_text())
        volume=(task['reset']['target_volume_mins'],task['reset']['target_volume_maxs'])
        offsets=keypoint_offsets(task['reward']['keypoint_scale']).to(device)
        boundary=torch.tensor(pregrasp_boundaries(d['obs'],valid_lengths),device=device)
        # Offsets must reproduce the demonstrated goal keypoints exactly.
        pool={t['candidate_id']:t for t in json.loads(args.waypoints.read_text())['trajectories']}
        check=0.
        for i in range(min(20,n_all)):
            g=pool[int(candidate_ids[i])];goal=quat_apply(torch.tensor(g['quat_wxyz'][-1],device=device)[None],offsets)+torch.tensor(g['pos'][-1],device=device)
            o=x[i,:int(valid_lengths[i])];check=max(check,float((object_keypoints(o)-o[:,125:137].reshape(-1,4,3)-goal).abs().max()))
        assert check<1e-4,f'Goal keypoint offsets disagree with cached observations: {check}'
        val_generator=torch.Generator(device=device).manual_seed(args.split_seed+12345)
        val_goals=sample_goal_keypoints(n_all,offsets,*volume,generator=val_generator,device=device)
    train_t=torch.tensor(train_idx,device=device)
    cfg=yaml.safe_load((VENDOR/'isaacsimenvs/cfg/train/SimToolRealSAPG.yaml').read_text())['params']
    model=model_builder.ModelBuilder().load(cfg).build(dict(actions_num=29,input_shape=(D+1,),
        num_seqs=args.batch_sequences,value_size=1,normalize_input=True,normalize_value=True,
        type='extra_param',coef_ids=torch.linspace(50,0,6,device=device),coef_id_idx=D)).to(device)
    model.train() # cuDNN LSTM backward requires training mode
    model.running_mean_std.eval();model.value_mean_std.eval() # freeze statistics only
    model.a2c_network.sigma.requires_grad_(False);model.a2c_network.sigma.data.fill_(-2.)
    for p in model.a2c_network.value.parameters():p.requires_grad_(False)
    valid_mask=torch.arange(x.shape[1],device=device)[None,:]<valid_lengths_t[:,None]
    flat=x[train_t][valid_mask[train_t]]
    if args.goal_relabel_prob>0:
        # Normalize goal features over both demonstrated and relabeled pre-grasp goals.
        goals=sample_goal_keypoints(len(train_idx),offsets,*volume,generator=torch.Generator(device=device).manual_seed(args.seed),device=device)
        pre=relabel(x[train_t],goals,0,boundary[train_t],torch.ones(len(train_idx),dtype=torch.bool,device=device))
        frames=torch.arange(x.shape[1],device=device)[None,:]
        flat=torch.cat([flat,pre[(frames<boundary[train_t,None])&valid_mask[train_t]]]);del pre
    model.running_mean_std.running_mean.copy_(flat.mean(0).double())
    model.running_mean_std.running_var.copy_(flat.var(0,unbiased=False).clamp_min(1e-4).double())
    model.running_mean_std.count.fill_(len(flat))
    init_epoch=0
    if args.resume_bc:
        assert not args.init_checkpoint
        initial=torch.load(args.resume_bc/'actor_init.pt',map_location=device,weights_only=False)
        assert initial['action_contract']=='simtoolreal_native_v1' and initial.get('rnn_context','window')==args.rnn_context
        model.load_state_dict(initial['model'])
    if args.init_checkpoint:
        initial=torch.load(args.init_checkpoint,map_location=device,weights_only=False)
        assert initial['action_contract']=='simtoolreal_native_v1'
        model.load_state_dict(initial['model']);init_epoch=initial.get('bc_epochs',0)
    frozen={k:v.detach().clone() for k,v in model.state_dict().items() if 'value' in k or 'running_mean_std' in k or k.endswith('.sigma')}
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=args.learning_rate)
    n,t,_=x.shape;length=args.sequence_length
    # Each epoch visits every frame once. Hidden state is rebuilt with a prefix
    # burn-in, so shuffled chunks do not pretend to start new episodes.
    starts=list(range(0,t,length))
    writer=None
    if args.wandb_project:
        import wandb
        writer=wandb.init(project=args.wandb_project,name=args.output.name,dir=str(args.output),
            save_code=False,settings=wandb.Settings(disable_git=True,disable_code=True,x_disable_meta=True,x_disable_stats=True,console='off'))
        save(args.output/'wandb_run.json',dict(url=writer.url,id=writer.id))
    audit=json.loads((args.output/'retiming_audit.json').read_text()) if (args.output/'retiming_audit.json').exists() else {}
    save(args.output/'stage_contract.json',dict(stage='pure_behavior_cloning',epochs=args.epochs,smoke_test=bool(args.smoke_steps),
        qualification=args.qualification,demo_max_seconds=args.demo_max_seconds,evaluation_reset_modes=args.eval_reset_modes,formal_sapg_horizon_unchanged=True,
        epoch_definition='One complete pass over all cached trajectory frames; each frame contributes once.',
        trajectories=n,frames_per_trajectory=valid_lengths.tolist(),actor_obs_dim=D,obs_layout=args.obs,num_actions=29,
        architecture='unmodified native SAPG LSTM + MLP + six coefficient embeddings',
        six_condition_ids=[50,40,30,20,10,0],condition_assignment='Rotate each trajectory through all IDs every six epochs',
        collection_envs=audit.get('collection_envs'),requested_sapg_num_envs=args.num_envs,blocks=6,block_size=args.block_size,
        critic_updates=0,policy_gradient_updates=0,reward_optimization=False,
        action_contract='simtoolreal_native_v1',sigma=-2.,learning_rate=args.learning_rate,
        sequence_length=length,burn_in=args.burn_in,rnn_context=args.rnn_context,
        sample_weights=str(args.sample_weights) if args.sample_weights else None,
        weighting='Fixed trajectory frame weights; global mean=1; original MSE logged separately',
        initialized_from=str(args.init_checkpoint) if args.init_checkpoint else None,initial_checkpoint_epoch=init_epoch,
        validation=dict(fraction=args.val_fraction,split='stratified by source_index',seed=args.split_seed,trajectories=len(val_idx),
            every_epochs=args.val_every,early_stop_evals=args.early_stop_evals,
            objective='mean(val MSE, val MSE with pre-grasp goals relabeled)' if args.goal_relabel_prob>0 else 'val MSE',
            note='full-sequence zero-state rollout on cached held-out trajectories; same-source augmentations are near duplicates'),
        goal_relabel=dict(probability=args.goal_relabel_prob,frames='before the first 30-step hold (truncated_selection proxy)',
            sampler='native absolute goal: uniform target volume position, uniform SO(3)',normalizer='train frames + one relabeled copy of pre-grasp frames') if args.goal_relabel_prob>0 else None,
        note='BC uses cached native Isaac observations; requested SAPG layout does not allocate simulation environments in cached BC mode.'))
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda',init_scale=256.)
    start=time.monotonic();video_process=None;updates=0;skipped=0;logged_evaluations=set()
    def evaluation_metrics():
        metrics={}
        for path in sorted((args.output/'videos').rglob('*.metrics.json')):
            if path not in logged_evaluations:
                mode=path.parent.name if path.parent.name in ('demo','native') else 'native'
                metrics.update({'eval/'+mode+'/'+k:v for k,v in json.loads(path.read_text()).items()})
                logged_evaluations.add(path)
        return metrics
    start_epoch=0
    if args.resume_bc:
        state=torch.load(args.resume_bc/'bc_optimizer.pt',map_location=device,weights_only=False)
        start_epoch=int(initial['bc_epochs'])
        assert state['epoch']==start_epoch and start_epoch<args.epochs
        optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
        updates=max(int(v['step']) for v in optimizer.state.values())
        save(args.output/'resume_check.json',dict(source=str(args.resume_bc),epoch=start_epoch,
            optimizer_steps=updates,optimizer_entries=len(optimizer.state),scaler_restored=True,
            model_exact=all(torch.equal(model.state_dict()[k],v) for k,v in initial['model'].items()),
            random_state='old checkpoint did not save RNG; shuffle restarts from seed'))
        print('BC_RESUMED',start_epoch,updates,flush=True)
    def balanced_loss(prediction,target,mask):
        error=(prediction.float()-target).square()
        return .5*(error[:,:,:7].mean(-1)+error[:,:,7:].mean(-1)),mask
    def validate():
        """Full-sequence rollout of cached held-out trajectories from zero LSTM state."""
        sums={k:0. for k in ('mse','relabel_mse','pre_mse','pre_relabel_mse')};counts={'all':0,'pre':0}
        for offset in range(0,len(val_idx),64):
            indices=val_idx[offset:offset+64].tolist();b=len(indices);steps=int(valid_lengths[indices].max())
            ids=torch.zeros(b,device=device)
            h=tuple(state[:,:1].to(device).expand(-1,b,-1).contiguous().zero_() for state in model.get_default_rnn_state())
            frames=torch.arange(steps,device=device)[None]
            mask=frames<valid_lengths_t[indices,None];pre=mask&(frames<boundary[indices,None])
            for name,xx in (('',x[indices,:steps]),('relabel_',relabel(x[indices,:steps],val_goals[indices],0,boundary[indices],torch.ones(b,dtype=torch.bool,device=device)))):
                inp=torch.cat([xx,ids[:,None,None].expand(b,steps,1)],-1).reshape(-1,D+1)
                with torch.no_grad(),torch.autocast(device_type=device.type,enabled=device.type=='cuda',dtype=torch.float16):
                    mu,_,_,_=model.a2c_network(dict(obs=model.norm_obs(inp),rnn_states=h,seq_length=steps))
                frame_loss,_=balanced_loss(mu.reshape(b,steps,29),y[indices,:steps],mask)
                sums[name+'mse']+=float((frame_loss*mask).sum());sums['pre_'+name+'mse']+=float((frame_loss*pre).sum())
            counts['all']+=int(mask.sum());counts['pre']+=int(pre.sum())
        out={k:v/max(counts['pre' if k.startswith('pre') else 'all'],1) for k,v in sums.items()}
        out['objective']=.5*(out['mse']+out['relabel_mse']) if args.goal_relabel_prob>0 else out['mse']
        return out
    best=dict(objective=float('inf'),epoch=None);stale=0;stop_reason='max_epochs'
    train_set=train_idx.tolist()
    for epoch in range(start_epoch+1,args.epochs+1):
        order=[train_set[i] for i in torch.randperm(len(train_set)).tolist()];loss_sum=0.;weighted_sum=0.;count=0
        if args.goal_relabel_prob>0:
            epoch_goals=sample_goal_keypoints(n_all,offsets,*volume,device=device)
            epoch_active=torch.rand(n_all,device=device)<args.goal_relabel_prob
        def augment(xx,indices,start):
            if args.goal_relabel_prob<=0:return xx
            ix=torch.as_tensor(indices,device=device)
            return relabel(xx,epoch_goals[ix],start,boundary[ix],epoch_active[ix])
        batches=[[(i,s) for i in order[offset:offset+args.batch_sequences]]
                 for s in starts for offset in range(0,len(order),args.batch_sequences)]
        state_bank=tuple(state[:,:1].to(device).expand(-1,n,-1).contiguous().zero_() for state in model.get_default_rnn_state()) if args.rnn_context=='carry' else None
        batch_order=range(len(batches)) if args.rnn_context=='carry' else torch.randperm(len(batches)).tolist()
        for iteration,batch_index in enumerate(batch_order):
            batch=[(i,s) for i,s in batches[batch_index] if s<valid_lengths[i]]
            if not batch:continue
            # Group equal chunk lengths and start times; a shuffled batch can
            # contain both a short last chunk and full-length chunks.
            groups={}
            for i,s in batch:groups.setdefault(s,[]).append(i)
            optimizer.zero_grad(set_to_none=True)
            batch_frames=sum(min(length,int(valid_lengths[i])-s) for i,s in batch)
            for s,indices in groups.items():
                b=len(indices);end=min(s+length,t);ids=50-10*((torch.tensor(indices,device=device)+epoch-1)%6)
                h=tuple(state[:,:1].to(device).expand(-1,b,-1).contiguous().zero_() for state in model.get_default_rnn_state())
                if state_bank is not None:h=tuple(state[:,indices].contiguous() for state in state_bank)
                def forward(xx,hh):
                    steps=xx.shape[1]
                    inp=torch.cat([xx,ids[:,None,None].expand(b,steps,1)],-1).reshape(-1,D+1)
                    mu,_,_,hh=model.a2c_network(dict(obs=model.norm_obs(inp),rnn_states=hh,seq_length=steps))
                    return mu.reshape(b,steps,29),hh
                with torch.autocast(device_type=device.type,enabled=device.type=='cuda',dtype=torch.float16):
                    with torch.no_grad():
                        if state_bank is None and s and args.burn_in:_,h=forward(augment(x[indices,max(0,s-args.burn_in):s],indices,max(0,s-args.burn_in)),h)
                    prediction,next_h=forward(augment(x[indices,s:end],indices,s),h)
                    error=(prediction.float()-y[indices,s:end]).square()
                    mask=torch.arange(s,end,device=device)[None,:]<valid_lengths_t[indices,None]
                    group_frames=int(mask.sum())
                    frame_loss=.5*(error[:,:,:7].mean(-1)+error[:,:,7:].mean(-1))
                    loss=(frame_loss*mask).sum()/group_frames
                    weighted_loss=(frame_loss*mask*frame_weights[indices,None]).sum()/group_frames
                    weighted=weighted_loss*(group_frames/batch_frames)
                scaler.scale(weighted).backward()
                if state_bank is not None:
                    for bank,state in zip(state_bank,next_h):bank[:,indices]=state.detach().to(bank.dtype)
                loss_sum+=float(loss.detach())*group_frames;count+=group_frames
                weighted_sum+=float(weighted_loss.detach())*group_frames
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=device.type!='cuda')
            scale_before=scaler.get_scale()
            scaler.step(optimizer);scaler.update()
            if scaler.get_scale()>=scale_before:updates+=1
            else:skipped+=1
            if iteration%100==0:print('BC_BATCH',epoch,iteration,len(batches),loss_sum/max(count,1),flush=True)
        assert count==int(valid_lengths[train_idx].sum())
        row=dict(epoch=epoch,action_balanced_mse=loss_sum/count,optimizer_steps=updates,
            weighted_action_mse=weighted_sum/count,
            supervised_frames=count,amp_skipped_updates=skipped,elapsed_s=time.monotonic()-start,critic_updates=0)
        improved=False
        if len(val_idx) and (epoch%args.val_every==0 or epoch==args.epochs or epoch==start_epoch+1):
            metrics=validate();row.update({'val_'+k:v for k,v in metrics.items()})
            if metrics['objective']<best['objective']:best=dict(metrics,epoch=epoch);stale=0;improved=True
            else:stale+=1
        with (args.output/'bc_progress.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print('BC',json.dumps(row),flush=True)
        if writer:writer.log(dict({'bc/'+k:v for k,v in row.items()},**evaluation_metrics()),step=epoch)
        def make_checkpoint():
            return dict(model=model.state_dict(),network=cfg['network'],action_contract='simtoolreal_native_v1',
                actor_obs_dim=D,obs_layout=args.obs,coef_ids=[50,40,30,20,10,0],bc_epochs=epoch,critic_trained=False,
                rnn_context=args.rnn_context,initial_checkpoint_epoch=init_epoch,initialized_from=str(args.init_checkpoint) if args.init_checkpoint else None,
                val_candidate_ids=candidate_ids[val_idx].tolist(),goal_relabel_prob=args.goal_relabel_prob)
        if improved:
            temporary=args.output/'actor_best.tmp';torch.save(dict(make_checkpoint(),validation=best),temporary);temporary.replace(args.output/'actor_best.pt')
        early=bool(args.early_stop_evals and stale>=args.early_stop_evals)
        if early:stop_reason='early_stop'
        if epoch%args.checkpoint_every==0 or epoch==args.epochs or early or (args.video_every and (epoch==1 or epoch%args.video_every==0)):
            assert all(torch.equal(model.state_dict()[k],v) for k,v in frozen.items()),'Frozen critic/sigma changed'
            assert all(torch.isfinite(v).all() for v in model.state_dict().values()),'Nonfinite actor checkpoint'
            checkpoint=make_checkpoint()
            path=args.output/'actor_init.pt';temporary=args.output/'actor_init.tmp';torch.save(checkpoint,temporary);temporary.replace(path)
            if args.keep_checkpoints:torch.save(checkpoint,args.output/f'actor_e{epoch:06d}.pt')
            torch.save(dict(optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),epoch=epoch),args.output/'bc_optimizer.pt')
            if args.video_every and (epoch==1 or epoch%args.video_every==0 or epoch==args.epochs):
                if epoch==args.epochs and video_process is not None and video_process.poll() is None:
                    video_process.wait()
                if video_process is None or video_process.poll() is not None:
                    snapshot=args.output/f'video_actor_{epoch:06d}.pt';torch.save(checkpoint,snapshot)
                    with (args.output/'video_evaluator.log').open('a') as log:
                        command=[sys.executable,str(ROOT/'evaluate_bc_modes.py'),
                            '--checkpoint',str(snapshot),'--dataset',str(args.output/'native_bc_dataset.npz'),
                            '--output-dir',str(args.output/'videos'),'--tag',f'epoch_{epoch:06d}',
                            '--waypoints',str(args.waypoints),'--frames',str(args.video_length),
                            '--reset-modes',args.eval_reset_modes]
                        if args.qualification=='full-path-grasp':command+=['--episode-seconds',str(args.demo_max_seconds)]
                        video_process=subprocess.Popen(command,stdout=log,stderr=log,start_new_session=True)
        if early:break
    if video_process is not None:
        status=video_process.wait()
        save(args.output/'last_video_status.json',dict(returncode=status))
        if writer:
            metrics=evaluation_metrics()
            if metrics:writer.log(metrics,step=args.epochs+1)
    save(args.output/'completed.json',dict(stage='pure_behavior_cloning',epochs=args.epochs,last_epoch=epoch,stop_reason=stop_reason,critic_updates=0,
        actor_checkpoint='actor_init.pt',best_checkpoint='actor_best.pt' if best['epoch'] else None,best_validation=best if best['epoch'] else None,
        later_stages_run=False))
    if writer:writer.finish()


def main():
    from isaaclab.app import AppLauncher
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path,default=PROJECT/'outputs/mustard_spatial_valid100/accepted.json')
    p.add_argument('--waypoints',type=Path,default=ROOT/'waypoints100.json')
    p.add_argument('--qualification',choices=['horizon-reaching','full-path-grasp'],default='horizon-reaching')
    p.add_argument('--demo-max-seconds',type=float,default=16.)
    p.add_argument('--expected-trajectories',type=int,default=100)
    p.add_argument('--num-envs',type=int,default=8184);p.add_argument('--block-size',type=int,default=1364)
    p.add_argument('--epochs',type=int,default=4000)
    p.add_argument('--learning-rate',type=float,default=1e-4)
    p.add_argument('--rnn-context',choices=['window','carry'],default='window',help='Carry recurrent state chronologically; detach gradients at chunk boundaries.')
    p.add_argument('--resume-bc',type=Path,help='Restore actor, Adam and AMP from a previous BC run into a fresh output.')
    p.add_argument('--init-checkpoint',type=Path,help='Actor warm-start for controlled fine-tuning; creates a fresh optimizer.')
    p.add_argument('--sequence-length',type=int,default=16)
    p.add_argument('--burn-in',type=int,default=16)
    p.add_argument('--batch-sequences',type=int,default=128)
    p.add_argument('--learner-device',default='cuda:0')
    p.add_argument('--checkpoint-every',type=int,default=100)
    p.add_argument('--eval-reset-modes',default='native',help='Comma-separated demo,native; recorded and logged separately.')
    p.add_argument('--video-every',type=int,default=100);p.add_argument('--video-length',type=int,default=300)
    p.add_argument('--wandb-project',default=None)
    p.add_argument('--dataset',type=Path,help='Reuse an audited native BC cache, bypass collection.')
    p.add_argument('--sample-weights',type=Path,help='Validated candidate-aligned fixed trajectory frame weights.')
    p.add_argument('--collect-only',action='store_true')
    p.add_argument('--retiming',choices=['continuous','legacy'],default='continuous')
    p.add_argument('--path-reference',choices=['realized-arm','control'],default='realized-arm')
    p.add_argument('--arm-smoothing',type=float,default=2.)
    p.add_argument('--teacher-control',choices=['native','absolute-diagnostic'],default='native')
    p.add_argument('--teacher-video',action='store_true')
    p.add_argument('--smoke-steps',type=int,default=0,help='Validation only: truncate collection; never set for formal training.')
    p.add_argument('--val-fraction',type=float,default=0.,help='Held-out trajectory fraction per source (stratified); 0 disables validation.')
    p.add_argument('--val-every',type=int,default=50)
    p.add_argument('--early-stop-evals',type=int,default=0,help='Stop after this many validations without improvement (0 disables).')
    p.add_argument('--goal-relabel-prob',type=float,default=0.,help='Per-trajectory probability of replacing the pre-grasp goal with a native random goal each epoch.')
    p.add_argument('--split-seed',type=int,default=0)
    p.add_argument('--object-z-offset',type=float,default=0.,help='Teacher collection: raise the cached MuJoCo object start pose (m) to avoid PhysX depenetration kicks.')
    p.add_argument('--keep-checkpoints',action='store_true',help='Also keep actor_eXXXXXX.pt at every checkpoint for closed-loop selection.')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--obs',choices=('native','adept'),default='native',help='adept: train on the exact ADEPT-layout conversion (adept_obs.py) of the cached native observations')
    AppLauncher.add_app_launcher_args(p)
    args=p.parse_args();args.output=args.output.resolve()
    modes=args.eval_reset_modes.split(',')
    assert len(set(modes))==len(modes) and set(modes)<=set(('demo','native'))
    assert args.num_envs==6*args.block_size and args.num_envs>=args.expected_trajectories
    assert min(args.epochs,args.sequence_length,args.batch_sequences,args.checkpoint_every)>0
    assert 0<=args.val_fraction<.5 and 0<=args.goal_relabel_prob<=1 and args.val_every>0 and args.early_stop_evals>=0
    assert args.demo_max_seconds>0 and args.arm_smoothing>=0 and args.smoke_steps>=0 and args.burn_in>=0 and args.video_every>=0 and args.learning_rate>0
    assert not args.output.exists(),'Use a fresh output directory.'
    args.output.mkdir(parents=True);save(args.output/'run_arguments.json',vars(args))
    assert args.teacher_control=='native' or args.collect_only,'Absolute diagnostic cannot train a native actor'
    if args.teacher_video:args.enable_cameras=True
    # Cached BC is a pure Torch learner; only collection/evaluation needs Kit.
    app=None if args.dataset else AppLauncher(args).app
    try:
        if args.dataset:
            import shutil
            shutil.copy2(args.dataset,args.output/'native_bc_dataset.npz')
            for name in ('grasp_validation.json','retiming_audit.json','source_manifest.json','selection.json','collection_environment.json'):
                path=args.dataset.parent/name
                if path.exists():shutil.copy2(path,args.output/name)
        else:collect(args)
        if args.collect_only:
            save(args.output/'completed.json',dict(stage='teacher_validation_only',actor_trained=False))
        else:learn(args)
    except BaseException as error:
        save(args.output/'failed.json',dict(error=type(error).__name__,message=str(error)))
        raise
    finally:
        video=globals().pop('_COLLECTION_VIDEO',None)
        if video is not None:video.close()
        env=globals().pop('_COLLECTION_ENV',None)
        if env is not None and not getattr(env,'_is_closed',False):env.close()
        if app is not None:app.close()


if __name__=='__main__':main()
