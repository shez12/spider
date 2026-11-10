"""Actor trajectory imitation using actual SimToolReal Isaac Lab/PhysX rollouts.

Uses the vendored SAPG actor backbone, supervised reference-target labels,
and teacher/student roll-ins. Critic and policy-gradient training are excluded.
"""
from pathlib import Path
import argparse
import json
import sys
import time


def main():
    # AppLauncher must precede all simulator-backed imports.
    from isaaclab.app import AppLauncher
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--assets',type=Path,default=Path('outputs/mustard_isaac_assets_v4'))
    parser.add_argument('--frozen',type=Path,default=Path('outputs/mustard_sapg_bridge20_v1'))
    parser.add_argument('--initial-run',type=Path,default=Path('outputs/mustard_actor_bc20_residual_v1'))
    parser.add_argument('--simtoolreal',type=Path,default=Path('/home/szc/work/simtoolreal_origin/simtoolreal'))
    parser.add_argument('--rounds',type=int,default=10)
    parser.add_argument('--epochs-per-round',type=int,default=25)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--physics-hz',type=int,default=120)
    parser.add_argument('--control-hz',type=int,default=60)
    parser.add_argument('--learner-device',default='cpu',help='Actor optimization device; PhysX stays on CUDA.')
    parser.add_argument('--checkpoint',type=Path,default=None)
    parser.add_argument('--learning-rate',type=float,default=1e-5)
    parser.add_argument('--contact-offset',type=float,default=.001)
    parser.add_argument('--decoder',choices=['previous-residual','absolute'],default='previous-residual')
    parser.add_argument('--ridge-head',action='store_true',help='Fit an absolute-target head to actual PhysX teacher features before fine-tuning.')
    parser.add_argument('--autoregressive',action='store_true',help='Backpropagate through predicted previous targets during full-sequence imitation.')
    parser.add_argument('--batch-episodes',type=int,default=16)
    parser.add_argument('--evaluate-only',action='store_true')
    parser.add_argument('--evaluation-repeats',type=int,default=3)
    parser.add_argument('--phase-input',action='store_true',help='Append the environment-native progress field for time-aligned trajectory imitation.')
    parser.add_argument('--train-candidate',type=int,default=None,help='Optional single-demo overfit diagnostic; must belong to the training split.')
    parser.add_argument('--initial-loss-weight',type=float,default=1.)
    parser.add_argument('--sapg',action='store_true',help='Run full vendored SAPG after constructing the aligned Isaac environment.')
    parser.add_argument('--num-envs',type=int,default=120)
    parser.add_argument('--sapg-epochs',type=int,default=1000)
    parser.add_argument('--critic-warmup-epochs',type=int,default=50)
    parser.add_argument('--sapg-device',default='cuda:0')
    parser.add_argument('--resume-sapg',type=Path,default=None)
    parser.add_argument('--capture-video',action='store_true')
    parser.add_argument('--video-interval',type=int,default=6000,help='Policy steps between recordings.')
    parser.add_argument('--video-length',type=int,default=300)
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args()
    if args.capture_video:args.enable_cameras=True
    args.output=args.output.resolve();args.assets=args.assets.resolve();args.frozen=args.frozen.resolve();args.initial_run=args.initial_run.resolve()
    assert not (args.output/'completed.json').exists(),'Run already completed; choose a new output.'
    args.output.mkdir(parents=True,exist_ok=True)
    sys.path[:0]=[str(args.simtoolreal),str(args.simtoolreal/'rl_games')]
    app=AppLauncher(args).app
    try:
        run(args,app)
    finally:
        app.close()


def run(args,app):
    import dataclasses
    import numpy as np
    import torch
    import yaml
    from isaaclab.envs import DirectRLEnv
    from isaaclab.utils.math import quat_apply,quat_error_magnitude
    from isaaclab.sim.converters import UrdfConverter,UrdfConverterCfg
    import isaacsimenvs.tasks.simtoolreal.simtoolreal_env as env_module
    import isaacsimenvs.tasks.simtoolreal.utils.scene_utils as scene_module
    from isaacsimenvs.tasks.simtoolreal.robot_spec import KUKA_SHARPA_SPEC
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg
    from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import compute_intermediate_values
    from rl_games.algos_torch import model_builder

    started=time.monotonic();torch.set_num_threads(4);torch.manual_seed(20261027)
    def save(name,data):
        (args.output/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    save('run_arguments.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    params=json.loads((args.assets/'robot_spec.json').read_text())
    schema=json.loads((args.initial_run/'schema.json').read_text())
    source=np.load(args.initial_run/'dataset_60hz.npz')
    initials=np.load(args.assets/'demo_initials.npz')
    manifest=json.loads((args.frozen/'manifest.json').read_text())
    names=params['joint_names'];hand=names[7:]
    assert args.physics_hz%args.control_hz==0
    assert args.control_hz==60,'Reference dataset for this version is 60 Hz.'
    right_spec=dataclasses.replace(KUKA_SHARPA_SPEC,name='kuka_sharpa_right',
        hand_joint_regex='right_.*',joint_names_canonical=tuple(names),
        palm_body_name=params['palm_body'],palm_center_offset=tuple(params['palm_offset']),
        fingertip_link_names=tuple(params['fingertip_bodies']),
        fingertip_offsets=tuple(tuple(x) for x in params['fingertip_offsets']),
        arm_joint_stiffness=dict(zip(names[:7],params['kp'][:7])),
        arm_joint_damping=dict(zip(names[:7],params['kd'][:7])),
        hand_joint_stiffness=dict(zip(hand,params['kp'][7:])),
        hand_joint_damping=dict(zip(hand,params['kd'][7:])),
        hand_joint_armature=dict(zip(hand,params['joint_armature'][7:])),
        hand_joint_friction=dict(zip(hand,params['joint_friction'][7:])),
        hand_joint_effort=dict(zip(hand,params['effort'][7:])),
        arm_default_joint_pos=dict(zip(names[:7],initials['joint_pos'][0,:7].tolist())),
        arm_higher_joint_deltas={},robot_base_pos=tuple(params['base_pos']),
        robot_base_rot_wxyz=tuple(params['base_quat']))
    # Keep changes local to this process; original SimToolReal checkout is read-only.
    original_resolve=env_module.resolve_robot_spec
    def resolve(path):return right_spec if str(path)==params['robot_urdf'] else original_resolve(path)
    env_module.resolve_robot_spec=resolve;scene_module.resolve_robot_spec=resolve
    original_convert=scene_module._convert_urdf_to_usd
    def convert(path,work,**kwargs):
        if str(path)!=params['robot_urdf']:return original_convert(path,work,**kwargs)
        prepared=scene_module._prepare_urdf_for_isaacsim(path,work)
        # Preserve the fixed palm body so its orientation and offset match source sites.
        cfg=UrdfConverterCfg(asset_path=prepared,usd_dir=str(work/'kuka_sharpa_right'),
            force_usd_conversion=True,fix_base=True,merge_fixed_joints=False,
            make_instanceable=False,self_collision=True,joint_drive=kwargs.get('joint_drive'))
        return UrdfConverter(cfg).usd_path
    scene_module._convert_urdf_to_usd=convert
    original_bake=scene_module._bake_usd
    def bake(*a,**kw):
        path=original_bake(*a,**kw)
        from pxr import Usd,UsdPhysics,PhysxSchema
        stage=Usd.Stage.Open(path)
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                collision=PhysxSchema.PhysxCollisionAPI.Apply(prim)
                collision.CreateContactOffsetAttr(args.contact_offset)
                collision.CreateRestOffsetAttr(0.)
        stage.GetRootLayer().Save()
        return path
    scene_module._bake_usd=bake
    original_adjacency=scene_module._load_adjacent_links_map
    def adjacency():
        result=original_adjacency()
        for a,b in params['exclusions']:
            result.setdefault(a,[]).append(b);result.setdefault(b,[]).append(a)
        return result
    scene_module._load_adjacent_links_map=adjacency
    original_ground=scene_module.spawn_ground_plane
    def ground(*a,**kw):
        kw['translation']=(0.,0.,-1.5)
        return original_ground(*a,**kw)
    scene_module.spawn_ground_plane=ground

    class DemoEnv(env_module.SimToolRealEnv):
        def __init__(self,cfg):
            self._demo_ready=False
            super().__init__(cfg,render_mode='rgb_array' if args.capture_video else None)
            self.demo_q=torch.tensor(initials['joint_pos'],device=self.device,dtype=torch.float32)
            self.demo_v=torch.tensor(initials['joint_vel'],device=self.device,dtype=torch.float32)
            self.demo_object=torch.tensor(initials['object_pose'],device=self.device,dtype=torch.float32)
            self.demo_goal=torch.tensor(initials['goal_pose'],device=self.device,dtype=torch.float32)
            object_velocity=initials['object_velocity'] if 'object_velocity' in initials else np.zeros((len(manifest),6))
            self.demo_object_velocity=torch.tensor(object_velocity,device=self.device,dtype=torch.float32)
            self._demo_ready=True
            if args.sapg:
                # Each exploration group starts from the same twenty initial states.
                for name in ('demo_q','demo_v','demo_object','demo_goal','demo_object_velocity'):
                    value=getattr(self,name)
                    setattr(self,name,value[torch.arange(self.num_envs,device=self.device)%len(manifest)])

        def _reset_idx(self,env_ids):
            if env_ids is None:env_ids=torch.arange(self.num_envs,device=self.device)
            env_ids=torch.as_tensor(env_ids,device=self.device,dtype=torch.long)
            DirectRLEnv._reset_idx(self,env_ids)
            if not self._demo_ready:return
            if args.sapg:
                from isaacsimenvs.tasks.simtoolreal.utils.action_utils import sample_log_uniform
                self._prev_episode_successes[env_ids]=self._successes[env_ids]
                self._lifted_object[env_ids]=False
                dr=self.cfg.domain_randomization
                self._random_force_prob[env_ids]=sample_log_uniform(dr.force_prob_range,len(env_ids)).to(self.device)
                self._random_torque_prob[env_ids]=sample_log_uniform(dr.torque_prob_range,len(env_ids)).to(self.device)
            q=self.demo_q[env_ids][:,self._perm_canon_to_lab]
            v=self.demo_v[env_ids][:,self._perm_canon_to_lab]
            self.robot.write_joint_state_to_sim(q,v,env_ids=env_ids)
            self.robot.set_joint_position_target(q,env_ids=env_ids)
            self._prev_targets[env_ids]=q;self._cur_targets[env_ids]=q
            for asset,poses in [(self.object,self.demo_object),(self.goal_viz,self.demo_goal)]:
                state=asset.data.default_root_state[env_ids].clone()
                state[:,:7]=poses[env_ids];state[:,:3]+=self.scene.env_origins[env_ids]
                state[:,7:]=0
                if asset is self.object:state[:,7:]=self.demo_object_velocity[env_ids]
                asset.write_root_state_to_sim(state,env_ids=env_ids)
            state=self.table.data.default_root_state[env_ids].clone()
            state[:,:3]=self.scene.env_origins[env_ids];state[:,3:7]=torch.tensor([1.,0,0,0],device=self.device);state[:,7:]=0
            self.table.write_root_state_to_sim(state,env_ids=env_ids)
            self._closest_keypoint_max_dist[env_ids]=-1;self._closest_fingertip_dist[env_ids]=-1
            self._successes[env_ids]=0;self._near_goal_steps[env_ids]=0
            self._object_init_z[env_ids]=self.demo_object[env_ids,2]
            self._table_z_per_env[env_ids]=0
            if args.sapg:
                self._is_success[env_ids]=False
                self._near_goal[env_ids]=False
                for name in ('_action_queue','_obs_queue','_object_state_queue','_object_forces','_object_torques'):
                    getattr(self,name)[env_ids]=0

        def _pre_physics_step(self,actions):
            if args.sapg:
                self.actions=actions.clone()
                dr=self.cfg.domain_randomization
                if dr.use_action_delay and dr.action_delay_max>0:
                    start=self.episode_length_buf==0
                    self._action_queue[start]=actions[start].unsqueeze(1)
                    self._action_queue=torch.roll(self._action_queue,1,1)
                    self._action_queue[:,0]=actions
                    delay=torch.randint(self._action_queue.shape[1],(self.num_envs,),device=self.device)
                    actions=self._action_queue[torch.arange(self.num_envs,device=self.device),delay]
                from isaacsimenvs.tasks.simtoolreal.utils.action_utils import apply_wrench_dr
                apply_wrench_dr(self)
                lo=self._joint_lower_canon;hi=self._joint_upper_canon
                previous=self._prev_targets[:,self._perm_lab_to_canon]
                normalized=2*(previous-lo)/(hi-lo)-1
                actions=normalized+.1*actions
            # The BC contract is normalized absolute targets for all joints.
            lo=self._joint_lower_canon;hi=self._joint_upper_canon
            targets=lo+.5*(actions.clamp(-1,1)+1)*(hi-lo)
            self._cur_targets[:]=targets[:,self._perm_canon_to_lab]
            self._prev_targets=self._cur_targets.clone()

        def _get_dones(self):
            if args.sapg:
                from isaacsimenvs.tasks.simtoolreal.utils.termination_utils import update_tolerance_curriculum
                update_tolerance_curriculum(self)
                compute_intermediate_values(self)
                self._successes+=self._is_success.long()
                fall=self.object.data.root_pos_w[:,2]-self.scene.env_origins[:,2]<-.05
                far=self._curr_fingertip_distances.max(-1).values>1.5
                timeout=self.episode_length_buf>=self.max_episode_length
                self._termination_reasons=dict(fall=fall,hand_far=far,success=self._is_success,timeout=timeout)
                return fall|far|self._is_success,timeout
            compute_intermediate_values(self)
            zeros=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
            return zeros,zeros

        def _get_rewards(self):
            if args.sapg:return super()._get_rewards()
            return torch.zeros(self.num_envs,device=self.device)

    cfg=SimToolRealEnvCfg()
    cfg.from_dict(yaml.safe_load((args.simtoolreal/'isaacsimenvs/cfg/task/SimToolReal.yaml').read_text()))
    cfg.scene.num_envs=len(manifest);cfg.scene.env_spacing=1.6
    if args.sapg:
        assert args.num_envs%120==0,'Six SAPG groups, each with a whole set of twenty initial states.'
        assert not args.phase_input,'Preserve the original 140-dimensional actor observations.'
        cfg.scene.num_envs=args.num_envs
        cfg.scene.env_spacing=1.2
        # Keep the original 10 s horizon, reward coefficients, success tolerance
        # curriculum and delay/noise/impulse randomization. Only object geometry
        # is calibrated to this bottle rather than the original tool.
        cfg.reward.fixed_size=tuple(schema['object_calibration']['bbox_dims_m'])
        if args.capture_video:
            cfg.viewer.origin_type='env'
            # Film the zero-exploration-coefficient leader, candidate 64.
            cfg.viewer.env_index=5*(args.num_envs//6)+16
            cfg.viewer.eye=(.65,-.8,.65)
            cfg.viewer.lookat=(0.,.1,.22)
    cfg.sim.dt=1/args.physics_hz;cfg.decimation=args.physics_hz//args.control_hz
    cfg.sim.render_interval=cfg.decimation
    cfg.assets.robot_urdf=params['robot_urdf'];cfg.assets.table_urdf=params['table_urdf']
    cfg.assets.object_urdf=str((args.frozen/'object_asset/mustard_bottle.urdf').resolve())
    cfg.assets.object_scale=tuple(schema['object_calibration']['policy_object_scale'])
    if not args.sapg:
        cfg.assets.robot_friction=1.;cfg.assets.finger_tip_friction=1.;cfg.assets.object_friction=1.;cfg.assets.table_friction=1.
    if args.phase_input:cfg.obs.obs_list=tuple(cfg.obs.obs_list)+('progress',)
    cfg.reset.table_reset_z=0.;cfg.reset.table_reset_z_range=0.
    dr=cfg.domain_randomization
    if not args.sapg:
        dr.use_obs_delay=False;dr.use_action_delay=False;dr.use_object_state_delay_noise=False
        dr.joint_velocity_obs_noise_std=0.;dr.force_scale=0.;dr.torque_scale=0.
        dr.fingertip_friction_scale_range=(1.,1.);dr.object_friction_scale_range=(1.,1.)
    save('config_summary.json',dict(backend='Isaac Sim / Isaac Lab / PhysX',num_envs=cfg.scene.num_envs,
        right_hand=True,physics_hz=args.physics_hz,control_hz=args.control_hz,output_contract='normalized absolute targets; no native arm EMA/increment conversion',
        teacher='time-indexed demonstration target, not an arbitrary-state expert',
        target_task='target pose reaching' if args.sapg else 'imitation of the fixed 20 spatial variants',
        critic_training=args.sapg,policy_gradient=args.sapg,
        contact_offset_m=args.contact_offset,object_initial_velocity_preserved='object_velocity' in initials,
        actor_initial_checkpoint=str(args.checkpoint or args.initial_run/'actor_best.pt'),
        learning_rate=args.learning_rate,learner_device=args.learner_device,
        phase_input=args.phase_input,autoregressive_training=args.autoregressive))
    if args.sapg:
        (args.output/'environment_config.json').write_text(json.dumps(cfg.to_dict(),ensure_ascii=False,indent=2,default=str)+'\n')
    env=DemoEnv(cfg)
    if args.sapg:
        from run_mustard_sapg import train_sapg
        train_sapg(args,env,schema,save)
        env.close()
        return
    device=env.device
    # Runtime limit/order verification before any learning.
    base_dim=env.cfg.observation_space
    assert base_dim==140+int(args.phase_input),base_dim
    assert len(env.robot.data.joint_names)==29
    learner_device=torch.device(args.learner_device)
    old=torch.load(args.checkpoint or args.initial_run/'actor_best.pt',map_location=learner_device,weights_only=False)
    network_cfg=yaml.safe_load((args.simtoolreal/'isaacsimenvs/cfg/train/SimToolRealSAPG.yaml').read_text())['params']
    coef_ids=torch.linspace(50,0,6,device=learner_device)
    model=model_builder.ModelBuilder().load(network_cfg).build(dict(actions_num=29,input_shape=(base_dim+1,),num_seqs=env.num_envs,
        value_size=1,normalize_input=True,normalize_value=True,type='extra_param',coef_ids=coef_ids,coef_id_idx=base_dim)).to(learner_device)
    state=model.state_dict();old_dim=len(old['model']['running_mean_std.running_mean'])
    for key,value in old['model'].items():
        if state[key].shape==value.shape:state[key]=value
        elif key in ('running_mean_std.running_mean','running_mean_std.running_var') and base_dim==old_dim+1:
            state[key][:old_dim]=value
        elif key.endswith('rnn.rnn.weight_ih_l0') and base_dim==old_dim+1:
            state[key][:,:old_dim]=value[:,:old_dim]
            state[key][:,old_dim]=0.
            state[key][:,old_dim+1:]=value[:,old_dim:]
        else:raise ValueError(f'Unsupported checkpoint shape change: {key} {value.shape} -> {state[key].shape}')
    model.load_state_dict(state);model.eval()
    model.a2c_network.sigma.requires_grad_(False)
    for p in model.a2c_network.value.parameters():p.requires_grad_(False)
    limits=torch.tensor(schema['joint_limits_rad'],device=device)
    learner_limits=limits.to(learner_device)
    decoder='previous-residual' if old.get('schema',schema).get('residual_decoder',True) else 'absolute'
    def forward(obs,hidden,ids):
        obs=obs.to(learner_device);ids=ids.to(learner_device)
        b,t,_=obs.shape
        net_obs=torch.cat([obs,ids[:,None,None].expand(b,t,1)],-1).reshape(-1,base_dim+1)
        if args.autoregressive:
            stats=model.running_mean_std
            normalized=(net_obs[:,:base_dim]-stats.running_mean.float())/torch.sqrt(stats.running_var.float()+stats.epsilon)
            net_obs=torch.cat([normalized.clamp(-5,5),net_obs[:,base_dim:]],-1)
        else:net_obs=model.norm_obs(net_obs)
        raw,_,_,h=model.a2c_network(dict(obs=net_obs,rnn_states=hidden,seq_length=t))
        raw=raw.reshape(b,t,29)
        if decoder=='absolute':return raw,h
        previous=2*(obs[:,:,58:87]-learner_limits[:,0])/(learner_limits[:,1]-learner_limits[:,0])-1
        return previous+.1*raw,h
    def zero_state(b):return tuple(s[:,:1].expand(-1,b,-1).contiguous().zero_().to(learner_device) for s in model.get_default_rnn_state())
    def autoregressive_forward(batch,ids):
        batch=batch.to(learner_device)
        h=zero_state(len(batch));previous=batch[:,0,58:87];predictions=[]
        for t in range(batch.shape[1]):
            current=torch.cat([batch[:,t,:58],previous,batch[:,t,87:]],-1)
            pred,h=forward(current[:,None],h,ids);pred=pred[:,0]
            predictions.append(pred)
            previous=learner_limits[:,0]+.5*(pred.clamp(-1,1)+1)*(learner_limits[:,1]-learner_limits[:,0])
        return torch.stack(predictions,1)
    targets=torch.tensor(source['targets_rad'],device=device)
    labels=2*(targets-limits[:,0])/(limits[:,1]-limits[:,0])-1
    train_ids=torch.as_tensor(np.flatnonzero(source['train']),device=device)
    val_ids=torch.as_tensor(np.flatnonzero(~source['train']),device=device)
    fit_ids=train_ids
    if args.train_candidate is not None:
        selected=np.flatnonzero(source['candidate_ids']==args.train_candidate)
        assert len(selected)==1 and source['train'][selected[0]],'Single-demo fit must use an original training episode.'
        fit_ids=torch.as_tensor(selected,device=device)
    frame_weights=torch.ones(1,labels.shape[1],1,device=learner_device)
    frame_weights[:,:10]=args.initial_loss_weight
    frame_weights/=frame_weights.mean()
    assert np.array_equal(initials['candidate_ids'],source['candidate_ids'])
    def observe():return env._get_observations()['policy']
    @torch.no_grad()
    def collect(teacher_prob,name):
        env.reset();h=zero_state(env.num_envs);observations=[];predictions=[];poses=[];joints=[]
        for t in range(labels.shape[1]):
            obs=observe();mu,h=forward(obs[:,None],h,torch.zeros(env.num_envs,device=device));mu=mu[:,0].to(device)
            choose=torch.rand(env.num_envs,1,device=device)<teacher_prob
            actions=torch.where(choose,labels[:,t],mu)
            observations.append(obs.clone());predictions.append(mu.clone())
            env.step(actions)
            poses.append(torch.cat([env.object.data.root_pos_w-env.scene.env_origins,env.object.data.root_quat_w],-1).clone())
            joints.append(env.robot.data.joint_pos[:,env._perm_lab_to_canon].clone())
        obs=torch.stack(observations,1);pred=torch.stack(predictions,1);poses=torch.stack(poses,1)
        goal=env.demo_goal
        poserr=torch.linalg.vector_norm(poses[:,-1,:3]-goal[:,:3],dim=-1)
        roterr=quat_error_magnitude(poses[:,-1,3:],goal[:,3:])*180/np.pi
        passed=(poserr<.03)&(roterr<20)
        summary=dict(name=name,teacher_probability=teacher_prob,train_pass=int(passed[train_ids].sum()),train_count=len(train_ids),
            validation_pass=int(passed[val_ids].sum()),validation_count=len(val_ids),
            mean_position_error_m=float(poserr.mean()),mean_orientation_error_deg=float(roterr.mean()),
            position_errors_m=poserr.cpu().tolist(),orientation_errors_deg=roterr.cpu().tolist(),
            elapsed_s=time.monotonic()-started)
        save(name+'.json',summary)
        np.savez_compressed(args.output/(name+'.npz'),obs=obs.cpu().numpy(),actor_predictions=pred.cpu().numpy(),
            object_poses=poses.cpu().numpy(),joint_positions=torch.stack(joints,1).cpu().numpy(),labels=labels.cpu().numpy())
        print('ROLLOUT',json.dumps(summary),flush=True)
        return obs,summary

    teacher_obs,teacher_report=collect(1.,'teacher_baseline')
    _,initial_report=collect(0.,'actor_initial')
    save('asset_runtime_check.json',dict(joint_names=list(env.robot.data.joint_names),body_names=list(env.robot.data.body_names),
        canonical_joint_names=names,limits_max_error_rad=float(torch.max(torch.abs(torch.stack([env._joint_lower_canon,env._joint_upper_canon],-1)-limits))),
        observation_dim=env.cfg.observation_space,simulation_device=str(device)))
    if args.smoke or args.evaluate_only:
        repeats=[]
        for i in range(args.evaluation_repeats if args.evaluate_only else 0):
            _,repeat=collect(0.,f'actor_repeat_{i:02d}')
            repeats.append(repeat)
        save('completed.json',dict(stage='Isaac Sim teacher/actor smoke only',teacher=teacher_report,actor=initial_report))
        if repeats:save('evaluation_repeats.json',repeats)
        env.close();return

    # Fit input statistics to actual target-engine training observations only.
    flat=teacher_obs[train_ids].reshape(-1,base_dim).to(learner_device)
    model.running_mean_std.running_mean.copy_(flat.mean(0).double())
    model.running_mean_std.running_var.copy_(flat.var(0).clamp_min(1e-4).double())
    model.running_mean_std.count.fill_(len(flat))
    decoder=args.decoder
    schema=dict(schema)
    schema['obs_dim']=base_dim
    schema['phase_conditioned']=args.phase_input
    if args.phase_input and not any(f[0]=='progress' for f in schema['fields']):schema['fields']=list(schema['fields'])+[['progress',1]]
    schema['residual_decoder']=decoder=='previous-residual'
    schema['target_contract']=('normalized absolute target = actor_mu; decode using joint limits' if decoder=='absolute'
        else 'normalized absolute target = normalized previous target + 0.1 * actor_mu; decode using joint limits')
    if args.ridge_head:
        assert decoder=='absolute'
        feature_batches=[];target_batches=[]
        captured=[]
        handle=model.a2c_network.mu.register_forward_pre_hook(lambda module,inputs:captured.append(inputs[0].detach()))
        with torch.no_grad():
            for cid in coef_ids:
                captured.clear()
                forward(teacher_obs[train_ids],zero_state(len(train_ids)),cid.repeat(len(train_ids)))
                feature_batches.append(captured[0].clone().double())
                target_batches.append(labels[train_ids].reshape(-1,29).to(learner_device).double())
        handle.remove()
        features=torch.cat(feature_batches);target=torch.cat(target_batches)
        mean=features.mean(0);std=features.std(0).clamp_min(.01)
        design=torch.cat([(features-mean)/std,torch.ones(len(features),1,device=learner_device,dtype=torch.float64)],-1)
        regularizer=torch.eye(design.shape[1],device=learner_device,dtype=torch.float64)*1.
        regularizer[-1,-1]=0.
        solution=torch.linalg.solve(design.T@design+regularizer,design.T@target)
        model.a2c_network.mu.weight.data.copy_((solution[:-1]/std[:,None]).T.float())
        model.a2c_network.mu.bias.data.copy_((solution[-1]-mean@(solution[:-1]/std[:,None])).float())
        save('head_fit.json',dict(method='ridge regression on frozen SAPG actor features from PhysX training episodes',
            training_rows=len(features),features=features.shape[1],validation_used_in_fit=False,
            fit_normalized_rmse=float(torch.mean((design@solution-target)**2).sqrt())))
    save('schema.json',schema)
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=args.learning_rate)
    history=[];best_score=(-1,float('-inf'));best_round=-1
    def score_of(e):
        v=val_ids.cpu().numpy()
        return (e['validation_pass'],-float(np.mean(np.array(e['position_errors_m'])[v]/.03+np.array(e['orientation_errors_deg'])[v]/20)))
    def make_checkpoint(round_idx):
        return dict(model=model.state_dict(),schema=schema,network_config=network_cfg['network'],coef_ids=coef_ids.cpu(),
            round=round_idx,optimizer=optimizer.state_dict(),backend='Isaac Sim / PhysX online trajectory-supervised actor imitation',
            output_contract=schema['target_contract'],native_sapg_actions=False)
    if args.ridge_head:
        _,head_evaluation=collect(0.,'actor_head_initial')
        best_score=score_of(head_evaluation)
        history.append(dict(round=-1,loss=None,evaluation=head_evaluation));save('history.json',history)
        torch.save(make_checkpoint(-1),args.output/'actor_best.pt')
    obs_bank=[teacher_obs]
    for round_idx in range(args.rounds):
        if round_idx:
            probability=max(.5,1-round_idx/max(1,args.rounds-2))
            rollin,_=collect(probability,f'rollin_{round_idx:02d}')
            obs_bank.append(rollin)
        # Aggregate recent roll-ins plus teacher; don't insert validation episodes into updates.
        bank=[obs_bank[0]]+obs_bank[-3:] if len(obs_bank)>1 else obs_bank
        for epoch in range(args.epochs_per_round):
            # Anchor most updates to demonstration-executed PhysX observations.
            chosen=bank[0] if epoch%5!=4 else bank[-1]
            order=torch.roll(fit_ids,shifts=epoch*args.batch_episodes)
            batch_ids=order[:min(args.batch_episodes,len(order))]
            batch=chosen[batch_ids]
            ids=coef_ids[(torch.arange(len(batch_ids),device=learner_device)+epoch)%6]
            if args.autoregressive:pred=autoregressive_forward(batch,ids)
            else:pred,_=forward(batch,zero_state(len(batch_ids)),ids)
            err=pred-labels[batch_ids].to(learner_device)
            loss=.5*((err[:,:,:7].square()*frame_weights).mean()+(err[:,:,7:].square()*frame_weights).mean())
            if not torch.isfinite(loss):raise RuntimeError('Non-finite actor imitation loss; preserving earlier checkpoints.')
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            if epoch%5==0:
                print('BC_STEP',json.dumps(dict(round=round_idx,epoch=epoch,loss=float(loss.detach()),learner_device=str(learner_device),elapsed_s=time.monotonic()-started)),flush=True)
        _,evaluation=collect(0.,f'actor_round_{round_idx:02d}')
        score=score_of(evaluation)
        history.append(dict(round=round_idx,loss=float(loss.detach()),evaluation=evaluation));save('history.json',history)
        checkpoint=make_checkpoint(round_idx)
        torch.save(checkpoint,args.output/'actor_latest.pt')
        if score>best_score:
            best_score=score;best_round=round_idx;torch.save(checkpoint,args.output/'actor_best.pt')
    save('completed.json',dict(stage='Isaac Sim actor supervised training',teacher=teacher_report,initial_actor=initial_report,
        best_round=best_round,best_validation_pose_pass=best_score[0],rounds=args.rounds,epochs_per_round=args.epochs_per_round,
        wall_time_s=time.monotonic()-started,source_episodes=1,spatial_variants=20,critic_trained=False))
    env.close()


if __name__=='__main__':main()
