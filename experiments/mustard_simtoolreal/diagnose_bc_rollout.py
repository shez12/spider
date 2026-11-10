"""Matched clean initial states: full-memory actor, training-memory actor, oracle."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))


def main():
 from isaaclab.app import AppLauncher
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--cases',type=int,default=10);p.add_argument('--cached-initial-state',action='store_true');p.add_argument('--video',type=Path)
 p.add_argument('--waypoints',type=Path,default=ROOT/'waypoints100.json');p.add_argument('--sample-random',action='store_true');p.add_argument('--sample-seed',type=int,default=42)
 AppLauncher.add_app_launcher_args(p);args=p.parse_args();args.enable_cameras=bool(args.video);app=AppLauncher(args).app;env=None;video=None
 try:
  import numpy as np,torch
  from environment import create_env
  from rl_games.algos_torch import model_builder
  from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import compute_intermediate_values
  source=np.load(args.run/'native_bc_dataset.npz');n=args.cases
  assert 0<n<=len(source['candidate_ids'])
  selection=np.random.default_rng(args.sample_seed).choice(len(source['candidate_ids']),n,replace=False) if args.sample_random else np.arange(n)
  x=source['obs'][selection];labels=source['actions'][selection];lengths=source['valid_lengths'][selection];candidate_ids=source['candidate_ids'][selection]
  pool={t['candidate_id']:t for t in json.loads(args.waypoints.read_text())['trajectories']}
  def configure(cfg):
   dr=cfg.domain_randomization;dr.use_action_delay=False;dr.use_obs_delay=False;dr.use_object_state_delay_noise=False;dr.joint_velocity_obs_noise_std=0.;dr.force_scale=0.;dr.torque_scale=0.;cfg.reset.table_reset_z_range=0.
   if args.video:cfg.viewer.env_index=0;cfg.sim.render_interval=12
  env=create_env(3*n,render_mode='rgb_array' if args.video else None,configure=configure);device=env.device;env.reset();indices=torch.arange(3*n,device=device)%n
  q=torch.tensor(x[:,0,58:87],device=device)[indices][:,env._perm_canon_to_lab]
  env.robot.write_joint_state_to_sim(q,torch.zeros_like(q));env.robot.set_joint_position_target(q);env._prev_targets[:]=q;env._cur_targets[:]=q
  pose=torch.tensor([pool[int(c)]['initial_pose'] for c in candidate_ids],device=device)[indices]
  state=env.object.data.default_root_state.clone();state[:,:7]=pose;state[:,:3]+=env.scene.env_origins;state[:,7:]=0;env.object.write_root_state_to_sim(state);env._object_init_z[:]=pose[:,2]
  goals=torch.tensor([pool[int(c)]['pos'][-1]+pool[int(c)]['quat_wxyz'][-1] for c in candidate_ids],device=device)[indices];goals[:,:3]+=env.scene.env_origins;env.goal_viz.write_root_pose_to_sim(goals)
  if args.cached_initial_state:
   initial=torch.tensor(x[:,0],device=device)
   lo=env._joint_lower_canon;hi=env._joint_upper_canon
   actual_q=(lo+.5*(initial[:,:29]+1)*(hi-lo))[indices][:,env._perm_canon_to_lab]
   actual_v=initial[:,29:58][indices][:,env._perm_canon_to_lab]
   env.robot.write_joint_state_to_sim(actual_q,actual_v)
   object_pose=torch.cat([initial[:,113:125].reshape(n,4,3).mean(1)+initial[:,87:90],initial[:,[97,94,95,96]]],-1)[indices]
   state[:,:7]=object_pose;state[:,:3]+=env.scene.env_origins;state[:,7:]=0
   env.object.write_root_state_to_sim(state)
   env.scene.write_data_to_sim();env.sim.forward();env.scene.update(0.)
  else:
   env.scene.write_data_to_sim();env.sim.step(render=False);env.scene.update(env.physics_dt)
  def no_dones():
   compute_intermediate_values(env);z=torch.zeros(3*n,device=device,dtype=torch.bool);return z,z.clone()
  env._get_dones=no_dones;env._get_rewards=lambda:torch.zeros(3*n,device=device)
  checkpoint=torch.load(args.checkpoint,map_location=device,weights_only=False)
  model=model_builder.ModelBuilder().load(dict(network=checkpoint['network'],model=dict(name='continuous_a2c_logstd'))).build(dict(actions_num=29,input_shape=(141,),num_seqs=n,value_size=1,normalize_input=True,normalize_value=True,type='extra_param',coef_ids=torch.linspace(50,0,6,device=device),coef_id_idx=140)).to(device);model.load_state_dict(checkpoint['model']);model.eval()
  def zero():return tuple(s.to(device).zero_() for s in model.get_default_rnn_state())
  def forward(obs,h):
   b,t,_=obs.shape;inp=torch.cat([obs,torch.zeros((b,t,1),device=device)],-1).reshape(-1,141);mu,_,_,h=model.a2c_network(dict(obs=model.norm_obs(inp),rnn_states=h,seq_length=t));return mu.reshape(b,t,29),h
  full_h=zero();window_h=zero();history=[];hold_run=torch.zeros(3*n,device=device,dtype=torch.long);max_hold=hold_run.clone();near=hold_run.clone();max_height=torch.full((3*n,),-100.,device=device);q_errors=[]
  if args.video:
   import imageio.v2 as imageio
   args.video.parent.mkdir(parents=True,exist_ok=True)
   video=imageio.get_writer(str(args.video),fps=10,codec='libx264',macro_block_size=1)
   for _ in range(20):
    if env.render(recompute=True).max()>0:break
  with torch.no_grad():
   for step in range(int(max(lengths))):
    obs=env._get_observations()['policy'].clamp(-10,10)
    if step==0:
     reference=torch.tensor(x[:,0],device=device)[indices]
     initial_error=float((obs-reference).abs().max())
     initial_fields={name:float((obs[:,start:end]-reference[:,start:end]).abs().max()) for name,start,end in [('joint_position_normalized',0,29),('joint_velocity_rad_s',29,58),('previous_target_rad',58,87),('palm_position_m',87,90),('palm_quaternion',90,94),('object_quaternion',94,98),('fingertips_m',98,113),('object_keypoints_m',113,125),('goal_keypoint_delta_m',125,137)]}
     group_initial_difference={name:float((obs[j*n:(j+1)*n]-obs[2*n:3*n]).abs().max()) for j,name in enumerate(('actor_full_history','actor_training_context'))}
    history.append(obs[n:2*n].clone())
    mu,full_h=forward(obs[:n,None],full_h)
    if step%16==0:
     window_h=zero()
     if step:_,window_h=forward(torch.stack(history[max(0,step-16):step],1),window_h)
    win,window_h=forward(obs[n:2*n,None],window_h)
    oracle=torch.tensor(labels[:,step],device=device)
    env.step(torch.cat([mu[:,0].clamp(-1,1),win[:,0].clamp(-1,1),oracle],0))
    observed=env._get_observations()['policy'];palm=observed[:,87:90];position=env.object.data.root_pos_w-env.scene.env_origins;height=position[:,2]-pose[:,2]
    held=(height>=.05)&(torch.linalg.vector_norm(position-palm,dim=-1)<.18)&(env._curr_fingertip_distances.sort(-1).values[:,:3].mean(-1)<.10)&(torch.linalg.vector_norm(env.object.data.root_lin_vel_w,dim=-1)<2.)
    active=torch.tensor(step<lengths,device=device)[indices]
    hold_run=torch.where(held&active,hold_run+1,0);max_hold=torch.maximum(max_hold,hold_run);max_height=torch.maximum(max_height,torch.where(active,height,-100.));near+=((env._keypoints_max_dist<=.1125)&active).long()
    if video is not None and step%6==0:video.append_data(env.render())
    if step%100==0:print('MATCHED_ROLLOUT',step,int(max(lengths)),flush=True)
  report=dict(initialization='cached_actor_state_with_zero_object_velocity' if args.cached_initial_state else 'demonstration_pose_then_half_physics_step',checkpoint_epoch=checkpoint['bc_epochs'],cases=n,candidate_ids=candidate_ids.tolist(),initial_observation_max_error=initial_error,initial_error_by_field=initial_fields,initial_actor_oracle_max_difference=group_initial_difference,groups={})
  for j,name in enumerate(('actor_full_history','actor_training_context','oracle_native_actions')):
   sl=slice(j*n,(j+1)*n);report['groups'][name]=dict(held_lift_30_steps=int((max_hold[sl]>=30).sum()),reached_initial_tolerance=int((near[sl]>=10).sum()),max_height_m=max_height[sl].cpu().tolist(),longest_hold_steps=max_hold[sl].cpu().tolist(),near_goal_steps=near[sl].cpu().tolist())
  args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
 finally:
  if video is not None:video.close()
  if env is not None:env.close()
  app.close()


if __name__=='__main__':main()
