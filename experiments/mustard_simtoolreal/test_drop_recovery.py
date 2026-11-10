"""Paired natural-drop diagnostic, differing only in recurrent memory at fork."""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT.parents[1]/'third_party/simtoolreal'),str(ROOT.parents[1]/'third_party/simtoolreal/rl_games')]

def main():
 from isaaclab.app import AppLauncher
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--waypoints',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--cases',type=int,default=30)
 p.add_argument('--bank',type=Path)
 AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app;env=None
 try:
  import numpy as np,torch
  from environment import create_env
  from bc_demo_reset import attach_demo_resets
  from rl_games.algos_torch import model_builder
  from isaacsimenvs.tasks.simtoolreal.utils import reset_utils
  from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import compute_intermediate_values
  ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);n=a.cases;count=2*n
  with np.load(a.dataset) as d:
   pick=np.random.default_rng(42).choice(len(d['candidate_ids']),n,replace=False);ids=d['candidate_ids'][pick]
  pool={t['candidate_id']:t for t in json.loads(a.waypoints.read_text())['trajectories']}
  def configure(c):
   dr=c.domain_randomization;dr.use_action_delay=False;dr.use_obs_delay=False;dr.use_object_state_delay_noise=False;dr.joint_velocity_obs_noise_std=0.;dr.force_scale=0.;dr.torque_scale=0.;c.reset.table_reset_z_range=0.
   c.episode_length_s=16.;c.termination.episode_length=960;c.termination.max_consecutive_successes=50;c.termination.tolerance_curriculum_interval=10**9;c.termination.eval_success_tolerance=.075
  env=create_env(count,configure=configure);dev=env.device
  mapping=SimpleNamespace(candidate_ids=torch.tensor(ids,device=dev),ids=torch.arange(count,device=dev)%n)
  reference,restore=attach_demo_resets(env,mapping,a.dataset);env.reset()
  goals=torch.tensor([pool[int(c)]['pos'][-1]+pool[int(c)]['quat_wxyz'][-1] for c in ids],device=dev,dtype=torch.float32).repeat(2,1);goals[:,:3]+=env.scene.env_origins
  env.goal_viz.write_root_pose_to_sim(goals);env.scene.write_data_to_sim();env.sim.forward();env.scene.update(0.)
  model=model_builder.ModelBuilder().load(dict(network=ck['network'],model=dict(name='continuous_a2c_logstd'))).build(dict(actions_num=29,input_shape=(141,),num_seqs=count,value_size=1,normalize_input=True,normalize_value=True,type='extra_param',coef_ids=torch.linspace(50,0,6,device=dev),coef_id_idx=140)).to(dev)
  model.load_state_dict(ck['model']);model.eval();h=[s.to(dev).zero_() for s in model.get_default_rnn_state()]
  initial=env._get_observations()['policy'].clamp(-10,10);initial_error=float((initial-reference()).abs().max());assert initial_error<1e-4
  startz=env.object.data.root_pos_w[:,2]-env.scene.env_origins[:,2]
  fork=torch.full((n,),-1,device=dev,dtype=torch.long);lift_run=torch.zeros(count,device=dev,dtype=torch.long);longest=lift_run.clone();postlong=lift_run.clone();postrun=lift_run.clone();near=lift_run.clone();settled=torch.zeros(n,device=dev,dtype=torch.long);eligible=torch.ones(n,device=dev,dtype=torch.bool)
  original_goal=reset_utils._reset_goal_pose
  def goal_fn(target,env_ids,mode):
   if target is env and mode=='delta':env_ids=env_ids[fork[env_ids%n]<0]
   if len(env_ids):return original_goal(target,env_ids,mode)
  reset_utils._reset_goal_pose=goal_fn
  old_dones=env._get_dones;terminal=torch.zeros(count,device=dev,dtype=torch.bool)
  def no_reset():
   nonlocal terminal
   done,trunc=old_dones();terminal=done|trunc;z=torch.zeros_like(done);return z,z.clone()
  env._get_dones=no_reset
  state_errors=[];bank=[]
  with torch.no_grad():
   for step in range(1920):
    obs=env._get_observations()['policy'].clamp(-10,10)
    inp=torch.cat([obs,torch.zeros((count,1),device=dev)],-1)
    mu,_,_,new_h=model.a2c_network(dict(obs=model.norm_obs(inp),rnn_states=h,seq_length=1));h=list(new_h)
    unbranched=torch.where(fork<0)[0]
    mu[unbranched+n]=mu[unbranched]
    for state in h:state[:,unbranched+n]=state[:,unbranched]
    env.step(mu.clamp(-1,1));compute_intermediate_values(env)
    pos=env.object.data.root_pos_w-env.scene.env_origins;palm=env._get_observations()['policy'][:,87:90]
    held=(pos[:,2]-startz>=.05)&(torch.linalg.vector_norm(pos-palm,dim=-1)<.18)&(env._curr_fingertip_distances.sort(-1).values[:,:3].mean(-1)<.1)&(torch.linalg.vector_norm(env.object.data.root_lin_vel_w,dim=-1)<2.)
    lift_run=torch.where(held,lift_run+1,0);longest=torch.maximum(longest,lift_run)
    active=(fork.repeat(2)>=0)&(step-fork.repeat(2)<=960)
    postrun=torch.where(held&active,postrun+1,0);postlong=torch.maximum(postlong,postrun)
    near+=((env._keypoints_max_dist<=.1125)&active).long()
    eligible&=~(terminal[:n]&(fork<0))
    detached=(torch.linalg.vector_norm(pos[:n]-palm[:n],dim=-1)>.18)|(env._curr_fingertip_distances[:n].sort(-1).values[:,:3].mean(-1)>.1)
    ontable=detached&(pos[:n,2]<=startz[:n]+.04)&(torch.linalg.vector_norm(env.object.data.root_lin_vel_w[:n],dim=-1)<.15)&(pos[:n,2]>.1)
    settled=torch.where(ontable&(longest[:n]>=30),settled+1,0)
    new=torch.where((fork<0)&(settled>=15)&eligible&(step<960))[0]
    if len(new):
     both=torch.cat([new,new+n]);src=torch.cat([new,new])
     q=env.robot.data.joint_pos[src].clone();v=env.robot.data.joint_vel[src].clone()
     root=env.object.data.root_state_w[src].clone();root[:,:3]+=env.scene.env_origins[both]-env.scene.env_origins[src]
     g=env.goal_viz.data.root_state_w[src,:7].clone();g[:,:3]+=env.scene.env_origins[both]-env.scene.env_origins[src]
     env.robot.write_joint_state_to_sim(q,v,env_ids=both);env.object.write_root_state_to_sim(root,env_ids=both);env.goal_viz.write_root_pose_to_sim(g,env_ids=both)
     for name in ['_prev_targets','_cur_targets','_successes','_near_goal_steps','episode_length_buf','_object_init_z']:
      value=getattr(env,name);value[both]=value[src].clone()
     env.robot.set_joint_position_target(env._prev_targets[both],env_ids=both)
     for state in h:state[:,new+n]=0
     fork[new]=step
     env.scene.write_data_to_sim();env.sim.forward();env.scene.update(0.)
     ob=env._get_observations()['policy'];state_errors.append(float((ob[new]-ob[new+n]).abs().max()))
     if a.bank:
      compute_intermediate_values(env)
      for index in new.cpu().tolist():
       entry=dict(candidate_id=int(ids[index]),q=env.robot.data.joint_pos[index].cpu().numpy().copy(),v=env.robot.data.joint_vel[index].cpu().numpy().copy(),
        root=env.object.data.root_state_w[index].cpu().numpy().copy(),goal=env.goal_viz.data.root_state_w[index,:7].cpu().numpy().copy(),
        previous=env._prev_targets[index].cpu().numpy().copy(),current=env._cur_targets[index].cpu().numpy().copy(),obs=env._get_observations()['policy'][index].cpu().numpy().copy(),
        origin=env.scene.env_origins[index].cpu().numpy().copy(),fork_step=step)
       for j,state in enumerate(h):entry['h'+str(j)]=state[:,index].cpu().numpy().copy()
       entry['root'][:3]-=entry['origin'];entry['goal'][:3]-=entry['origin'];bank.append(entry)
      a.bank.parent.mkdir(parents=True,exist_ok=True)
      np.savez_compressed(a.bank,**{k:np.stack([e[k] for e in bank]) for k in bank[0]})
     print('DROP_FORK',step,new.cpu().tolist(),flush=True)
    if step==959:first_grasp_count=int((longest[:n]>=30).sum())
    if step%200==0:print('RECOVERY_PROGRESS',step,'initial_grasps',int((longest[:n]>=30).sum()),'drops',int((fork>=0).sum()),flush=True)
    if step>=960 and (not (fork>=0).any() or bool(((step-fork[fork>=0])>=960).all())):break
  rows=[]
  for i,c in enumerate(ids):
   rows.append(dict(candidate_id=int(c),first_grasp=bool(longest[i]>=30),drop_step=int(fork[i]),
    keep_memory_regrasp=bool(postlong[i]>=30) if fork[i]>=0 else None,zero_memory_regrasp=bool(postlong[i+n]>=30) if fork[i]>=0 else None,
    keep_memory_reach=bool(near[i]>=10) if fork[i]>=0 else None,zero_memory_reach=bool(near[i+n]>=10) if fork[i]>=0 else None))
  report=dict(training_epoch=ck.get('training_epoch'),cases=n,policy='deterministic mean; leader coefficient ID=0',initial_observation_error=initial_error,
   natural_first_grasps=first_grasp_count,natural_settled_drops=sum(x['drop_step']>=0 for x in rows),
   keep_memory_regrasp=sum(x['keep_memory_regrasp'] is True for x in rows),zero_memory_regrasp=sum(x['zero_memory_regrasp'] is True for x in rows),
   keep_memory_reach=sum(x['keep_memory_reach'] is True for x in rows),zero_memory_reach=sum(x['zero_memory_reach'] is True for x in rows),
   max_fork_observation_error=max(state_errors,default=None),rows=rows,
   protocol='Observe natural drops within first 16s; after 15 settled and detached steps on table, rewrite both branches to matching joint/object/goal/command state; keep vs zero LSTM. Allow up to16s after fork and hold current target. Not native episode success rate; PhysX contact cache is not cloned.')
  a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
 finally:
  if env is not None:env.close()
  app.close()
if __name__=='__main__':main()
