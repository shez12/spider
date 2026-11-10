"""Compare BC and SAPG at matched saved drop states and demo initial states."""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT.parents[1]/'third_party/simtoolreal'),str(ROOT.parents[1]/'third_party/simtoolreal/rl_games')]

def main():
 from isaaclab.app import AppLauncher
 p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--bc',type=Path,required=True);p.add_argument('--sapg',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--waypoints',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app;env=None
 try:
  import numpy as np,torch
  from environment import create_env
  from bc_demo_reset import attach_demo_resets
  from rl_games.algos_torch import model_builder
  from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import compute_intermediate_values
  bank=np.load(a.bank);ids=bank['candidate_id'];n=len(ids);count=5*n;dev='cuda:0'
  def configure(c):
   dr=c.domain_randomization;dr.use_action_delay=False;dr.use_obs_delay=False;dr.use_object_state_delay_noise=False;dr.joint_velocity_obs_noise_std=0.;dr.force_scale=0.;dr.torque_scale=0.;c.reset.table_reset_z_range=0.;c.episode_length_s=16.;c.termination.tolerance_curriculum_interval=10**9
  env=create_env(count,configure=configure);dev=env.device
  mapping=SimpleNamespace(candidate_ids=torch.tensor(ids,device=dev),ids=torch.arange(count,device=dev)%n)
  reference,restore=attach_demo_resets(env,mapping,a.dataset);env.reset()
  pool={t['candidate_id']:t for t in json.loads(a.waypoints.read_text())['trajectories']}
  goals=torch.tensor([pool[int(c)]['pos'][-1]+pool[int(c)]['quat_wxyz'][-1] for c in ids],device=dev,dtype=torch.float32).repeat(5,1);goals[:,:3]+=env.scene.env_origins
  env.goal_viz.write_root_pose_to_sim(goals)
  dropids=torch.arange(3*n,device=dev);select=torch.arange(3*n,device=dev)%n
  def val(k):return torch.tensor(bank[k],device=dev)[select]
  root=val('root');root[:,:3]+=env.scene.env_origins[dropids];goal=val('goal');goal[:,:3]+=env.scene.env_origins[dropids]
  env.robot.write_joint_state_to_sim(val('q'),val('v'),env_ids=dropids);env.object.write_root_state_to_sim(root,env_ids=dropids);env.goal_viz.write_root_pose_to_sim(goal,env_ids=dropids)
  env._prev_targets[dropids]=val('previous');env._cur_targets[dropids]=val('current');env.robot.set_joint_position_target(val('previous'),env_ids=dropids)
  env.scene.write_data_to_sim();env.sim.forward();env.scene.update(0.);compute_intermediate_values(env)
  initial=env._get_observations()['policy'].clone();paired_error=float((initial[:n]-initial[n:2*n]).abs().max())
  bank_error=float((initial[:n]-torch.tensor(bank['obs'],device=dev)).abs().max());assert paired_error<1e-4 and bank_error<1e-4,(paired_error,bank_error)
  def no_reset():
   compute_intermediate_values(env);z=torch.zeros(count,device=dev,dtype=torch.bool);return z,z.clone()
  env._get_dones=no_reset;env._get_rewards=lambda:torch.zeros(count,device=dev)
  checkpoints=[torch.load(a.bc,map_location=dev,weights_only=False),torch.load(a.sapg,map_location=dev,weights_only=False)]
  models=[];states=[]
  for ck in checkpoints:
   m=model_builder.ModelBuilder().load(dict(network=ck['network'],model=dict(name='continuous_a2c_logstd'))).build(dict(actions_num=29,input_shape=(141,),num_seqs=count,value_size=1,normalize_input=True,normalize_value=True,type='extra_param',coef_ids=torch.linspace(50,0,6,device=dev),coef_id_idx=140)).to(dev);m.load_state_dict(ck['model']);m.eval();models.append(m)
   states.append([s.to(dev).zero_() for s in m.get_default_rnn_state()])
  for j,state in enumerate(states[1]):state[:,2*n:3*n]=torch.tensor(bank['h'+str(j)],device=dev).transpose(0,1)
  names=['bc_drop_zero','sapg_drop_zero','sapg_drop_keep','bc_demo_zero','sapg_demo_zero']
  liftbase=env.object.data.root_pos_w[:,2].clone();run=torch.zeros(count,device=dev,dtype=torch.long);longest=run.clone();near=run.clone();min_distance=torch.full((count,),100.,device=dev)
  with torch.no_grad():
   for step in range(960):
    obs=env._get_observations()['policy'].clamp(-10,10);inp=torch.cat([obs,torch.zeros((count,1),device=dev)],-1)
    output=[]
    for j,m in enumerate(models):
     mu,_,_,st=m.a2c_network(dict(obs=m.norm_obs(inp),rnn_states=states[j],seq_length=1));states[j]=list(st);output.append(mu)
    actions=output[1].clone();actions[:n]=output[0][:n];actions[3*n:4*n]=output[0][3*n:4*n]
    env.step(actions.clamp(-1,1));compute_intermediate_values(env)
    pos=env.object.data.root_pos_w-env.scene.env_origins;palm=env._get_observations()['policy'][:,87:90]
    held=(env.object.data.root_pos_w[:,2]-liftbase>=.05)&(torch.linalg.vector_norm(pos-palm,dim=-1)<.18)&(env._curr_fingertip_distances.sort(-1).values[:,:3].mean(-1)<.1)&(torch.linalg.vector_norm(env.object.data.root_lin_vel_w,dim=-1)<2.)
    run=torch.where(held,run+1,0);longest=torch.maximum(longest,run);near+=(env._keypoints_max_dist<=.1125).long();min_distance=torch.minimum(min_distance,env._curr_fingertip_distances.sort(-1).values[:,:3].mean(-1))
    if step%200==0:print('COMPARE_DROP',step,flush=True)
  report=dict(cases=n,sapg_epoch=checkpoints[1].get('training_epoch'),bc_total_epochs=4200,paired_initial_error=paired_error,bank_observation_error=bank_error,groups={},candidate_ids=ids.tolist(),protocol='16s deterministic rollout; same drop physical states, fixed current goal; zero LSTM comparison plus SAPG saved-memory control; same candidate demo initial baselines. No stochastic policy, no automatic resets, no native 50-goal success rate.')
  for j,name in enumerate(names):
   sl=slice(j*n,(j+1)*n);report['groups'][name]=dict(grasped_30_steps=int((longest[sl]>=30).sum()),reached_initial_tolerance=int((near[sl]>=10).sum()),longest_hold_steps=longest[sl].cpu().tolist(),min_fingertip_distance_m=min_distance[sl].cpu().tolist())
  a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
 finally:
  if env is not None:env.close()
  app.close()
if __name__=='__main__':main()
