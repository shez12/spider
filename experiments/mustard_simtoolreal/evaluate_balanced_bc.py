"""Paired fixed-window BC diagnosis, independent of the live SAPG process."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT.parents[1]/'third_party/simtoolreal'),str(ROOT.parents[1]/'third_party/simtoolreal/rl_games')]


def main():
    from isaaclab.app import AppLauncher
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints',type=Path,nargs='+',required=True)
    p.add_argument('--names',nargs='+',required=True)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--waypoints',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True);p.add_argument('--drop-bank',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--native-cases',type=int,default=256)
    p.add_argument('--frames',type=int,default=960);p.add_argument('--seed',type=int,default=917)
    AppLauncher.add_app_launcher_args(p);a=p.parse_args()
    assert len(a.checkpoints)==len(a.names) and len(set(a.names))==len(a.names)
    a.output.mkdir(parents=True,exist_ok=False);app=AppLauncher(a).app;env=None
    try:
        import numpy as np
        import torch
        from environment import create_env
        from bc_demo_reset import attach_demo_resets
        from isaacsimenvs.tasks.simtoolreal.utils import reset_utils
        from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import compute_intermediate_values
        from isaacsimenvs.tasks.simtoolreal.utils.reward_utils import compute_rewards
        from rl_games.algos_torch import model_builder
        cache=np.load(a.dataset);bank=np.load(a.drop_bank);audit=json.loads(a.audit.read_text())
        assert audit['dataset_sha256']==hashlib.sha256(a.dataset.read_bytes()).hexdigest()
        ids=cache['candidate_ids'];nd=len(ids);nn=a.native_cases;nr=len(bank['candidate_id']);n=nd+nn+nr;m=len(a.names);count=n*m
        rng=np.random.default_rng(a.seed);native_indices=rng.integers(0,nd,nn)
        lookup={int(c):i for i,c in enumerate(ids)}
        indices=np.r_[np.arange(nd),native_indices,[lookup[int(c)] for c in bank['candidate_id']]]
        def configure(c):
            c.seed=a.seed;c.episode_length_s=16.;c.termination.episode_length=960
            c.termination.tolerance_curriculum_interval=10**9;c.reset.table_reset_z_range=0.
            dr=c.domain_randomization;dr.use_action_delay=False;dr.use_obs_delay=False;dr.use_object_state_delay_noise=False
            dr.joint_velocity_obs_noise_std=0.;dr.force_scale=0.;dr.torque_scale=0.
        env=create_env(count,configure=configure);dev=env.device
        mapping=SimpleNamespace(candidate_ids=torch.tensor(ids,device=dev),ids=torch.tensor(np.tile(indices,m),device=dev))
        attach_demo_resets(env,mapping,a.dataset);env.reset()
        pool={int(t['candidate_id']):t for t in json.loads(a.waypoints.read_text())['trajectories']}
        goals=torch.tensor([pool[int(ids[i])]['pos'][-1]+pool[int(ids[i])]['quat_wxyz'][-1] for i in indices],device=dev,dtype=torch.float32)
        goals[:,:3]+=env.scene.env_origins[:n];env.goal_viz.write_root_pose_to_sim(goals,env_ids=torch.arange(n,device=dev))
        native=torch.arange(nd,nd+nn,device=dev);env.cfg.reset.table_reset_z_range=.01
        reset_utils._reset_table_pose(env,native);reset_utils._reset_object_pose(env,native);reset_utils._reset_goal_pose(env,native,'absolute')
        drop=torch.arange(nd+nn,n,device=dev)
        def tensor(k):return torch.tensor(bank[k],device=dev)
        root=tensor('root');root[:,:3]+=env.scene.env_origins[drop]
        goal=tensor('goal');goal[:,:3]+=env.scene.env_origins[drop]
        env.robot.write_joint_state_to_sim(tensor('q'),tensor('v'),env_ids=drop)
        env.object.write_root_state_to_sim(root,env_ids=drop);env.goal_viz.write_root_pose_to_sim(goal,env_ids=drop)
        env._prev_targets[drop]=tensor('previous');env._cur_targets[drop]=tensor('current')
        env.robot.set_joint_position_target(tensor('previous'),env_ids=drop)
        env.scene.write_data_to_sim();env.sim.forward();env.scene.update(0.)
        # Duplicate all initial physical states in simultaneous isolated envs.
        q=env.robot.data.joint_pos[:n].clone();v=env.robot.data.joint_vel[:n].clone()
        previous=env._prev_targets[:n].clone();current=env._cur_targets[:n].clone()
        object_state=env.object.data.root_state_w[:n].clone();object_state[:,:3]-=env.scene.env_origins[:n]
        table_pose=env.table.data.root_state_w[:n,:7].clone();table_pose[:,:3]-=env.scene.env_origins[:n]
        goal_pose=env.goal_viz.data.root_state_w[:n,:7].clone();goal_pose[:,:3]-=env.scene.env_origins[:n]
        for j in range(m):
            ei=torch.arange(j*n,(j+1)*n,device=dev)
            env.robot.write_joint_state_to_sim(q,v,env_ids=ei);env.robot.set_joint_position_target(previous,env_ids=ei)
            env._prev_targets[ei]=previous;env._cur_targets[ei]=current
            for asset,state,full in [(env.object,object_state,True),(env.table,table_pose,False),(env.goal_viz,goal_pose,False)]:
                value=state.clone();value[:,:3]+=env.scene.env_origins[ei]
                if full:asset.write_root_state_to_sim(value,env_ids=ei)
                else:asset.write_root_pose_to_sim(value,env_ids=ei)
            env._table_z_per_env[ei]=env._table_z_per_env[:n].clone()
        env._object_init_z=object_state[:,2].repeat(m)
        env._closest_keypoint_max_dist.fill_(-1);env._closest_fingertip_dist.fill_(-1);env._successes.zero_();env._near_goal_steps.zero_()
        env.scene.write_data_to_sim();env.sim.forward();env.scene.update(0.);compute_intermediate_values(env)
        initial=env._get_observations()['policy'].clone();initial_groups=initial.reshape(m,n,-1)
        initial_error=float((initial_groups-initial_groups[:1]).abs().max());assert initial_error<1e-4,initial_error
        ref=torch.tensor(cache['obs'][:,0],device=dev);demo_error=float((initial[:nd]-ref).abs().max());assert demo_error<1e-4,demo_error
        np.savez_compressed(a.output/'evaluation_initial_states.npz',candidate_ids=ids[indices],
            q=q.cpu().numpy(),v=v.cpu().numpy(),previous=previous.cpu().numpy(),current=current.cpu().numpy(),
            root=object_state.cpu().numpy(),goal=goal_pose.cpu().numpy(),table=table_pose.cpu().numpy(),obs=initial[:n].cpu().numpy())
        # Compute intermediate values once per control step, retain native reward
        # tracker updates, but hold goal fixed and do not automatically reset.
        def no_reset():
            compute_intermediate_values(env);z=torch.zeros(count,device=dev,dtype=torch.bool);return z,z.clone()
        env._get_dones=no_reset
        env._get_rewards=lambda:compute_rewards(env)
        models=[];states=[]
        for path in a.checkpoints:
            ck=torch.load(path,map_location=dev,weights_only=False)
            model=model_builder.ModelBuilder().load(dict(network=ck['network'],model=dict(name='continuous_a2c_logstd'))).build(dict(
                actions_num=29,input_shape=(141,),num_seqs=n,value_size=1,normalize_input=True,normalize_value=True,
                type='extra_param',coef_ids=torch.linspace(50,0,6,device=dev),coef_id_idx=140)).to(dev)
            model.load_state_dict(ck['model']);model.eval();models.append(model)
            states.append([s.to(dev).zero_() for s in model.get_default_rnn_state()])
        run=torch.zeros(count,device=dev,dtype=torch.long);longest=run.clone();near=run.clone();near_run=run.clone();near_longest=run.clone()
        min_z=env.object.data.root_pos_w[:,2].clone();min_goal=torch.full((count,),100.,device=dev)
        tolerance=float(env._current_success_tolerance*env.cfg.reward.keypoint_scale)
        start_obs=initial
        with torch.no_grad():
            for step in range(a.frames):
                obs=start_obs if step==0 else env._get_observations()['policy']
                actions=[]
                for j,model in enumerate(models):
                    inp=torch.cat([obs[j*n:(j+1)*n].clamp(-10,10),torch.zeros((n,1),device=dev)],-1)
                    mu,_,_,st=model.a2c_network(dict(obs=model.norm_obs(inp),rnn_states=states[j],seq_length=1));states[j]=list(st);actions.append(mu)
                env.step(torch.cat(actions).clamp(-1,1))
                pos=env.object.data.root_pos_w-env.scene.env_origins;palm=env._get_observations()['policy'][:,87:90]
                min_z=torch.minimum(min_z,env.object.data.root_pos_w[:,2])
                liftbase=object_state[:,2].repeat(m)+env.scene.env_origins[:,2]
                # Native spawns drop from above the table: use the running low
                # height to measure a subsequent lift instead of spawn height.
                for j in range(m):liftbase[j*n+nd:j*n+nd+nn]=min_z[j*n+nd:j*n+nd+nn]
                held=(env.object.data.root_pos_w[:,2]-liftbase>=.05)&(torch.linalg.vector_norm(pos-palm,dim=-1)<.18)&(env._curr_fingertip_distances.sort(-1).values[:,:3].mean(-1)<.1)&(torch.linalg.vector_norm(env.object.data.root_lin_vel_w,dim=-1)<2.)
                run=torch.where(held,run+1,0);longest=torch.maximum(longest,run)
                close=env._keypoints_max_dist<=tolerance;near+=close.long();near_run=torch.where(close,near_run+1,0);near_longest=torch.maximum(near_longest,near_run)
                min_goal=torch.minimum(min_goal,env._keypoints_max_dist)
                if step%120==0:print('BALANCE_EVAL',step,flush=True)
        success_steps=env.cfg.termination.success_steps
        final_root=env.object.data.root_state_w.clone();final_root[:,:3]-=env.scene.env_origins
        final_goal=env.goal_viz.data.root_state_w[:,:7].clone();final_goal[:,:3]-=env.scene.env_origins
        np.savez_compressed(a.output/'evaluation_final_states.npz',model_names=np.array(a.names),cases_per_model=np.array(n),
            candidate_ids=np.tile(ids[indices],m),q=env.robot.data.joint_pos.cpu().numpy(),v=env.robot.data.joint_vel.cpu().numpy(),
            previous=env._prev_targets.cpu().numpy(),current=env._cur_targets.cpu().numpy(),root=final_root.cpu().numpy(),
            goal=final_goal.cpu().numpy(),obs=env._get_observations()['policy'].cpu().numpy(),
            h0=torch.cat([s[0].transpose(0,1) for s in states]).cpu().numpy(),h1=torch.cat([s[1].transpose(0,1) for s in states]).cpu().numpy())
        def summary(mask,grasp,reach):
            return dict(cases=int(mask.sum()),grasped=int(grasp[mask].sum()),reached=int(reach[mask].sum()),
                grasp_rate=float(grasp[mask].mean()) if mask.any() else None,reach_rate=float(reach[mask].mean()) if mask.any() else None)
        groups=np.array([r['group'] for r in audit['records']]);strata=np.array([r['stratum'] for r in audit['records']])
        report=dict(seed=a.seed,frames=a.frames,paired_initial_observation_max_error=initial_error,demo_observation_max_error=demo_error,
            tolerance_parameter=env._current_success_tolerance,effective_keypoint_tolerance_m=tolerance,success_steps=success_steps,
            protocol='Fixed goal/no automatic resets; native actions/reward tracker; deterministic leader; zero initial LSTM. Native object reset plus demo robot. Not 50-target task success. Grasp is a geometric proxy, not a contact-force certificate.',models={})
        for j,name in enumerate(a.names):
            sl=slice(j*n,(j+1)*n);grasp=(longest[sl]>=30).cpu().numpy();reach=(near[sl]>=success_steps).cpu().numpy()
            rows=dict(checkpoint=str(a.checkpoints[j]),sha256=hashlib.sha256(a.checkpoints[j].read_bytes()).hexdigest(),suites={},groups={},strata={},
                cases=[dict(candidate_id=int(ids[indices[i]]),suite='demo' if i<nd else 'native_object' if i<nd+nn else 'drop',
                    grasped=bool(grasp[i]),reached=bool(reach[i]),longest_hold_steps=int(longest[sl][i]),near_steps=int(near[sl][i]),
                    consecutive_near_steps=int(near_longest[sl][i]),minimum_goal_distance_m=float(min_goal[sl][i])) for i in range(n)])
            for suite,begin,end in [('demo',0,nd),('native_object',nd,nd+nn),('drop',nd+nn,n)]:
                mask=(np.arange(n)>=begin)&(np.arange(n)<end);rows['suites'][suite]=summary(mask,grasp,reach)
            for label,values,dest in [('group',groups,'groups'),('stratum',strata,'strata')]:
                for value in np.unique(values):
                    mask=np.r_[values==value,np.zeros(nn+nr,dtype=bool)];rows[dest][str(int(value))]=summary(mask,grasp,reach)
            report['models'][name]=rows
        # Paired case bootstrap: sampling cases, not frames, preserves pairing.
        report['paired_differences']={};rr=np.random.default_rng(20261103)
        pair=('native_only','mixed_reset') if 'native_only' in report['models'] else ('baseline','balanced')
        if all(name in report['models'] for name in pair):
            report['paired_comparison_names']=list(pair)
            for suite in ('demo','native_object','drop'):
                comparisons={}
                for metric in ('grasped','reached'):
                    va=np.array([r[metric] for r in report['models'][pair[0]]['cases'] if r['suite']==suite],dtype=float)
                    vb=np.array([r[metric] for r in report['models'][pair[1]]['cases'] if r['suite']==suite],dtype=float);delta=vb-va
                    samples=delta[rr.integers(0,len(delta),(10000,len(delta)))].mean(1)
                    comparisons[metric]=dict(balanced_minus_baseline=float(delta.mean()),paired_bootstrap_95=np.quantile(samples,[.025,.975]).tolist(),wins=int((delta>0).sum()),losses=int((delta<0).sum()))
                report['paired_differences'][suite]=comparisons
        (a.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({name:r['suites'] for name,r in report['models'].items()},indent=2),flush=True)
        (a.output/'completed.json').write_text(json.dumps({'cases_per_model':n,'models':a.names})+'\n')
    except BaseException as e:
        (a.output/'failed.json').write_text(json.dumps({'error':repr(e)})+'\n');raise
    finally:
        if env is not None:env.close()
        app.close()


if __name__=='__main__':main()
