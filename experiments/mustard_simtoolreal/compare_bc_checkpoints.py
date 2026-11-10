"""Compare BC checkpoints in parallel from identical cached actor states."""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))


def main():
    from isaaclab.app import AppLauncher
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints',nargs='+',required=True,help='name=checkpoint pairs')
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cases',type=int,default=100)
    p.add_argument('--stochastic-names',nargs='*',default=[],help='Named models use native Normal(mu, exp(logstd)) actions.')
    p.add_argument('--noise-scales',nargs='*',default=[],help='Optional name=scale multipliers for sampled standard deviation.')
    AppLauncher.add_app_launcher_args(p);args=p.parse_args();app=AppLauncher(args).app;env=None
    try:
        import numpy as np,torch
        from environment import create_env
        from bc_demo_reset import attach_demo_resets
        from rl_games.algos_torch import model_builder
        from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import compute_intermediate_values
        pairs=[item.split('=',1) for item in args.checkpoints]
        assert len({name for name,_ in pairs})==len(pairs)
        assert set(args.stochastic_names)<={name for name,_ in pairs}
        noise_scales={name:float(value) for name,value in (item.split('=',1) for item in args.noise_scales)}
        assert set(noise_scales)<=set(args.stochastic_names) and all(value>=0 for value in noise_scales.values())
        data=np.load(args.run/'native_bc_dataset.npz');n=args.cases
        ids=data['candidate_ids'][:n];lengths=data['valid_lengths'][:n];labels=data['actions'][:n]
        count=(len(pairs)+1)*n
        def configure(cfg):
            dr=cfg.domain_randomization;dr.use_action_delay=False;dr.use_obs_delay=False;dr.use_object_state_delay_noise=False
            dr.joint_velocity_obs_noise_std=0.;dr.force_scale=0.;dr.torque_scale=0.;cfg.reset.table_reset_z_range=0.
        env=create_env(count,configure=configure);device=env.device
        mapping=SimpleNamespace(candidate_ids=torch.tensor(ids,device=device),ids=torch.arange(count,device=device)%n)
        reference,restore=attach_demo_resets(env,mapping,args.run/'native_bc_dataset.npz');env.reset()
        pool={t['candidate_id']:t for t in json.loads((ROOT/'waypoints100.json').read_text())['trajectories']}
        sample_ids=torch.arange(count,device=device)%n
        goals=torch.tensor([pool[int(c)]['pos'][-1]+pool[int(c)]['quat_wxyz'][-1] for c in ids],device=device)[sample_ids]
        goals[:,:3]+=env.scene.env_origins;env.goal_viz.write_root_pose_to_sim(goals)
        env.scene.write_data_to_sim();env.sim.forward();env.scene.update(0.)
        def no_dones():
            compute_intermediate_values(env);z=torch.zeros(count,device=device,dtype=torch.bool);return z,z.clone()
        env._get_dones=no_dones;env._get_rewards=lambda:torch.zeros(count,device=device)
        models=[];states=[];metadata={}
        for name,path in pairs:
            ck=torch.load(path,map_location=device,weights_only=False)
            assert ck['action_contract']=='simtoolreal_native_v1'
            model=model_builder.ModelBuilder().load(dict(network=ck['network'],model=dict(name='continuous_a2c_logstd'))).build(dict(actions_num=29,input_shape=(141,),num_seqs=n,value_size=1,normalize_input=True,normalize_value=True,type='extra_param',coef_ids=torch.linspace(50,0,6,device=device),coef_id_idx=140)).to(device)
            model.load_state_dict(ck['model']);model.eval();models.append(model)
            states.append(tuple(s.to(device).zero_() for s in model.get_default_rnn_state()))
            metadata[name]=dict(path=path,epoch=ck['bc_epochs'],initial_checkpoint_epoch=ck.get('initial_checkpoint_epoch'),rnn_context=ck.get('rnn_context','window'))
        hold_run=torch.zeros(count,device=device,dtype=torch.long);longest=hold_run.clone();near=hold_run.clone()
        initial_z=torch.tensor([pool[int(c)]['initial_pose'][2] for c in ids],device=device)[sample_ids]
        first=env._get_observations()['policy'].clamp(-10,10)
        initial_error=float((first-reference()).abs().max());assert initial_error<1e-4,initial_error
        valid=torch.tensor(lengths,device=device)[sample_ids]
        with torch.no_grad():
            for step in range(int(max(lengths))):
                obs=env._get_observations()['policy'].clamp(-10,10);actions=[]
                for j,model in enumerate(models):
                    inp=torch.cat([obs[j*n:(j+1)*n],torch.zeros((n,1),device=device)],-1)
                    mu,logstd,_,states[j]=model.a2c_network(dict(obs=model.norm_obs(inp),rnn_states=states[j],seq_length=1))
                    action=mu+torch.randn_like(mu)*logstd.exp()*noise_scales.get(pairs[j][0],1.) if pairs[j][0] in args.stochastic_names else mu
                    actions.append(action.clamp(-1,1))
                actions.append(torch.tensor(labels[:,step],device=device));env.step(torch.cat(actions))
                post=env._get_observations()['policy'];position=env.object.data.root_pos_w-env.scene.env_origins
                held=(position[:,2]-initial_z>=.05)&(torch.linalg.vector_norm(position-post[:,87:90],dim=-1)<.18)
                held&=env._curr_fingertip_distances.sort(-1).values[:,:3].mean(-1)<.1
                held&=torch.linalg.vector_norm(env.object.data.root_lin_vel_w,dim=-1)<2.
                active=step<valid;hold_run=torch.where(held&active,hold_run+1,0);longest=torch.maximum(longest,hold_run)
                near+=((env._keypoints_max_dist<=.1125)&active).long()
                if step%100==0:print('CHECKPOINT_COMPARISON',step,int(max(lengths)),flush=True)
        report=dict(cases=n,stochastic_names=args.stochastic_names,noise_scales=noise_scales,initial_observation_max_error=initial_error,initialization='cached_actor_state_object_velocity_zero',models=metadata,groups={})
        for j,name in enumerate([name for name,_ in pairs]+['oracle']):
            sl=slice(j*n,(j+1)*n);report['groups'][name]=dict(held_lift_30_steps=int((longest[sl]>=30).sum()),reached_initial_tolerance=int((near[sl]>=10).sum()),candidate_ids=ids.tolist(),hold_pass=(longest[sl]>=30).cpu().tolist(),reaching_pass=(near[sl]>=10).cpu().tolist())
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({name:{k:v for k,v in row.items() if not isinstance(v,list)} for name,row in report['groups'].items()},indent=2),flush=True)
        restore()
    finally:
        if env is not None:env.close()
        app.close()


if __name__=='__main__':main()
