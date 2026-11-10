"""Locate the failed transitions between native, near-object, grasp and goal resets."""
import argparse,json,hashlib,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parents[1]
sys.path[:0]=[str(ROOT),str(PROJECT/'third_party/simtoolreal'),str(PROJECT/'third_party/simtoolreal/rl_games')]


def main():
    from isaaclab.app import AppLauncher
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--checkpoints',type=Path,nargs='+',required=True);p.add_argument('--names',nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cases-per-stage',type=int,default=128)
    p.add_argument('--frames',type=int,default=960);p.add_argument('--seed',type=int,default=1103)
    p.add_argument('--native-robot-reset',action='store_true')
    p.add_argument('--coefs',type=float,nargs='+',help='SAPG block coefficient per checkpoint; 50 = leader (default), 0 = last follower (pre-2026-11-05 behaviour)')
    p.add_argument('--obs',choices=('native','adept'),default='native',help='adept: ADEPT observation layout (adept_obs.py, 64 points)')
    p.add_argument('--align-left-physics',action='store_true',help='physics_align.install() before building the env (match training)')
    p.add_argument('--perturb-xy',type=float,default=0.,help='bank suites: uniform +-m planar shift. Object-on-table suites (near_teacher, '
        'recovery) move object and goal together; held suites (grasp, near_goal) move only the goal (moving the object would leave the hand)')
    p.add_argument('--perturb-yaw-deg',type=float,default=0.,help='bank suites: uniform +-deg rotation about the vertical axis through the object '
        'center, applied with the shift (object+goal or goal only, as above)')
    p.add_argument('--perturb-joint',type=float,default=0.,help='bank suites: uniform +-rad on all 29 joint positions and the matching controller targets')
    AppLauncher.add_app_launcher_args(p);a=p.parse_args();assert len(a.names)==len(a.checkpoints)
    a.coefs=a.coefs or [50.]*len(a.checkpoints);assert len(a.coefs)==len(a.checkpoints) and all(c in (50.,40.,30.,20.,10.,0.) for c in a.coefs)
    a.output.mkdir(parents=True,exist_ok=False);app=AppLauncher(a).app;env=None
    try:
        import numpy as np
        import torch
        from environment import create_env
        from diverse_resets import capture_states,restore_states,clear_trackers,hold_actions,environment_signature,table_top_z
        from bc_demo_reset import attach_demo_resets
        from isaacsimenvs.tasks.simtoolreal.utils import reset_utils
        from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import compute_intermediate_values
        from isaacsimenvs.tasks.simtoolreal.utils.reward_utils import compute_rewards
        from rl_games.algos_torch import model_builder
        with np.load(a.bank) as archive:bank={k:archive[k] for k in archive.files}
        assert str(bank['environment_sha256'])==environment_signature()
        records=[r for r in json.loads((a.bank.parent/'validation.json').read_text())['records'] if r['accepted']]
        assert len(records)==len(bank['kind'])
        teacher=np.array([r['origin']=='native_teacher_cache' for r in records]);rng=np.random.default_rng(a.seed)
        suites=[('native',None),('near_teacher',np.flatnonzero((bank['kind']==1)&teacher)),
            ('recovery',np.flatnonzero((bank['kind']==1)&~teacher)),('grasp',np.flatnonzero(bank['kind']==2)),('near_goal',np.flatnonzero(bank['kind']==3))]
        labels=[];entries=[]
        for name,pool in suites:
            sampled=np.full(a.cases_per_stage,-1) if pool is None else rng.choice(pool,min(a.cases_per_stage,len(pool)),replace=False)
            labels.extend([name]*len(sampled));entries.extend(sampled.tolist())
        n=len(entries);names=a.names+['hold_command'];m=len(names);total=n*m
        def configure(c):
            c.seed=a.seed;c.reset.table_reset_z_range=0.;c.episode_length_s=16.;c.termination.episode_length=960
            c.termination.tolerance_curriculum_interval=10**9
            dr=c.domain_randomization;dr.use_action_delay=False;dr.use_obs_delay=False;dr.use_object_state_delay_noise=False
            dr.joint_velocity_obs_noise_std=0.;dr.force_scale=0.;dr.torque_scale=0.
        if a.align_left_physics:
            import physics_align;physics_align.install()
        if a.obs=='adept':
            from adept_obs import create_adept_env;env=create_adept_env(create_env,total,configure,a.output)
        else:env=create_env(total,configure=configure)
        dev=env.device;allids=torch.arange(total,device=dev);D=env.cfg.observation_space
        with np.load(a.dataset) as archive:dataset_ids=archive['candidate_ids']
        mapping=SimpleNamespace(candidate_ids=torch.tensor(dataset_ids,device=dev),ids=torch.tensor(rng.integers(0,len(dataset_ids),total),device=dev))
        attach_demo_resets(env,mapping,a.dataset);env.reset()
        native=torch.tensor([i for i,e in enumerate(entries) if e<0],device=dev)
        env.cfg.reset.table_reset_z_range=.01
        robot_check=None
        if a.native_robot_reset:
            # Reuse the original sampler and verify the actual random draw.
            torch.manual_seed(a.seed+17);rng_state=torch.cuda.get_rng_state(dev)
            limits=env.robot.data.joint_pos_limits[native];default=env.robot.data.default_joint_pos[native]
            expected=default*.9+.1*(limits[:,:,0]+(limits[:,:,1]-limits[:,:,0])*torch.rand_like(default))
            expected_v=torch.empty_like(default).uniform_(-.5,.5);torch.cuda.set_rng_state(rng_state,dev)
            reset_utils._randomize_robot_dof_state(env,native)
            qerr=float((env.robot.data.joint_pos[native]-expected).abs().max());verr=float((env.robot.data.joint_vel[native]-expected_v).abs().max())
            assert qerr<1e-6 and verr<1e-6
            robot_check=dict(formula='0.9*default+0.1*uniform(limits)',velocity_bound=.5,position_error=qerr,velocity_error=verr)
        reset_utils._reset_table_pose(env,native);reset_utils._reset_object_pose(env,native);reset_utils._reset_goal_pose(env,native,'absolute')
        selected=torch.tensor([i for i,e in enumerate(entries) if e>=0],device=dev)
        indices=np.array([e for e in entries if e>=0])
        state={k:torch.tensor(bank[k][indices],device=dev) for k in ('q','v','previous','current','root','goal','table','table_z')}
        if a.perturb_xy or a.perturb_yaw_deg or a.perturb_joint:
            from isaaclab.utils.math import quat_apply,quat_mul
            g=torch.Generator(device=dev).manual_seed(a.seed+29);k=len(indices)
            u=lambda *s:2*torch.rand(*s,generator=g,device=dev)-1
            shift=torch.zeros(k,3,device=dev);shift[:,:2]=a.perturb_xy*u(k,2);yaw=torch.deg2rad(torch.tensor(a.perturb_yaw_deg,device=dev))*u(k)
            qz=torch.stack([torch.cos(yaw/2),torch.zeros_like(yaw),torch.zeros_like(yaw),torch.sin(yaw/2)],-1)
            sel_labels=[labels[i] for i in range(len(entries)) if entries[i]>=0];table_suite=torch.tensor([s in ('near_teacher','recovery') for s in sel_labels],device=dev)
            c=state['root'][:,:3].clone();rot=lambda v:quat_apply(qz,v)
            goal=state['goal'].clone();goal[:,:3]=c+rot(goal[:,:3]-c)+shift;goal[:,3:7]=quat_mul(qz,goal[:,3:7]);state['goal']=goal
            root=state['root'].clone();moved=root.clone()
            moved[:,:3]=c+shift;moved[:,3:7]=quat_mul(qz,root[:,3:7]);moved[:,7:10]=rot(root[:,7:10]);moved[:,10:13]=rot(root[:,10:13])
            state['root']=torch.where(table_suite[:,None],moved,root)
            dq=a.perturb_joint*u(k,29);lo=env._joint_lower_canon;hi=env._joint_upper_canon
            for key in ('q','previous','current'):state[key]=torch.max(torch.min(state[key]+dq,hi),lo)
            perturbation=dict(xy=a.perturb_xy,yaw_deg=a.perturb_yaw_deg,joint=a.perturb_joint,
                object_moved_suites=['near_teacher','recovery'],goal_only_suites=['grasp','near_goal'])
        else:perturbation=None
        restore_states(env,selected,state)
        env.scene.write_data_to_sim();env.sim.forward();env.scene.update(0.)
        base=capture_states(env,torch.arange(n,device=dev))
        for j in range(m):restore_states(env,torch.arange(j*n,(j+1)*n,device=dev),base)
        env.scene.write_data_to_sim();env.sim.forward();env.scene.update(0.);compute_intermediate_values(env)
        obs=env._get_observations()['policy'];error=float((obs.reshape(m,n,-1)-obs[:n]).abs().max());assert error<2e-5,error
        np.savez_compressed(a.output/'initial_states.npz',**{k:v.cpu().numpy() for k,v in base.items()},entries=np.array(entries),suites=np.array(labels))
        def no_done():
            compute_intermediate_values(env);z=torch.zeros(total,device=dev,dtype=torch.bool);return z,z.clone()
        env._get_dones=no_done;env._get_rewards=lambda:compute_rewards(env)
        models=[];states=[]
        for path in a.checkpoints:
            ck=torch.load(path,map_location=dev,weights_only=False)
            if 'model' not in ck:ck=ck[0]
            # Raw SAPG checkpoints (sapg_latest.pth) carry no network spec; use the run's sapg_config.json.
            network=ck.get('network') or json.loads((path.parent/'sapg_config.json').read_text())['params']['network']
            model=model_builder.ModelBuilder().load(dict(network=network,model=dict(name='continuous_a2c_logstd'))).build(dict(
                actions_num=29,input_shape=(D+1,),num_seqs=n,value_size=1,normalize_input=True,normalize_value=True,
                type='extra_param',coef_ids=torch.linspace(50,0,6,device=dev),coef_id_idx=D)).to(dev)
            model.load_state_dict(ck['model']);model.eval();models.append(model);states.append([s.to(dev).zero_() for s in model.get_default_rnn_state()])
        vertices=[]
        for path in (ROOT/'assets/object').glob('right_object_*.obj'):
            if path.name!='right_object_visual.obj':vertices.extend([list(map(float,s.split()[1:4])) for s in path.read_text().splitlines() if s.startswith('v ')])
        vertices=torch.tensor(vertices,device=dev,dtype=torch.float32).T.contiguous()
        def geometric_hold():
            root=env.object.data.root_state_w;q=root[:,3:7];w,x,y,z=q.unbind(-1)
            row=torch.stack([2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)],-1)
            bottom=(row@vertices).min(-1).values+root[:,2]-env.scene.env_origins[:,2]
            pos=root[:,:3]-env.scene.env_origins;palm=env._get_observations()['policy'][:,87:90]
            return (bottom-table_top_z(env)>.02)&((pos-palm).norm(dim=-1)<.18)&(env._curr_fingertip_distances.sort(-1).values[:,:3].mean(-1)<.1)&(env.object.data.root_lin_vel_w.norm(dim=-1)<2.)
        initial_held=geometric_hold();run=torch.zeros(total,device=dev,dtype=torch.long);longest=run.clone();near=run.clone();lost_run=run.clone()
        reached=torch.zeros(total,device=dev,dtype=torch.bool);dropped=reached.clone();terminated=reached.clone()
        first_goal=torch.full((total,),-1,device=dev,dtype=torch.long);minimum=torch.full((total,),100.,device=dev)
        with torch.no_grad():
            for step in range(a.frames):
                ob=env._get_observations()['policy'];actions=[]
                for j,model in enumerate(models):
                    inp=torch.cat([ob[j*n:(j+1)*n].clamp(-10,10),torch.full((n,1),a.coefs[j],device=dev)],-1)
                    mu,_,_,h=model.a2c_network(dict(obs=model.norm_obs(inp),rnn_states=states[j],seq_length=1));states[j]=list(h);actions.append(mu)
                actions.append(hold_actions(env)[-n:]);env.step(torch.cat(actions).clamp(-1,1))
                active=~(reached|terminated);held=geometric_hold()
                run=torch.where(held&active,run+1,0);longest=torch.maximum(longest,run)
                close=env._keypoints_max_dist<=env._current_success_tolerance*env.cfg.reward.keypoint_scale
                near+=(close&active).long();hit=(near>=env.cfg.termination.success_steps)&~reached
                first_goal[hit]=step;reached|=hit
                lost_run=torch.where(~held&active&initial_held,lost_run+1,0);dropped|=lost_run>=10
                minimum=torch.where(active,torch.minimum(minimum,env._keypoints_max_dist),minimum)
                terminated|=active&((env.object.data.root_pos_w[:,2]-env.scene.env_origins[:,2]<.1)|(env._curr_fingertip_distances.max(-1).values>1.5))
                if step%120==0:print('STAGE_EVAL',step,'reached',int(reached.sum()),flush=True)
        labels=np.array(labels);report=dict(cases_per_model=n,frames=a.frames,paired_initial_error=error,native_robot_reset=robot_check,
            bank_sha256=hashlib.sha256(a.bank.read_bytes()).hexdigest(),models={},coefs=dict(zip(a.names,a.coefs)),perturbation=perturbation,
            protocol='Fixed goal, mean action of the SAPG block given by coefs (50 = leader), zero LSTM, no automatic resets. Metrics stop at first success or native physical failure. Hold-command branch validates physical support. Not full 50-goal task success; geometric hold proxy requires mesh bottom >2cm above table.')
        for j,name in enumerate(names):
            sl=slice(j*n,(j+1)*n);h=initial_held[sl].cpu().numpy();g=(longest[sl]>=30).cpu().numpy();r=reached[sl].cpu().numpy();drop=dropped[sl].cpu().numpy();rows={}
            for suite in np.unique(labels):
                mask=labels==suite;rows[suite]=dict(cases=int(mask.sum()),held_initial=int(h[mask].sum()),
                    sustained_hold_before_goal=int(g[mask].sum()),first_goal_reached=int(r[mask].sum()),lost_initial_hold_before_goal=int(drop[mask].sum()))
            report['models'][name]=dict(suites=rows,cases=[dict(suite=str(labels[i]),entry=int(entries[i]),held_initial=bool(h[i]),held_30=bool(g[i]),reached=bool(r[i]),dropped=bool(drop[i]),first_goal_step=int(first_goal[sl][i]),minimum_goal_distance_m=float(minimum[sl][i])) for i in range(n)])
        (a.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
        (a.output/'completed.json').write_text(json.dumps({'cases_per_model':n,'models':names})+'\n')
        print(json.dumps({k:v['suites'] for k,v in report['models'].items()},indent=2),flush=True)
    except BaseException as e:
        (a.output/'failed.json').write_text(json.dumps({'error':repr(e)})+'\n');raise
    finally:
        if env is not None:env.close()
        app.close()


if __name__=='__main__':main()
