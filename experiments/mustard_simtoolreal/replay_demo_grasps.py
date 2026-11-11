"""Scripted replay of the pipeline grasps, placed relative to the bottle, from native random starts of the UNMODIFIED
left SimToolReal env (demo_relative_grasps.py extract -> demos.npz).  No learning.

Per env: native reset, settle, then pick the demo (and bottle half-turn) whose grasp wrist pose, moved with the rigid
transform that takes the demo's initial bottle pose to this env's bottle pose, has an arm IK solution closest to the
native arm configuration.  The arm is driven through joint waypoints with the native 29-D action space, so the
0.0025 rad/step arm limit of the action mapping applies:
  native config -> pre-grasp pose raised by --raise-m -> pre-grasp pose (demo frame lift-PRE) -> every --stride-th
  demo frame up to the end of the demo (IK seeded with the previous waypoint).
Hand targets are the demo's joint targets of the same frame (mirrored, clipped to the left limits), low-passed by the
env's own action filter.  Envs without a feasible demo keep still.  DR, wrench impulses and resets are off.
Run from the original repo root.
"""
import argparse,json,sys,time
from pathlib import Path
PRISTINE=Path('/home/szc/work/newsimtoolreal/simtoolreal');HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(PRISTINE),str(PRISTINE/'rl_games')];sys.path.append(str(HERE))
ARM_STEP=.1*1.5/60  # arm_moving_average * dof_speed_scale * dt


def main():
    from isaaclab.app import AppLauncher
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--demos',type=Path,default=HERE.parents[1]/'outputs/demo_relative_v1/demos.npz')
    p.add_argument('--envs',type=int,default=1024);p.add_argument('--settle',type=int,default=60);p.add_argument('--steps',type=int,default=1500);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--max-tilt-deg',type=float,default=15.);p.add_argument('--pre-frames',type=int,default=50);p.add_argument('--stride',type=int,default=2)
    p.add_argument('--raise-m',type=float,default=.08);p.add_argument('--min-steps',type=int,default=6,help='env steps spent on every demo waypoint at least')
    p.add_argument('--object-dir',type=Path)
    AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.headless=True;a.output.mkdir(parents=True,exist_ok=True)
    app=AppLauncher(a).app;code=1
    try:run(a);code=0
    except BaseException:
        import traceback;traceback.print_exc()
    finally:sys.stdout.flush();sys.stderr.flush();import os;os._exit(code)


def run(a):
    import numpy as np,torch,yaml
    from scipy.spatial import ConvexHull
    from isaaclab.utils.math import matrix_from_quat
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env import SimToolRealEnv
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg
    from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import compute_intermediate_values
    from isaacsimenvs.tasks.simtoolreal.utils.termination_utils import compute_terminations
    from adept_obs import object_dir_of,object_urdf
    from mirror_right_bank import LEFT_URDF,left_canonical_names,mesh_vertices,urdf_limits
    from resolve_arm_config import arm_chain,fk,solve
    from demo_relative_grasps import BASE,C2,mat,inv
    from diverse_resets import capture_states
    from left_reset_bank import signature
    torch.manual_seed(a.seed);np.random.seed(a.seed);V,cal=mesh_vertices(object_dir_of(a.object_dir))
    cfg=SimToolRealEnvCfg();cfg.from_dict(yaml.safe_load((PRISTINE/'isaacsimenvs/cfg/task/SimToolReal.yaml').read_text()))
    N=a.envs;cfg.seed=a.seed;cfg.scene.num_envs=N
    cfg.assets.object_urdf=str(object_urdf(a.object_dir));cfg.assets.object_scale=tuple(cal['policy_object_scale']);cfg.reward.fixed_size=tuple(cal['bbox_dims_m'])
    dr=cfg.domain_randomization;dr.use_action_delay=False;dr.use_obs_delay=False;dr.use_object_state_delay_noise=False
    dr.joint_velocity_obs_noise_std=0.;dr.force_scale=0.;dr.torque_scale=0.
    cfg.episode_length_s=1e5;cfg.termination.episode_length=10**8;cfg.termination.tolerance_curriculum_interval=10**9
    env=SimToolRealEnv(cfg);dev=env.device
    def no_done():
        compute_intermediate_values(env);compute_terminations(env);z=torch.zeros(N,device=dev,dtype=torch.bool);return z,z.clone()
    env._get_dones=no_done;env.reset();l2c=env._perm_lab_to_canon
    hold=torch.zeros(N,29,device=dev);hold[:,7:]=-1.
    for _ in range(a.settle):env.step(hold)
    hull=torch.tensor(V[np.unique(ConvexHull(V).vertices)],device=dev,dtype=torch.float32)
    def measure():
        pos=env.object.data.root_pos_w-env.scene.env_origins;quat=env.object.data.root_quat_w
        return torch.cat([pos,quat],-1),(matrix_from_quat(quat)[:,2,:]@hull.T).min(1).values+pos[:,2]-(env._table_z_per_env+.15)
    root_t,clear0=measure();root=root_t.cpu().numpy().astype(np.float64);q0=env.robot.data.joint_pos[:,l2c].cpu().numpy().astype(np.float64)
    lo=env._joint_lower_canon;hi=env._joint_upper_canon
    # ---- plan (CPU) ----
    t0=time.time();d=np.load(a.demos);left=left_canonical_names();chain=arm_chain(LEFT_URDF,left[:7]);lim=urdf_limits(LEFT_URDF,left)[:7];lim=np.stack([lim[:,0]+np.deg2rad(2),lim[:,1]-np.deg2rad(2)],1)
    assert np.allclose(lo.cpu().numpy(),urdf_limits(LEFT_URDF,left)[:,0],atol=1e-4),'canonical joint order differs from the URDF order used for the demos'
    K=len(d['q']);lf=d['lift_frame'];L=d['length'];Td=mat(d['obj'][:,0].astype(np.float64));Wd=d['wrist'].copy();Wd[...,:3,3]-=BASE;Td[:,:3,3]-=BASE
    To=mat(root);To[:,:3,3]-=BASE;on_table=np.abs(clear0.cpu().numpy())<.02;qa=q0[:,:7]
    best=np.full(N,np.inf);choice=np.full((N,2),-1);rng=np.random.default_rng(a.seed)
    for sym,C in enumerate((np.eye(4),C2)):
        G=np.einsum('nij,kjl->nkil',To,inv(Td@C));tilt=np.degrees(np.arccos(np.clip(G[...,2,2],-1,1)));ni,ki=np.nonzero((tilt<a.max_tilt_deg)&on_table[:,None])
        if not len(ni):continue
        tg=np.einsum('pij,pjk->pik',G[ni,ki],Wd[ki,lf[ki]])
        for trial in range(2):
            seed=qa[ni] if trial==0 else np.clip(qa[ni]+rng.uniform(-1.,1.,(len(ni),7)),lim[:,0],lim[:,1])
            q,ep,er=solve(chain,tg,seed,qa[ni],lim);dist=np.where((ep<2e-3)&(er<1.),np.abs(q-qa[ni]).max(1),np.inf)
            for j in np.flatnonzero(np.isfinite(dist)):
                if dist[j]<best[ni[j]]:best[ni[j]]=dist[j];choice[ni[j]]=(ki[j],sym)
    idx=np.flatnonzero(np.isfinite(best));n=len(idx);k=choice[idx,0];Cs=np.where(choice[idx,1,None,None]==1,C2,np.eye(4));G=To[idx]@inv(Td[k]@Cs)
    tpre=np.maximum(lf[k]-a.pre_frames,0);J=int(np.ceil((L[k]-1-tpre).max()/a.stride))+1;frames=np.minimum(tpre[:,None]+a.stride*np.arange(J)[None],(L[k]-1)[:,None])
    P=np.zeros((n,J+1,7));H=np.zeros((n,J+1,22));err=np.zeros((n,J+1,2))
    T=G@Wd[k,tpre];T[:,2,3]+=a.raise_m;P[:,0],e1,e2=solve(chain,T,qa[idx],qa[idx],lim);err[:,0]=np.c_[e1,e2];H[:,0]=d['ctrl'][k,tpre,7:]
    for j in range(J):
        P[:,j+1],e1,e2=solve(chain,G@Wd[k,frames[:,j]],P[:,j],P[:,j],lim,iters=60 if j==0 else 25);err[:,j+1]=np.c_[e1,e2];H[:,j+1]=d['ctrl'][k,frames[:,j],7:]
    path_ok=(err[...,0].max(1)<5e-3)&(err[...,1].max(1)<2.);jlift=np.array([int(np.argmax(frames[i]>=lf[k[i]]))+1 for i in range(n)])
    print(f'PLAN {time.time()-t0:.0f}s envs {N} on_table {int(on_table.sum())} grasp-feasible {n} path-feasible {int(path_ok.sum())} waypoints {J+1}',flush=True)
    idx,k,P,H,jlift,dist0=idx[path_ok],k[path_ok],P[path_ok],H[path_ok],jlift[path_ok],best[idx][path_ok];n=len(idx);last=P.shape[1]-1
    ix=torch.tensor(idx,device=dev);Pt=torch.tensor(P,device=dev,dtype=torch.float32);Ht=torch.tensor(H,device=dev,dtype=torch.float32);ar=torch.arange(n,device=dev)
    w=torch.zeros(n,device=dev,dtype=torch.long);dwell=torch.zeros(n,device=dev,dtype=torch.long);done_step=torch.full((n,),-1,device=dev,dtype=torch.long);lift_wp_step=torch.full((n,),-1,device=dev,dtype=torch.long)
    first_air=torch.full((N,),-1,device=dev,dtype=torch.long);first_flag=first_air.clone();maxclear=torch.zeros(N,device=dev);jl=torch.tensor(jlift,device=dev)
    rec={x:[] for x in ('q','v','previous','current','root','action')};fixed=capture_states(env,ix)
    for step in range(a.steps):
        prev=env._prev_targets[:,l2c];act=hold.clone()
        delta=Pt[ar,w]-prev[ix,:7];m=delta.abs().max(1).values;act[ix,:7]=(delta*torch.clamp(ARM_STEP/m.clamp(min=1e-9),max=1.)[:,None]/ARM_STEP).clamp(-1,1)
        act[ix,7:]=(2*(Ht[ar,w]-lo[7:])/(hi[7:]-lo[7:])-1).clamp(-1,1)
        s=capture_states(env,ix)  # state before the action: restoring it and applying the action reproduces the step
        for x in ('q','v','previous','current','root'):rec[x].append(s[x].cpu())
        rec['action'].append(act[ix].cpu());env.step(act)
        dwell+=1;adv=(m<=ARM_STEP*1.001)&(dwell>=a.min_steps)&(w<last);finished=(m<=ARM_STEP*1.001)&(w==last)&(done_step<0);done_step[finished]=step
        w[adv]+=1;dwell[adv]=0;hit=(w>=jl)&(lift_wp_step<0);lift_wp_step[hit]=step
        r,c=measure();maxclear=torch.maximum(maxclear,c);air=(c>.03)&(first_air<0);first_air[air]=step;fl=env._lifted_object&(first_flag<0);first_flag[fl]=step
    r,c=measure();c=c.cpu().numpy();fa=first_air.cpu().numpy();ff=first_flag.cpu().numpy();ds=done_step.cpu().numpy();held=c[idx]>.03
    src=d['source'][k];rep=dict(envs=N,on_table=int(on_table.sum()),grasp_feasible=int(np.isfinite(best).sum()),attempted=n,steps=a.steps,
        path_finished=float((ds>=0).mean()),airborne_ever=float((fa[idx]>=0).mean()),airborne_at_end=float(held.mean()),lifted_flag_ever=float((ff[idx]>=0).mean()),
        of_all_envs=dict(airborne_at_end=float(held.sum()/N),lifted_flag_ever=float((ff[idx]>=0).sum()/N),lifted_flag_within_600_steps=float(((ff[idx]>=0)&(ff[idx]<600)).sum()/N)),
        first_airborne_step_median=float(np.median(fa[idx][fa[idx]>=0])) if (fa[idx]>=0).any() else None,lifted_flag_step_median=float(np.median(ff[idx][ff[idx]>=0])) if (ff[idx]>=0).any() else None,
        lifted_flag_within_600_steps=float(((ff[idx]>=0)&(ff[idx]<600)).mean()),arm_distance_rad_median=float(np.median(dist0)),
        idle_envs=dict(n=int(N-n),airborne_ever=float((np.delete(fa,idx)>=0).mean()) if n<N else None),
        by_source={int(s):dict(n=int((src==s).sum()),airborne_at_end=float(held[src==s].mean()),lifted_flag_ever=float((ff[idx][src==s]>=0).mean())) for s in sorted(set(src))})
    print('REPORT',json.dumps(rep),flush=True);(a.output/'report.json').write_text(json.dumps(rep,indent=1)+'\n')
    np.savez_compressed(a.output/'replay.npz',env=idx,demo=k,source=src,arm_distance=dist0,waypoints=P,hand_waypoints=H,lift_waypoint=jlift,start_q=q0[idx],start_root=root[idx],
        first_airborne=fa[idx],first_flag=ff[idx],path_done=ds,lift_waypoint_step=lift_wp_step.cpu().numpy(),final_clearance=c[idx],max_clearance=maxclear.cpu().numpy()[idx],
        goal=fixed['goal'].cpu().numpy(),table=fixed['table'].cpu().numpy(),table_z=fixed['table_z'].cpu().numpy(),
        signature=np.array(signature(PRISTINE/cfg.assets.robot_urdf,cfg.assets.object_urdf,PRISTINE/'isaacsimenvs/cfg/task/SimToolReal.yaml')),
        **{x:torch.stack(v,1).numpy().astype(np.float32) for x,v in rec.items()})


if __name__=='__main__':main()
