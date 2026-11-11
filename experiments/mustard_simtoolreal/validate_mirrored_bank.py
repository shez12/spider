"""Physically validate a mirror_right_bank.py bank in the original left-hand SimToolReal env and keep the survivors.

Same procedure as the right-hand build (build_diverse_reset_bank.py): restore each state, hold the stored targets for
--settle steps, capture the settled state, restore it and hold for another --hold steps.  A state is accepted when
  all kinds : finite, joints within limits, object speed < 2 m/s, joint speed < 8 rad/s, object still on/above the table
  kind 2, 3 : the object stays off the table (lowest mesh point > 1 cm), near the fingers and slow for >= 25 of the
              --hold steps, and does not sink more than 3 cm.
Accepted *settled* states (simulator states of the left env) are written to <output>/reset_bank.npz in
left_reset_bank.py format.  Per-state measurements go to <output>/states.npz so criteria can be changed offline.

--relax-limits widens the left hand joint limits to the ones stored in the bank (limit_lower/limit_upper, i.e. the
right-hand abduction range).  That is NOT the original left-hand setting.
--fk-check <geometry_states.npz of the right-hand build> compares simulator link positions of the mirrored joints
with the mirrored right-hand link positions (needs --relax-limits to be exact for the hand).
Run from the original repo root (relative asset paths).
"""
import argparse,json,sys
from pathlib import Path
PRISTINE=Path('/home/szc/work/newsimtoolreal/simtoolreal');HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(PRISTINE),str(PRISTINE/'rl_games')];sys.path.append(str(HERE))


def main():
    from isaaclab.app import AppLauncher
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bank',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--envs',type=int,default=1024)
    p.add_argument('--settle',type=int,default=30);p.add_argument('--hold',type=int,default=30);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--relax-limits',action='store_true');p.add_argument('--fk-check',type=Path);p.add_argument('--object-dir',type=Path)
    p.add_argument('--wrench',action='store_true',help='keep the training random object force/torque impulses (also before lift)')
    AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.headless=True;a.output.mkdir(parents=True,exist_ok=True)
    app=AppLauncher(a).app
    code=1
    try:run(a);code=0
    except BaseException:
        import traceback;traceback.print_exc()
    finally:
        sys.stdout.flush();sys.stderr.flush();import os;os._exit(code)


def run(a):
    import numpy as np,torch,yaml
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation as Rot
    from isaaclab.utils.math import matrix_from_quat
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env import SimToolRealEnv
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg
    from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import compute_intermediate_values
    from isaacsimenvs.tasks.simtoolreal.utils.termination_utils import compute_terminations
    from diverse_resets import capture_states,restore_states,hold_actions
    from left_reset_bank import signature,FIELDS
    from adept_obs import object_dir_of,object_urdf
    from mirror_right_bank import mesh_vertices,mirror_joints,joint_maps,M
    torch.manual_seed(a.seed);d=np.load(a.bank);total=len(d['kind']);N=min(a.envs,total)
    V,cal=mesh_vertices(object_dir_of(a.object_dir))
    cfg=SimToolRealEnvCfg();cfg.from_dict(yaml.safe_load((PRISTINE/'isaacsimenvs/cfg/task/SimToolReal.yaml').read_text()))
    cfg.seed=a.seed;cfg.scene.num_envs=N
    cfg.assets.object_urdf=str(object_urdf(a.object_dir));cfg.assets.object_scale=tuple(cal['policy_object_scale']);cfg.reward.fixed_size=tuple(cal['bbox_dims_m'])
    dr=cfg.domain_randomization;dr.use_action_delay=False;dr.use_obs_delay=False;dr.use_object_state_delay_noise=False
    dr.joint_velocity_obs_noise_std=0.
    if a.wrench:dr.force_only_when_lifted=False;dr.torque_only_when_lifted=False  # training-scale impulses from the first step
    else:dr.force_scale=0.;dr.torque_scale=0.
    print('wrench',a.wrench,dr.force_scale,dr.torque_scale,dr.force_prob_range,dr.torque_prob_range,flush=True)
    cfg.episode_length_s=1e5;cfg.termination.episode_length=10**8;cfg.termination.tolerance_curriculum_interval=10**9
    env=SimToolRealEnv(cfg);dev=env.device
    sig=signature(PRISTINE/cfg.assets.robot_urdf,cfg.assets.object_urdf,PRISTINE/'isaacsimenvs/cfg/task/SimToolReal.yaml')
    assert str(d['signature'])==sig,'bank was mirrored for a different environment'
    def no_done():
        compute_intermediate_values(env);compute_terminations(env);z=torch.zeros(N,device=dev,dtype=torch.bool);return z,z.clone()
    env._get_dones=no_done;env.reset();ix=torch.arange(N,device=dev);l2c=env._perm_lab_to_canon;c2l=env._perm_canon_to_lab
    if a.relax_limits:
        lim=env.robot.data.joint_pos_limits.clone();lim[:,:,0]=torch.tensor(d['limit_lower'],device=dev)[c2l];lim[:,:,1]=torch.tensor(d['limit_upper'],device=dev)[c2l]
        env.robot.write_joint_position_limit_to_sim(lim,warn_limit_violation=False);lim=env.robot.data.joint_pos_limits
        env._joint_lower_canon=lim[0,:,0][l2c];env._joint_upper_canon=lim[0,:,1][l2c]
        env._arm_lower=lim[:,env._arm_joint_ids,0];env._arm_upper=lim[:,env._arm_joint_ids,1];env._hand_lower=lim[:,env._hand_joint_ids,0];env._hand_upper=lim[:,env._hand_joint_ids,1]
        assert torch.allclose(env._joint_lower_canon,torch.tensor(d['limit_lower'],device=dev),atol=1e-5)
    lo=env._joint_lower_canon;hi=env._joint_upper_canon
    print('LIMITS_DEG',{n:[round(float(x),1) for x in np.rad2deg([lo[i].item(),hi[i].item()])] for i,n in enumerate(np.array(env.robot.joint_names)[l2c.cpu().numpy()]) if 'AA' in n},flush=True)
    def sync():env.scene.write_data_to_sim();env.sim.forward();env.scene.update(0.)
    report=dict(bank=str(a.bank),relax_limits=bool(a.relax_limits),settle=a.settle,hold=a.hold,envs=N)
    if a.fk_check:
        g=np.load(a.fk_check);rng=np.random.default_rng(a.seed);rows=rng.choice(len(g['q']),N,replace=False);_,_,perm=joint_maps()
        q=torch.tensor(mirror_joints(g['q'][rows],perm,True),device=dev,dtype=torch.float32)
        env.robot.write_joint_state_to_sim(q[:,c2l],torch.zeros_like(q),env_ids=ix);sync()
        body=(env.robot.data.body_link_state_w[:,:,:3]-env.scene.env_origins[:,None,:]).cpu().numpy();left=list(env.robot.body_names);right=[str(x) for x in g['body_names']]
        pairs=[(left.index(n.replace('right_','left_')),i,n) for i,n in enumerate(right) if n.replace('right_','left_') in left]
        err=np.stack([np.linalg.norm(body[:,l]-g['body_positions'][rows][:,r]@M,axis=1) for l,r,_ in pairs],1)
        report['fk_check']=dict(states=int(N),links=len(pairs),unmatched=[n for n in right if n.replace('right_','left_') not in left],max_m=float(err.max()),median_m=float(np.median(err)),
            per_link_max_mm={n:round(float(err[:,k].max())*1000,3) for k,(_,_,n) in enumerate(pairs)})
        print('FK_CHECK',json.dumps(report['fk_check']),flush=True)
    hull=torch.tensor(V[np.unique(ConvexHull(V).vertices)],device=dev,dtype=torch.float32);hand=env.robot.find_bodies('iiwa14_link_7')[0][0]
    def measure():
        pos=env.object.data.root_pos_w-env.scene.env_origins;quat=env.object.data.root_quat_w
        low=(matrix_from_quat(quat)[:,2,:]@hull.T).min(1).values+pos[:,2]-(env._table_z_per_env+.15)
        tips=env.robot.data.body_link_state_w[:,env._fingertip_body_ids,:3]-env.scene.env_origins[:,None,:]
        h=env.robot.data.body_link_state_w[:,hand,:7].clone();h[:,:3]-=env.scene.env_origins
        return dict(obj=torch.cat([pos,quat],-1),clearance=low,tips=tips,hand=h,speed=env.object.data.root_lin_vel_w.norm(dim=-1))
    rec={};saved={k:[] for k in FIELDS}
    def put(name,value,count):rec.setdefault(name,[]).append(value[:count].cpu().numpy())
    for offset in range(0,total,N):
        idx=np.arange(offset,offset+N)%total;count=min(N,total-offset)
        restore_states(env,ix,{k:torch.tensor(d[k][idx],device=dev) for k in FIELDS});sync()
        m=measure()
        for k in ('obj','clearance','tips','hand'):put('start_'+k,m[k],count)
        for _ in range(a.settle):env.step(hold_actions(env))
        settled=capture_states(env,ix);m=measure()
        for k in ('obj','clearance','tips','hand'):put('settled_'+k,m[k],count)
        restore_states(env,ix,settled);sync()
        minz=settled['root'][:,2].clone();maxspeed=torch.zeros(N,device=dev);run_len=torch.zeros(N,device=dev,dtype=torch.long);minclear=torch.full((N,),1e3,device=dev)
        for _ in range(a.hold):
            env.step(hold_actions(env));m=measure();minz=torch.minimum(minz,m['obj'][:,2]);maxspeed=torch.maximum(maxspeed,m['speed']);minclear=torch.minimum(minclear,m['clearance'])
            near=(m['tips']-m['obj'][:,None,:3]).norm(dim=-1).sort(-1).values[:,:3].mean(-1)
            run_len=torch.where((m['clearance']>.01)&(near<.1)&(m['speed']<.5),run_len+1,0)
        final=capture_states(env,ix)
        for k in ('obj','clearance','tips','hand'):put('final_'+k,m[k],count)
        finite=torch.isfinite(torch.cat([final['q'],final['v'],final['root']],-1)).all(-1)
        basic=finite&(final['q']>=lo-1e-3).all(-1)&(final['q']<=hi+1e-3).all(-1)&(final['v'].abs().max(-1).values<8)&(maxspeed<2.)&(minz>.4)&(settled['root'][:,:2].abs()<.45).all(-1)
        for name,value in [('basic',basic),('run',run_len),('maxspeed',maxspeed),('minz',minz),('minclear',minclear),('settled_z',settled['root'][:,2]),('settled_q',settled['q']),('final_q',final['q'])]:put(name,value,count)
        for k in FIELDS:saved[k].append(settled[k][:count].cpu().numpy())
        print('VALIDATE_BATCH',offset,count,flush=True)
    rec={k:np.concatenate(v) for k,v in rec.items()};saved={k:np.concatenate(v) for k,v in saved.items()};kind=d['kind']
    held=(rec['run']>=min(25,a.hold))&(rec['minz']>=rec['settled_z']-.03);accept=rec['basic']&np.where(kind>=2,held,True)
    def relative(tag):  # object pose in the hand (iiwa link 7) frame
        Rh=Rot.from_quat(rec[tag+'_hand'][:,[4,5,6,3]]);Ro=Rot.from_quat(rec[tag+'_obj'][:,[4,5,6,3]]);return Rh.inv().apply(rec[tag+'_obj'][:,:3]-rec[tag+'_hand'][:,:3]),Rh.inv()*Ro
    p0,r0=relative('start');p1,r1=relative('settled');p2,r2=relative('final')
    rec['slip_settle_m']=np.linalg.norm(p1-p0,axis=1);rec['slip_settle_deg']=np.rad2deg((r0.inv()*r1).magnitude())
    rec['slip_total_m']=np.linalg.norm(p2-p0,axis=1);rec['slip_total_deg']=np.rad2deg((r0.inv()*r2).magnitude());rec['accept']=accept;rec['held']=held;rec['kind']=kind
    np.savez_compressed(a.output/'states.npz',**rec)
    pct=lambda x:[round(float(v),4) for v in np.percentile(x,[50,90])] if len(x) else None
    report['by_kind']={str(k):dict(states=int((kind==k).sum()),basic=int(rec['basic'][kind==k].sum()),held=int(held[kind==k].sum()),accepted=int(accept[kind==k].sum()),
        airborne_start=int((rec['start_clearance'][kind==k]>.01).sum()),airborne_final=int((rec['final_clearance'][kind==k]>.01).sum()),
        slip_total_m_p50_p90=pct(rec['slip_total_m'][kind==k]),slip_total_deg_p50_p90=pct(rec['slip_total_deg'][kind==k]),
        accepted_slip_m_p50_p90=pct(rec['slip_total_m'][(kind==k)&accept]),accepted_slip_deg_p50_p90=pct(rec['slip_total_deg'][(kind==k)&accept])) for k in (1,2,3)}
    report['accepted']=int(accept.sum());report['states']=int(total)
    np.savez_compressed(a.output/'reset_bank.npz',**{k:v[accept] for k,v in saved.items()},kind=kind[accept],source=np.column_stack([d['source'][accept],np.flatnonzero(accept)]),
        signature=np.array(sig),format_version=np.array(1),relaxed_limits=np.array(bool(a.relax_limits)),limit_lower=lo.cpu().numpy(),limit_upper=hi.cpu().numpy())
    (a.output/'reset_bank.json').write_text(json.dumps(report,indent=1)+'\n');print('VALIDATE_REPORT',json.dumps(report['by_kind'],indent=1),flush=True)
    env.close()


if __name__=='__main__':main()
