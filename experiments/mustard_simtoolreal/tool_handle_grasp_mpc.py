"""Sampling-based MPC (MPPI) handle grasps for SimToolReal's NATIVE procedural tools (handle + head primitives) in the
UNMODIFIED env (left KUKA+Sharpa, /home/szc/work/newsimtoolreal), from native random initial states.

Same MPPI loop as left_grasp_mpc.py (G start states x S samples, native 29-D actions, restore/rollout/reweight), with
three differences:
  * The grasp must be on the handle. The procedural URDF puts the handle centred at the object origin with its axis
    along +x and the head attached at the +x end; the env's object bounding box (object_scales obs, reward keypoints)
    is the handle's box. The cost uses the fingertip distance to the SURFACE of that handle box (thumb + the two
    closest other fingertips) and penalises fingertips close to the head box.
  * Lift is the clearance of the object's LOWEST point above the table top (handle and head geometry), so standing a
    long tool upright does not count as lifting it. Centre height only enters as a small shaping term.
  * Envs get pool objects by env index, so the S samples of a group are chosen among envs that share one object.

Success is evaluated per executed step and reported three ways (the per-step arrays are saved for re-evaluation):
lifted (clearance), lifted while the thumb and >=2 other fingertips are within --tip-tol of the handle box, and the
latter with no fingertip within --head-tol of the head box. Run from the original repo root (relative asset paths).
"""
import argparse,json,shutil,sys,time
from pathlib import Path
PRISTINE=Path('/home/szc/work/newsimtoolreal/simtoolreal');HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(PRISTINE),str(PRISTINE/'rl_games')];sys.path.append(str(HERE))
TASK_YAML=PRISTINE/'isaacsimenvs/cfg/task/SimToolReal.yaml'


def urdf_parts(path):
    """Corner/rim points (K,3) of every <collision> primitive in the object frame; part 0 is the handle, part 1 the head."""
    import numpy as np,xml.etree.ElementTree as ET
    def rot(r,p,y):
        cr,sr,cp,sp,cy,sy=np.cos(r),np.sin(r),np.cos(p),np.sin(p),np.cos(y),np.sin(y)
        return np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],[sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]])
    parts=[]
    for c in ET.parse(path).getroot().iter('collision'):
        o=c.find('origin');get=lambda k:[float(x) for x in (o.get(k,'0 0 0') if o is not None else '0 0 0').split()]
        box=c.find('geometry/box');cyl=c.find('geometry/cylinder')
        if box is not None:
            half=np.array([float(x) for x in box.get('size').split()])/2
            pts=np.array([[i,j,k] for i in (-1,1) for j in (-1,1) for k in (-1,1)])*half
        else:
            assert cyl is not None,f'unsupported collision geometry in {path}'
            L=float(cyl.get('length'));R=float(cyl.get('radius'));ang=np.linspace(0,2*np.pi,16,endpoint=False)
            pts=np.array([[R*np.cos(t),R*np.sin(t),s*L/2] for s in (-1,1) for t in ang])
        parts.append(pts@rot(*get('rpy')).T+np.array(get('xyz')))
    assert len(parts) in (1,2),f'expected handle (+head) in {path}'
    return parts


def tool_cfg(a,num_envs):
    import yaml
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg
    cfg=SimToolRealEnvCfg();cfg.from_dict(yaml.safe_load(TASK_YAML.read_text()))
    cfg.seed=a.seed;cfg.scene.num_envs=num_envs
    cfg.assets.handle_head_types=tuple(a.tool_types);cfg.assets.num_assets_per_type=a.assets_per_type
    return cfg


class ToolGeometry:
    """Per-env handle box, head box and hull points of the procedural tool actually spawned in each env."""
    def __init__(self,env):
        import numpy as np,torch
        dev=env.device;self.env=env;self.paths=[Path(p) for p in env._object_urdf_paths];P=len(self.paths)
        parts=[urdf_parts(p) for p in self.paths];K=max(sum(len(x) for x in q) for q in parts)
        pts=np.zeros((P,K,3));head_lo=np.zeros((P,3));head_hi=np.zeros((P,3));has_head=np.zeros(P,bool);half=np.zeros((P,3))
        for i,q in enumerate(parts):
            allp=np.concatenate(q);pts[i]=np.concatenate([allp,np.repeat(allp[:1],K-len(allp),0)])
            half[i]=np.abs(q[0]).max(0);assert np.allclose(q[0].max(0),-q[0].min(0),atol=1e-6),'handle is not centred at the origin'
            if len(q)==2:has_head[i]=True;head_lo[i]=q[1].min(0);head_hi[i]=q[1].max(0)
        self.asset=env._object_asset_index_per_env.clone();t=lambda x:torch.tensor(x,device=dev,dtype=torch.float32)
        self.half=t(half)[self.asset];self.pts=t(pts)[self.asset];self.head_lo=t(head_lo)[self.asset];self.head_hi=t(head_hi)[self.asset]
        self.has_head=torch.tensor(has_head,device=dev)[self.asset]
        # The env's own bounding box (policy object scale x object_base_size) must be the handle box parsed from the URDF.
        env_half=env._object_scale_per_env*env.cfg.reward.object_base_size*.5
        assert torch.allclose(env_half,self.half,atol=1e-5),'env object scale does not match the handle box of its URDF'
        names=[env.robot.body_names[i] for i in env._fingertip_body_ids];self.finger_names=names
        self.thumb=[i for i,n in enumerate(names) if 'thumb' in n.lower()];assert len(self.thumb)==1,names
        self.others=[i for i in range(len(names)) if i not in self.thumb]

    def measure(self):
        """clearance of the lowest point above z=0 of the env frame, centre z, fingertip->handle box, fingertip->head box, palm->handle box."""
        import torch
        env=self.env;org=env.scene.env_origins;pos=env.object.data.root_pos_w-org;w,x,y,z=env.object.data.root_quat_w.unbind(-1)
        R=torch.stack([1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)],-1).view(-1,3,3)
        lowest=torch.einsum('nkj,nj->nk',self.pts,R[:,2]).min(1).values+pos[:,2]
        body=env.robot.data.body_state_w
        world=torch.cat([body[:,env._fingertip_body_ids,:3],body[:,env._palm_body_id,:3].unsqueeze(1)],1)-org.unsqueeze(1)
        local=torch.einsum('nij,nfi->nfj',R,world-pos.unsqueeze(1))
        d_handle=(local.abs()-self.half.unsqueeze(1)).clamp(min=0).norm(dim=-1)
        d_head=torch.maximum(self.head_lo.unsqueeze(1)-local,local-self.head_hi.unsqueeze(1)).clamp(min=0).norm(dim=-1)
        d_head=torch.where(self.has_head.unsqueeze(1),d_head,torch.ones_like(d_head))
        return lowest,pos[:,2],d_handle[:,:-1],d_head[:,:-1],d_handle[:,-1]


def main():
    from isaaclab.app import AppLauncher
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tool-types',nargs='+',default=['hammer','screwdriver'],help='SimToolReal handle_head_types to draw the pool from')
    p.add_argument('--assets-per-type',type=int,default=2,help='objects per size distribution (a type can have a cuboid and a cylinder distribution)')
    p.add_argument('--groups',type=int,default=8,help='start states; a multiple of the pool size, split evenly over the objects')
    p.add_argument('--samples',type=int,default=256)
    p.add_argument('--horizon',type=int,default=40);p.add_argument('--execute',type=int,default=10);p.add_argument('--steps',type=int,default=360)
    p.add_argument('--iterations',type=int,default=2);p.add_argument('--temperature',type=float,default=.05)
    p.add_argument('--arm-sigma',type=float,default=.6);p.add_argument('--hand-sigma',type=float,default=.35);p.add_argument('--knot',type=int,default=5)
    p.add_argument('--settle',type=int,default=60);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--lift-weight',type=float,default=4.);p.add_argument('--centre-weight',type=float,default=1.)
    p.add_argument('--head-weight',type=float,default=10.);p.add_argument('--head-margin',type=float,default=.03)
    p.add_argument('--clear',type=float,default=.03,help='lowest-point clearance above the table that counts as lifted [m]')
    p.add_argument('--tip-tol',type=float,default=.03,help='fingertip-link distance to the handle box that counts as on the handle [m]')
    p.add_argument('--head-tol',type=float,default=.015,help='fingertip-link distance to the head box that counts as touching the head [m]')
    AppLauncher.add_app_launcher_args(p);a=p.parse_args()
    a.output=a.output.resolve();assert not a.output.exists(),'Use a fresh output directory.';a.output.mkdir(parents=True)
    app=AppLauncher(a).app;code=0
    try:run(a)
    except BaseException:
        import traceback;traceback.print_exc();code=1
    finally:sys.stdout.flush();sys.stderr.flush();import os;os._exit(code)


def run(a):
    import numpy as np,torch
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env import SimToolRealEnv
    from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import compute_intermediate_values
    from isaacsimenvs.tasks.simtoolreal.utils.termination_utils import compute_terminations
    from diverse_resets import capture_states,restore_states,table_top_z
    torch.manual_seed(a.seed)
    (a.output/'run_arguments.json').write_text(json.dumps(vars(a),indent=1,default=str)+'\n')
    G,S=a.groups,a.samples;N=G*S;cfg=tool_cfg(a,N)
    dr=cfg.domain_randomization;dr.use_action_delay=False;dr.use_obs_delay=False;dr.use_object_state_delay_noise=False
    dr.joint_velocity_obs_noise_std=0.;dr.force_scale=0.;dr.torque_scale=0.
    cfg.episode_length_s=1e5;cfg.termination.episode_length=10**8;cfg.termination.tolerance_curriculum_interval=10**9
    env=SimToolRealEnv(cfg);dev=env.device;geo=ToolGeometry(env);P=len(geo.paths)
    (a.output/'objects').mkdir()
    for i,path in enumerate(geo.paths):shutil.copy(path,a.output/'objects'/f'{i:03d}_{path.name}')
    def no_done():
        # Vendor bookkeeping (termination reasons, goal trackers) for logging; never terminate or reset.
        compute_intermediate_values(env);compute_terminations(env);z=torch.zeros(N,device=dev,dtype=torch.bool);return z,z.clone()
    env._get_dones=no_done
    # Groups: envs sharing one pool object, S per group, G/P groups per object.
    assert G%P==0,f'--groups must be a multiple of the pool size {P}'
    counts=torch.bincount(geo.asset,minlength=P);assert bool((counts==N//P).all()),f'uneven object assignment {counts.tolist()}'
    perm=torch.argsort(geo.asset,stable=True);group=torch.empty(N,device=dev,dtype=torch.long);group[perm]=torch.arange(N,device=dev)//S
    first=perm[torch.arange(G,device=dev)*S];assert bool((geo.asset==geo.asset[first][group]).all())
    by_group=lambda x:x[perm].view(G,S,*x.shape[1:])
    env.reset()
    def broadcast(state):restore_states(env,torch.arange(N,device=dev),{k:v[group] for k,v in state.items()});env.scene.write_data_to_sim();env.sim.forward();env.scene.update(0.)
    # Settle each group's native start (object dropped from 0.25 m) with the arm still and the hand open.
    broadcast(capture_states(env,first))
    hold=torch.zeros(N,29,device=dev);hold[:,7:]=-1.
    for _ in range(a.settle):env.step(hold)
    start=capture_states(env,first);broadcast(start);top=table_top_z(env)[first]
    lowest,centre,_,_,_=geo.measure();rest_centre=centre[first].clone();rest_clear=(lowest[first]-top)
    names=[p.name for p in geo.paths];group_asset=geo.asset[first].cpu().numpy()
    print('TOOLS',json.dumps(dict(pool=names,group_asset=group_asset.tolist(),handle_box_m=(2*geo.half[first]).cpu().numpy().round(4).tolist(),
        rest_clearance_m=rest_clear.cpu().numpy().round(4).tolist(),fingertips=geo.finger_names)),flush=True)
    # A native start can roll off the table while settling (cylinders); such groups are kept but never count.
    on_table=(rest_centre>top-.05).cpu().numpy();print('START_ON_TABLE',on_table.astype(int).tolist(),flush=True)
    assert float(rest_clear.abs()[torch.tensor(on_table,device=dev)].max())<.01,'settled object is not resting on the table: geometry/asset mapping is wrong'
    def metrics():
        lowest,centre,d_handle,d_head,d_palm=geo.measure()
        clear=(lowest-top[group]).clamp(min=0.);rise=(centre-rest_centre[group]).clamp(min=0.)
        fell=centre<top[group]-.05
        return clear,rise,d_handle,d_head,d_palm,fell
    def step_cost():
        clear,rise,d_handle,d_head,d_palm,fell=metrics()
        grasp=(d_handle[:,geo.thumb[0]]+d_handle[:,geo.others].sort(-1).values[:,:2].sum(-1))/3
        head=(a.head_margin-d_head).clamp(min=0.).sum(-1)
        cost=grasp+.3*d_palm-a.lift_weight*clear.clamp(max=.2)-a.centre_weight*rise.clamp(max=.2)+a.head_weight*head+5.*fell.float()
        return cost,(clear,rise,d_handle,d_head,d_palm)
    H,C,K=a.horizon,a.execute,a.knot;knots=H//K+1
    sigma=torch.tensor([a.arm_sigma]*7+[a.hand_sigma]*22,device=dev)
    U=torch.zeros(G,H,29,device=dev);U[:,:,7:]=-1.
    def noise():
        k=torch.randn(N,knots,29,device=dev)*sigma
        t=torch.arange(H,device=dev)/K;i=t.floor().long().clamp(max=knots-2);f=(t-i)[None,:,None]
        e=k[:,i]*(1-f)+k[:,i+1]*f;e[first]=0.;return e   # one env of each group keeps the nominal
    executed=[];actions=[];log=[];state=start;t0=time.monotonic()
    for t in range(0,a.steps,C):
        for _ in range(a.iterations):
            broadcast(state);eps=noise();seq=(U[group]+eps).clamp(-1,1);cost=torch.zeros(N,device=dev)
            for h in range(H):
                env.step(seq[:,h]);c,_=step_cost();cost+=c*(1.+2.*(h==H-1))
            cg=by_group(cost);w=torch.softmax(-(cg-cg.min(1,keepdim=True).values)/(a.temperature*cg.std(1,keepdim=True).clamp_min(1e-6)),1)
            U=(w[:,:,None,None]*by_group(seq)).sum(1)
        broadcast(state)
        for h in range(C):
            env.step(U[group][:,h]);executed.append({k:v.cpu() for k,v in capture_states(env,first).items()});actions.append(U[:,h].cpu())
            _,m=step_cost();log.append([x[first].cpu() for x in m])
        state=capture_states(env,first);U=torch.cat([U[:,C:],U[:,-1:].repeat(1,C,1)],1)
        clear,_,d_handle,_,_=log[-1]
        print('MPC',t+C,'clear_mean_cm',round(float(clear.mean())*100,1),f'clear>{a.clear*100:.0f}cm',int((clear>a.clear).sum()),'of',int(on_table.sum()),
            'thumb_to_handle_cm',round(float(d_handle[:,geo.thumb[0]].mean())*100,1),'s',round(time.monotonic()-t0),flush=True)
    clear,rise,d_handle,d_head,d_palm=(torch.stack([x[i] for x in log],1).numpy() for i in range(5))   # (G,T,...)
    lifted=(clear>a.clear)&on_table[:,None]
    on_handle=(d_handle[:,:,geo.thumb[0]]<a.tip_tol)&((d_handle[:,:,geo.others]<a.tip_tol).sum(-1)>=2)
    off_head=d_head.min(-1)>a.head_tol
    def longest(mask):
        run=np.zeros(G,int);best=np.zeros(G,int)
        for col in mask.T:run=(run+1)*col;best=np.maximum(best,run)
        return best
    runs=dict(lifted=longest(lifted),lifted_on_handle=longest(lifted&on_handle),lifted_on_handle_off_head=longest(lifted&on_handle&off_head))
    out={k:torch.stack([e[k] for e in executed],1).numpy() for k in executed[0]}
    np.savez_compressed(a.output/'trajectories.npz',**out,actions=torch.stack(actions,1).numpy(),clearance=clear,centre_rise=rise,
        handle_distance=d_handle,head_distance=d_head,palm_handle_distance=d_palm,group_asset=group_asset,handle_box=(2*geo.half[first]).cpu().numpy(),
        has_head=geo.has_head[first].cpu().numpy(),start_on_table=on_table,table_top_z=top.cpu().numpy(),rest_clearance=rest_clear.cpu().numpy(),
        thumb_index=geo.thumb[0],**{'longest_'+k:v for k,v in runs.items()})
    report=dict(groups=G,starts_on_table=int(on_table.sum()),start_on_table=on_table.astype(int).tolist(),samples=S,steps=a.steps,tool_types=a.tool_types,pool=names,group_asset=group_asset.tolist(),
        success_30_steps={k:int((v>=30).sum()) for k,v in runs.items()},longest_run={k:v.tolist() for k,v in runs.items()},
        max_clearance_m=clear.max(1).round(3).tolist(),final_clearance_m=clear[:,-1].round(3).tolist(),
        final_thumb_to_handle_m=d_handle[:,-1,geo.thumb[0]].round(3).tolist(),final_min_head_distance_m=d_head[:,-1].min(-1).round(3).tolist(),
        thresholds=dict(clear_m=a.clear,tip_tol_m=a.tip_tol,head_tol_m=a.head_tol),elapsed_s=time.monotonic()-t0)
    (a.output/'report.json').write_text(json.dumps(report,indent=1)+'\n');print('MPC_DONE',json.dumps(report),flush=True)


if __name__=='__main__':main()
