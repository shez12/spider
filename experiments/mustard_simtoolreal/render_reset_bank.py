"""Render states of a left_reset_bank.py bank: per kind, --per-kind states from distinct trajectories, two close-up
views each (side view of palm->object, and from behind the object looking at the palm). No physics step is taken:
each state is restored into its own env and only rendered. Run from the original repo root (relative asset paths).
"""
import argparse,json,sys
from pathlib import Path
PRISTINE=Path('/home/szc/work/newsimtoolreal/simtoolreal');HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(PRISTINE),str(PRISTINE/'rl_games')];sys.path.append(str(HERE))


def main():
    from isaaclab.app import AppLauncher
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--object-dir',type=Path)
    p.add_argument('--per-kind',type=int,default=12);p.add_argument('--seed',type=int,default=0);p.add_argument('--label',default='')
    p.add_argument('--distance',type=float,default=.42);p.add_argument('--settle-frames',type=int,default=6);p.add_argument('--cell',type=int,nargs=2,default=(480,360))
    AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.enable_cameras=True
    a.output.mkdir(parents=True,exist_ok=True);app=AppLauncher(a).app;code=0
    try:run(a)
    except BaseException:
        import traceback;traceback.print_exc();code=1
    finally:sys.stdout.flush();sys.stderr.flush();import os;os._exit(code)


def run(a):
    import cv2,numpy as np,torch,yaml
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env import SimToolRealEnv
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg
    from diverse_resets import restore_states
    from left_reset_bank import signature,FIELDS,KINDS
    from adept_obs import object_dir_of,object_urdf
    d=np.load(a.bank);rep=json.loads(a.bank.with_suffix('.json').read_text());kind=d['kind'];src=d['source']
    traj={i:np.load(Path(r)/'trajectories.npz') for i,r in enumerate(rep['runs'])}
    rng=np.random.default_rng(a.seed);picks=[]
    for k in (1,2,3):
        idx=np.flatnonzero(kind==k);tid=src[idx,0]*10000+src[idx,1];chosen=rng.permutation(np.unique(tid))[:a.per_kind]
        picks+=[int(rng.choice(idx[tid==t])) for t in chosen]
    n=len(picks);cal=json.loads((object_dir_of(a.object_dir)/'calibration.json').read_text())
    cfg=SimToolRealEnvCfg();cfg.from_dict(yaml.safe_load((PRISTINE/'isaacsimenvs/cfg/task/SimToolReal.yaml').read_text()))
    cfg.seed=a.seed;cfg.scene.num_envs=n
    cfg.assets.object_urdf=str(object_urdf(a.object_dir));cfg.assets.object_scale=tuple(cal['policy_object_scale']);cfg.reward.fixed_size=tuple(cal['bbox_dims_m'])
    cfg.viewer.origin_type='world';cfg.viewer.resolution=tuple(a.cell)
    env=SimToolRealEnv(cfg,render_mode='rgb_array');dev=env.device
    sig=signature(PRISTINE/cfg.assets.robot_urdf,cfg.assets.object_urdf,PRISTINE/'isaacsimenvs/cfg/task/SimToolReal.yaml')
    assert str(d['signature'])==sig,'bank was built in a different environment (wrong --object-dir?)'
    try:  # the mustard meshes have no texture: tint the object so it stands out from the grey table
        import isaaclab.sim as sim_utils
        sim_utils.PreviewSurfaceCfg(diffuse_color=(.95,.72,.05),roughness=.6).func('/World/Looks/BankObject',sim_utils.PreviewSurfaceCfg(diffuse_color=(.95,.72,.05),roughness=.6))
        for prim in sim_utils.find_matching_prim_paths(env.object.cfg.prim_path):sim_utils.bind_visual_material(prim,'/World/Looks/BankObject',stronger_than_descendants=True)
    except Exception as exc:print('TINT_FAILED',repr(exc),flush=True)
    env.reset();ids=torch.arange(n,device=dev)
    restore_states(env,ids,{f:torch.tensor(d[f][picks],device=dev) for f in FIELDS})
    hidden=env.goal_viz.data.root_state_w[:,:7].clone();hidden[:,2]-=50.;env.goal_viz.write_root_pose_to_sim(hidden,env_ids=ids)
    env.scene.write_data_to_sim();env.sim.forward();env.scene.update(0.)
    obj=env.object.data.root_pos_w.clone();palm=env.robot.data.body_state_w[:,env._palm_body_id,:3].clone()
    away=obj-palm;away[:,2]=0.;away=away/away.norm(dim=-1,keepdim=True).clamp_min(1e-6);side=torch.stack([-away[:,1],away[:,0],torch.zeros(n,device=dev)],-1)
    centre=.5*(obj+palm);centre[:,2]=obj[:,2]
    for warm in range(80):
        f=env.render()
        if f is not None and f.size and f.max()>0:break
    print('RENDER_WARMUP',warm,None if f is None else (f.shape,int(f.max())),flush=True)
    def shot(eye,target):
        env.sim.set_camera_view(eye=tuple(eye.tolist()),target=tuple(target.tolist()))
        for _ in range(a.settle_frames):f=env.render()
        return cv2.cvtColor(np.ascontiguousarray(f[...,:3]),cv2.COLOR_RGB2BGR)
    up=torch.tensor([0.,0.,.22],device=dev);cells=[];meta=[]
    for i,e in enumerate(picks):
        r,g,t=(int(x) for x in src[e]);lift=float(traj[r]['lift'][g,t]);tips=float(traj[r]['tip_distance'][g,t])
        views=[shot(centre[i]+a.distance*side[i]+up,centre[i]),shot(obj[i]+a.distance*away[i]+up,obj[i])]
        txt=f'{KINDS[int(kind[e])]}  lift {lift*100:.1f} cm  tips {tips*100:.1f} cm  run {r} traj {g} t {t}'
        for v in views:cv2.rectangle(v,(0,0),(v.shape[1],22),(0,0,0),-1)
        cv2.putText(views[0],txt,(6,16),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(views[1],'from behind the object',(6,16),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
        cell=np.concatenate(views,1);cv2.rectangle(cell,(0,0),(cell.shape[1]-1,cell.shape[0]-1),(255,255,255),1);cells.append(cell)
        meta.append(dict(bank_index=e,kind=KINDS[int(kind[e])],run=r,trajectory=g,step=t,lift_m=lift,tip_distance_m=tips))
    for j,k in enumerate((1,2,3)):
        part=cells[j*a.per_kind:(j+1)*a.per_kind];cols=3;rows=[np.concatenate(part[c:c+cols],1) for c in range(0,len(part),cols) if len(part[c:c+cols])==cols]
        grid=np.concatenate(rows,0);bar=np.zeros((34,grid.shape[1],3),np.uint8)
        cv2.putText(bar,f'{a.label}  {KINDS[k]}  ({int((kind==k).sum())} states in bank, {len(part)} shown, one per trajectory)',(8,23),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2,cv2.LINE_AA)
        cv2.imwrite(str(a.output/f'{KINDS[k]}.jpg'),np.concatenate([bar,grid],0),[cv2.IMWRITE_JPEG_QUALITY,88])
    (a.output/'states.json').write_text(json.dumps(meta,indent=1)+'\n');print('RENDER_DONE',n,'states',flush=True)


if __name__=='__main__':main()
