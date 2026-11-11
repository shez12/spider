"""Render states of a tool_handle_grasp_mpc.py run: per start state (group) the middle of its longest
lifted-on-the-handle stretch (or its highest-clearance step if there is none) and the final step, two close-up views
each. No physics step is taken: each state is restored into an env that holds the same pool object and rendered.
The procedural tools are brown (handle) and grey (head). Run from the original repo root (relative asset paths).
"""
import argparse,json,sys
from pathlib import Path
PRISTINE=Path('/home/szc/work/newsimtoolreal/simtoolreal');HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(PRISTINE),str(PRISTINE/'rl_games')];sys.path.append(str(HERE))


def main():
    from isaaclab.app import AppLauncher
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--label',default='')
    p.add_argument('--distance',type=float,default=.42);p.add_argument('--settle-frames',type=int,default=6);p.add_argument('--cell',type=int,nargs=2,default=(480,360))
    AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.enable_cameras=True
    a.output.mkdir(parents=True,exist_ok=True);app=AppLauncher(a).app;code=0
    try:run(a)
    except BaseException:
        import traceback;traceback.print_exc();code=1
    finally:sys.stdout.flush();sys.stderr.flush();import os;os._exit(code)


def run(a):
    import cv2,numpy as np,torch
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env import SimToolRealEnv
    from diverse_resets import restore_states
    from left_reset_bank import FIELDS
    from tool_handle_grasp_mpc import tool_cfg,ToolGeometry
    ra=argparse.Namespace(**json.loads((a.run/'run_arguments.json').read_text()));d=np.load(a.run/'trajectories.npz')
    clear=d['clearance'];dh=d['handle_distance'];dhead=d['head_distance'];th=int(d['thumb_index']);G,T=clear.shape
    others=[i for i in range(dh.shape[-1]) if i!=th]
    good=(clear>ra.clear)&(dh[:,:,th]<ra.tip_tol)&((dh[:,:,others]<ra.tip_tol).sum(-1)>=2)&d['start_on_table'][:,None]
    picks=[]   # (group, step, tag)
    for g in np.flatnonzero(d['start_on_table']):
        best=(0,0);start=None
        for t in range(T+1):
            if t<T and good[g,t]:start=t if start is None else start
            elif start is not None:best=max(best,(t-start,start));start=None
        mid=(best[1]+best[0]//2,f'middle of {best[0]}-step handle lift') if best[0] else (int(clear[g].argmax()),'no handle lift: highest step')
        picks+=[(int(g),int(mid[0]),mid[1]),(int(g),T-1,'final step')]
    asset=d['group_asset'];P=int(asset.max())+1;per_asset=max(sum(1 for g,_,_ in picks if asset[g]==i) for i in range(P))
    cfg=tool_cfg(ra,P*per_asset);cfg.viewer.origin_type='world';cfg.viewer.resolution=tuple(a.cell)
    env=SimToolRealEnv(cfg,render_mode='rgb_array');dev=env.device;geo=ToolGeometry(env)
    names=sorted(x.name[4:] for x in (a.run/'objects').iterdir());assert names==sorted(x.name for x in geo.paths),'object pool differs from the run'
    env.reset();free={i:torch.nonzero(geo.asset==i).flatten().tolist() for i in range(P)};slot=[free[int(asset[g])].pop(0) for g,_,_ in picks]
    assert torch.allclose(2*geo.half[slot].cpu(),torch.tensor(d['handle_box'][[g for g,_,_ in picks]]),atol=1e-5),'handle boxes differ from the run'
    ids=torch.tensor(slot,device=dev);gi=[g for g,_,_ in picks];ti=[t for _,t,_ in picks]
    restore_states(env,ids,{f:torch.tensor(d[f][gi,ti],device=dev) for f in FIELDS})
    every=torch.arange(env.num_envs,device=dev);hidden=env.goal_viz.data.root_state_w[:,:7].clone();hidden[:,2]-=50.;env.goal_viz.write_root_pose_to_sim(hidden,env_ids=every)
    env.scene.write_data_to_sim();env.sim.forward();env.scene.update(0.)
    obj=env.object.data.root_pos_w[ids].clone();palm=env.robot.data.body_state_w[ids,env._palm_body_id,:3].clone();n=len(picks)
    away=obj-palm;away[:,2]=0.;away=away/away.norm(dim=-1,keepdim=True).clamp_min(1e-6);side=torch.stack([-away[:,1],away[:,0],torch.zeros(n,device=dev)],-1)
    centre=.5*(obj+palm);centre[:,2]=obj[:,2]
    for warm in range(80):
        f=env.render()
        if f is not None and f.size and f.max()>0:break
    def shot(eye,target):
        env.sim.set_camera_view(eye=tuple(eye.tolist()),target=tuple(target.tolist()))
        for _ in range(a.settle_frames):f=env.render()
        return cv2.cvtColor(np.ascontiguousarray(f[...,:3]),cv2.COLOR_RGB2BGR)
    up=torch.tensor([0.,0.,.22],device=dev);cells=[];meta=[]
    for i,(g,t,tag) in enumerate(picks):
        views=[shot(centre[i]+a.distance*side[i]+up,centre[i]),shot(obj[i]+a.distance*away[i]+up,obj[i])]
        tool=geo.paths[int(asset[g])].name.split('_')[1];box=' x '.join(f'{x*100:.1f}' for x in d['handle_box'][g])
        l1=f'start {g}  {tool}  handle {box} cm  t {t}  ({tag})'
        l2=f'clearance {clear[g,t]*100:.1f} cm   thumb-handle {dh[g,t,th]*100:.1f} cm   fingers on handle {int((dh[g,t,others]<ra.tip_tol).sum())}/4   closest tip to head {dhead[g,t].min()*100:.1f} cm'
        for v,txt in zip(views,(l1,l2)):
            cv2.rectangle(v,(0,0),(v.shape[1],22),(0,0,0),-1);cv2.putText(v,txt,(6,16),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1,cv2.LINE_AA)
        cell=np.concatenate(views,1);cv2.rectangle(cell,(0,0),(cell.shape[1]-1,cell.shape[0]-1),(255,255,255),1);cells.append(cell)
        meta.append(dict(group=g,step=t,tag=tag,tool=tool,clearance_m=float(clear[g,t]),thumb_handle_m=float(dh[g,t,th]),min_head_m=float(dhead[g,t].min())))
    grid=np.concatenate([np.concatenate(cells[i:i+2],1) for i in range(0,len(cells),2)],0);bar=np.zeros((34,grid.shape[1],3),np.uint8)
    cv2.putText(bar,f'{a.label}  left: best handle-lift step, right: final step (brown = handle, grey = head)',(8,23),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2,cv2.LINE_AA)
    cv2.imwrite(str(a.output/'handle_grasps.jpg'),np.concatenate([bar,grid],0),[cv2.IMWRITE_JPEG_QUALITY,88])
    (a.output/'states.json').write_text(json.dumps(meta,indent=1)+'\n');print('RENDER_DONE',n,'states',flush=True)


if __name__=='__main__':main()
