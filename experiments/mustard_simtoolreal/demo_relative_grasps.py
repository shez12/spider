"""Object-relative left-hand grasp references from the right-hand pipeline trajectories (MuJoCo, KUKA + Sharpa).

Only what survives the embodiment/scene change is kept: the wrist (iiwa link 7) pose and the hand joints as a
function of time, expressed relative to the bottle.  Per accepted trajectory of mustard_multisource_valid100:
  joints  right MuJoCo order == right canonical order -> left canonical (mirror_right_bank.mirror_joints)
  object  MuJoCo body frame -> Isaac asset frame (bbox centre, calibration.json) -> world shifted so the table top is
          at z = 0.53 (the robot base is 0.53 m below the table top in both scenes) -> mirrored scene (mirror_pose)
  wrist   left URDF forward kinematics of the mirrored arm joints, robot base at (0, 0.8, 0)
CPU only, no simulator.  `reach` additionally measures, for native settled starts of the left env, how far the arm
has to move in joint space to bring the wrist to the demo grasp pose placed relative to the native bottle pose.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from mirror_right_bank import HERE,LEFT_URDF,M,fit_mirror_plane,joint_maps,left_canonical_names,mesh_vertices,mirror_joints,mirror_pose,urdf_limits
from resolve_arm_config import arm_chain,fk,solve

OUT=HERE.parents[1]/'outputs';BASE=np.array([0.,.8,0.]);TABLE_TOP=.53
C2=np.diag([-1.,-1.,1.,1.])  # the bottle is symmetric under a half turn about its long (z) axis (vertex error ~1 mm)


def mat(pose):
    """(N, 7) pos + wxyz -> (N, 4, 4)."""
    T=np.broadcast_to(np.eye(4),(len(pose),4,4)).copy();T[:,:3,:3]=Rot.from_quat(pose[:,[4,5,6,3]]).as_matrix();T[:,:3,3]=pose[:,:3];return T


def inv(T):
    R=T[...,:3,:3].swapaxes(-1,-2);out=np.zeros_like(T);out[...,:3,:3]=R;out[...,:3,3]=-np.einsum('...ij,...j->...i',R,T[...,:3,3]);out[...,3,3]=1;return out


def up_face(T):
    """Object axis that points up in the world, as 'y', 'x', 'z' (sign dropped: +/- are the same up to C2 or a flip)."""
    up=T[...,2,:3];return np.array(list('xyz'))[np.abs(up).argmax(-1)],np.abs(up).max(-1)


def extract(a):
    V,cal=mesh_vertices(a.object_dir);n,c,_=fit_mirror_plane(V);offset=np.array(cal['source_to_new_root_center_in_source_m'])
    left,_,perm=joint_maps();lim=urdf_limits(LEFT_URDF,left);chain=arm_chain(LEFT_URDF,left[:7])
    acc=json.loads((a.campaign/'accepted.json').read_text());rng=np.random.default_rng(0);by={}
    for e in acc:by.setdefault(int(e['source_index']),[]).append(e)
    rows=[]
    for s,es in sorted(by.items()):
        for e in [es[i] for i in sorted(rng.choice(len(es),min(len(es),a.per_source),replace=False))]:
            z=np.load(e[a.key]);qpos=z['qpos'].astype(np.float64);ctrl=z['ctrl'].astype(np.float64);assert qpos.shape[1]==36 and float(z['control_dt'])==.01
            R=Rot.from_quat(qpos[:,[33,34,35,32]]);pose=np.concatenate([qpos[:,29:32]+R.apply(offset)+[0,0,TABLE_TOP],qpos[:,32:36]],1)
            q=mirror_joints(qpos[:,:29],perm,True);u=mirror_joints(ctrl,perm,True)
            over=float(np.maximum(lim[:7,0]-q[:,:7],q[:,:7]-lim[:7,1]).max())
            W=fk(chain,q[:,:7]);W[:,:3,3]+=BASE
            rows.append(dict(source=s,candidate=int(e['candidate_id']),q=np.clip(q,lim[:,0],lim[:,1]),ctrl=np.clip(u,lim[:,0],lim[:,1]),obj=mirror_pose(pose,n,c),wrist=W,arm_over=over))
    T=max(len(r['q']) for r in rows);pad=lambda x:np.concatenate([x,np.repeat(x[-1:],T-len(x),0)])
    out={k:np.stack([pad(r[k]) for r in rows]).astype(np.float32 if k!='wrist' else np.float64) for k in ('q','ctrl','obj','wrist')}
    out.update(length=np.array([len(r['q']) for r in rows]),source=np.array([r['source'] for r in rows]),candidate=np.array([r['candidate'] for r in rows]),arm_over=np.array([r['arm_over'] for r in rows]))
    lift=out['obj'][:,:,2]-out['obj'][:,:1,2];out['lift_frame']=np.array([int(np.argmax(l>.01)) if (l>.01).any() else -1 for l in lift])
    face,comp=up_face(mat(out['obj'][:,0].astype(np.float64)));out['face']=face;out['face_comp']=comp
    a.output.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output/'demos.npz',**out)
    rep={int(s):dict(n=int((out['source']==s).sum()),face=str(face[out['source']==s][0]),face_comp=float(comp[out['source']==s].min()),lift_frame=int(np.median(out['lift_frame'][out['source']==s])),
        max_lift_m=float(lift[out['source']==s].max(1).min()),arm_over_limit_rad=float(out['arm_over'][out['source']==s].max())) for s in sorted(set(out['source']))}
    (a.output/'demos.json').write_text(json.dumps(dict(campaign=str(a.campaign),key=a.key,demos=len(rows),per_source=rep),indent=1)+'\n');print(json.dumps(rep,indent=1))


def reach(a):
    d=np.load(a.output/'demos.npz');left=left_canonical_names();chain=arm_chain(LEFT_URDF,left[:7]);lim=urdf_limits(LEFT_URDF,left)[:7]
    lim=np.stack([lim[:,0]+np.deg2rad(2),lim[:,1]-np.deg2rad(2)],1)
    nat=[np.load(f) for f in a.native];q0=np.concatenate([z['q'][:,0,:7] for z in nat]).astype(np.float64);root=np.concatenate([z['root'][:,0,:7] for z in nat]).astype(np.float64)
    top=np.concatenate([z['table_top_z'] for z in nat]);ok=np.abs(root[:,2]-top)<.12;q0,root=q0[ok],root[ok];To=mat(root);nface,_=up_face(To)
    K=len(d['q']);lf=d['lift_frame'];Td=mat(d['obj'][:,0].astype(np.float64));Wg=d['wrist'][np.arange(K),lf];Wp=d['wrist'][np.arange(K),np.maximum(lf-a.pre_frames,0)]
    rng=np.random.default_rng(0);best=np.full(len(q0),np.inf);arg=np.full((len(q0),2),-1);tilt_ok=np.zeros(len(q0),bool);allq=np.zeros((len(q0),7));pre=np.full(len(q0),np.inf)
    for sym,C in enumerate((np.eye(4),C2)):
        G=np.einsum('nij,kjl->nkil',To,inv(Td@C))  # demo world -> native world, per (native, demo)
        tilt=np.degrees(np.arccos(np.clip(G[...,2,2],-1,1)));m=tilt<a.max_tilt_deg;ni,ki=np.nonzero(m)
        if not len(ni):continue
        tilt_ok|=m.any(1);tg=np.einsum('pij,pjk->pik',G[ni,ki],Wg[ki]);tp=np.einsum('pij,pjk->pik',G[ni,ki],Wp[ki])
        tg[:,:3,3]-=BASE;tp[:,:3,3]-=BASE  # world -> robot root frame of the URDF chain
        for trial in range(a.seeds):
            seed=q0[ni] if trial==0 else np.clip(q0[ni]+rng.uniform(-1.,1.,(len(ni),7)),lim[:,0],lim[:,1])
            q,ep,er=solve(chain,tg,seed,q0[ni],lim);good=(ep<2e-3)&(er<1.);dist=np.abs(q-q0[ni]).max(1)
            qp,epp,erp=solve(chain,tp,q,q0[ni],lim);dp=np.where((epp<2e-3)&(erp<1.),np.abs(qp-q0[ni]).max(1),np.inf)
            for j in np.flatnonzero(good):
                if dist[j]<best[ni[j]]:best[ni[j]]=dist[j];arg[ni[j]]=(ki[j],sym);allq[ni[j]]=q[j];pre[ni[j]]=dp[j]
    speed=.1*1.5  # rad/s: arm_moving_average * dof_speed_scale, the largest arm target speed the action space allows
    fin=np.isfinite(best);rep=dict(native=int(len(q0)),native_faces={f:float((nface==f).mean()) for f in 'xyz'},tilt_compatible=float(tilt_ok.mean()),ik_feasible=float(fin.mean()),
        arm_distance_rad={str(p):float(np.percentile(best[fin],p)) for p in (10,25,50,75,90)} if fin.any() else None,
        within={str(t):float((best<=t).mean()) for t in (.3,.5,.75,1.,1.5)},seconds_at_max_speed_median=float(np.median(best[fin])/speed) if fin.any() else None,
        pregrasp_distance_rad={str(p):float(np.percentile(pre[np.isfinite(pre)],p)) for p in (10,50,90)} if np.isfinite(pre).any() else None,
        best_source={int(s):int((d['source'][arg[fin,0]]==s).sum()) for s in sorted(set(d['source']))},
        by_native_face={f:dict(n=int((nface==f).sum()),feasible=float(fin[nface==f].mean()),median_rad=float(np.median(best[fin&(nface==f)])) if (fin&(nface==f)).any() else None) for f in 'xyz'})
    print(json.dumps(rep,indent=1));(a.output/'reach.json').write_text(json.dumps(rep,indent=1)+'\n')
    np.savez(a.output/'reach.npz',q0=q0,root=root,best=best,arg=arg,q_grasp=allq,pre=pre)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter);p.add_argument('command',choices=('extract','reach'))
    p.add_argument('--campaign',type=Path,default=OUT/'mustard_multisource_valid100');p.add_argument('--key',default='replay_trajectory');p.add_argument('--per-source',type=int,default=8)
    p.add_argument('--object-dir',type=Path,default=HERE/'assets/object');p.add_argument('--output',type=Path,default=OUT/'demo_relative_v1')
    p.add_argument('--native',type=Path,nargs='+',default=sorted(OUT.glob('left_grasp_mpc_v2_s*/trajectories.npz'))+[OUT/'left_grasp_mpc_v1/trajectories.npz'])
    p.add_argument('--max-tilt-deg',type=float,default=15.);p.add_argument('--pre-frames',type=int,default=40);p.add_argument('--seeds',type=int,default=4)
    a=p.parse_args();{'extract':extract,'reach':reach}[a.command](a)
