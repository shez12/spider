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


def symmetries(cal):
    """(S, 4, 4) object-frame rotations that map the mesh onto itself, identity first.

    calibration.json 'symmetry_rotations' (make_dexycb_object.py); assets without it are the mustard bottle (C2)."""
    return np.stack([np.eye(4)]+[np.asarray(t,dtype=np.float64) for t in cal.get('symmetry_rotations',[C2])])


def mat(pose):
    """(N, 7) pos + wxyz -> (N, 4, 4)."""
    T=np.broadcast_to(np.eye(4),(len(pose),4,4)).copy();T[:,:3,:3]=Rot.from_quat(pose[:,[4,5,6,3]]).as_matrix();T[:,:3,3]=pose[:,:3];return T


def inv(T):
    R=T[...,:3,:3].swapaxes(-1,-2);out=np.zeros_like(T);out[...,:3,:3]=R;out[...,:3,3]=-np.einsum('...ij,...j->...i',R,T[...,:3,3]);out[...,3,3]=1;return out


def up_face(T):
    """Object axis that points up in the world, as 'y', 'x', 'z' (sign dropped: +/- are the same up to C2 or a flip)."""
    up=T[...,2,:3];return np.array(list('xyz'))[np.abs(up).argmax(-1)],np.abs(up).max(-1)


def finger_chains():
    """Per finger: (index into the 29 left canonical joints, chain from the robot root to the fingertip link)."""
    import xml.etree.ElementTree as ET
    left=left_canonical_names();J={j.get('name'):j for j in ET.parse(LEFT_URDF).getroot().findall('joint')};out=[]
    for f,tip in (('thumb','left_thumb_fingertip'),('index','left_index_fingertip'),('middle','left_middle_fingertip'),('ring','left_ring_fingertip'),('pinky','left_pinky_fingertip')):
        fj=[n for n in left[7:] if f in n];ch=arm_chain(LEFT_URDF,left[:7]+fj)
        for name in (f'left_{f}_elastomer_fix_joint',f'left_{f}_fingertip_fix_joint'):
            o=J[name].find('origin');T=np.eye(4);T[:3,3]=[float(x) for x in o.get('xyz').split()];T[:3,:3]=Rot.from_euler('xyz',[float(x) for x in o.get('rpy').split()]).as_matrix();ch.append((T,None))
        assert J[f'left_{f}_fingertip_fix_joint'].find('child').get('link')==tip
        out.append((np.array([left.index(n) for n in left[:7]+fj]),ch))
    return out


def retarget_fingers(x,lim,chains,iters=150,damp=1e-4,null=.05):
    """Left-hand finger joints whose fingertips (left URDF FK) reach the fingertips of the unclipped mirrored joints x,
    within the left limits: the limit-respecting replacement of clipping.  The arm is not changed.
    Returns (joints, fingertip error after retargeting (N, 5) m, fingertip error of plain clipping (N, 5) m)."""
    xc=np.clip(x,lim[:,0],lim[:,1]);q=xc.copy();e_ret=np.zeros((len(x),5));e_clip=np.zeros((len(x),5))
    for f,(ix,ch) in enumerate(chains):
        tip=lambda z:fk(ch,z[:,ix])[:,:3,3];tgt=tip(x);e_clip[:,f]=np.linalg.norm(tip(xc)-tgt,axis=1);fi=ix[7:];m=len(fi);I=np.eye(m)
        for it in range(iters):
            p=tip(q);e=tgt-p;Jc=np.empty((len(q),3,m))
            for k,j in enumerate(fi):
                dq=q.copy();dq[:,j]+=1e-6;Jc[:,:,k]=(tip(dq)-p)/1e-6
            JT=Jc.transpose(0,2,1);pinv=JT@np.linalg.inv(Jc@JT+damp*np.eye(3))
            d=np.einsum('nij,nj->ni',pinv,e)+np.einsum('nij,nj->ni',I-pinv@Jc,null*(xc[:,fi]-q[:,fi]))
            q[:,fi]=np.clip(q[:,fi]+.5*d,lim[fi,0],lim[fi,1])
        e_ret[:,f]=np.linalg.norm(tip(q)-tgt,axis=1)
    return q,e_ret,e_clip


def extract(a):
    V,cal=mesh_vertices(a.object_dir);n,c,_=fit_mirror_plane(V);offset=np.array(cal['source_to_new_root_center_in_source_m'])
    left,_,perm=joint_maps();lim=urdf_limits(LEFT_URDF,left);chain=arm_chain(LEFT_URDF,left[:7])
    if a.kinematic:  # IK-feasible augmented references that were not (or not yet) physically optimised / validated
        acc=[dict(source_index=int(c.parent.name[:3]),candidate_id=int(c.name[-5:]),reference=str(c/'reference.npz')) for c in sorted((a.campaign/'groups').glob('*/candidate_*'))
            if (c/'state.json').exists() and (c/'reference.npz').exists() and json.loads((c/'state.json').read_text()).get('ik_pass')]
    elif a.no_final_orientation:  # physically optimised candidates that pass every validation check except the final object orientation
        ok=lambda r:r['object_rmse_m']<.03 and r['final_object_position_error_m']<.03 and max(r['max_arm_table_penetration_m'],r['max_hand_table_penetration_m'],r['max_hand_self_penetration_m'])<.001
        acc=[]
        for folder in sorted((a.campaign/'groups').glob('*/candidate_*')):
            state=json.loads((folder/'state.json').read_text()) if (folder/'state.json').exists() else {}
            if 'selected_metrics' in state and 'replay_metrics' in state and ok(state['selected_metrics']) and ok(state['replay_metrics']) and (folder/'gpu/replay_trajectory.npz').exists():
                acc.append(dict(source_index=int(state['source_index']),candidate_id=int(state['candidate_id']),replay_trajectory=str(folder/'gpu/replay_trajectory.npz'),trajectory=str(folder/'gpu/trajectory.npz')))
    else:acc=json.loads((a.campaign/'accepted.json').read_text())
    rng=np.random.default_rng(0);by={}
    for e in acc:by.setdefault(int(e['source_index']),[]).append(e)
    rows=[]
    for s,es in sorted(by.items()):
        for e in [es[i] for i in sorted(rng.choice(len(es),min(len(es),a.per_source),replace=False))]:
            if a.kinematic:z=np.load(e['reference']);ctrl=z['ctrl_nominal'].astype(np.float64);qpos=np.concatenate([z['joint_ik'],z['object_qpos']],1)[:len(ctrl)].astype(np.float64)
            else:z=np.load(e[a.key]);qpos=z['qpos'].astype(np.float64);ctrl=z['ctrl'].astype(np.float64);assert float(z['control_dt'])==.01
            assert qpos.shape[1]==36
            R=Rot.from_quat(qpos[:,[33,34,35,32]]);pose=np.concatenate([qpos[:,29:32]+R.apply(offset)+[0,0,TABLE_TOP],qpos[:,32:36]],1)
            q=mirror_joints(qpos[:,:29],perm,True);u=mirror_joints(ctrl,perm,True);tips={}
            if a.finger_retarget:
                fc=fc if 'fc' in dir() else finger_chains()
                (q,er,ec),(u,eru,ecu)=retarget_fingers(q,lim,fc),retarget_fingers(u,lim,fc);tips=dict(tip_err_retarget=er,tip_err_clip=ec,tip_err_retarget_ctrl=eru,tip_err_clip_ctrl=ecu)
            over=float(np.maximum(lim[:7,0]-q[:,:7],q[:,:7]-lim[:7,1]).max())
            W=fk(chain,q[:,:7]);W[:,:3,3]+=BASE
            rows.append(dict(source=s,candidate=int(e['candidate_id']),q=np.clip(q,lim[:,0],lim[:,1]),ctrl=np.clip(u,lim[:,0],lim[:,1]),obj=mirror_pose(pose,n,c),wrist=W,arm_over=over,**tips))
    T=max(len(r['q']) for r in rows);pad=lambda x:np.concatenate([x,np.repeat(x[-1:],T-len(x),0)])
    out={k:np.stack([pad(r[k]) for r in rows]).astype(np.float32 if k!='wrist' else np.float64) for k in ('q','ctrl','obj','wrist')}
    out.update(length=np.array([len(r['q']) for r in rows]),source=np.array([r['source'] for r in rows]),candidate=np.array([r['candidate'] for r in rows]),arm_over=np.array([r['arm_over'] for r in rows]))
    lift=out['obj'][:,:,2]-out['obj'][:,:1,2];out['lift_frame']=np.array([int(np.argmax(l>.01)) if (l>.01).any() else -1 for l in lift])
    face,comp=up_face(mat(out['obj'][:,0].astype(np.float64)));out['face']=face;out['face_comp']=comp
    a.output.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output/'demos.npz',**out)
    if a.finger_retarget:
        st=lambda k:{f:dict(p50_mm=float(1e3*np.median(v)),p95_mm=float(1e3*np.quantile(v,.95)),max_mm=float(1e3*v.max())) for f,v in zip(('thumb','index','middle','ring','pinky'),np.concatenate([r[k] for r in rows]).T)}
        (a.output/'finger_retarget.json').write_text(json.dumps({k:st(k) for k in ('tip_err_clip','tip_err_retarget','tip_err_clip_ctrl','tip_err_retarget_ctrl')},indent=1)+'\n')
    rep={int(s):dict(n=int((out['source']==s).sum()),face=str(face[out['source']==s][0]),face_comp=float(comp[out['source']==s].min()),lift_frame=int(np.median(out['lift_frame'][out['source']==s])),
        max_lift_m=float(lift[out['source']==s].max(1).min()),arm_over_limit_rad=float(out['arm_over'][out['source']==s].max())) for s in sorted(set(out['source']))}
    (a.output/'demos.json').write_text(json.dumps(dict(campaign=str(a.campaign),key='kinematic reference (not validated)' if a.kinematic else a.key+(' (final object orientation not checked)' if a.no_final_orientation else ''),demos=len(rows),per_source=rep),indent=1)+'\n');print(json.dumps(rep,indent=1))


def reach(a):
    d=np.load(a.output/'demos.npz');left=left_canonical_names();chain=arm_chain(LEFT_URDF,left[:7]);lim=urdf_limits(LEFT_URDF,left)[:7]
    lim=np.stack([lim[:,0]+np.deg2rad(2),lim[:,1]-np.deg2rad(2)],1)
    nat=[np.load(f) for f in a.native];q0=np.concatenate([z['q'][:,0,:7] for z in nat]).astype(np.float64);root=np.concatenate([z['root'][:,0,:7] for z in nat]).astype(np.float64)
    top=np.concatenate([z['table_top_z'] for z in nat]);ok=np.abs(root[:,2]-top)<.12;q0,root=q0[ok],root[ok];To=mat(root);nface,_=up_face(To)
    K=len(d['q']);lf=d['lift_frame'];Td=mat(d['obj'][:,0].astype(np.float64));Wg=d['wrist'][np.arange(K),lf];Wp=d['wrist'][np.arange(K),np.maximum(lf-a.pre_frames,0)]
    rng=np.random.default_rng(0);best=np.full(len(q0),np.inf);arg=np.full((len(q0),2),-1);tilt_ok=np.zeros(len(q0),bool);allq=np.zeros((len(q0),7));pre=np.full(len(q0),np.inf)
    for sym,C in enumerate(symmetries(json.loads((a.object_dir/'calibration.json').read_text()))):
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
    p.add_argument('--kinematic',action='store_true',help="extract: use the campaign's IK-feasible kinematic references (groups/*/candidate_*/reference.npz) instead of the validated trajectories")
    p.add_argument('--no-final-orientation',action='store_true',help='extract: use every physically optimised candidate that passes the validation except the final object orientation (< 20 deg), which the grasp does not need')
    p.add_argument('--finger-retarget',action='store_true',help='extract: instead of clipping the mirrored finger joints to the left limits, solve left finger joints (within the limits) whose fingertips reach the fingertips of the unclipped mirror; arm unchanged')
    p.add_argument('--object-dir',type=Path,default=HERE/'assets/object');p.add_argument('--output',type=Path,default=OUT/'demo_relative_v1')
    p.add_argument('--native',type=Path,nargs='+',default=sorted(OUT.glob('left_grasp_mpc_v2_s*/trajectories.npz'))+[OUT/'left_grasp_mpc_v1/trajectories.npz'])
    p.add_argument('--max-tilt-deg',type=float,default=15.);p.add_argument('--pre-frames',type=int,default=40);p.add_argument('--seeds',type=int,default=4)
    a=p.parse_args();{'extract':extract,'reach':reach}[a.command](a)
