"""Mirror a right-hand reset bank (diverse_resets format: canonical joints, env-local world frame) into the
left-hand original SimToolReal env.  CPU only, no simulator.

Both envs have the robot base at (0, 0.8, 0) with identity rotation and the table centred on x=0, so the whole scene
is reflected in the world plane x=0 (M = diag(-1, 1, 1)):
  arm     q_left = -q_right, then joint 7 += J7_OFFSET (the hand mount differs by 30 deg between the two URDFs)
  hand    copied by joint name (the left URDF hand is the mirror image of the right one)
  object  the reflected mesh is replaced by the original mesh through the bottle's own mirror plane S (n.x = c in
          the object frame, fitted here):  R' = M R S_n,  t' = M (t + 2 c R n),  v' = M v,  w' = -M w.
          The goal pose is transformed the same way.  The table (x=0, identity) is unchanged.
Outputs (left_reset_bank.py format, signature of the left env):
  mirrored_raw.npz      hand joints exactly as in the right-hand data.  Most states exceed the left URDF abduction
                        limits; only usable with the relaxed limits stored in limit_lower / limit_upper.
  mirrored_clamped.npz  q / previous / current clipped to the left URDF limits (fingertips move, grasps may open).
Neither file is physically validated: run validate_mirrored_bank.py before training on them.
"""
import argparse,ast,json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

PRISTINE=Path('/home/szc/work/newsimtoolreal/simtoolreal');HERE=Path(__file__).resolve().parent
LEFT_URDF=PRISTINE/'assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf'
RIGHT_URDF=HERE/'assets/robot/kuka_sharpa_right.urdf'
J7_OFFSET=np.deg2rad(-30.)
M=np.diag([-1.,1.,1.])


def left_canonical_names():
    tree=ast.parse((PRISTINE/'isaacsimenvs/tasks/simtoolreal/utils/scene_utils.py').read_text())
    for node in tree.body:
        if isinstance(node,ast.AnnAssign) and getattr(node.target,'id','')=='JOINT_NAMES_CANONICAL':return list(ast.literal_eval(node.value))
    raise KeyError('JOINT_NAMES_CANONICAL')


def joint_key(name):
    """'right_thumb_CMC_FE' and 'left_1_thumb_CMC_FE' -> 'thumb_CMC_FE'; arm joints keep their name."""
    parts=name.split('_')
    if parts[0] not in ('left','right'):return name
    return '_'.join(parts[2:] if parts[1].isdigit() else parts[1:])


def urdf_limits(path,names):
    d={j.get('name'):(float(j.find('limit').get('lower')),float(j.find('limit').get('upper'))) for j in ET.parse(path).getroot().findall('joint') if j.get('type')!='fixed'}
    return np.array([d[n] for n in names])


def joint_maps():
    right=json.loads((HERE/'assets/robot/robot_spec.json').read_text())['joint_names'];left=left_canonical_names()
    keys=[joint_key(n) for n in right];perm=np.array([keys.index(joint_key(n)) for n in left])
    assert sorted(perm.tolist())==list(range(29)) and (perm[:7]==np.arange(7)).all()
    return left,right,perm


def mirror_joints(x,perm,position):
    """x: (..., 29) right canonical -> left canonical.  position=False for velocities (no joint-7 offset)."""
    y=np.asarray(x,dtype=np.float64)[...,perm].copy();y[...,:7]*=-1
    if position:y[...,6]+=J7_OFFSET
    return y


def fit_mirror_plane(vertices):
    """Vertical plane n.x = c (n in the object xy-plane) that best maps the mesh onto itself."""
    tree=cKDTree(vertices);sub=vertices[np.random.default_rng(0).choice(len(vertices),min(len(vertices),4000),replace=False)]
    def error(deg,c):
        n=np.array([np.cos(np.deg2rad(deg)),np.sin(np.deg2rad(deg)),0.]);d=tree.query(sub-2*((sub@n)-c)[:,None]*n)[0];return float(np.percentile(d,95))
    best=min((error(g,c),g,c) for g in np.arange(0,180,1.) for c in np.arange(-.006,.0061,.001))
    best=min((error(g,c),g,c) for g in np.arange(best[1]-1,best[1]+1.01,.05) for c in np.arange(best[2]-.001,best[2]+.00101,.0002))
    n=np.array([np.cos(np.deg2rad(best[1])),np.sin(np.deg2rad(best[1])),0.]);d=tree.query(vertices-2*((vertices@n)-best[2])[:,None]*n)[0]
    return n,float(best[2]),dict(normal_deg_from_x=float(best[1]),offset_m=float(best[2]),median_m=float(np.median(d)),p95_m=float(np.percentile(d,95)),max_m=float(d.max()))


def mirror_pose(pose,n,c):
    """(N, 7+) pos + wxyz quat (+ linear, angular velocity) of the original mesh -> pose of the same mesh in the mirrored scene."""
    pose=np.asarray(pose,dtype=np.float64);out=pose.copy();R=Rot.from_quat(pose[:,[4,5,6,3]]).as_matrix();S=np.eye(3)-2*np.outer(n,n)
    R2=M@R@S;assert np.allclose(np.linalg.det(R2),1,atol=1e-6)
    out[:,:3]=(pose[:,:3]+2*c*(R@n))@M;q=Rot.from_matrix(R2).as_quat();q*=np.where(q[:,3:]<0,-1,1);out[:,3:7]=q[:,[3,0,1,2]]
    if pose.shape[1]>=13:out[:,7:10]=pose[:,7:10]@M;out[:,10:13]=-pose[:,10:13]@M
    return out


def mesh_vertices(object_dir):
    cal=json.loads((object_dir/'calibration.json').read_text());obj=sorted(object_dir.glob('*visual*.obj'));assert len(obj)==1,obj
    V=np.array([[float(x) for x in l.split()[1:4]] for l in obj[0].read_text().splitlines() if l.startswith('v ')])
    return V,cal  # policy_object_scale is the bbox observation scale, not a mesh scale: the URDF meshes are in metres


def clearance(root,vertices,table_z):
    """Lowest mesh point above the table top (table_narrow is 0.30 m thick)."""
    R=Rot.from_quat(root[:,[4,5,6,3]]).as_matrix();hull=vertices[np.unique(__import__('scipy.spatial').spatial.ConvexHull(vertices).vertices)]
    return (np.einsum('nij,kj->nki',R,hull)[...,2].min(1)+root[:,2])-(table_z+.15)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bank',type=Path,default=HERE.parents[1]/'outputs/mustard_diverse_reset_bank_v41/merged/reset_bank.npz')
    p.add_argument('--object-dir',type=Path,default=HERE/'assets/object');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    import sys;sys.path.append(str(HERE))
    from left_reset_bank import signature,FIELDS
    from adept_obs import object_urdf
    left,right,perm=joint_maps();d=np.load(a.bank);V,cal=mesh_vertices(a.object_dir);n,c,plane=fit_mirror_plane(V)
    assert np.abs(d['table'][:,[0]]).max()<1e-6 and np.abs(d['table'][:,3:]-[1,0,0,0]).max()<1e-6,'table must be centred on x=0 and unrotated'
    out={k:mirror_joints(d[k],perm,True) for k in ('q','previous','current')};out['v']=mirror_joints(d['v'],perm,False)
    out['root']=mirror_pose(d['root'],n,c);out['goal']=mirror_pose(d['goal'],n,c);out['table']=d['table'].astype(np.float64);out['table_z']=d['table_z'].astype(np.float64)
    lim=urdf_limits(LEFT_URDF,left);rlim=urdf_limits(RIGHT_URDF,right)[perm]
    relaxed=lim.copy();relaxed[7:,0]=np.minimum(lim[7:,0],rlim[7:,0]);relaxed[7:,1]=np.maximum(lim[7:,1],rlim[7:,1])
    over=np.maximum(lim[:,0]-out['q'],out['q']-lim[:,1]).clip(min=0);arm_ok=(over[:,:7]<1e-4).all(1)
    over_relaxed=np.maximum(relaxed[:,0]-out['q'],out['q']-relaxed[:,1]).clip(min=0)
    keep=arm_ok&(over_relaxed[:,7:]<1e-3).all(1);kind=d['kind'][keep];source=np.flatnonzero(keep)
    sig=signature(LEFT_URDF,object_urdf(a.object_dir),PRISTINE/'isaacsimenvs/cfg/task/SimToolReal.yaml')
    common=dict(kind=kind,source=source,signature=np.array(sig),format_version=np.array(1),limit_lower=relaxed[:,0].astype(np.float32),limit_upper=relaxed[:,1].astype(np.float32))
    raw={k:out[k][keep].astype(np.float32) for k in FIELDS};clamped=dict(raw)
    for k in ('q','previous','current'):clamped[k]=np.clip(out[k][keep],lim[:,0],lim[:,1]).astype(np.float32)
    np.savez_compressed(a.output/'mirrored_raw.npz',**raw,**common);np.savez_compressed(a.output/'mirrored_clamped.npz',**clamped,**common)
    clr=clearance(out['root'][keep],V,out['table_z'][keep]);clr0=clearance(d['root'][keep].astype(np.float64),V,d['table_z'][keep].astype(np.float64))
    report=dict(bank=str(a.bank),states_in=int(len(keep)),states_out=int(keep.sum()),dropped_arm_limit=int((~arm_ok).sum()),dropped_hand_beyond_relaxed=int((arm_ok&~keep).sum()),
        counts={str(k):int((kind==k).sum()) for k in (1,2,3)},mirror_plane=plane,joint7_offset_deg=float(np.rad2deg(J7_OFFSET)),signature=sig,
        left_joint_names=left,relaxed_joints={left[i]:dict(left_deg=np.rad2deg(lim[i]).round(2).tolist(),relaxed_deg=np.rad2deg(relaxed[i]).round(2).tolist()) for i in range(29) if np.abs(relaxed[i]-lim[i]).max()>1e-6},
        beyond_left_limit={left[i]:dict(median_deg=float(np.rad2deg(np.median(over[keep,i]))),frac_over_2deg=float((over[keep,i]>np.deg2rad(2)).mean())) for i in range(29) if (over[keep,i]>np.deg2rad(2)).mean()>.01},
        clearance_check=dict(max_abs_change_m=float(np.abs(clr-clr0).max()),airborne_over_1cm={str(k):float((clr[kind==k]>.01).mean()) for k in (1,2,3)}))
    (a.output/'mirror_report.json').write_text(json.dumps(report,indent=1)+'\n');print(json.dumps(report,indent=1))


if __name__=='__main__':main()
