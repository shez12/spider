"""Re-solve the 7 arm joints of a left-hand reset bank so the arm sits in the joint-space branch of the native starts.

The mirrored right-hand bank (mirror_right_bank.py) reproduces every link pose, but its arm is on another IK branch
than the native resets of the left env (joint 1 differs by ~pi).  For every state this keeps the pose of the link
after arm joint 7 (so the hand, fingers and bottle are untouched) and searches, inside the URDF limits, for the arm
solution closest to a reference configuration (median native start).  CPU only: damped least squares with a
null-space pull towards the reference, started from several seeds; the converged solution with the smallest
max-joint distance to the reference wins.

Arm targets (previous / current) are set to the new q and arm velocities to 0: the old joint-space tracking error and
velocity have no meaning on another branch.  States without a solution keep their kind out of the output.
Not physically validated: run validate_mirrored_bank.py on the output.
"""
import argparse,json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from mirror_right_bank import LEFT_URDF,left_canonical_names,urdf_limits


def arm_chain(urdf,arm):
    """[(4x4 origin, axis or None)] from the root link to the child link of the last arm joint."""
    J={j.find('child').get('link'):j for j in ET.parse(urdf).getroot().findall('joint')};by={j.get('name'):j for j in J.values()}
    link=by[arm[-1]].find('child').get('link');chain=[]
    while link in J:
        j=J[link];o=j.find('origin');T=np.eye(4)
        if o is not None:
            T[:3,3]=[float(x) for x in o.get('xyz','0 0 0').split()];T[:3,:3]=Rot.from_euler('xyz',[float(x) for x in o.get('rpy','0 0 0').split()]).as_matrix()
        axis=np.array([float(x) for x in j.find('axis').get('xyz').split()]) if j.get('type')!='fixed' else None
        chain.append((j.get('name'),T,axis));link=j.find('parent').get('link')
    chain.reverse();assert [n for n,_,a in chain if a is not None]==list(arm),[n for n,_,a in chain if a is not None]
    return [(T,a) for _,T,a in chain]


def fk(chain,q):
    """q: (N, 7) -> (N, 4, 4)."""
    T=np.broadcast_to(np.eye(4),(len(q),4,4)).copy();i=0
    for origin,axis in chain:
        T=T@origin
        if axis is not None:
            R=np.broadcast_to(np.eye(4),(len(q),4,4)).copy();R[:,:3,:3]=Rot.from_rotvec(q[:,i,None]*axis).as_matrix();T=T@R;i+=1
    return T


def pose_error(T,target):
    """(N, 6): position error and rotation vector of target * T^-1, world frame."""
    return np.concatenate([target[:,:3,3]-T[:,:3,3],Rot.from_matrix(target[:,:3,:3]@T[:,:3,:3].transpose(0,2,1)).as_rotvec()],1)


def solve(chain,target,seed,ref,lim,iters=300,damp=1e-3,null=.15,step=.5):
    q=seed.copy();I=np.eye(7)
    for it in range(iters):
        T=fk(chain,q);e=pose_error(T,target);Jc=np.empty((len(q),6,7))
        for k in range(7):
            dq=q.copy();dq[:,k]+=1e-5;Jc[:,:,k]=-(pose_error(fk(chain,dq),target)-e)/1e-5
        JT=Jc.transpose(0,2,1);pinv=JT@np.linalg.inv(Jc@JT+damp*np.eye(6))
        pull=null*(ref-q) if it<iters-40 else 0*q  # final iterations: pure pose convergence
        d=np.einsum('nij,nj->ni',pinv,e)+np.einsum('nij,nj->ni',I-pinv@Jc,pull)
        n=np.abs(d).max(1,keepdims=True);q=np.clip(q+step*d*np.minimum(1,.3/np.maximum(n,1e-9)),lim[:,0],lim[:,1])
    e=pose_error(fk(chain,q),target)
    return q,np.linalg.norm(e[:,:3],axis=1),np.degrees(np.linalg.norm(e[:,3:],axis=1))


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bank',type=Path,required=True);p.add_argument('--native',type=Path,nargs='+',required=True,help='left_grasp_mpc trajectories.npz (first frame = settled native start)')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--pos-tol',type=float,default=1e-3);p.add_argument('--rot-tol-deg',type=float,default=.5)
    p.add_argument('--margin-deg',type=float,default=2.,help='keep arm solutions this far inside the URDF limits')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    names=left_canonical_names();lim=urdf_limits(LEFT_URDF,names)[:7];lim=np.stack([lim[:,0]+np.deg2rad(a.margin_deg),lim[:,1]-np.deg2rad(a.margin_deg)],1)
    chain=arm_chain(LEFT_URDF,names[:7]);d=dict(np.load(a.bank,allow_pickle=True));q0=d['q'][:,:7].astype(np.float64)
    nat=np.concatenate([np.load(f)['q'][:,0,:7] for f in a.native]).astype(np.float64);ref=np.median(nat,0);target=fk(chain,q0)
    flip=lambda q,i:np.stack([np.where(k==i,q[:,k]+np.pi*np.sign(-q[:,k]+1e-9),np.where(k==i+1,-q[:,k],np.where(k==i+2,q[:,k]+np.pi*np.sign(-q[:,k]+1e-9),q[:,k]))) for k in range(7)],1)
    rng=np.random.default_rng(0);seeds=[np.broadcast_to(ref,q0.shape),q0,flip(q0,0),flip(q0,4),flip(flip(q0,0),4)]+[nat[rng.integers(len(nat),size=len(q0))] for _ in range(3)]
    R=np.broadcast_to(ref,q0.shape);best=np.full(len(q0),np.inf);out=q0.copy();ok=np.zeros(len(q0),bool);pe=np.zeros(len(q0));re=np.zeros(len(q0))
    for s in seeds:
        q,ep,er=solve(chain,target,np.clip(s,lim[:,0],lim[:,1]),R,lim);dist=np.abs(q-ref).max(1);good=(ep<a.pos_tol)&(er<a.rot_tol_deg)&(dist<best)
        out[good]=q[good];best[good]=dist[good];ok|=good;pe[good]=ep[good];re[good]=er[good]
    nat_dist=np.abs(nat-ref).max(1);kind=d['kind']
    report=dict(bank=str(a.bank),states=int(len(q0)),solved=int(ok.sum()),reference=ref.round(3).tolist(),native_starts=int(len(nat)),
        native_max_joint_dist_rad_p50_p95_max=[float(np.percentile(nat_dist,50)),float(np.percentile(nat_dist,95)),float(nat_dist.max())],
        before_max_joint_dist_rad_p50=float(np.median(np.abs(q0-ref).max(1))),
        by_kind={str(k):dict(states=int((kind==k).sum()),solved=int(ok[kind==k].sum()),
            max_joint_dist_rad_p10_p50_p90=np.percentile(best[ok&(kind==k)],[10,50,90]).round(3).tolist() if (ok&(kind==k)).any() else None,
            within_native_p95=float((best[kind==k]<=np.percentile(nat_dist,95)).mean()),within_native_max=float((best[kind==k]<=nat_dist.max()).mean()),within_1rad=float((best[kind==k]<1.).mean()),
            per_joint_abs_dist_p50=np.median(np.abs(out-ref)[ok&(kind==k)],0).round(2).tolist() if (ok&(kind==k)).any() else None) for k in sorted(set(kind.tolist()))},
        pose_error_solved=dict(pos_m_max=float(pe[ok].max()) if ok.any() else None,rot_deg_max=float(re[ok].max()) if ok.any() else None))
    new={k:v[ok] if getattr(v,'ndim',0)>=1 and len(v)==len(q0) else v for k,v in d.items()}
    for k in ('q','previous','current'):new[k]=new[k].copy();new[k][:,:7]=out[ok].astype(np.float32)
    new['v']=new['v'].copy();new['v'][:,:7]=0
    np.savez_compressed(a.output/'resolved.npz',**new,arm_dist_to_native=best[ok].astype(np.float32))
    (a.output/'resolve_report.json').write_text(json.dumps(report,indent=1)+'\n');print(json.dumps(report,indent=1))


if __name__=='__main__':main()
