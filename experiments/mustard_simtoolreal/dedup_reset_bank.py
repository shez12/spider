"""Remove near-duplicate states from a merged, validated reset bank (greedy, bank order = priority).

A state is a duplicate of an already kept state of the same kind when all hold: object position
< --pos m, object orientation < --rot deg, arm joint RMS < --arm rad, hand joint RMS < --hand rad
and goal position < --goal m. Validation records are filtered in the same order.
"""
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R


def duplicates(data,pos,rot,arm,hand,goal):
    keep=np.zeros(len(data['kind']),dtype=bool);root=data['root'];q=data['q'];g=data['goal'][:,:3]
    rots=R.from_quat(root[:,[4,5,6,3]])
    for k in np.unique(data['kind']):
        kept=[]
        for i in np.flatnonzero(data['kind']==k):
            if kept:
                j=np.array(kept)
                close=(np.linalg.norm(root[j,:3]-root[i,:3],axis=1)<pos)&(np.linalg.norm(g[j]-g[i],axis=1)<goal)
                close&=np.sqrt(((q[j,:7]-q[i,:7])**2).mean(1))<arm
                close&=np.sqrt(((q[j,7:]-q[i,7:])**2).mean(1))<hand
                if close.any():
                    jj=j[close];close_rot=np.degrees((rots[jj].inv()*rots[i]).magnitude())<rot
                    if close_rot.any():continue
            kept.append(i);keep[i]=True
    return keep


def main(a):
    assert not a.output.exists()
    with np.load(a.source/'reset_bank.npz') as z:data={k:z[k] for k in z.files}
    report=json.loads((a.source/'validation.json').read_text());records=report['records'];assert len(records)==len(data['kind'])
    keep=duplicates(data,a.pos,a.rot,a.arm,a.hand,a.goal)
    out={k:(v[keep] if v.ndim>0 and len(v)==len(keep) else v) for k,v in data.items()}
    a.output.mkdir(parents=True);np.savez_compressed(a.output/'reset_bank.npz',**out)
    removed=Counter((Path(r.get('source_bank','?')).name,r['kind']) for r,k in zip(records,keep) if not k)
    summary=dict(source=str(a.source.resolve()),before=len(keep),after=int(keep.sum()),removed=int((~keep).sum()),
        removed_by_bank_kind={f'{b}/kind{k}':v for (b,k),v in sorted(removed.items())},
        counts={str(k):int((out['kind']==k).sum()) for k in (1,2,3)},
        thresholds=dict(object_pos_m=a.pos,object_rot_deg=a.rot,arm_rms_rad=a.arm,hand_rms_rad=a.hand,goal_pos_m=a.goal),
        bank_sha256=hashlib.sha256((a.output/'reset_bank.npz').read_bytes()).hexdigest())
    (a.output/'validation.json').write_text(json.dumps(dict(summary,records=[r for r,k in zip(records,keep) if k]),indent=2)+'\n')
    (a.output/'completed.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pos',type=float,default=.01);p.add_argument('--rot',type=float,default=5.)
    p.add_argument('--arm',type=float,default=.05);p.add_argument('--hand',type=float,default=.1);p.add_argument('--goal',type=float,default=.03)
    main(p.parse_args())
