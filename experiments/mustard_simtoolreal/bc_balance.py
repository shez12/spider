"""Offline grasp-configuration audit and bounded trajectory loss weights."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def bounded_weights(groups, lengths, mix=.5):
    groups=np.asarray(groups);lengths=np.asarray(lengths)
    assert 0<=mix<=1 and (lengths>0).all()
    _,inverse,counts=np.unique(groups,return_inverse=True,return_counts=True)
    mass=(1-mix)/len(groups)+mix/(len(counts)*counts[inverse])
    weights=mass/lengths*lengths.sum()
    weights=np.clip(weights,.25,3.)
    return weights/(weights@lengths/lengths.sum())


def load_weights(path, candidate_ids, lengths, dataset):
    data=np.load(path,allow_pickle=False)
    assert np.array_equal(data['candidate_ids'],candidate_ids),'Weight candidate ID mismatch'
    assert np.array_equal(data['valid_lengths'],lengths),'Weight length mismatch'
    assert str(data['dataset_sha256'])==hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),'Weight dataset mismatch'
    w=data['weights']
    assert w.shape==lengths.shape and np.isfinite(w).all() and (w>0).all()
    assert abs(float(w@lengths/lengths.sum())-1)<1e-5
    return w


def rotation(q):
    # Cached actor quaternion order is xyzw. Return object-to-world matrix.
    x,y,z,w=np.asarray(q).T
    return np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                     2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                     2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],axis=-1).reshape(-1,3,3)


def kmeans(features,k,seed=42):
    rng=np.random.default_rng(seed);best=None
    for _ in range(12):
        centers=features[rng.choice(len(features),k,replace=False)].copy()
        for _ in range(80):
            dist=((features[:,None]-centers[None])**2).sum(-1);labels=dist.argmin(1)
            new=np.stack([features[labels==j].mean(0) if (labels==j).any() else centers[j] for j in range(k)])
            if np.allclose(new,centers):break
            centers=new
        cost=float(((features-centers[labels])**2).sum())
        if best is None or cost<best[0]:best=(cost,labels.copy())
    labels=best[1]
    # Small clusters are not independently oversampled.
    while True:
        ids,counts=np.unique(labels,return_counts=True)
        if len(ids)==1 or counts.min()>=12:break
        small=ids[counts.argmin()];others=ids[ids!=small]
        center=features[labels==small].mean(0)
        dest=min(others,key=lambda j:float(((features[labels==j].mean(0)-center)**2).sum()))
        labels[labels==small]=dest
    return np.unique(labels,return_inverse=True)[1]


def audit(dataset, output):
    output.mkdir(parents=True,exist_ok=False)
    d=np.load(dataset);x=d['obs'];lengths=d['valid_lengths'];ids=d['candidate_ids']
    features=[];records=[];phase_counts=np.zeros(4,dtype=int)
    for i,length in enumerate(lengths):
        ob=x[i,:length];rel=ob[:,113:125].reshape(-1,4,3).mean(1);pos=rel+ob[:,87:90]
        tips=ob[:,98:113].reshape(-1,5,3);dist=np.sort(np.linalg.norm(tips-rel[:,None],axis=-1),axis=1)[:,:3].mean(1)
        speed=np.r_[0,np.linalg.norm(np.diff(pos,axis=0),axis=1)*60]
        lift=pos[:,2]-pos[0,2]
        held=(lift>=.05)&(np.linalg.norm(rel,axis=1)<.18)&(dist<.1)&(speed<2.)
        run=np.convolve(held.astype(int),np.ones(30,dtype=int),mode='valid')
        starts=np.flatnonzero(run==30);assert len(starts),f'No stable hold: {ids[i]}'
        first=int(starts[0]);frame=first+29;sl=slice(first,frame+1)
        rot=rotation(ob[sl,94:98])
        palm_obj=np.einsum('nji,nj->ni',rot,-rel[sl])
        tips_obj=np.einsum('nji,nkj->nki',rot,tips[sl]-rel[sl,None])
        # Block-normalized, physically scaled descriptors; no label from success.
        feat=np.r_[ob[sl,7:29].mean(0)/np.sqrt(22),
                   palm_obj.mean(0)/(.1*np.sqrt(3)),tips_obj.mean(0).reshape(-1)/(.1*np.sqrt(15))]
        features.append(feat)
        phases=np.where(np.arange(length)>frame,3,np.where(lift>=.02,2,np.where(dist<.06,1,0)))
        counts=np.bincount(phases,minlength=4);phase_counts+=counts
        c=float(rotation(ob[:1,94:98])[0,2,2]);stratum=0 if c>.5 else 2 if c<-.5 else 1
        records.append(dict(candidate_id=int(ids[i]),valid_frames=int(length),first_hold_frame=frame,
            initial_axis_z_cos=c,stratum=stratum,initial_object_xyz=pos[0].tolist(),phase_frames=counts.tolist()))
    f=np.stack(features);strata=np.array([r['stratum'] for r in records]);groups=np.zeros(len(ids),dtype=int);offset=0
    for stratum in np.unique(strata):
        selected=np.flatnonzero(strata==stratum);labels=kmeans(f[selected],min(4,max(1,len(selected)//20)))
        groups[selected]=labels+offset;offset+=len(np.unique(labels))
    w=bounded_weights(groups,lengths);digest=hashlib.sha256(dataset.read_bytes()).hexdigest()
    np.savez(output/'weights.npz',candidate_ids=ids,valid_lengths=lengths,weights=w,groups=groups,dataset_sha256=np.array(digest))
    summaries=[]
    for g in np.unique(groups):
        sel=np.flatnonzero(groups==g);center=f[sel].mean(0);rep=sel[np.argsort(((f[sel]-center)**2).sum(1))[:3]]
        summaries.append(dict(group=int(g),stratum=int(strata[sel[0]]),trajectories=len(sel),frames=int(lengths[sel].sum()),
            original_frame_mass=float(lengths[sel].sum()/lengths.sum()),weighted_frame_mass=float(w[sel]@lengths[sel]/lengths.sum()),
            representative_candidate_ids=ids[rep].tolist()))
    for i,r in enumerate(records):r.update(group=int(groups[i]),weight=float(w[i]))
    report=dict(dataset=str(dataset),dataset_sha256=digest,trajectories=len(ids),valid_frames=int(lengths.sum()),
        strata_names=['local_z_up','local_z_side','local_z_down'],phase_names=['approach','near_contact','lifting','after_first_hold'],
        phase_frames=phase_counts.tolist(),groups=summaries,weight_min=float(w.min()),weight_max=float(w.max()),
        weighted_effective_trajectories=float((w*lengths).sum()**2/((w*lengths)**2).sum()),records=records,
        interpretation='Configuration clusters, not manual grasp labels. Cached data lacks contact forces. All frames are retained. Fixed frame weights are not exact minibatch-resampled training.')
    (output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.dataset.resolve(),a.output.resolve())
