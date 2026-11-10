"""Feedback-guided spatial proposals; never changes references or validity gates."""
import numpy as np

SCALE=np.array([10.,.05,.05])


def guidance(feedback):
    physical=[row for row in feedback if row['stage']=='physical_success']
    ik=[row for row in feedback if row['stage']=='ik_success']
    if physical:return physical,'physical_success'
    if ik:return ik,'ik_only'
    failed=sorted((row for row in feedback if row.get('geometry_score') is not None),key=lambda row:row['geometry_score'])
    if failed:return failed[:max(4,int(np.ceil(len(failed)*.1)))],'geometry_near_miss_not_success'
    return [],'no_feedback'


def proposals(feedback,count,seed,adaptive,min_separation=.1):
    """Exact 1-in-4 global proposal slots, plus disclosed adaptive fallbacks.

    Distance is Euclidean in yaw/10deg, dx/.05m, dy/.05m. The same novelty
    constraint is applied to both test arms against prior accepted and new points.
    """
    if count%4:raise ValueError('Count must be divisible by four for exact 25/75 proposal slots')
    rng=np.random.default_rng(seed);anchors,stage=guidance(feedback)
    prior=[np.asarray(row['point'])/SCALE for row in feedback if row['stage']=='physical_success']
    chosen=[];result=[]
    for index in range(count):
        local=bool(adaptive and index%4!=0)
        fallback=False;point=None
        for attempt in range(2000):
            if local and anchors and attempt<1500:
                anchor=np.asarray(anchors[int(rng.integers(len(anchors)))]['point'])/SCALE
                raw=anchor+rng.normal(0,[.25,.15,.15],3)
                if (np.abs(raw)>1).any():continue  # rejection, not boundary clipping
            else:
                raw=rng.uniform(-1,1,3)
                if local:fallback=True
            others=prior+chosen
            if others and np.linalg.norm(np.asarray(others)-raw,axis=1).min()<min_separation:continue
            point=raw;break
        if point is None:raise RuntimeError('Cannot satisfy diversity guard; do not silently relax it')
        chosen.append(point)
        result.append(dict(index=index+1,point=(point*SCALE).tolist(),proposal_slot='adaptive' if local else 'global',
            proposal_used='global_fallback' if fallback else ('adaptive' if local else 'global'),
            guidance_stage=stage if local else 'global_exploration',min_normalized_separation=min_separation))
    return result


def coverage(points):
    points=np.asarray(points,dtype=float).reshape(-1,3)
    if not len(points):return dict(count=0,occupied_bins=0,min_normalized_distance=None,span=None)
    normalized=points/SCALE
    bins=np.minimum(3,np.floor((normalized+1)*2).astype(int))
    distance=np.linalg.norm(normalized[:,None]-normalized[None,:],axis=-1)
    np.fill_diagonal(distance,np.inf)
    return dict(count=len(points),occupied_bins=len({tuple(x)for x in bins}),total_bins=64,
        min_normalized_distance=float(distance.min()) if len(points)>1 else None,
        span=(points.max(0)-points.min(0)).tolist())
