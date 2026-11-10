"""Pre-grasp goal relabeling for BC: the demonstrated approach does not depend on the goal.

Observation keypoints_rel_goal (obs[125:137]) = object keypoints - goal keypoints, both
world-frame fixed-size corners (obs_utils.KEYPOINT_CORNERS * 0.5 * keypoint_scale *
fixed_size). Object keypoints are recovered exactly as keypoints_rel_palm + palm.
"""
import json
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
CORNERS=((1,1,1),(1,1,-1),(-1,-1,1),(-1,-1,-1))
GOAL_SLICE=slice(125,137)


def keypoint_offsets(keypoint_scale=1.5,fixed_size=None):
    if fixed_size is None:fixed_size=json.loads((ROOT/'assets/object/calibration.json').read_text())['bbox_dims_m']
    return torch.tensor(CORNERS,dtype=torch.float32)*(.5*keypoint_scale*torch.tensor(fixed_size,dtype=torch.float32))


def quat_apply(q,v):
    """Rotate v (...,3) by wxyz quaternions q broadcastable to v[...,:1]."""
    w,u=q[...,:1],q[...,1:];t=2*torch.cross(u.expand_as(v),v,dim=-1)
    return v+w*t+torch.cross(u.expand_as(v),t,dim=-1)


def sample_goal_keypoints(n,offsets,mins,maxs,generator=None,device='cpu'):
    """Native absolute goal sampler: uniform position in the target volume, uniform SO(3)."""
    mins=torch.as_tensor(mins,dtype=torch.float32,device=device);maxs=torch.as_tensor(maxs,dtype=torch.float32,device=device)
    pos=mins+(maxs-mins)*torch.rand((n,3),generator=generator,device=device)
    q=torch.randn((n,4),generator=generator,device=device);q=q/q.norm(dim=-1,keepdim=True)
    return quat_apply(q[:,None],offsets.to(device)[None].expand(n,-1,-1))+pos[:,None]


def object_keypoints(obs):
    return obs[...,113:125].reshape(*obs.shape[:-1],4,3)+obs[...,None,87:90]


def relabel(obs,goal_kp,start,boundary,active):
    """Replace the goal observation on frames start+k < boundary of active trajectories.

    obs (B,T,140) slice beginning at absolute frame `start`; goal_kp (B,4,3); boundary,active (B,).
    """
    frames=start+torch.arange(obs.shape[1],device=obs.device)
    mask=(active[:,None]&(frames[None]<boundary[:,None]))[...,None]
    new=(object_keypoints(obs)-goal_kp[:,None]).reshape(*obs.shape[:2],12)
    out=obs.clone();out[...,GOAL_SLICE]=torch.where(mask,new,obs[...,GOAL_SLICE])
    return out


def pregrasp_boundaries(obs,lengths):
    """Obs index of the first held state of the first 30-step hold (truncated_selection proxy); length if never held."""
    from truncated_selection import HOLD_STEPS,first_hold_end,hold_mask
    out=[]
    for o,length in zip(obs,lengths):
        end,_=first_hold_end(hold_mask(o),length)
        out.append(int(length) if end is None else end-HOLD_STEPS+2)
    return np.array(out)


def stratified_split(candidate_ids,groups,fraction,seed=0):
    """Hold out round(fraction*n) trajectories per group (>=1 for groups with >=10)."""
    rng=np.random.default_rng(seed);val=[]
    groups=np.asarray(groups)
    for g in np.unique(groups):
        idx=np.flatnonzero(groups==g);k=int(round(fraction*len(idx)))
        if len(idx)>=10:k=max(k,1)
        val.extend(rng.permutation(idx)[:k].tolist())
    val=np.array(sorted(val),dtype=int);train=np.setdiff1d(np.arange(len(candidate_ids)),val)
    return train,val
