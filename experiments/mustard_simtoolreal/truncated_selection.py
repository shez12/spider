"""Truncate long native teacher demonstrations after the grasp instead of discarding them.

The teacher collection (train_bc.collect) evaluates its hold proxy online after
each action t, i.e. on cached observation t+1: object rise >=5cm, object within
18cm of the palm, mean nearest-3 fingertip distance <10cm, object speed <2m/s,
for 30 consecutive steps inside the valid trajectory. Object velocity is not in
the actor observation, so speed is reconstructed by finite differences.
"""
import numpy as np

VERSION='truncate_after_hold_v1'
HOLD_STEPS=30


def hold_mask(obs,dt=1/60):
    """Per-action hold flags (length T-1) from cached observations (T,140)."""
    palm=obs[:,87:90];rel=obs[:,113:125].reshape(-1,4,3).mean(1)
    position=rel+palm;tips=obs[:,98:113].reshape(-1,5,3)+palm[:,None]
    nxt=slice(1,None)
    height=position[nxt,2]-position[0,2]
    close=np.linalg.norm(rel[nxt],axis=-1)<.18
    fingers=np.sort(np.linalg.norm(tips[nxt]-position[nxt,None],axis=-1),axis=-1)[:,:3].mean(-1)<.10
    speed=np.linalg.norm(np.diff(position,axis=0),axis=-1)/dt
    return (height>=.05)&close&fingers&(speed<2.)


def first_hold_end(hold,valid_length,steps=HOLD_STEPS):
    """Action index at which the first full hold completes inside the valid range, else None; plus the longest run."""
    run=longest=0;end=None
    for t in range(min(len(hold),int(valid_length))):
        run=run+1 if hold[t] else 0;longest=max(longest,run)
        if end is None and run>=steps:end=t
    return end,longest


def cutoff(valid_length,hold_end,max_steps=960,margin=60):
    """Keep full demos that fit; truncate longer ones `margin` steps after the hold; None means reject."""
    if hold_end is None:return None
    if valid_length<=max_steps:return int(valid_length)
    cut=min(int(valid_length),hold_end+1+margin)
    return cut if cut<=max_steps else None
