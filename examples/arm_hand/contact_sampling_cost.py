"""Experimental GPU cost for the same hand-self contacts audited at export."""
import torch


def hand_self_penalty(geoms,distances,worlds,active_count,hand_geoms,nworld,tolerance=.0005):
    a,b=geoms[:,0].long(),geoms[:,1].long();w=worlds.long();ng=len(hand_geoms)
    valid=((torch.arange(len(distances),device=distances.device)<active_count)&
           (a>=0)&(a<ng)&(b>=0)&(b<ng)&(w>=0)&(w<nworld))
    pair=hand_geoms[a.clamp(0,ng-1)]&hand_geoms[b.clamp(0,ng-1)]
    values=torch.where(valid&pair,(-distances-tolerance).clamp_min(0).square(),0)
    result=distances.new_zeros(nworld)
    return result.scatter_add_(0,w.clamp(0,nworld-1),values)
