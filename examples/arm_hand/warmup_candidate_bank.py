"""Shortlist actual warmup rollouts without imposing a final-orientation gate.

Changes feasible export selection only. Original SPIDER costs, sampled actions,
search updates and random draws remain untouched. Every shortlisted trace still
requires full geometry/grasp audit and independent GPU replay.
"""
import hashlib
import torch
from examples.arm_hand.strict_candidate_bank import StrictCandidateBank


class WarmupCandidateBank(StrictCandidateBank):
    def __init__(self,capacity):
        super().__init__(capacity)
        self.statistics=dict(observe_calls=0,worlds_observed=0,eligible_worlds=0,bank_entries_staged=0)

    def observe(self,offsets,costs,commands,qpos,qvel,object_reference):
        if self.capacity<=0:return
        target=torch.as_tensor(object_reference,dtype=qpos.dtype,device=qpos.device)
        rmse=(qpos[:,:,-7:-4]-target[None,1:,:3]).square().sum(-1).mean(-1).sqrt()
        final=(qpos[:,-1,-7:-4]-target[-1,:3]).norm(dim=-1)
        good=(torch.isfinite(costs)&torch.isfinite(qpos).flatten(1).all(1)&
              torch.isfinite(qvel).flatten(1).all(1)&torch.isfinite(commands).flatten(1).all(1)&
              (rmse<.030001)&(final<.030001)&(qpos[:,-1,-4:].norm(dim=-1)>1e-6))
        self.statistics['observe_calls']+=1
        self.statistics['worlds_observed']+=len(costs)
        self.statistics['eligible_worlds']+=int(good.sum())
        ids=torch.nonzero(good).flatten()
        if not len(ids):return
        ids=ids[torch.argsort(costs[ids])[:self.capacity]]
        existing={r['signature'] for r in self.rows}
        for index in ids.tolist():
            signature=hashlib.sha256(commands[index].detach().cpu().numpy().tobytes()).hexdigest()
            if signature in self.checked or signature in existing:continue
            self.rows.append(dict(signature=signature,cost=float(costs[index]),offsets=offsets[index].clone(),
                rollout=dict(qpos=qpos[index].clone(),qvel=qvel[index].clone(),ctrl=commands[index].clone())))
            self.statistics['bank_entries_staged']+=1
            existing.add(signature)
        self.rows.sort(key=lambda item:item['cost']);del self.rows[self.capacity:]
