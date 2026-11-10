"""Keep actual candidate rollouts satisfying cheap tracking gates for exact certification."""
import hashlib
import math
import torch


class StrictCandidateBank:
    def __init__(self,capacity):
        self.capacity=capacity;self.rows=[];self.checked=set()

    def observe(self,offsets,costs,commands,qpos,qvel,object_reference):
        if self.capacity<=0:return
        target=torch.as_tensor(object_reference,dtype=qpos.dtype,device=qpos.device)
        rmse=(qpos[:,:,-7:-4]-target[None,1:,:3]).square().sum(-1).mean(-1).sqrt()
        final=(qpos[:,-1,-7:-4]-target[-1,:3]).norm(dim=-1)
        quat=qpos[:,-1,-4:];quat=quat/quat.norm(dim=-1,keepdim=True).clamp_min(1e-12)
        goal=target[-1,3:];goal=goal/goal.norm().clamp_min(1e-12)
        dot=(quat*goal).sum(-1).abs()
        good=torch.isfinite(costs)&(rmse<.030001)&(final<.030001)&(dot>math.cos(math.radians(10.0001)))
        ids=torch.nonzero(good).flatten()
        if not len(ids):return
        ids=ids[torch.argsort(costs[ids])[:self.capacity]]
        existing={r['signature'] for r in self.rows}
        for index in ids.tolist():
            signature=hashlib.sha256(commands[index].detach().cpu().numpy().tobytes()).hexdigest()
            if signature in self.checked or signature in existing:continue
            self.rows.append(dict(signature=signature,cost=float(costs[index]),offsets=offsets[index].clone(),
                rollout=dict(qpos=qpos[index].clone(),qvel=qvel[index].clone(),ctrl=commands[index].clone())))
            existing.add(signature)
        self.rows.sort(key=lambda item:item['cost']);del self.rows[self.capacity:]

    def take(self):
        rows=self.rows;self.rows=[]
        self.checked.update(row['signature'] for row in rows)
        return rows
