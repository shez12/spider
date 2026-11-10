"""Frozen-model gradients versus previously logged matched-state real GPU perturbations.
Diagnostic only: reused model-validation data, no fresh held-out success claim.
"""
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
import torch
from examples.arm_hand.world_model import load_model

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--world',type=Path,required=True);p.add_argument('--collection',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2);start=time.monotonic();net,ck=load_model(a.world,'cpu');nq=ck['spec']['nq']
 for v in net.parameters():v.requires_grad_(False)
 rows=[]
 for rec in json.loads(a.collection.read_text())['records']:
  if rec['source_index']!=3:continue
  path=Path(rec['path']);z=np.load(path);states=z['state'];actions=z['action'];context=torch.tensor(z['context'])[None];N=len(actions);half=(N-2)//2
  if N!=128:raise ValueError('Paired layout not preserved')
  assert np.allclose(states[:,0],states[0,0],atol=1e-6)
  ref=np.load(Path(rec['case'])/'reference.npz')['object_qpos'];frame=rec['frame']
  for H in [8,16]:
   target=torch.tensor(ref[frame+1:frame+H+1,:3],dtype=torch.float32);actual=3600*((states[:,1:H+1,nq-7:nq-4]-target.numpy())**2).sum(-1).mean(-1);noise=abs(float(actual[-1]-actual[-2]));delta=(actions[:half,:H]-actions[half:2*half,:H])/2
   for anchored in [False,True]:
    control=torch.tensor(actions[-1,:H],requires_grad=True);state=torch.tensor(states[-1,0])[None];loss=0.
    for t in range(H):
     nxt=net(state,control[t:t+1],context)
     if anchored:
      with torch.no_grad():bias=torch.tensor(states[-1,t+1:t+2])-net(torch.tensor(states[-1,t:t+1]),torch.tensor(actions[-1,t:t+1]),context)
      nxt=nxt+bias;q=nxt[:,nq-4:nq];nxt=torch.cat((nxt[:,:nq-4],q/q.norm(dim=-1,keepdim=True).clamp_min(1e-8),nxt[:,nq:]),-1)
     state=nxt;loss=loss+3600*((state[:,nq-7:nq-4]-target[t])**2).sum()/H
    grad=torch.autograd.grad(loss,control)[0].detach().numpy();pred=np.einsum('htu,tu->h',delta,grad);truth=(actual[:half]-actual[half:2*half])/2;usable=np.isfinite(pred)&np.isfinite(truth)&(np.abs(truth)>max(5*noise,1e-7));den=np.linalg.norm(pred[usable])*np.linalg.norm(truth[usable]);chosen=np.where(pred<=0,np.arange(half),np.arange(half)+half);base=float(actual[-2:].mean());new=actual[chosen];rows.append(dict(frame=frame,horizon=H,anchored=anchored,pairs=half,resolved_pairs=int(usable.sum()),duplicate_gpu_cost_noise=noise,sign_agreement=float(np.mean(pred[usable]*truth[usable]>0)) if usable.any() else None,directional_cosine=float(pred[usable]@truth[usable]/den) if den>0 else None,baseline_real_cost=base,gradient_selected_real_mean_cost=float(new.mean()),random_pair_real_mean_cost=float(actual[:2*half].mean()),gradient_selected_improves_baseline=int((new<base-max(5*noise,1e-7)).sum()),case=rec['case'],shard_sha256=sha(path)))
 result=dict(kind='GPU logged paired gradient diagnostic; source003 was model validation, not unseen test; position objective only, no acceptance or speed claim',world=str(a.world.resolve()),world_sha256=sha(a.world),collection_sha256=sha(a.collection),rows=rows,wall_s=time.monotonic()-start)
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));print(json.dumps(dict(output=str(a.output),rows=len(rows),wall_s=result['wall_s'])))
if __name__=='__main__':main()
