"""Diagnostic action gradients against CPU MuJoCo; does not certify GPU trajectories."""
import argparse
import json
from pathlib import Path
import time
import mujoco
import numpy as np
import torch
from examples.arm_hand.world_model import load_model,scene_spec


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['world','case','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--anchor-world',action='store_true')
    p.add_argument('--frame',type=int);p.add_argument('--horizon',type=int,default=8)
    p.add_argument('--directions',type=int,default=24);p.add_argument('--epsilon',type=float,default=.01)
    a=p.parse_args();start=time.monotonic();net,ck=load_model(a.world,'cpu');model,spec,context=scene_spec(a.case/'scene.xml')
    if spec!=ck['spec']:raise ValueError('Scene topology differs')
    for parameter in net.parameters():parameter.requires_grad_(False)
    with np.load(a.case/'gpu/trajectory.npz') as f:tr={k:f[k] for k in f.files}
    with np.load(a.case/'reference.npz') as f:ref={k:f[k] for k in f.files}
    T=len(tr['ctrl']);frame=a.frame if a.frame is not None else T//2;H=min(a.horizon,T-frame)
    if H<1:raise ValueError('No probe horizon')
    cpu=mujoco.MjData(model);mujoco.mj_setState(model,cpu,tr['initial_state'],mujoco.mjtState.mjSTATE_FULLPHYSICS)
    q=cpu.qpos.copy() if frame==0 else tr['qpos'][frame-1].copy();v=cpu.qvel.copy() if frame==0 else tr['qvel'][frame-1].copy()
    controls=tr['ctrl'][frame:frame+H];targets=ref['object_qpos'][frame+1:frame+H+1];nu=model.nu;nq=model.nq
    low,high=model.actuator_ctrlrange.T;rng=np.random.default_rng(42);directions=rng.normal(size=(a.directions,nu))
    directions[:,:nu-22]*=.3;directions/=np.linalg.norm(directions,axis=-1,keepdims=True)
    state=torch.tensor(np.r_[q,v],dtype=torch.float32)[None];delta=torch.zeros(1,nu,requires_grad=True)
    c=torch.tensor(context)[None];objective=0.
    for t in range(H):
        control=(torch.tensor(controls[t])[None]+delta).clamp(torch.tensor(low,dtype=torch.float32),torch.tensor(high,dtype=torch.float32))
        next_state=net(state,control,c)
        if a.anchor_world:
            anchor_q=q if t==0 else tr['qpos'][frame+t-1];anchor_v=v if t==0 else tr['qvel'][frame+t-1]
            anchor=torch.tensor(np.r_[anchor_q,anchor_v],dtype=torch.float32)[None]
            observed=torch.tensor(np.r_[tr['qpos'][frame+t],tr['qvel'][frame+t]],dtype=torch.float32)[None]
            with torch.no_grad():bias=observed-net(anchor,torch.tensor(controls[t],dtype=torch.float32)[None],c)
            next_state=next_state+bias;quat=next_state[:,nq-4:nq];quat=quat/quat.norm(dim=-1,keepdim=True).clamp_min(1e-8)
            next_state=torch.cat((next_state[:,:nq-4],quat,next_state[:,nq:]),-1)
        state=next_state;target=torch.tensor(targets[t,:3],dtype=torch.float32)
        objective=objective+3600*(state[:,nq-7:nq-4]-target).square().sum()/H
    objective.backward();predicted=directions@delta.grad.detach().numpy()[0]
    def physical(offset):
        mujoco.mj_resetData(model,cpu);cpu.qpos[:]=q;cpu.qvel[:]=v;mujoco.mj_forward(model,cpu);cpu.qacc_warmstart[:]=0
        cost=0.
        for t in range(H):
            cpu.ctrl[:]=np.clip(controls[t]+offset,low,high)
            for sub in range(5):mujoco.mj_step(model,cpu)
            cost+=3600*np.sum((cpu.qpos[-7:-4]-targets[t,:3])**2)/H
        return cost
    zero=physical(np.zeros(nu));actual=np.array([(physical(a.epsilon*d)-physical(-a.epsilon*d))/(2*a.epsilon) for d in directions])
    denom=np.linalg.norm(predicted)*np.linalg.norm(actual)
    result=dict(kind='CPU-only gradient diagnostic, not GPU acceptance',world=str(a.world.resolve()),checkpoint_update=ck.get('update'),anchored_world=a.anchor_world,frame=frame,horizon=H,
        predicted_objective=float(objective.detach()),cpu_objective=zero,epsilon=a.epsilon,
        directional_cosine=float(np.dot(predicted,actual)/denom) if denom>1e-12 else None,
        sign_agreement=float(np.mean((predicted*actual)>0)),predicted_directional_derivatives=predicted.tolist(),
        cpu_directional_derivatives=actual.tolist(),elapsed_s=time.monotonic()-start)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
