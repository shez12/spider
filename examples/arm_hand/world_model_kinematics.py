"""Differentiable MuJoCo hinge/slide FK for exact reference tracking costs."""
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation


class Kinematics:
    def __init__(self,model,site_names,device):
        self.sites=[model.site(name).id for name in site_names]
        needed={0}
        for site in self.sites:
            b=int(model.site_bodyid[site])
            while b:
                needed.add(b);b=int(model.body_parentid[b])
        self.entries=[];self.order=[0];depth={0:0}
        tensor=lambda x:torch.as_tensor(np.asarray(x).copy(),dtype=torch.float32,device=device)
        for b in sorted(needed-{0}):
            jid=int(model.body_jntadr[b]);nj=int(model.body_jntnum[b]);parent=int(model.body_parentid[b])
            if nj>1 or (nj and model.jnt_type[jid] not in (mujoco.mjtJoint.mjJNT_HINGE,mujoco.mjtJoint.mjJNT_SLIDE)):
                raise ValueError('FK supports static, hinge and slide bodies')
            depth[b]=depth[parent]+1
            quat=model.body_quat[b]
            rot=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
            self.entries.append(dict(body=b,parent=parent,depth=depth[b],offset=model.body_pos[b],rot=rot,
                axis=model.jnt_axis[jid] if nj else [0,0,1],anchor=model.jnt_pos[jid] if nj else [0,0,0],
                qadr=int(model.jnt_qposadr[jid]) if nj else 0,
                q0=model.qpos0[model.jnt_qposadr[jid]] if nj else 0,
                hinge=bool(nj and model.jnt_type[jid]==mujoco.mjtJoint.mjJNT_HINGE),
                slide=bool(nj and model.jnt_type[jid]==mujoco.mjtJoint.mjJNT_SLIDE)))
        self.levels=[]
        for d in range(1,max(depth.values(),default=0)+1):
            es=[e for e in self.entries if e['depth']==d]
            level={k:tensor([e[k] for e in es]) for k in ['offset','rot','axis','anchor','q0','hinge','slide']}
            level['parent']=torch.tensor([self.order.index(e['parent']) for e in es],device=device)
            level['qadr']=torch.tensor([e['qadr'] for e in es],device=device)
            self.order.extend(e['body'] for e in es);self.levels.append(level)
        self.site_bodies=[self.order.index(int(model.site_bodyid[s])) for s in self.sites]
        self.site_pos=tensor(model.site_pos[self.sites])
        self.site_rot=tensor([Rotation.from_quat(model.site_quat[s][[1,2,3,0]]).as_matrix() for s in self.sites])

    def __call__(self,q):
        shape=q.shape[:-1];pos=q.new_zeros(*shape,1,3)
        rot=torch.eye(3,device=q.device,dtype=q.dtype).expand(*shape,1,3,3)
        eye=torch.eye(3,device=q.device,dtype=q.dtype)
        for e in self.levels:
            parent=e['parent'];pp=pos[...,parent,:];pr=rot[...,parent,:,:]
            angle=(q[...,e['qadr']]-e['q0'])*e['hinge']
            x,y,z=e['axis'].unbind(-1);zero=torch.zeros_like(x)
            skew=torch.stack((zero,-z,y,z,zero,-x,-y,x,zero),-1).reshape(-1,3,3)
            rr=eye+angle.sin()[...,None,None]*skew+(1-angle.cos())[...,None,None]*(skew@skew)
            localrot=e['rot']@rr
            anchor=e['anchor'];off=e['offset']+(e['rot']@(anchor-(rr@anchor[...,None]).squeeze(-1))[...,None]).squeeze(-1)
            off=off+(e['rot']@e['axis'][...,None]).squeeze(-1)*((q[...,e['qadr']]-e['q0'])*e['slide'])[...,None]
            nextpos=pp+(pr@off[...,None]).squeeze(-1);nextrot=pr@localrot
            pos=torch.cat((pos,nextpos),-2);rot=torch.cat((rot,nextrot),-3)
        sp=pos[...,self.site_bodies,:]+(rot[...,self.site_bodies,:,:]@self.site_pos[...,None]).squeeze(-1)
        sr=rot[...,self.site_bodies,:,:]@self.site_rot
        return sp,sr
