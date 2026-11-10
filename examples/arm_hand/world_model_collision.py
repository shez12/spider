"""First-order differentiable penalties from the original MuJoCo contact geometry.

Only robot/table and hand self contacts audited by the production verifier count.
This is a training objective, never a replacement for independent physical audits.
"""
import hashlib
import mujoco
import numpy as np
import torch
from torch import nn


class _ContactCost(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qpos, geometry):
        shape=qpos.shape;poses=qpos.detach().cpu().double().numpy().reshape(-1,shape[-1])
        costs=np.zeros(len(poses));gradient=np.zeros_like(poses)
        model=geometry.model;data=geometry.data
        jac1=np.zeros((3,model.nv));jac2=np.zeros_like(jac1)
        for i,pose in enumerate(poses):
            data.qpos[:]=pose;mujoco.mj_fwdPosition(model,data)
            # Match the verifier's maximum penetration per category, not number of contacts.
            worst={}
            for c in data.contact:
                a,b=int(c.geom1),int(c.geom2);category=geometry.category(a,b)
                if category is None or c.dist>=0:continue
                if category not in worst or c.dist<worst[category].dist:worst[category]=c
            for c in worst.values():
                a,b=int(c.geom1),int(c.geom2);distance=float(c.dist)
                mujoco.mj_jac(model,data,jac1,None,c.pos,int(model.geom_bodyid[a]))
                mujoco.mj_jac(model,data,jac2,None,c.pos,int(model.geom_bodyid[b]))
                derivative=c.frame[:3]@(jac2-jac1)
                costs[i]+=distance*distance
                # Robot coordinates are scalar hinge/slide qpos; object contacts are excluded.
                gradient[i,:geometry.robot_dofs]+=2*distance*derivative[:geometry.robot_dofs]
        ctx.save_for_backward(torch.as_tensor(gradient.reshape(shape),device=qpos.device,dtype=qpos.dtype))
        return torch.as_tensor(costs.reshape(shape[:-1]),device=qpos.device,dtype=qpos.dtype)

    @staticmethod
    def backward(ctx, incoming):
        gradient,=ctx.saved_tensors
        return incoming.unsqueeze(-1)*gradient,None


class ContactPenalty(nn.Module):
    def __init__(self,model,robot_dofs):
        super().__init__();self.model=model;self.data=mujoco.MjData(model);self.robot_dofs=robot_dofs
        if any(model.jnt_type[j] not in [mujoco.mjtJoint.mjJNT_HINGE,mujoco.mjtJoint.mjJNT_SLIDE] for j in range(model.njnt) if model.jnt_dofadr[j]<robot_dofs):
            raise ValueError('Robot contact gradients require scalar hinge/slide joints')
        self.floor=model.geom('floor').id
        self.names=[model.geom(i).name or '' for i in range(model.ngeom)]

    def category(self,a,b):
        na,nb=self.names[a],self.names[b]
        if self.floor in [a,b]:
            other=nb if a==self.floor else na
            if other.startswith('collision_kuka_'):return 'arm_table'
            if other.startswith('collision_sharpa_'):return 'hand_table'
        if na.startswith('collision_sharpa_') and nb.startswith('collision_sharpa_'):return 'hand_self'
        return None

    def forward(self,qpos):return _ContactCost.apply(qpos,self)


def geometry_signature(model):
    """Reject sharing a contact model if any audited geometry or robot frame differs."""
    h=hashlib.sha256();bodies=set()
    for g in range(model.ngeom):
        name=model.geom(g).name or ''
        if name!='floor' and not name.startswith(('collision_kuka_','collision_sharpa_')):continue
        h.update(name.encode())
        for array in [model.geom_type,model.geom_size,model.geom_pos,model.geom_quat,model.geom_contype,model.geom_conaffinity]:h.update(np.asarray(array[g]).tobytes())
        if model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH:
            mesh=int(model.geom_dataid[g]);v=int(model.mesh_vertadr[mesh]);n=int(model.mesh_vertnum[mesh]);f=int(model.mesh_faceadr[mesh]);nf=int(model.mesh_facenum[mesh])
            h.update(model.mesh_vert[v:v+n].tobytes());h.update(model.mesh_face[f:f+nf].tobytes())
        body=int(model.geom_bodyid[g])
        while body:
            bodies.add(body);body=int(model.body_parentid[body])
    for body in sorted(bodies):
        for array in [model.body_parentid,model.body_pos,model.body_quat]:h.update(np.asarray(array[body]).tobytes())
    for j in range(model.njnt):
        if int(model.jnt_bodyid[j]) not in bodies:continue
        for array in [model.jnt_type,model.jnt_pos,model.jnt_axis,model.jnt_qposadr,model.jnt_dofadr]:h.update(np.asarray(array[j]).tobytes())
        adr=int(model.jnt_qposadr[j]);h.update(model.qpos0[adr:adr+1].tobytes())
    h.update(model.exclude_signature.tobytes());h.update(model.pair_geom1.tobytes());h.update(model.pair_geom2.tobytes())
    return h.hexdigest()
