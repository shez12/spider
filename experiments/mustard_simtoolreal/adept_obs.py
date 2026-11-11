"""ADEPT-style object observations (arXiv 2608.19182, Tab. 6/7) for SimToolReal; reward/success unchanged.

Works with both the unmodified SimToolReal checkout and third_party/simtoolreal: it only patches obs_utils helpers.
Import after AppLauncher starts Isaac Sim.
"""
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
# obs_utils.KEYPOINT_CORNERS (copied: importing obs_utils needs a running Kit app).
KEYPOINT_CORNERS=((1,1,1),(1,1,-1),(-1,-1,1),(-1,-1,-1))
ADEPT_OBS=('joint_pos','joint_vel','prev_action_targets','palm_pos','palm_rot','fingertip_pos_rel_palm',
    'object_pos','object_rot','goal_pos','goal_rot','object_scales','object_pointcloud')
ADEPT_STATE=('joint_pos','joint_vel','prev_action_targets','palm_pos','palm_rot','palm_vel','object_pos','object_rot','object_vel',
    'fingertip_pos_rel_palm','goal_pos','goal_rot','object_scales','object_pointcloud','closest_keypoint_max_dist',
    'closest_fingertip_dist','lifted_object','progress','successes','reward')


def object_dir_of(object_dir=None):
    return Path(object_dir).resolve() if object_dir else HERE/'assets/object'


def object_urdf(object_dir=None):
    d=object_dir_of(object_dir);u=sorted(d.glob('*.urdf'));assert len(u)==1,f'expected exactly one URDF in {d}';return u[0]


def mustard_surface_points(k,seed=0,object_dir=None):
    """K farthest-point-sampled surface points of the mustard visual mesh, in the URDF link frame (m)."""
    import trimesh,xml.etree.ElementTree as ET
    root=ET.parse(object_urdf(object_dir)).getroot();vis=root.find('.//visual')
    origin=np.array([float(x) for x in vis.find('origin').get('xyz').split()]) if vis.find('origin') is not None else np.zeros(3)
    mesh=trimesh.load(object_dir_of(object_dir)/vis.find('.//mesh').get('filename'),force='mesh')
    dense=np.asarray(trimesh.sample.sample_surface_even(mesh,8192,seed=seed)[0])
    idx=[int(np.argmax(np.linalg.norm(dense-dense.mean(0),axis=1)))];d=np.linalg.norm(dense-dense[idx[0]],axis=1)
    for _ in range(k-1):idx.append(int(np.argmax(d)));d=np.minimum(d,np.linalg.norm(dense-dense[idx[-1]],axis=1))
    return (dense[idx]+origin).astype(np.float32)


def _quat_apply_xyzw(q,v):
    """Rotate v (...,K,3) by xyzw quaternions q (...,4)."""
    import torch
    u=q[...,None,:3];w=q[...,None,3:]
    t=2*torch.cross(u.expand_as(v),v,dim=-1)
    return v+w*t+torch.cross(u.expand_as(v),t,dim=-1)


def _matrix_to_quat_xyzw(R):
    """Rotation matrices (...,3,3) -> unit xyzw quaternions with w>=0."""
    import torch
    m=R;t=m[...,0,0]+m[...,1,1]+m[...,2,2]
    q=torch.stack([1+m[...,0,0]-m[...,1,1]-m[...,2,2],1-m[...,0,0]+m[...,1,1]-m[...,2,2],1-m[...,0,0]-m[...,1,1]+m[...,2,2],1+t],-1)
    i=q.argmax(-1,keepdim=True)
    xyzw=torch.stack([
        torch.stack([1+m[...,0,0]-m[...,1,1]-m[...,2,2],m[...,0,1]+m[...,1,0],m[...,0,2]+m[...,2,0],m[...,2,1]-m[...,1,2]],-1),
        torch.stack([m[...,0,1]+m[...,1,0],1-m[...,0,0]+m[...,1,1]-m[...,2,2],m[...,1,2]+m[...,2,1],m[...,0,2]-m[...,2,0]],-1),
        torch.stack([m[...,0,2]+m[...,2,0],m[...,1,2]+m[...,2,1],1-m[...,0,0]-m[...,1,1]+m[...,2,2],m[...,1,0]-m[...,0,1]],-1),
        torch.stack([m[...,2,1]-m[...,1,2],m[...,0,2]-m[...,2,0],m[...,1,0]-m[...,0,1],1+t],-1)],-2)
    q=torch.gather(xyzw,-2,i[...,None].expand(*i.shape[:-1],1,4)).squeeze(-2)
    q=q/q.norm(dim=-1,keepdim=True)
    return q*torch.where(q[...,3:]<0,-1.,1.)


def native_to_adept(obs,pts,object_base_size=.04,keypoint_scale=1.5):
    """Exact conversion of clean native 140-D actor observations to the 318-D ADEPT layout.

    Native layout: joint_pos/vel/prev 0:87, palm_pos 87:90, palm_rot 90:94, object_rot 94:98, fingertips 98:113,
    keypoints_rel_palm 113:125, keypoints_rel_goal 125:137, object_scales 137:140. The 4 keypoint corners sum to zero,
    so their mean is the object/goal position; the goal rotation is the Kabsch fit of the (coplanar, rank-2) corners.
    """
    import torch
    assert obs.shape[-1]==140
    lead=obs.shape[:-1];o=obs.reshape(-1,140);pts=torch.as_tensor(pts,device=o.device,dtype=o.dtype)
    palm=o[:,87:90];obj_rot=o[:,94:98];scales=o[:,137:140]
    obj_kp=o[:,113:125].reshape(-1,4,3)+palm[:,None];goal_kp=obj_kp-o[:,125:137].reshape(-1,4,3)
    obj_pos=obj_kp.mean(1);goal_pos=goal_kp.mean(1)
    local=torch.tensor(KEYPOINT_CORNERS,device=o.device,dtype=o.dtype)[None]*(scales*object_base_size*keypoint_scale*.5)[:,None]
    H=local.transpose(1,2)@(goal_kp-goal_pos[:,None])
    U,_,Vh=torch.linalg.svd(H.double());V=Vh.transpose(1,2)
    d=torch.sign(torch.linalg.det(V@U.transpose(1,2)))
    D=torch.diag_embed(torch.stack([torch.ones_like(d),torch.ones_like(d),d],-1))
    goal_rot=_matrix_to_quat_xyzw(V@D@U.transpose(1,2)).to(o.dtype)
    cloud=_quat_apply_xyzw(obj_rot,pts[None].expand(len(o),-1,-1))+obj_pos[:,None]
    out=torch.cat([o[:,:94],o[:,98:113],obj_pos,obj_rot,goal_pos,goal_rot,scales,cloud.reshape(len(o),-1)],-1)
    return out.reshape(*lead,out.shape[-1])


def create_adept_env(create_env,num_envs,configure,output,k=64):
    """environment.create_env with ADEPT observation lists.

    environment.py (hashed into the reset-bank signature) asserts the native 140/162 dims, so the env instance is
    captured and the assertion replaced by an exact check against the ADEPT dims.
    """
    import isaacsimenvs.tasks.simtoolreal.simtoolreal_env as module
    dims=install_adept_obs(k,output);built=[];cls=module.SimToolRealEnv
    def configure_adept(cfg):
        configure(cfg);cfg.obs.obs_list=list(ADEPT_OBS);cfg.obs.state_list=list(ADEPT_STATE)
    module.SimToolRealEnv=lambda *x,**kw:built.append(cls(*x,**kw)) or built[-1]
    try:return create_env(num_envs,configure=configure_adept)
    except AssertionError:
        env=built[-1];assert (env.cfg.observation_space,env.cfg.state_space)==dims,(env.cfg.observation_space,env.cfg.state_space,dims)
        return env
    finally:module.SimToolRealEnv=cls


def install_adept_obs(k,output,object_dir=None):
    """Adds object_pos, goal_pos, goal_rot (xyzw) and object_pointcloud (K points, env frame) observation fields.

    Actor fields use the DR-delayed/noisy object pose (same noise as the native keypoint fields); critic fields are clean.
    The noisy pose is captured from _apply_object_state_dr; the actor stack is recognised by its object_vel tensor.
    Returns the expected (actor, critic) dims for the default 29-joint/5-fingertip robot.
    """
    import torch
    from isaaclab.utils.math import convert_quat
    from isaacsimenvs.tasks.simtoolreal.utils import obs_utils as ou
    pts=mustard_surface_points(k,object_dir=object_dir);np.save(Path(output)/'pointcloud_object_frame.npy',pts)
    ou.OBS_FIELD_SIZES.update(object_pos=3,goal_pos=3,goal_rot=4,object_pointcloud=3*k)
    cap={};orig_joint,orig_dr,orig_stack=ou._canonical_joint_obs,ou._apply_object_state_dr,ou._stack_obs_dict
    def joint(env):
        cap.clear();cap['env']=env;return orig_joint(env)
    def dr(env,*args):
        out=orig_dr(env,*args);cap['noisy']=out;return out
    def stack(d,fields):
        env=cap['env'];o=env.scene.env_origins;noisy=('noisy' in cap and d['object_vel'] is cap['noisy'][2])
        pos,rot=(cap['noisy'][0],cap['noisy'][1]) if noisy else (env.object.data.root_pos_w-o,env.object.data.root_quat_w)
        offs=torch.as_tensor(pts,device=env.device).unsqueeze(0)*env._object_scale_multiplier.unsqueeze(1)
        goal_rot=convert_quat(env.goal_viz.data.root_quat_w,to='xyzw');goal_rot=goal_rot*torch.where(goal_rot[:,3:]<0,-1.,1.)
        d=dict(d,object_pos=pos,goal_pos=env.goal_viz.data.root_pos_w-o,goal_rot=goal_rot,
            object_pointcloud=ou._keypoints_world(pos,rot,offs.expand(env.num_envs,-1,-1)))
        return orig_stack(d,fields)
    ou._canonical_joint_obs,ou._apply_object_state_dr,ou._stack_obs_dict=joint,dr,stack
    return ou.compute_obs_dim(ADEPT_OBS),ou.compute_obs_dim(ADEPT_STATE)
