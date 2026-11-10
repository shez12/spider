"""Dump the effective robot setup of the pristine left-hand env (0b) or our right-hand env (0c) for comparison.

--side left : /home/szc/work/newsimtoolreal (run with cwd=/home/szc/work/newsimtoolreal/simtoolreal), as train_pristine_mustard.py
--side right: third_party/simtoolreal + environment.create_env, as train_scratch.py --pristine-settings
Reads joint limits / PD / armature / friction / effort / velocity limits as loaded in PhysX, body masses, contact and
rest offsets, max depenetration velocity, then puts the arm at its default pose and reports palm-center / fingertip
geometry, and how fingertips move when each finger flexes.
"""
import argparse,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[1]
PRISTINE=Path('/home/szc/work/newsimtoolreal/simtoolreal');VENDOR=PROJECT/'third_party/simtoolreal'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--side',choices=('left','right'),required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--num-envs',type=int,default=4)
    p.add_argument('--align-left-physics',action='store_true',help='right only: apply physics_align.install() first')
    p.add_argument('--restrict-abduction',action='store_true',help='right only, with --align-left-physics')
    root=PRISTINE if p.parse_known_args()[0].side=='left' else VENDOR
    sys.path[:0]=[str(root),str(root/'rl_games'),str(HERE)]
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
    try:run(a)
    except BaseException:
        import traceback;traceback.print_exc();raise
    finally:sys.stdout.flush();sys.stderr.flush();app.close()


def tolist(x):
    import torch
    return x.detach().cpu().tolist() if torch.is_tensor(x) else x


def run(a):
    import torch,yaml
    cal=json.loads((HERE/'assets/object/calibration.json').read_text())
    if a.side=='left':
        from isaacsimenvs.tasks.simtoolreal.simtoolreal_env import SimToolRealEnv
        from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg
        from isaacsimenvs.tasks.simtoolreal.utils import obs_utils
        cfg=SimToolRealEnvCfg();cfg.from_dict(yaml.safe_load((PRISTINE/'isaacsimenvs/cfg/task/SimToolReal.yaml').read_text()))
        cfg.seed=42;cfg.scene.num_envs=a.num_envs
        cfg.assets.object_urdf=str(HERE/'assets/object/mustard_bottle.urdf');cfg.assets.object_scale=tuple(cal['policy_object_scale'])
        cfg.reward.fixed_size=tuple(cal['bbox_dims_m'])
        env=SimToolRealEnv(cfg);palm_offset=obs_utils.PALM_CENTER_OFFSET;tip_offsets=[obs_utils.FINGERTIP_OFFSET]*5
    else:
        from environment import create_env
        if a.align_left_physics:
            import physics_align;physics_align.install(restrict_abduction=a.restrict_abduction)
        def configure(c):c.domain_randomization.torque_scale=2.
        env=create_env(a.num_envs,configure=configure);spec=env._robot_spec
        palm_offset=spec.palm_center_offset;tip_offsets=list(spec.fingertip_offsets)
    r=env.robot;d=r.data;dev=env.device
    lab=list(d.joint_names);canon=[lab[i] for i in env._perm_canon_to_lab.tolist()]
    def joint_field(name):
        v=getattr(d,name,None)
        return None if v is None else [tolist(v[0,i]) for i in env._perm_canon_to_lab.tolist()]
    out=dict(side=a.side,joints_canonical=canon,limits=joint_field('joint_pos_limits'),
        stiffness=joint_field('joint_stiffness'),damping=joint_field('joint_damping'),armature=joint_field('joint_armature'),
        friction=joint_field('joint_friction_coeff'),effort=joint_field('joint_effort_limits'),
        velocity=joint_field('joint_vel_limits') or joint_field('joint_velocity_limits'),
        default_q=joint_field('default_joint_pos'))
    view=r.root_physx_view;bodies=list(d.body_names)
    out['bodies']=bodies;out['masses']=tolist(view.get_masses()[0])
    out['palm_body']=bodies[env._palm_body_id];out['fingertip_bodies']=[bodies[i] for i in env._fingertip_body_ids]
    out['palm_offset']=list(palm_offset);out['fingertip_offsets']=[list(t) for t in tip_offsets]
    co=view.get_contact_offsets()[0];ro=view.get_rest_offsets()[0]
    out['robot_contact_offset_range']=[float(co.min()),float(co.max())];out['robot_rest_offset_range']=[float(ro.min()),float(ro.max())]
    ov=env.object.root_physx_view
    out['object_mass']=float(ov.get_masses()[0].sum());out['object_contact_offset']=tolist(ov.get_contact_offsets()[0])
    out['robot_shape_count']=int(view.max_shapes)
    mats=view.get_material_properties()[0];out['robot_friction_values']=sorted({round(float(x),3) for x in mats[:,0]})
    out['object_friction']=sorted({round(float(x),3) for x in ov.get_material_properties()[0][:,0]})
    from pxr import UsdPhysics
    import isaaclab.sim as sim_utils
    stage=sim_utils.get_current_stage();dep={}
    for prim in stage.Traverse():
        path=str(prim.GetPath())
        if not path.startswith('/World/envs/env_0/') or not prim.HasAPI(UsdPhysics.RigidBodyAPI):continue
        at=prim.GetAttribute('physxRigidBody:maxDepenetrationVelocity')
        if at and at.HasAuthoredValue():dep[path.split('/')[4]]=dep.get(path.split('/')[4],set())|{round(float(at.Get()),3)}
    out['max_depenetration_velocity']={k:sorted(v) for k,v in dep.items()}
    from isaaclab.utils.math import quat_apply
    ids=torch.arange(env.num_envs,device=dev)
    def pose(q):
        r.write_joint_state_to_sim(q,torch.zeros_like(q),env_ids=ids);env.scene.write_data_to_sim();env.sim.forward();env.scene.update(0.)
        bs=d.body_state_w[0];o=env.scene.env_origins[0]
        pp=bs[env._palm_body_id,:3]-o;pr=bs[env._palm_body_id,3:7]
        pc=pp+quat_apply(pr[None],torch.tensor(palm_offset,device=dev)[None])[0]
        tips=[]
        for k,i in enumerate(env._fingertip_body_ids):
            tips.append(bs[i,:3]-o+quat_apply(bs[i,3:7][None],torch.tensor(tip_offsets[k],device=dev)[None])[0])
        return pc,torch.stack(tips),pr,bs
    q0=d.default_joint_pos.clone()
    pc,tips,pr,bs=pose(q0)
    rotm=lambda q:torch.stack([quat_apply(q[None],torch.eye(3,device=dev)[i][None])[0] for i in range(3)],1)
    out['default_pose']=dict(palm_center=tolist(pc),palm_body_rot_wxyz=tolist(pr),palm_body_axes_world_cols=tolist(rotm(pr)),
        tips=tolist(tips),tips_minus_palm=tolist(tips-pc),finger_dir=tolist(torch.nn.functional.normalize(tips[[0,1,2]].mean(0)-pc,dim=0)),
        thumb_dir=tolist(torch.nn.functional.normalize(tips[3]-pc,dim=0)),
        table_top_z=float(env._table_z_per_env[0]+0.15) if hasattr(env,'_table_z_per_env') else None,
        lowest_body_z=float((bs[:,2]-env.scene.env_origins[0,2]).min()))
    flex={}
    for finger in ('thumb','index','middle','ring','pinky'):
        q=q0.clone()
        for j,name in enumerate(lab):
            if finger in name and any(s in name for s in ('FE','PIP','DIP','IP')) and 'AA' not in name:
                lo,hi=d.joint_pos_limits[0,j];q[:,j]=lo+.6*(hi-lo)
        _,t2,_,_=pose(q);k=[n for n in out['fingertip_bodies']].index(next(n for n in out['fingertip_bodies'] if finger in n))
        dv=t2[k]-tips[k];flex[finger]=dict(tip_move=tolist(dv),dist_palm_before=float((tips[k]-pc).norm()),dist_palm_after=float((t2[k]-pc).norm()))
    pose(q0);out['flex_60pct']=flex
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=1)+'\n');print('WROTE',a.output,flush=True)
    env.close()


if __name__=='__main__':main()
