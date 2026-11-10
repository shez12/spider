"""Dynamic finger-closure test (no object contact): command hand targets through the native action path and
measure joint tracking, left (unmodified SimToolReal) vs right (our assets, optionally physics_align).

Hand actions are absolute targets in [-1,1] over each joint's range; the arm action (delta) is 0. The object is
parked far away so only self-collision, effort and drive limits can stop the fingers.
"""
import argparse,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--side',choices=('left','right'),required=True);p.add_argument('--align-left-physics',action='store_true')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=120)
    p.add_argument('--no-self-collision',action='store_true',help='disable robot self-collision (diagnostic only)')
    p.add_argument('--filter',nargs='*',default=[],choices=('palm-thumb','thumb-index','palm-fingers','pinky-ring','adjacent-fingers'),
        help='right only: extra self-collision filtered pairs (diagnostic)')
    p.add_argument('--palm-hull',action='store_true');p.add_argument('--official-hand',action='store_true')
    p.add_argument('--solo',choices=('thumb',),help='only these joints follow the level; all other hand joints stay open (-1)')
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(p);a=p.parse_args()
    vendor=Path('/home/szc/work/newsimtoolreal/simtoolreal') if a.side=='left' else PROJECT/'third_party/simtoolreal'
    sys.path[:0]=[str(vendor),str(vendor/'rl_games'),str(HERE)]
    app=AppLauncher(a).app
    try:run(a,vendor)
    finally:sys.stdout.flush();import os;os._exit(0)


def run(a,vendor):
    import torch,yaml
    levels=(-1.,0.,.3,.6,1.);n=len(levels)*8
    def configure(cfg):
        dr=cfg.domain_randomization;dr.use_action_delay=False;dr.use_obs_delay=False;dr.use_object_state_delay_noise=False
        dr.joint_velocity_obs_noise_std=0.;dr.force_scale=0.;dr.torque_scale=0.;cfg.termination.episode_length=10**6;cfg.episode_length_s=10**4
    if a.side=='left':
        from isaacsimenvs.tasks.simtoolreal.simtoolreal_env import SimToolRealEnv
        from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg
        cfg=SimToolRealEnvCfg();cfg.from_dict(yaml.safe_load((vendor/'isaacsimenvs/cfg/task/SimToolReal.yaml').read_text()))
        cal=json.loads((HERE/'assets/object/calibration.json').read_text())
        cfg.assets.object_urdf=str(HERE/'assets/object/mustard_bottle.urdf');cfg.assets.object_scale=tuple(cal['policy_object_scale'])
        cfg.scene.num_envs=n;configure(cfg);env=SimToolRealEnv(cfg)
    else:
        if a.no_self_collision:
            import isaaclab.sim.converters as conv,isaacsimenvs.tasks.simtoolreal.utils.scene_utils as scene
            Cfg=conv.UrdfConverterCfg;conv.UrdfConverterCfg=lambda **kw:Cfg(**dict(kw,self_collision=False))
            bake=scene._bake_usd
            scene._bake_usd=lambda *x,props=None,**kw:bake(*x,props=dict(props,enabled_self_collisions=False) if props and 'enabled_self_collisions' in props else props,**kw)
        if a.palm_hull:
            import palm_hull;print('PALM_HULL',palm_hull.install(),flush=True)
        if a.filter:
            import self_collision_filters;self_collision_filters.install(a.filter)
        if a.align_left_physics:
            import physics_align;physics_align.install()
        from environment import create_env
        if a.official_hand:from official_hand import create_env
        env=create_env(n,configure=configure)
    env.reset();dev=env.device
    # Park the object 2 m to the side so the hand never touches it.
    st=env.object.data.root_state_w.clone();st[:,0]+=2.;st[:,7:]=0;env.object.write_root_state_to_sim(st)
    level=torch.tensor(levels,device=dev).repeat_interleave(8)
    p=env._perm_lab_to_canon;names=[env.robot.data.joint_names[i] for i in p[7:].tolist()]
    follow=torch.tensor([a.solo is None or a.solo in nm for nm in names],device=dev)
    cmd=torch.where(follow[None],level[:,None],-torch.ones(1,device=dev))
    act=torch.zeros(n,29,device=dev);act[:,7:]=cmd
    for _ in range(a.steps):env.step(act)
    q=env.robot.data.joint_pos[:,p];lo=env._joint_lower_canon;hi=env._joint_upper_canon
    target=lo[7:]+(cmd+1)/2*(hi[7:]-lo[7:])
    err=(q[:,7:]-target).abs();rng=(hi[7:]-lo[7:])
    out=dict(solo=a.solo,side=a.side,aligned=a.align_left_physics,steps=a.steps,levels=levels,
        mean_abs_error_frac_by_level={str(l):float((err/rng)[level==l].mean()) for l in levels},
        no_self_collision=a.no_self_collision,filters=a.filter,
        thumb_pinky_q_by_level={names[j]:{str(l):[round(float(q[level==l,7+j].mean()),3),round(float(target[level==l,j].mean()),3)] for l in levels}
            for j in range(22) if names[j].split('_',1)[-1] in ('thumb_CMC_AA','pinky_CMC','thumb_CMC_FE','5_pinky_CMC','1_thumb_CMC_FE')},
        worst_joints_at_full_close=sorted([(names[j],float((err/rng)[level==1.,j].mean())) for j in range(22)],key=lambda x:-x[1])[:8],
        joint_vel_rms_end=float(env.robot.data.joint_vel[:,p][:,7:].pow(2).mean().sqrt()))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=1)+'\n');print('FINGER',json.dumps(out),flush=True)


if __name__=='__main__':main()
