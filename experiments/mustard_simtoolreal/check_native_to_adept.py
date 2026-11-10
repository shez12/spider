"""Check adept_obs.native_to_adept against live ADEPT observations in the right-hand mustard env (no obs DR)."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent;VENDOR=ROOT.parents[1]/'third_party/simtoolreal'
sys.path[:0]=[str(VENDOR),str(VENDOR/'rl_games'),str(ROOT)]


def main():
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True).app
    import torch,tempfile
    from environment import create_env
    from adept_obs import install_adept_obs,native_to_adept,mustard_surface_points,ADEPT_OBS,ADEPT_STATE
    import isaacsimenvs.tasks.simtoolreal.simtoolreal_env as module
    from isaacsimenvs.tasks.simtoolreal.utils import obs_utils
    tmp=Path(tempfile.mkdtemp());install_adept_obs(64,tmp);built=[];cls=module.SimToolRealEnv
    def configure(cfg):
        dr=cfg.domain_randomization;dr.use_action_delay=False;dr.use_obs_delay=False;dr.use_object_state_delay_noise=False
        dr.joint_velocity_obs_noise_std=0.;dr.force_scale=0.;dr.torque_scale=0.
        cfg.obs.obs_list=list(ADEPT_OBS);cfg.obs.state_list=list(ADEPT_STATE)
    module.SimToolRealEnv=lambda *x,**kw:built.append(cls(*x,**kw)) or built[-1]
    try:create_env(64,configure=configure)
    except AssertionError:pass
    env=built[-1];module.SimToolRealEnv=cls;env.reset()
    import adept_obs;assert tuple(map(tuple,obs_utils.KEYPOINT_CORNERS))==adept_obs.KEYPOINT_CORNERS
    native=['joint_pos','joint_vel','prev_action_targets','palm_pos','palm_rot','object_rot','fingertip_pos_rel_palm','keypoints_rel_palm','keypoints_rel_goal','object_scales']
    pts=mustard_surface_points(64);worst={}
    for step in range(120):
        env.step(torch.rand(64,29,device=env.device)*2-1)
        if step%20:continue
        env.cfg.obs.obs_list=native;n=obs_utils.build_observations(env)['policy']
        env.cfg.obs.obs_list=list(ADEPT_OBS);a=obs_utils.build_observations(env)['policy']
        c=native_to_adept(n,pts);assert c.shape==a.shape==(64,318)
        names=ADEPT_OBS;sizes=obs_utils.OBS_FIELD_SIZES;off=0
        for f in names:
            s=sizes[f] if f not in('joint_pos','joint_vel','prev_action_targets') else 29
            if f=='fingertip_pos_rel_palm':s=15
            worst[f]=max(worst.get(f,0.),float((c[:,off:off+s]-a[:,off:off+s]).abs().max()));off+=s
    print('NATIVE_TO_ADEPT',json.dumps(worst),flush=True)
    assert max(worst.values())<1e-4,worst
    import os;os._exit(0)


if __name__=='__main__':main()
