"""Smoke-check the copied environment with the native controller and lifecycle."""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
VENDOR=PROJECT/'third_party/simtoolreal'
sys.path[:0]=[str(VENDOR),str(VENDOR/'rl_games'),str(ROOT)]


def main():
    from isaaclab.app import AppLauncher
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-envs',type=int,default=12)
    parser.add_argument('--steps',type=int,default=64)
    parser.add_argument('--output',type=Path,default=PROJECT/'outputs/mustard_local_native_smoke_v1')
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args();app=AppLauncher(args).app
    try:
        import torch
        import isaacsimenvs
        import rl_games
        from environment import create_env
        env=create_env(args.num_envs)
        assert Path(isaacsimenvs.__file__).resolve().is_relative_to(VENDOR)
        assert Path(rl_games.__file__).resolve().is_relative_to(VENDOR)
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/'environment_config.json').write_text(json.dumps(env.cfg.to_dict(),indent=2,default=str)+'\n')
        obs,_=env.reset()
        for _ in range(args.steps):
            obs,reward,done,truncated,info=env.step(torch.zeros((env.num_envs,29),device=env.device))
            assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
        result=dict(isaacsimenvs=str(isaacsimenvs.__file__),rl_games=str(rl_games.__file__),
            environment_class=type(env).__module__+'.'+type(env).__name__,
            native_hooks=list(('action','reset','reward','observation','termination','curriculum')),
            steps=args.steps,num_envs=env.num_envs,observation_dim=env.cfg.observation_space,
            critic_dim=env.cfg.state_space,action_dim=env.cfg.action_space,
            arm_moving_average=env.cfg.action.arm_moving_average,
            min_hand_effort_nm=min(env._robot_spec.hand_joint_effort.values()),
            max_hand_effort_nm=max(env._robot_spec.hand_joint_effort.values()),
            torque_scale=env.cfg.domain_randomization.torque_scale,
            reset='native randomized reset; no demonstration state injection',
            goals='native random absolute then delta after success')
        (args.output/'completed.json').write_text(json.dumps(result,indent=2)+'\n')
        print('LOCAL_NATIVE_SMOKE',json.dumps(result),flush=True)
        env.close()
    finally:app.close()


if __name__=='__main__':main()
