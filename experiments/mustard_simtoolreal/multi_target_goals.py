"""Restore native delta-goal continuation while retaining demonstration resets."""


LEVELS=[dict(max_targets=1,delta_m=.05,rotation_deg=30.),
        dict(max_targets=2,delta_m=.05,rotation_deg=30.),
        dict(max_targets=5,delta_m=.075,rotation_deg=45.),
        dict(max_targets=10,delta_m=.1,rotation_deg=90.),
        dict(max_targets=50,delta_m=.1,rotation_deg=90.)]


class MultiTargetGoals:
    def __init__(self,env,paired,level=0):
        self.env=env;self.paired=paired;self.old=paired.set_goals
        self.level=0;self.transitions=0
        def goals(ids,mode):
            if mode=='absolute':return self.old(ids,mode)
            # Exactly the original native sampler and lifecycle after success.
            # Native termination will end the episode at max_consecutive_successes.
            self.transitions+=len(ids)
            paired.index[ids]+=1
            paired.old_goal(env,ids,'delta')
            paired.completed+=int((env._successes[ids]>=env.cfg.termination.max_consecutive_successes).sum())
        paired.set_goals=goals
        self.set_level(level)

    def set_level(self,level):
        assert 0<=level<len(LEVELS)
        self.level=level;spec=LEVELS[level]
        self.env.cfg.termination.max_consecutive_successes=spec['max_targets']
        self.env.cfg.reset.delta_goal_distance=spec['delta_m']
        self.env.cfg.reset.delta_rotation_degrees=spec['rotation_deg']

    def restore(self):self.paired.set_goals=self.old


def verify_lifecycle(env,paired,controller):
    """Force native successes, not fake action rollouts, to test state lifecycle."""
    import torch
    from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import compute_intermediate_values
    controller.set_level(1);env.reset()
    ids=torch.arange(env.num_envs,device=env.device)
    before_counts=paired.reset_counts.clone()
    for number in (1,2):
        goal=env.goal_viz.data.root_state_w[:,:7].clone()
        env.object.write_root_pose_to_sim(goal)
        env.object.write_root_velocity_to_sim(torch.zeros((env.num_envs,6),device=env.device))
        env.scene.write_data_to_sim();env.sim.forward();env.scene.update(0.)
        compute_intermediate_values(env)
        env._near_goal_steps[:]=env.cfg.termination.success_steps-1
        q=env.robot.data.joint_pos.clone();target=env._prev_targets.clone()
        env.episode_length_buf[:]=100
        terminated,truncated=env._get_dones()
        assert torch.all(env._successes==number),'Forced goal dwell must increment native successes.'
        assert torch.all(env.episode_length_buf==0),'Native timeout must restart after each target.'
        assert torch.equal(before_counts,paired.reset_counts),'Target success must not reset the scene.'
        assert torch.equal(q,env.robot.data.joint_pos) and torch.equal(target,env._prev_targets)
        assert float((env.object.data.root_state_w[:,:7]-goal).abs().max())<1e-5
        if number==1:
            assert not terminated.any() and not truncated.any()
            assert float((env.goal_viz.data.root_pos_w-goal[:,:3]).abs().max())>0
        else:assert terminated.all() and not truncated.any()
    controller.set_level(4);env.reset();env._is_success[:]=False
    env.episode_length_buf[:]=env.max_episode_length
    _,truncated=env._get_dones();assert truncated.all()
    controller.set_level(0);env.reset()
    return dict(success_advances_goal_without_robot_or_object_reset=True,
        second_success_terminates_at_cap_two=True,timeout_since_last_success=True,
        native_final_cap=LEVELS[-1]['max_targets'],native_final_delta_m=LEVELS[-1]['delta_m'],
        native_final_rotation_deg=LEVELS[-1]['rotation_deg'])
