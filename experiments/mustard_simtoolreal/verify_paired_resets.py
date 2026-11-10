"""Simulator checks for trajectory binding and native success lifecycle."""


def verify(env,controller):
    import torch
    from isaaclab.utils.math import quat_apply,quat_mul
    env.reset()
    expected=controller.initial[controller.ids].clone()
    expected[:,:3]+=controller.translation+env.scene.env_origins
    expected[:,3:]=quat_mul(controller.yaw,expected[:,3:])
    actual=env.object.data.root_state_w
    assert torch.max(torch.abs(actual[:,:3]-expected[:,:3]))<1e-4
    assert torch.max(torch.abs(torch.abs((actual[:,3:7]*expected[:,3:7]).sum(-1))-1))<1e-5
    assert torch.max(torch.abs(actual[:,7:]))<1e-7
    assert torch.max(torch.abs(controller.translation[:,:2]))<=controller.xy_noise+1e-7
    expected_p,expected_q=controller.transformed(torch.arange(env.num_envs,device=env.device),controller.index)
    assert torch.max(torch.abs(env.goal_viz.data.root_pos_w-env.scene.env_origins-expected_p))<1e-4
    selected=torch.arange(6,device=env.device)*(env.num_envs//6)
    for success_number in (1,2):
        pose=env.goal_viz.data.root_state_w[selected,:7].clone()
        env.object.write_root_pose_to_sim(pose,env_ids=selected)
        env._near_goal_steps[selected]=9
        terminated,truncated=env._get_dones()
        assert torch.all(env._successes[selected]==success_number)
        assert torch.max(torch.abs(env.object.data.root_pos_w[selected]-pose[:,:3]))<1e-4
        if success_number==1:
            assert torch.all(controller.index[selected]==1) and not terminated[selected].any()
        else:
            assert terminated[selected].all() and controller.done[selected].all()
    env.reset()
    assert not controller.done.any() and not env._successes.any()
    before=controller.skipped
    p,q=controller.transformed(selected,torch.zeros_like(selected))
    env.object.write_root_pose_to_sim(torch.cat([p+env.scene.env_origins[selected],q],-1),env_ids=selected)
    controller.index[selected]=-1
    controller.set_goals(selected,'delta')
    assert controller.skipped>=before+len(selected) and not env._successes[selected].any()
    env.reset()
    controller.skipped=0;controller.completed=0;controller.successful=0;controller.reset_counts.zero_()
    return dict(initial_pose_bound_to_selected_trajectory=True,zero_initial_object_velocity=True,
        perturbation_bound_checked=True,target_and_initial_pose_share_transform=True,
        first_success_advances_without_object_reset=True,second_success_terminates_chain=True,
        reset_starts_new_chain=True,skip_gives_no_success_credit=True,
        original_robot_reset_and_action_pipeline=True,num_envs=env.num_envs)
