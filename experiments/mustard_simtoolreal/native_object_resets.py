"""Use vendor object/goal reset after the experiment's demo robot reset."""

def attach_native_object_resets(env,paired,*,native_robot=False,after=None):
    import torch
    from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import compute_intermediate_values
    original=env._reset_idx
    def reset(env_ids):
        original(env_ids)
        ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else torch.as_tensor(env_ids,device=env.device,dtype=torch.long)
        if native_robot:
            from isaacsimenvs.tasks.simtoolreal.utils.reset_utils import _randomize_robot_dof_state
            _randomize_robot_dof_state(env,ids)
        # The BC wrapper otherwise overwrites object pose after vendor reset.
        paired.old_object(env,ids)
        paired.old_goal(env,ids,'absolute')
        if after is not None:after(ids)
        paired.done[ids]=False;paired.index[ids]=0
        env._closest_fingertip_dist[ids]=-1;env._closest_keypoint_max_dist[ids]=-1
        env._near_goal_steps[ids]=0;env._is_success[ids]=False
        env.scene.write_data_to_sim();env.sim.forward();env.scene.update(0.)
    env._reset_idx=reset
    def restore():env._reset_idx=original
    return restore


def audit_native_robot_resets(env,repeats=2):
    import torch
    positions=[];velocities=[];errors=[]
    lower=env.robot.data.joint_pos_limits[:,:,0];upper=env.robot.data.joint_pos_limits[:,:,1]
    default=env.robot.data.default_joint_pos
    scale=torch.zeros_like(default);scale[:,env._arm_joint_ids]=env.cfg.reset.reset_dof_pos_random_interval_arm
    scale[:,env._hand_joint_ids]=env.cfg.reset.reset_dof_pos_random_interval_fingers
    assert torch.allclose(scale,torch.full_like(scale,.1))
    for _ in range(repeats):
        env.reset();q=env.robot.data.joint_pos.clone();v=env.robot.data.joint_vel.clone()
        normalized=(q-((1-scale)*default+scale*lower))/(scale*(upper-lower))
        assert float(normalized.min())>=-2e-5 and float(normalized.max())<=1+2e-5
        positions.append(normalized);velocities.append(v)
        errors.append(float((env._prev_targets-q).abs().max()));errors.append(float((env._cur_targets-q).abs().max()))
    u=torch.cat(positions);v=torch.cat(velocities)
    assert float(v.abs().max())<=.50001 and max(errors)<1e-6
    assert bool((u.min(0).values<.03).all()) and bool((u.max(0).values>.97).all())
    assert bool((v.min(0).values<-.45).all()) and bool((v.max(0).values>.45).all())
    return dict(samples=len(u),sampler='unmodified vendor _randomize_robot_dof_state',position_scale=.1,
        position_formula='0.9*default + 0.1*uniform(lower,upper)',velocity_range=[float(v.min()),float(v.max())],
        position_uniform_coordinate_range=[float(u.min()),float(u.max())],previous_current_target_max_error=max(errors))


def audit_native_object_resets(env,repeats=2):
    import torch
    positions=[];quats=[];tables=[];velocities=[]
    for _ in range(repeats):
        env.reset()
        positions.append((env.object.data.root_pos_w-env.scene.env_origins).clone())
        quats.append(env.object.data.root_quat_w.clone());tables.append(env._table_z_per_env.clone())
        velocities.append(env.object.data.root_state_w[:,7:].clone())
    p=torch.cat(positions);q=torch.cat(quats);t=torch.cat(tables);v=torch.cat(velocities);cfg=env.cfg.reset
    delta=p[:,2]-t-cfg.table_object_z_offset
    assert float(p[:,0].abs().max())<=cfg.reset_position_noise_x+1e-5
    assert float(p[:,1].abs().max())<=cfg.reset_position_noise_y+1e-5
    assert float(delta.abs().max())<=cfg.reset_position_noise_z+1e-5
    assert float((t-cfg.table_reset_z).abs().max())<=cfg.table_reset_z_range+1e-5
    assert float((q.norm(dim=1)-1).abs().max())<1e-5 and float(v.abs().max())<1e-5
    assert bool((p[:,:2].min(dim=0).values<-.08).all()) and bool((p[:,:2].max(dim=0).values>.08).all())
    assert bool((q.std(dim=0)>.2).all()),'Native full rotation must vary on all quaternion axes.'
    return dict(samples=len(p),xy_min=p[:,:2].min(dim=0).values.cpu().tolist(),xy_max=p[:,:2].max(dim=0).values.cpu().tolist(),
        z_relative_table_offset_min=float(delta.min()),z_relative_table_offset_max=float(delta.max()),
        table_z_min=float(t.min()),table_z_max=float(t.max()),quaternion_std=q.std(dim=0).cpu().tolist(),
        max_initial_velocity=float(v.abs().max()),object_sampler='unmodified vendor _reset_object_pose',
        goal_sampler='unmodified vendor _reset_goal_pose absolute; delta after success',robot_reset='cached demo joint position/velocity/previous target')
