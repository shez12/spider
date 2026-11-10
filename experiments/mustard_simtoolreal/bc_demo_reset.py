"""Restore cached actor-visible demo state without stepping physics.

Object velocities are explicitly zero, not a claim of restoring a PhysX snapshot.
"""
import numpy as np
import torch


def attach_demo_resets(env,paired,dataset,perturbations=None):
    archive=np.load(dataset)
    assert str(archive['action_contract'])=='simtoolreal_native_v1'
    by_id={int(c):i for i,c in enumerate(archive['candidate_ids'])}
    lookup=torch.tensor([by_id[int(c)] for c in paired.candidate_ids.cpu().tolist()],device=env.device)
    initial=torch.tensor(archive['obs'][:,0],device=env.device)
    # Native-position curriculum must blend from the measured cached pose,
    # which may differ from the pre-simulation source pose after settling.
    centers=initial[:,113:125].reshape(-1,4,3).mean(1)+initial[:,87:90]
    paired.actor_initial_centers=centers[lookup]
    old_reset=env._reset_idx
    def reference():return initial[lookup[paired.ids]]
    def reset(env_ids):
        old_reset(env_ids)
        if env_ids is None:env_ids=torch.arange(env.num_envs,device=env.device)
        ids=torch.as_tensor(env_ids,device=env.device,dtype=torch.long)
        obs=initial[lookup[paired.ids[ids]]]
        lo=env._joint_lower_canon;hi=env._joint_upper_canon
        q=(lo+.5*(obs[:,:29]+1)*(hi-lo))[:,env._perm_canon_to_lab]
        velocity=obs[:,29:58][:,env._perm_canon_to_lab]
        previous=obs[:,58:87][:,env._perm_canon_to_lab]
        if perturbations is not None and perturbations.get('joint_rad',0)>0:
            width=perturbations['joint_rad']
            delta=torch.empty_like(q).uniform_(-width,width)
            limits=env.robot.data.joint_pos_limits[ids]
            q=(q+delta).clamp(limits[:,:,0],limits[:,:,1])
            previous=(previous+delta).clamp(limits[:,:,0],limits[:,:,1])
        env.robot.write_joint_state_to_sim(q,velocity,env_ids=ids)
        env.robot.set_joint_position_target(previous,env_ids=ids)
        env._prev_targets[ids]=previous;env._cur_targets[ids]=previous
        state=env.object.data.default_root_state[ids].clone()
        state[:,:3]=obs[:,113:125].reshape(-1,4,3).mean(1)+obs[:,87:90]
        env._object_init_z[ids]=state[:,2]
        state[:,:3]+=env.scene.env_origins[ids]
        state[:,3:7]=obs[:,[97,94,95,96]];state[:,7:]=0
        if perturbations is not None:
            from isaaclab.utils.math import quat_mul
            # Rotate around each demonstration's object center, matching paired
            # goal transforms; do not rotate the fixed robot base.
            state[:,:3]+=paired.translation[ids]
            state[:,3:7]=quat_mul(paired.yaw[ids],state[:,3:7])
            env._object_init_z[ids]=state[:,2]-env.scene.env_origins[ids,2]
        env.object.write_root_state_to_sim(state,env_ids=ids)
        env._closest_fingertip_dist[ids]=-1;env._closest_keypoint_max_dist[ids]=-1
        env.scene.write_data_to_sim();env.sim.forward();env.scene.update(0.)
    env._reset_idx=reset
    def restore():env._reset_idx=old_reset
    return reference,restore
