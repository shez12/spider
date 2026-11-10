"""Experiment-only goal sampler. Native control/reward/reset/termination stay intact."""
import json


def install_waypoint_goals(env,path):
    import torch
    from isaaclab.utils.math import quat_apply
    from isaacsimenvs.tasks.simtoolreal.utils import reset_utils
    payload=json.loads(path.read_text());sequences=payload['trajectories']
    positions=[torch.tensor(t['pos'],device=env.device,dtype=torch.float32) for t in sequences]
    rotations=[torch.tensor(t['quat_wxyz'],device=env.device,dtype=torch.float32) for t in sequences]
    controller=dict(enabled=True,skipped=0,completed=0,assigned=0)
    ids=torch.zeros(env.num_envs,device=env.device,dtype=torch.long)
    indices=torch.zeros_like(ids);active=torch.ones(env.num_envs,device=env.device,dtype=torch.bool)
    original=reset_utils._reset_goal_pose
    margin=float(payload['online_skip_margin_m'])
    def sample(target,env_ids,mode):
        if target is not env or not controller['enabled']:return original(target,env_ids,mode)
        if mode=='absolute':
            ids[env_ids]=torch.randint(len(sequences),(len(env_ids),),device=env.device)
            indices[env_ids]=0;active[env_ids]=True
        else:indices[env_ids]+=active[env_ids].long()
        fallback=[]
        # Goal switches happen on resets/successes, not on every physics step.
        for i in env_ids.tolist():
            if not active[i]:fallback.append(i);continue
            sequence=int(ids[i]);index=int(indices[i]);offset=env._keypoint_offsets_fixed[i]
            obj_pos=env.object.data.root_pos_w[i]-env.scene.env_origins[i]
            obj_rot=env.object.data.root_quat_w[i]
            object_kp=obj_pos+quat_apply(obj_rot.expand(len(offset),-1),offset)
            tolerance=float(env._current_success_tolerance)*env.cfg.reward.keypoint_scale+margin
            # Skip already-satisfied points; never repeatedly pay for the final
            # point after the sequence ends. Continue native random goals instead.
            while index<len(positions[sequence]):
                p=positions[sequence][index];q=rotations[sequence][index]
                goal_kp=p+quat_apply(q.expand(len(offset),-1),offset)
                if float(torch.linalg.vector_norm(goal_kp-object_kp,dim=-1).max())>tolerance:break
                index+=1;controller['skipped']+=1
            indices[i]=index
            if index==len(positions[sequence]):
                active[i]=False;controller['completed']+=1;fallback.append(i);continue
            pose=torch.cat([positions[sequence][index]+env.scene.env_origins[i],rotations[sequence][index]])[None]
            env.goal_viz.write_root_pose_to_sim(pose,env_ids=torch.tensor([i],device=env.device))
            controller['assigned']+=1
        if fallback:original(env,torch.tensor(fallback,device=env.device,dtype=torch.long),mode)
    reset_utils._reset_goal_pose=sample
    controller['active']=active
    controller['indices']=indices
    controller['trajectory_ids']=ids
    controller['restore']=lambda:setattr(reset_utils,'_reset_goal_pose',original)
    return controller


def verify_transitions(env,controller,path):
    """Controlled simulator checks; never enable for a production training run."""
    import torch
    from isaacsimenvs.tasks.simtoolreal.utils import reset_utils
    env.reset();device=env.device;one=torch.tensor([0],device=device)
    payload=json.loads(path.read_text())['trajectories'][0]
    poses=torch.tensor([p+q for p,q in zip(payload['pos'],payload['quat_wxyz'])],device=device)
    assert len(poses)>=2
    def place(index):
        pose=poses[index].clone();pose[:3]+=env.scene.env_origins[0]
        env.object.write_root_pose_to_sim(pose[None],env_ids=one)
    controller['trajectory_ids'][0]=0;controller['indices'][0]=0;controller['active'][0]=True
    place(0)
    reset_utils._reset_goal_pose(env,one,'delta')
    assert int(controller['indices'][0])==1 and bool(controller['active'][0])
    previous_completed=controller['completed'];place(1)
    reset_utils._reset_goal_pose(env,one,'delta')
    assert not bool(controller['active'][0]) and controller['completed']==previous_completed+1
    reset_utils._reset_goal_pose(env,one,'delta')
    assert controller['completed']==previous_completed+1,'Final waypoint was counted repeatedly.'
    # Put the object exactly on the first point. It must be skipped, not offered
    # as a new target already satisfied by the same physical state.
    previous_skipped=controller['skipped'];controller['trajectory_ids'][0]=0
    controller['indices'][0]=-1;controller['active'][0]=True;place(0)
    reset_utils._reset_goal_pose(env,one,'delta')
    assert controller['skipped']>previous_skipped and int(controller['indices'][0])>=1
    env.reset()
    return dict(advance_after_success=True,chain_end_uses_native_random_targets=True,
        no_repeated_last_point=True,already_satisfied_point_skipped=True,production_state_reset_after_test=True)
