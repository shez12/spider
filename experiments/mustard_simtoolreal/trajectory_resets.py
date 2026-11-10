"""Batched, paired object reset + waypoint targets for actor-only pretraining.

Original action/reward/robot reset stay intact. The experiment intentionally
ends each two-goal chain and uses an attainable curriculum success threshold.
"""
import json


class PairedTrajectoryResets:
    def __init__(self,env,path,xy_noise=.005,yaw_noise_deg=2.):
        import torch
        from isaacsimenvs.tasks.simtoolreal.utils import reset_utils
        import isaacsimenvs.tasks.simtoolreal.simtoolreal_env as module
        self.env=env;self.torch=torch;self.xy_noise=xy_noise;self.yaw_noise_deg=yaw_noise_deg
        payload=json.loads(path.read_text());items=payload['trajectories']
        counts={len(t['pos']) for t in items}
        assert counts=={2},'This actor-only protocol requires two targets per trajectory.'
        self.initial=torch.tensor([t['initial_pose'] for t in items],device=env.device,dtype=torch.float32)
        self.positions=torch.tensor([t['pos'] for t in items],device=env.device,dtype=torch.float32)
        self.rotations=torch.tensor([t['quat_wxyz'] for t in items],device=env.device,dtype=torch.float32)
        self.candidate_ids=torch.tensor([t['candidate_id'] for t in items],device=env.device,dtype=torch.long)
        n=env.num_envs;device=env.device
        self.ids=torch.zeros(n,device=device,dtype=torch.long)
        self.index=torch.zeros_like(self.ids)
        self.translation=torch.zeros((n,3),device=device)
        self.yaw=torch.zeros((n,4),device=device);self.yaw[:,0]=1
        self.done=torch.zeros(n,device=device,dtype=torch.bool)
        self.reset_counts=torch.zeros((6,len(items)),device=device,dtype=torch.long)
        self.skipped=0;self.completed=0;self.successful=0
        self.old_object=reset_utils._reset_object_pose
        self.old_goal=reset_utils._reset_goal_pose
        self.old_termination=module.compute_terminations
        def reset_object(target,env_ids):
            if target is env:return self.reset_object(env_ids)
            return self.old_object(target,env_ids)
        def reset_goal(target,env_ids,mode):
            if target is env:return self.set_goals(env_ids,mode)
            return self.old_goal(target,env_ids,mode)
        def terminate(target):
            terminated,truncated=self.old_termination(target)
            if target is env:
                target._termination_reasons['waypoint_chain']=self.done.clone()
                terminated=terminated|self.done
            return terminated,truncated
        reset_utils._reset_object_pose=reset_object
        reset_utils._reset_goal_pose=reset_goal
        module.compute_terminations=terminate

    def transformed(self,env_ids,index):
        from isaaclab.utils.math import quat_apply,quat_mul
        origin=self.initial[self.ids[env_ids],:3]
        p=self.positions[self.ids[env_ids],index]
        q=self.rotations[self.ids[env_ids],index]
        return origin+self.translation[env_ids]+quat_apply(self.yaw[env_ids],p-origin),quat_mul(self.yaw[env_ids],q)

    def reset_object(self,env_ids):
        from isaaclab.utils.math import quat_mul
        t=self.torch;env=self.env;n=len(env_ids)
        self.ids[env_ids]=t.randint(len(self.initial),(n,),device=env.device)
        self.index[env_ids]=0;self.done[env_ids]=False
        self.translation[env_ids,:2]=t.empty((n,2),device=env.device).uniform_(-self.xy_noise,self.xy_noise)
        # Preserve the trajectory's height above the actual randomized tabletop.
        self.translation[env_ids,2]=env._table_z_per_env[env_ids]-env.cfg.reset.table_reset_z
        z_noise=getattr(self,'z_noise',0.)
        native_mix=getattr(self,'native_position_mix',0.)
        if native_mix:
            centers=getattr(self,'actor_initial_centers',self.initial[:,:3])
            initial=centers[self.ids[env_ids]]
            self.translation[env_ids,:2]-=native_mix*initial[:,:2]
            native_z=env.cfg.reset.table_reset_z+env.cfg.reset.table_object_z_offset
            self.translation[env_ids,2]+=native_mix*(native_z-initial[:,2])
        if z_noise:self.translation[env_ids,2]+=t.empty(n,device=env.device).uniform_(-z_noise,z_noise)
        angle=t.empty(n,device=env.device).uniform_(-self.yaw_noise_deg,self.yaw_noise_deg)*(.5*t.pi/180)
        self.yaw[env_ids]=0;self.yaw[env_ids,0]=t.cos(angle);self.yaw[env_ids,3]=t.sin(angle)
        pose=self.initial[self.ids[env_ids]].clone()
        pose[:,:3]+=self.translation[env_ids];pose[:,3:]=quat_mul(self.yaw[env_ids],pose[:,3:])
        env._object_init_z[env_ids]=pose[:,2]
        pose[:,:3]+=env.scene.env_origins[env_ids]
        env.object.write_root_pose_to_sim(pose,env_ids=env_ids)
        env.object.write_root_velocity_to_sim(t.zeros((n,6),device=env.device),env_ids=env_ids)
        block=t.div(env_ids,env.num_envs//6,rounding_mode='floor')
        flat=block*len(self.initial)+self.ids[env_ids]
        self.reset_counts+=t.bincount(flat,minlength=6*len(self.initial)).reshape(6,-1)

    def set_goals(self,env_ids,mode):
        from isaaclab.utils.math import quat_apply
        t=self.torch;env=self.env
        if mode=='absolute':self.index[env_ids]=0
        else:self.index[env_ids]+=1
        remaining=env_ids[~self.done[env_ids]]
        for _ in range(self.positions.shape[1]+1):
            ended=remaining[self.index[remaining]>=2]
            if len(ended):
                self.done[ended]=True;self.completed+=len(ended)
                self.successful+=int((env._successes[ended]>=2).sum())
            remaining=remaining[self.index[remaining]<2]
            if not len(remaining):break
            p,q=self.transformed(remaining,self.index[remaining])
            offsets=env._keypoint_offsets_fixed[remaining]
            objpos=env.object.data.root_pos_w[remaining]-env.scene.env_origins[remaining]
            objq=env.object.data.root_quat_w[remaining]
            targetkp=p[:,None]+quat_apply(q[:,None].expand(-1,4,-1).reshape(-1,4),offsets.reshape(-1,3)).reshape(-1,4,3)
            objectkp=objpos[:,None]+quat_apply(objq[:,None].expand(-1,4,-1).reshape(-1,4),offsets.reshape(-1,3)).reshape(-1,4,3)
            distance=t.linalg.vector_norm(targetkp-objectkp,dim=-1).max(-1).values
            skip=distance<=env._current_success_tolerance*env.cfg.reward.keypoint_scale+.01
            valid=remaining[~skip]
            if len(valid):
                poses=t.cat([p[~skip]+env.scene.env_origins[valid],q[~skip]],-1)
                env.goal_viz.write_root_pose_to_sim(poses,env_ids=valid)
            remaining=remaining[skip]
            self.skipped+=len(remaining);self.index[remaining]+=1
            if not len(remaining):break

    def restore(self):
        from isaacsimenvs.tasks.simtoolreal.utils import reset_utils
        import isaacsimenvs.tasks.simtoolreal.simtoolreal_env as module
        reset_utils._reset_object_pose=self.old_object;reset_utils._reset_goal_pose=self.old_goal
        module.compute_terminations=self.old_termination
