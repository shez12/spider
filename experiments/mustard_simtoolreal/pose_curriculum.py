"""Demo-to-native initial-state curriculum for native-kind resets.

Each native reset blends the cached demonstration start toward the vendor native
sample with a per-episode mix alpha: joint positions, velocities, previous
targets and object XY are interpolated, object orientation is slerped. The
first goal is optionally blended the same way from the demonstration target to
the native absolute goal. alpha=0 is the demonstration start (and target),
alpha=1 is the unmodified native reset. alpha_max
only increases, gated on the success of frontier (alpha==alpha_max) episodes.
"""
from collections import deque
import torch


def slerp(q0,q1,t):
    """Shortest-path slerp for wxyz quaternions; t broadcastable to (N,1)."""
    dot=(q0*q1).sum(-1,keepdim=True);q1=torch.where(dot<0,-q1,q1);dot=dot.abs().clamp(max=1.)
    theta=torch.acos(dot);s=torch.sin(theta)
    out=torch.where(s<1e-6,torch.lerp(q0,q1,t),(torch.sin((1-t)*theta)*q0+torch.sin(t*theta)*q1)/s.clamp_min(1e-9))
    return out/out.norm(dim=-1,keepdim=True)


def blend(demo,native,alpha):
    """Blend canonical-free state dicts; z stays at the demo rest height only for alpha=0."""
    a=alpha[:,None]
    out={k:torch.lerp(demo[k],native[k],a) for k in ('q','v')}
    out['previous']=torch.lerp(demo['previous'],native['q'],a)
    out['pos']=torch.lerp(demo['pos'],native['pos'],a)
    out['pos'][:,2]=torch.where(alpha>0,native['pos'][:,2],demo['pos'][:,2])
    out['quat']=slerp(demo['quat'],native['quat'],a)
    if 'goal_pos' in demo:
        out['goal_pos']=torch.lerp(demo['goal_pos'],native['goal_pos'],a)
        out['goal_quat']=slerp(demo['goal_quat'],native['goal_quat'],a)
    return out


class PoseCurriculum:
    def __init__(self,env,reference,alpha=0.,step=.1,frontier_probability=.5,threshold=.5,window=500,demo_goal=None):
        """demo_goal(ids)->(local pos, wxyz quat) also blends the first goal from the demo target toward the native absolute goal."""
        assert 0<=alpha<=1 and 0<step<=1 and 0<=frontier_probability<=1 and 0<threshold<=1 and window>0
        n=env.num_envs;device=env.device
        self.env=env;self.reference=reference;self.demo_goal=demo_goal;self.alpha_max=float(alpha);self.step=step
        self.frontier_probability=frontier_probability;self.threshold=threshold
        self.current=torch.full((n,),-1.,device=device);self.last_completed=torch.full((n,),-1.,device=device)
        self.outcomes=deque(maxlen=window);self.enabled=True;self.last_window_rate=None

    def sample(self,n):
        device=self.env.device
        if self.alpha_max>=1:return torch.ones(n,device=device)
        frontier=torch.rand(n,device=device)<self.frontier_probability
        return torch.where(frontier,torch.full((n,),self.alpha_max,device=device),torch.rand(n,device=device)*self.alpha_max)

    def apply(self,ids):
        """Call right after the native robot/object reset has written its sample."""
        env=self.env;self.last_completed[ids]=self.current[ids]
        alpha=self.sample(len(ids)) if self.enabled else torch.ones(len(ids),device=env.device)
        self.current[ids]=alpha if self.enabled else -1.
        if bool((alpha>=1).all()):return
        obs=self.reference()[ids];p=env._perm_canon_to_lab
        lo=env._joint_lower_canon;hi=env._joint_upper_canon
        origins=env.scene.env_origins[ids]
        pos=obs[:,113:125].reshape(-1,4,3).mean(1)+obs[:,87:90]
        pos[:,2]+=env._table_z_per_env[ids]-env.cfg.reset.table_reset_z
        demo=dict(q=(lo+.5*(obs[:,:29]+1)*(hi-lo))[:,p],v=obs[:,29:58][:,p],previous=obs[:,58:87][:,p],
            pos=pos,quat=obs[:,[97,94,95,96]])
        native=dict(q=env.robot.data.joint_pos[ids].clone(),v=env.robot.data.joint_vel[ids].clone(),
            pos=env.object.data.root_pos_w[ids]-origins,quat=env.object.data.root_quat_w[ids].clone())
        if self.demo_goal is not None:
            demo['goal_pos'],demo['goal_quat']=self.demo_goal(ids)
            native.update(goal_pos=env.goal_viz.data.root_pos_w[ids]-origins,goal_quat=env.goal_viz.data.root_quat_w[ids].clone())
        s=blend(demo,native,alpha)
        if self.demo_goal is not None:
            env.goal_viz.write_root_pose_to_sim(torch.cat([s['goal_pos']+origins,s['goal_quat']],-1),env_ids=ids)
        env.robot.write_joint_state_to_sim(s['q'],s['v'],env_ids=ids)
        env.robot.set_joint_position_target(s['previous'],env_ids=ids)
        env._prev_targets[ids]=s['previous'];env._cur_targets[ids]=s['previous']
        state=torch.zeros((len(ids),13),device=env.device)
        state[:,:3]=s['pos']+origins;state[:,3:7]=s['quat']
        env.object.write_root_state_to_sim(state,env_ids=ids)
        env._object_init_z[ids]=s['pos'][:,2]

    def record(self,alphas,reached):
        frontier=alphas>=min(self.alpha_max,1.)-1e-6
        self.outcomes.extend(reached[frontier].cpu().tolist())

    def frontier_rate(self):
        return sum(self.outcomes)/len(self.outcomes) if self.outcomes else None

    def maybe_advance(self):
        """Decide once per full window of fresh frontier episodes (non-overlapping), then start a new window."""
        if self.alpha_max>=1 or len(self.outcomes)<self.outcomes.maxlen:return None
        rate=self.frontier_rate();self.last_window_rate=rate;self.outcomes.clear()
        if rate<self.threshold:return None
        old=self.alpha_max;self.alpha_max=min(1.,round(self.alpha_max+self.step,6))
        return dict(old_alpha_max=old,new_alpha_max=self.alpha_max,window_rate=rate)

    def eligible(self):
        """Only fully native episodes may drive the tolerance curriculum."""
        return self.last_completed>=1.


def audit_pose_resets(env,pose,bank=None):
    """At alpha_max=0, native-kind resets must reproduce the demo start exactly."""
    saved=pose.alpha_max;pose.alpha_max=0.
    try:
        env.reset()
        native=pose.current==0
        if bank is not None:native&=bank.current_kind==0
        ids=torch.where(native)[0];assert len(ids),'No native-kind reset to audit'
        obs=pose.reference()[ids];lo=env._joint_lower_canon;hi=env._joint_upper_canon
        q=(lo+.5*(obs[:,:29]+1)*(hi-lo))[:,env._perm_canon_to_lab]
        pos=obs[:,113:125].reshape(-1,4,3).mean(1)+obs[:,87:90]
        pos[:,2]+=env._table_z_per_env[ids]-env.cfg.reset.table_reset_z
        quat=obs[:,[97,94,95,96]];actual=env.object.data.root_quat_w[ids]
        errors=dict(joint_rad=float((env.robot.data.joint_pos[ids]-q).abs().max()),
            previous_target_rad=float((env._prev_targets[ids]-obs[:,58:87][:,env._perm_canon_to_lab]).abs().max()),
            object_m=float((env.object.data.root_pos_w[ids]-env.scene.env_origins[ids]-pos).abs().max()),
            object_quat=float((1-(actual*quat).sum(-1).abs()).abs().max()))
        if pose.demo_goal is not None:
            goal_pos,goal_quat=pose.demo_goal(ids)
            errors.update(goal_m=float((env.goal_viz.data.root_pos_w[ids]-env.scene.env_origins[ids]-goal_pos).abs().max()),
                goal_quat=float((1-(env.goal_viz.data.root_quat_w[ids]*goal_quat).sum(-1).abs()).abs().max()))
        assert max(errors.values())<1e-4,errors
    finally:pose.alpha_max=saved
    return dict(audited_native_resets=len(ids),max_errors=errors,restored_alpha_max=saved)
