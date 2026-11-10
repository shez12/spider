"""Goal-rotation curriculum for SimToolReal (ADR-style, cf. ADEPT): alpha in [0,1] scales how much the object
must be re-oriented. alpha=1 is the unmodified task.

- Full-reset (first) goals: position unchanged (vendor absolute sample); orientation = slerp(object orientation at
  reset, vendor random orientation, alpha).
- Later goals: vendor delta sampler with delta_rotation_degrees = alpha * original value (distance unchanged).
- alpha rises by `step` once the any-goal rate of completed native episodes (bank kind 0 if a bank is attached)
  averaged over `window` epochs reaches `threshold`; never decreases. Applied identically to baseline and warm start.
"""
import torch


def slerp(q0,q1,t):
    dot=(q0*q1).sum(-1,keepdim=True);q1=torch.where(dot<0,-q1,q1);dot=dot.abs().clamp(max=1.)
    th=torch.acos(dot);s=torch.sin(th)
    out=torch.where(s<1e-6,torch.lerp(q0,q1,t),(torch.sin((1-t)*th)*q0+torch.sin(t*th)*q1)/s.clamp_min(1e-9))
    return out/out.norm(dim=-1,keepdim=True)


class RotationCurriculum:
    def __init__(self,env,alpha=0.,step=.1,threshold=.2,window=50,min_epochs=100):
        self.env=env;self.alpha=float(alpha);self.step=step;self.threshold=threshold;self.window=window;self.min_epochs=min_epochs
        self.base_delta=float(env.cfg.reset.delta_rotation_degrees);self.rates=[];self.since=0;self.changes=[]
        self.apply_cfg();original=env._reset_idx
        def reset(env_ids):
            original(env_ids)
            if self.alpha>=1:return
            ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else torch.as_tensor(env_ids,device=env.device,dtype=torch.long)
            kind=getattr(env,'_curriculum_kind',None)
            if kind is not None:ids=ids[kind()[ids]==0]  # bank episodes already get delta first goals
            if not len(ids):return
            o=env.scene.env_origins[ids];g=env.goal_viz.data.root_pos_w[ids];gq=env.goal_viz.data.root_quat_w[ids]
            q=slerp(env.object.data.root_quat_w[ids],gq,torch.full((len(ids),1),self.alpha,device=env.device))
            env.goal_viz.write_root_pose_to_sim(torch.cat([g,q],-1),env_ids=ids)
        env._reset_idx=reset

    def apply_cfg(self):self.env.cfg.reset.delta_rotation_degrees=self.alpha*self.base_delta

    def update(self,epoch,native_any_goal_rate):
        """Call once per epoch; returns a change record or None."""
        self.rates.append(native_any_goal_rate);self.rates=self.rates[-self.window:];self.since+=1
        if self.alpha>=1 or self.since<self.min_epochs or len(self.rates)<self.window:return None
        rate=sum(self.rates)/len(self.rates)
        if rate<self.threshold:return None
        old=self.alpha;self.alpha=min(1.,round(self.alpha+self.step,6));self.apply_cfg();self.since=0;self.rates=[]
        rec=dict(epoch=epoch,old_alpha=old,new_alpha=self.alpha,window_rate=rate,delta_rotation_degrees=self.env.cfg.reset.delta_rotation_degrees)
        self.changes.append(rec);return rec
