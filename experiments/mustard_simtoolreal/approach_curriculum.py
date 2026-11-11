"""Start-state curriculum that retreats to the native start, from scripted demo-relative approaches.

Build (CPU):  python approach_curriculum.py build --replays outputs/demo_relative_v1/replay_bank_s*/replay.npz --output ...
  replay_demo_grasps.py drives the arm, from a native random start, to the pipeline grasp placed relative to the
  bottle.  For every replay that latched the env's lifted flag, the simulator state at fractions `level` = 0.1 ... 1.0
  of the time between the native start (0) and the moment the script starts lifting (1) is kept.  No state after
  the lift starts is kept, and every state lies on a path that begins at a native start.

ApproachCurriculum(env, path) wraps env._reset_idx (vendor native reset first, as left_reset_bank.StateResetBank):
  with probability `share` the episode starts from a bank state whose level is in (stage - width, stage]; all other
  episodes are plain native starts.  `stage` starts at 1 and moves `step` towards 0 whenever at least
  `min_episodes` bank episodes of the current stage finished and `threshold` of them latched the lifted flag.
  At stage 0 the bank is switched off: only native starts remain.
With path=None nothing is restored; the class only counts, per finished episode, whether the lifted flag latched
(the per-episode native lift rate of a from-scratch run).
"""
import argparse,json
from pathlib import Path
import numpy as np
from left_reset_bank import FIELDS


def build(a):
    out={k:[] for k in FIELDS};level=[];source=[];sig=None;stats=dict(replays=[],attempted=0,successful=0)
    for ri,path in enumerate(a.replays):
        with np.load(path) as z:d={k:z[k] for k in z.files}  # NpzFile decompresses on every access
        s=str(d['signature']);assert sig is None or s==sig,'replays from different environments';sig=s
        ok=(d['first_flag']>=0)&(d['lift_waypoint_step']>0);stats['replays'].append(str(path));stats['attempted']+=len(ok);stats['successful']+=int(ok.sum())
        for i in np.flatnonzero(ok):
            for p in a.levels:
                t=min(int(round(p*d['lift_waypoint_step'][i])),d['q'].shape[1]-1)
                for k in ('q','v','previous','current','root'):out[k].append(d[k][i,t])
                for k in ('goal','table','table_z'):out[k].append(d[k][i])
                level.append(p);source.append((ri,int(d['env'][i]),t,int(d['source'][i])))
    arrays={k:np.stack(v).astype(np.float32) for k,v in out.items()};a.output.parent.mkdir(parents=True,exist_ok=True)
    # bank states must be before the lift: drop whole paths on which the bottle is off the table at any kept level
    from mirror_right_bank import HERE,clearance,mesh_vertices
    L=len(a.levels);c=clearance(arrays['root'].astype(np.float64),mesh_vertices(a.object_dir or HERE/'assets/object')[0],arrays['table_z'].astype(np.float64)).reshape(-1,L)
    keep=np.repeat(np.abs(c).max(1)<a.max_clearance,L);stats['dropped_bottle_off_table']=int((~keep).sum())//L
    arrays={k:v[keep] for k,v in arrays.items()};level=list(np.array(level)[keep]);source=[s for s,m in zip(source,keep) if m]
    np.savez_compressed(a.output,**arrays,level=np.array(level,dtype=np.float32),kind=np.ones(len(level),dtype=np.int64),source=np.array(source),signature=np.array(sig),format_version=np.array(1))
    src=np.array(source)[:,3];report=dict(stats,levels=list(a.levels),states=len(level),paths=len(level)//len(a.levels),paths_per_demo_source={int(s):int((src==s).sum())//len(a.levels) for s in sorted(set(src))},signature=sig)
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=1)+'\n');print(json.dumps(report))


class ApproachCurriculum:
    def __init__(self,env,path=None,share=.3,threshold=.5,step=.1,width=.2,min_episodes=2000,expected_signature=None):
        import torch
        from diverse_resets import restore_states
        self.env=env;self.path=path;dev=env.device;N=env.num_envs;self.share,self.threshold,self.step,self.width,self.min_episodes=share,threshold,step,width,min_episodes
        self.data=None;self.stage=0.
        if path is not None:
            with np.load(path) as d:
                if expected_signature is not None:assert str(d['signature'])==expected_signature,'bank was built in a different environment'
                self.data={k:torch.tensor(d[k],device=dev) for k in FIELDS};self.level=torch.tensor(d['level'],device=dev)
            self.stage=1.
        self.current_kind=torch.zeros(N,device=dev,dtype=torch.long);self.last_completed_kind=torch.full((N,),-1,device=dev,dtype=torch.long)
        self.env_stage=torch.zeros(N,device=dev);self.started=torch.zeros(N,device=dev,dtype=torch.bool)
        self.count=torch.zeros(2,2,device=dev);self.stage_count=torch.zeros(2,device=dev);self.changes=[];self.original=env._reset_idx
        def reset(env_ids):
            ids=torch.arange(N,device=dev) if env_ids is None else torch.as_tensor(env_ids,device=dev,dtype=torch.long)
            kind=self.current_kind[ids];lifted=env._lifted_object[ids]&self.started[ids];done=self.started[ids]
            for k in (0,1):m=done&(kind==k);self.count[k,0]+=m.sum();self.count[k,1]+=(m&lifted).sum()
            m=done&(kind==1)&(self.env_stage[ids]==self.stage);self.stage_count[0]+=m.sum();self.stage_count[1]+=(m&lifted).sum()
            self.last_completed_kind[ids]=torch.where(done,kind,torch.full_like(kind,-1));self.started[ids]=True;self.original(env_ids);self.current_kind[ids]=0
            if self.stage<=0:return
            sel=ids[torch.rand(len(ids),device=dev)<self.share]
            if not len(sel):return
            pool=torch.where((self.level>self.stage-self.width-1e-6)&(self.level<=self.stage+1e-6))[0];e=pool[torch.randint(len(pool),(len(sel),),device=dev)]
            restore_states(env,sel,{f:v[e] for f,v in self.data.items()});self.current_kind[sel]=1;self.env_stage[sel]=self.stage
            # vendor lift reference (spawn height), as in StateResetBank; bank states are taken before the lift starts
            rc=env.cfg.reset;init=env._table_z_per_env[sel]+rc.table_object_z_offset;env._object_init_z[sel]=init
            env._lifted_object[sel]=(.05+self.data['root'][e,2]-init)>env.cfg.reward.lifting_bonus_threshold
            env.scene.write_data_to_sim();env.sim.forward();env.scene.update(0.)
        env._reset_idx=reset;env._curriculum_eligible_mask=lambda:self.last_completed_kind==0

    def update(self,epoch,frames):
        """Call once per epoch.  Returns the log fields and resets the per-log counters."""
        c=self.count.cpu().tolist();self.count.zero_()
        row=dict(native_episodes=int(c[0][0]),native_episode_lift_rate=c[0][1]/c[0][0] if c[0][0] else None,bank_episodes=int(c[1][0]),
            bank_episode_lift_rate=c[1][1]/c[1][0] if c[1][0] else None,approach_stage=self.stage)
        n,l=self.stage_count.cpu().tolist()
        if self.stage>0 and n>=self.min_episodes:
            rate=l/n;self.stage_count.zero_()
            if rate>=self.threshold:
                change=dict(epoch=epoch,frames=frames,old=self.stage,new=max(round(self.stage-self.step,6),0.),bank_lift_rate=rate,episodes=int(n))
                self.stage=change['new'];self.changes.append(change);row['approach_stage_change']=change
        return row

    def report(self):
        return dict(bank=str(self.path),share=self.share,threshold=self.threshold,step=self.step,width=self.width,min_episodes=self.min_episodes,
            states=0 if self.data is None else int(len(self.level)),levels=[] if self.data is None else sorted(set(np.round(self.level.cpu().numpy(),3).tolist())))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter);sub=p.add_subparsers(dest='cmd',required=True)
    b=sub.add_parser('build');b.add_argument('--replays',type=Path,nargs='+',required=True);b.add_argument('--output',type=Path,required=True)
    b.add_argument('--object-dir',type=Path);b.add_argument('--max-clearance',type=float,default=.01)
    b.add_argument('--levels',type=float,nargs='+',default=[round(.1*i,1) for i in range(1,11)])
    build(p.parse_args())
