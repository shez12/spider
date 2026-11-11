"""Stage 2 (carry to the target pose): start states with the bottle already held in the air + a first goal that
grows from "next to the held bottle" to the vendor distribution.

Build (CPU):  python carry_curriculum.py build --replays outputs/demo_relative_v1/replay_bank_s*/replay.npz --output ...
  From replay_demo_grasps.py paths that latched the lifted flag AND still hold the bottle at the end: simulator
  states every --stride steps after the flag, kept while the lowest bottle point is > --min-clearance above the table.

CarryCurriculum(env, path) wraps env._reset_idx (vendor native reset first):
  with probability `share` the episode starts from a held state (lifted flag set: no lift bonus) and its first goal is
  the vendor delta sampler around the restored bottle pose with (distance, rotation) = LEVELS[level], resampled
  while the bottle is within 1.2x the success tolerance of it (a goal reached without moving).  All other episodes are unmodified native episodes (absolute first goal).
  The level goes up when >= `min_episodes` held-start episodes of the level finished and `threshold` of them reached
  their first goal.  Later goals of every episode are the vendor delta goals.
"""
import argparse,json
from pathlib import Path
import numpy as np
from left_reset_bank import FIELDS

# vendor first goal: median 0.27 m, 133 deg.  Smaller than (0.10 m, 45 deg) is mostly inside the 0.1125 m keypoint tolerance.
LEVELS=((.10,45.),(.15,70.),(.20,100.),(.30,140.),(.40,180.))


def build(a):
    from mirror_right_bank import HERE,clearance,mesh_vertices
    V=mesh_vertices(a.object_dir or HERE/'assets/object')[0];out={k:[] for k in FIELDS};source=[];sig=None;stats=dict(replays=[],lifted=0,held_to_end=0)
    for ri,path in enumerate(a.replays):
        with np.load(path) as z:d={k:z[k] for k in z.files}
        s=str(d['signature']);assert sig is None or s==sig,'replays from different environments';sig=s;T=d['q'].shape[1]
        ok=(d['first_flag']>=0);held=ok&(d['final_clearance']>.03);stats['replays'].append(str(path));stats['lifted']+=int(ok.sum());stats['held_to_end']+=int(held.sum())
        for i in np.flatnonzero(held):
            ts=np.arange(int(d['first_flag'][i])+a.stride,T,a.stride)
            if not len(ts):continue
            c=clearance(d['root'][i,ts].astype(np.float64),V,np.full(len(ts),float(d['table_z'][i])))
            for t in ts[c>a.min_clearance]:
                for k in ('q','v','previous','current','root'):out[k].append(d[k][i,t])
                for k in ('goal','table','table_z'):out[k].append(d[k][i])
                source.append((ri,int(d['env'][i]),int(t),int(d['source'][i])))
    arrays={k:np.stack(v).astype(np.float32) for k,v in out.items()};a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,**arrays,kind=np.full(len(source),3,dtype=np.int64),source=np.array(source),signature=np.array(sig),format_version=np.array(1))
    src=np.array(source);report=dict(stats,stride=a.stride,min_clearance=a.min_clearance,states=len(source),paths=len(set(map(tuple,src[:,:2].tolist()))),signature=sig)
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=1)+'\n');print(json.dumps(report))


class CarryCurriculum:
    def __init__(self,env,path,share=.5,threshold=.3,min_episodes=2000,expected_signature=None,margin=1.2,tries=20):
        import torch
        from diverse_resets import restore_states
        from isaacsimenvs.tasks.simtoolreal.utils.goal_sampling import sample_delta_goal_pose
        from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import _keypoints_world
        self.env=env;self.path=path;dev=env.device;N=env.num_envs;self.share,self.threshold,self.min_episodes=share,threshold,min_episodes;self.level=0
        with np.load(path) as d:
            if expected_signature is not None:assert str(d['signature'])==expected_signature,'bank was built in a different environment'
            self.data={k:torch.tensor(d[k],device=dev) for k in FIELDS}
        self.states=len(self.data['q']);self.current_kind=torch.zeros(N,device=dev,dtype=torch.long);self.last_completed_kind=torch.full((N,),-1,device=dev,dtype=torch.long)
        self.env_level=torch.zeros(N,device=dev,dtype=torch.long);self.started=torch.zeros(N,device=dev,dtype=torch.bool)
        self.count=torch.zeros(2,3,device=dev);self.level_count=torch.zeros(2,device=dev);self.trivial=self.sampled=0;self.original=env._reset_idx
        def reset(env_ids):
            ids=torch.arange(N,device=dev) if env_ids is None else torch.as_tensor(env_ids,device=dev,dtype=torch.long)
            kind=self.current_kind[ids];done=self.started[ids];lifted=env._lifted_object[ids]&done;goal=(env._successes[ids]>=1)&done
            for k in (0,1):m=done&(kind==k);self.count[k,0]+=m.sum();self.count[k,1]+=(m&lifted).sum();self.count[k,2]+=(m&goal).sum()
            m=done&(kind==1)&(self.env_level[ids]==self.level);self.level_count[0]+=m.sum();self.level_count[1]+=(m&goal).sum()
            self.last_completed_kind[ids]=torch.where(done,kind,torch.full_like(kind,-1));self.started[ids]=True;self.original(env_ids);self.current_kind[ids]=0
            sel=ids[torch.rand(len(ids),device=dev)<self.share]
            if not len(sel):return
            e=torch.randint(self.states,(len(sel),),device=dev);restore_states(env,sel,{f:v[e] for f,v in self.data.items()})
            self.current_kind[sel]=1;self.env_level[sel]=self.level
            rc=env.cfg.reset;env._object_init_z[sel]=env._table_z_per_env[sel]+rc.table_object_z_offset;env._lifted_object[sel]=True
            root=self.data['root'][e];o=env.scene.env_origins;dist,rot=LEVELS[self.level]
            tol=env._current_success_tolerance*env.cfg.reward.keypoint_scale*margin
            kp=env._keypoint_offsets_fixed if env.cfg.reward.fixed_size_keypoint_reward else env._keypoint_offsets
            obj_kp=_keypoints_world(root[:,:3],root[:,3:7],kp[sel]);todo=torch.arange(len(sel),device=dev)
            for _ in range(tries):
                p,q=sample_delta_goal_pose(prev_pos=root[todo,:3],prev_quat_wxyz=root[todo,3:7],delta_distance=dist,delta_rotation_degrees=rot,
                    mins=rc.target_volume_mins,maxs=rc.target_volume_maxs,scale=rc.target_volume_region_scale)
                env.goal_viz.write_root_pose_to_sim(torch.cat([p+o[sel[todo]],q],-1),env_ids=sel[todo])
                todo=todo[(obj_kp[todo]-_keypoints_world(p,q,kp[sel[todo]])).norm(dim=-1).max(-1).values<=tol]
                if not len(todo):break
            self.trivial+=len(todo);self.sampled+=len(sel);env.scene.write_data_to_sim();env.sim.forward();env.scene.update(0.)
        env._reset_idx=reset;env._curriculum_eligible_mask=lambda:self.last_completed_kind==0

    def update(self,epoch,frames):
        c=self.count.cpu().tolist();self.count.zero_();f=lambda a,b:a/b if b else None
        row=dict(native_episodes=int(c[0][0]),native_episode_lift_rate=f(c[0][1],c[0][0]),native_episode_goal_rate=f(c[0][2],c[0][0]),
            carry_episodes=int(c[1][0]),carry_episode_goal_rate=f(c[1][2],c[1][0]),carry_level=self.level,carry_trivial_goal_share=f(self.trivial,self.sampled))
        self.trivial=self.sampled=0
        n,g=self.level_count.cpu().tolist()
        if n>=self.min_episodes:
            rate=g/n;self.level_count.zero_()
            if rate>=self.threshold and self.level<len(LEVELS)-1:
                row['carry_level_change']=dict(epoch=epoch,frames=frames,old=self.level,new=self.level+1,carry_goal_rate=rate,episodes=int(n));self.level+=1
        return row

    def report(self):
        return dict(bank=str(self.path),share=self.share,threshold=self.threshold,min_episodes=self.min_episodes,states=self.states,levels=[list(l) for l in LEVELS])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter);sub=p.add_subparsers(dest='cmd',required=True)
    b=sub.add_parser('build');b.add_argument('--replays',type=Path,nargs='+',required=True);b.add_argument('--output',type=Path,required=True)
    b.add_argument('--stride',type=int,default=20);b.add_argument('--min-clearance',type=float,default=.03);b.add_argument('--object-dir',type=Path)
    build(p.parse_args())
