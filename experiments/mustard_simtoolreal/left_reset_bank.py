"""Reset bank from left_grasp_mpc.py trajectories, and a reset sampler for the unmodified SimToolReal env.

Build (CPU):  python left_reset_bank.py build --runs outputs/left_grasp_mpc_v1 [...] --output outputs/left_reset_bank_v1/reset_bank.npz
Kinds (states taken from successful trajectories, every --stride steps; captured from the simulator, so physically valid):
  1 near_object : before the first lift above 1 cm, nearest-3 fingertip mean < 8 cm
  2 grasp       : object lifted 1-6 cm with fingertips within 8 cm
  3 lifted      : object lifted > 6 cm with fingertips within 8 cm
StateResetBank(env, path, probabilities) wraps env._reset_idx like diverse_resets.DiverseResetBank (kind 0 = vendor
native reset) and checks the bank signature against the live env assets instead of the right-hand asset hash.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

FIELDS=('q','v','previous','current','root','goal','table','table_z')
KINDS=('native','near_object','grasp','lifted')


def signature(robot_urdf,object_urdf,task_yaml):
    h=hashlib.sha256()
    for p in (robot_urdf,object_urdf,task_yaml):h.update(Path(p).name.encode());h.update(Path(p).read_bytes())
    return h.hexdigest()


def build(a):
    out={k:[] for k in FIELDS};kind=[];source=[];sig=None;stats=dict(runs=[],groups=0,successful=0)
    for run in a.runs:
        d=np.load(run/'trajectories.npz');args=json.loads((run/'run_arguments.json').read_text());stats['runs'].append(str(run))
        sig_run=(run/'signature.txt').read_text().strip() if (run/'signature.txt').exists() else None
        assert sig is None or sig_run==sig,'runs from different environments';sig=sig_run
        lift=d['lift'];tips=d['tip_distance'];ok=d['success'];stats['groups']+=len(ok);stats['successful']+=int(ok.sum())
        for g in np.flatnonzero(ok):
            first=int(np.argmax(lift[g]>.01))
            for t in range(0,lift.shape[1],a.stride):
                k=0
                if tips[g,t]<.08:
                    if t<first:k=1
                    elif .01<lift[g,t]<=.06:k=2
                    elif lift[g,t]>.06:k=3
                if not k:continue
                for f in FIELDS:out[f].append(d[f][g,t])
                kind.append(k);source.append((len(stats['runs'])-1,int(g),t))
    arrays={f:np.stack(v).astype(np.float32) for f,v in out.items()}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,**arrays,kind=np.array(kind),source=np.array(source),signature=np.array(sig or ''),format_version=np.array(1))
    counts={KINDS[k]:int((np.array(kind)==k).sum()) for k in (1,2,3)}
    report=dict(stats,stride=a.stride,states=len(kind),per_kind=counts,signature=sig)
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=1)+'\n');print(json.dumps(report))


class StateResetBank:
    def __init__(self,env,path,probabilities=(.5,.2,.15,.15),expected_signature=None,delta_first_goal=False):
        """delta_first_goal: bank episodes get a first goal sampled like every later native goal (vendor delta
        sampler: 0.1 m, <=90 deg) around the restored object pose, instead of the stored absolute goal."""
        import torch
        from isaacsimenvs.tasks.simtoolreal.utils.reset_utils import _reset_goal_pose
        self.delta_first_goal=delta_first_goal
        self.env=env;self.path=Path(path);dev=env.device
        with np.load(path) as d:
            if expected_signature is not None:assert str(d['signature'])==expected_signature,'bank was built in a different environment'
            self.data={k:torch.tensor(d[k],device=dev) for k in FIELDS};kinds=torch.tensor(d['kind'],device=dev,dtype=torch.long)
        self.p=torch.tensor(probabilities,device=dev,dtype=torch.float32);assert len(probabilities)==4 and abs(float(self.p.sum())-1)<1e-6
        for k in (1,2,3):
            if not bool((kinds==k).any()):self.p[k]=0.
        self.p/=self.p.sum();self.indices={k:torch.where(kinds==k)[0] for k in (1,2,3)}
        self.current_kind=torch.zeros(env.num_envs,device=dev,dtype=torch.long);self.last_completed_kind=torch.full((env.num_envs,),-1,device=dev,dtype=torch.long)
        self.original=env._reset_idx
        from diverse_resets import restore_states
        def reset(env_ids):
            ids=torch.arange(env.num_envs,device=dev) if env_ids is None else torch.as_tensor(env_ids,device=dev,dtype=torch.long)
            self.last_completed_kind[ids]=self.current_kind[ids];self.original(env_ids)
            kinds=torch.multinomial(self.p,len(ids),replacement=True);self.current_kind[ids]=kinds
            for k in (1,2,3):
                sel=ids[kinds==k]
                if not len(sel):continue
                pool=self.indices[k];e=pool[torch.randint(len(pool),(len(sel),),device=dev)]
                restore_states(env,sel,{f:v[e] for f,v in self.data.items()})
                # restore_states sets the lift reference to the current height; use the vendor reference instead
                # (spawn height = table centre z + table_object_z_offset) and latch states already above the
                # lift threshold, so held states count as lifted and get no free one-shot lift bonus.
                rc=env.cfg.reset;init=env._table_z_per_env[sel]+rc.table_object_z_offset;env._object_init_z[sel]=init
                z=self.data['root'][e,2]  # env-local, as stored by capture_states
                env._lifted_object[sel]=(.05+z-init)>env.cfg.reward.lifting_bonus_threshold
                if self.delta_first_goal:self._delta_goal(sel,self.data['root'][e])
            env.scene.write_data_to_sim();env.sim.forward();env.scene.update(0.)
        env._reset_idx=reset
        env._curriculum_eligible_mask=lambda:self.last_completed_kind==0

    def _delta_goal(self,sel,root,margin=1.2,tries=8):
        """Vendor delta goal around the restored object pose, resampled while the object already satisfies it
        (initial keypoint max error <= margin x success threshold): hold-still would otherwise 'succeed' (12-27%)."""
        import torch
        from isaacsimenvs.tasks.simtoolreal.utils.reset_utils import _reset_goal_pose
        from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import _keypoints_world
        env=self.env;o=env.scene.env_origins;tol=env._current_success_tolerance*env.cfg.reward.keypoint_scale*margin
        kp=env._keypoint_offsets_fixed if env.cfg.reward.fixed_size_keypoint_reward else env._keypoint_offsets
        obj_kp=_keypoints_world(root[:,:3],root[:,3:7],kp[sel]);todo=torch.arange(len(sel),device=env.device)
        for _ in range(tries):
            ids=sel[todo];env.goal_viz.write_root_pose_to_sim(torch.cat([root[todo,:3]+o[ids],root[todo,3:7]],-1),env_ids=ids)
            _reset_goal_pose(env,ids,'delta')
            g=env.goal_viz.data.root_pos_w[ids]-o[ids];gq=env.goal_viz.data.root_quat_w[ids]
            d=(obj_kp[todo]-_keypoints_world(g,gq,kp[ids])).norm(dim=-1).max(-1).values
            todo=todo[d<=tol]
            if not len(todo):break
        self.trivial_goals_left=getattr(self,'trivial_goals_left',0)+len(todo)

    def report(self):
        return dict(bank=str(self.path),probabilities=self.p.cpu().tolist(),states_per_kind={KINDS[k]:len(v) for k,v in self.indices.items()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
    b=sub.add_parser('build');b.add_argument('--runs',type=Path,nargs='+',required=True);b.add_argument('--output',type=Path,required=True)
    b.add_argument('--stride',type=int,default=5)
    a=p.parse_args();build(a)
