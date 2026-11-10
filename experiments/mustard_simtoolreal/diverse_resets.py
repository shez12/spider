"""GPU cached physical resets for the existing native-action SAPG task."""
import hashlib
import json
from pathlib import Path
import numpy as np

FIELDS=('q','v','previous','current','root','goal','table')


def environment_signature():
    root=Path(__file__).resolve().parent
    files=sorted(p for p in (root/'assets').rglob('*') if p.suffix in ('.obj','.urdf','.json'))
    files.extend([root/'environment.py',root.parents[1]/'third_party/simtoolreal/isaacsimenvs/cfg/task/SimToolReal.yaml',
        root.parents[1]/'third_party/simtoolreal/assets/urdf/table_narrow.urdf'])
    digest=hashlib.sha256()
    for path in files:digest.update(path.name.encode());digest.update(path.read_bytes())
    return digest.hexdigest()


def clear_trackers(env,ids):
    env._closest_fingertip_dist[ids]=-1;env._closest_keypoint_max_dist[ids]=-1
    env._near_goal_steps[ids]=0;env._is_success[ids]=False;env._successes[ids]=0
    env._lifted_object[ids]=False;env.episode_length_buf[ids]=0;env.reward_buf[ids]=0
    env._object_init_z[ids]=env.object.data.root_pos_w[ids,2]-env.scene.env_origins[ids,2]


def restore_states(env,ids,state):
    """States use canonical joints and env-local world-frame object/table poses."""
    p=env._perm_canon_to_lab
    env.robot.write_joint_state_to_sim(state['q'][:,p],state['v'][:,p],env_ids=ids)
    previous=state['previous'][:,p];current=state['current'][:,p]
    env._prev_targets[ids]=previous;env._cur_targets[ids]=current
    env.robot.set_joint_position_target(previous,env_ids=ids)
    for name in ('root','goal','table'):
        value=state[name].clone();value[:,:3]+=env.scene.env_origins[ids]
        asset=env.object if name=='root' else env.goal_viz if name=='goal' else env.table
        if name=='root':asset.write_root_state_to_sim(value,env_ids=ids)
        else:asset.write_root_pose_to_sim(value,env_ids=ids)
    # Native reset height tracks the box CENTER; the top face is +0.15m.
    env._table_z_per_env[ids]=state['table_z']
    clear_trackers(env,ids)


def table_top_z(env):
    """Actual local top face of table_narrow (0.30m thick, yaw-only rotation)."""
    import torch
    assert bool((env.table.data.root_quat_w[:,1:3].abs()<1e-5).all()),'Tilted table needs a plane-distance calculation'
    return env.table.data.root_pos_w[:,2]-env.scene.env_origins[:,2]+.15


def capture_states(env,ids):
    p=env._perm_lab_to_canon
    state={k:getattr(env.robot.data,'joint_pos' if k=='q' else 'joint_vel')[ids][:,p].clone() for k in ('q','v')}
    state.update(previous=env._prev_targets[ids][:,p].clone(),current=env._cur_targets[ids][:,p].clone())
    for name,asset in [('root',env.object),('goal',env.goal_viz),('table',env.table)]:
        value=asset.data.root_state_w[ids].clone()
        if name!='root':value=value[:,:7]
        value[:,:3]-=env.scene.env_origins[ids];state[name]=value
    state['table_z']=env._table_z_per_env[ids].clone()
    return state


def hold_actions(env):
    """Hold the filter command through the original 29-D action mapping."""
    import torch
    p=env._perm_lab_to_canon;prev=env._prev_targets[:,p]
    lo=env._joint_lower_canon;hi=env._joint_upper_canon
    a=torch.zeros_like(prev);a[:,7:]=2*(prev[:,7:]-lo[7:])/(hi[7:]-lo[7:])-1
    return a.clamp(-1,1)


class DiverseResetBank:
    def __init__(self,env,path,probabilities=(.4,.25,.2,.15)):
        import torch
        self.env=env;self.path=Path(path);self.probabilities=torch.tensor(probabilities,device=env.device)
        assert len(probabilities)==4 and all(p>=0 for p in probabilities) and abs(sum(probabilities)-1)<1e-6
        with np.load(path,allow_pickle=False) as data:
            assert int(data['format_version'])==1
            assert str(data['environment_sha256'])==environment_signature(),'Reset bank assets/environment changed'
            self.data={k:torch.tensor(data[k],device=env.device) for k in (*FIELDS,'table_z')}
            self.kinds=torch.tensor(data['kind'],device=env.device,dtype=torch.long)
        assert all(torch.isfinite(v).all() for v in self.data.values())
        self.indices={kind:torch.where(self.kinds==kind)[0] for kind in (1,2,3)}
        assert all(len(v)>0 for v in self.indices.values()),'Each reset type needs validated states'
        self.counts=torch.zeros((6,4),device=env.device,dtype=torch.long)
        self.current_kind=torch.zeros(env.num_envs,device=env.device,dtype=torch.long)
        self.last_completed_kind=torch.full((env.num_envs,),-1,device=env.device,dtype=torch.long)
        self.current_entry=torch.full((env.num_envs,),-1,device=env.device,dtype=torch.long)
        self.original=env._reset_idx
        def reset(env_ids):
            ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else torch.as_tensor(env_ids,device=env.device,dtype=torch.long)
            self.last_completed_kind[ids]=self.current_kind[ids]
            self.original(env_ids)
            kinds=torch.multinomial(self.probabilities,len(ids),replacement=True)
            self.current_kind[ids]=kinds;self.current_entry[ids]=-1
            for kind in (1,2,3):
                selected=ids[kinds==kind]
                if not len(selected):continue
                pool=self.indices[kind];entries=pool[torch.randint(len(pool),(len(selected),),device=env.device)]
                self.current_entry[selected]=entries
                restore_states(env,selected,{k:v[entries] for k,v in self.data.items()})
            block=torch.div(ids,max(1,env.num_envs//6),rounding_mode='floor').clamp(max=5)
            self.counts+=torch.bincount(block*4+kinds,minlength=24).reshape(6,4)
            env.scene.write_data_to_sim();env.sim.forward();env.scene.update(0.)
        env._reset_idx=reset
        env._curriculum_eligible_mask=self.curriculum_eligible_mask

    def curriculum_eligible_mask(self):
        """Gate tolerance using the latest completed native episode only."""
        return self.last_completed_kind == 0

    def report(self):
        return dict(bank=str(self.path),sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
            names=['native','near_object','stable_grasp','near_goal'],probabilities=self.probabilities.cpu().tolist(),
            states_per_kind={str(k):len(v) for k,v in self.indices.items()},counts_per_block=self.counts.cpu().tolist())

    def restore(self):
        self.env._reset_idx=self.original
        del self.env._curriculum_eligible_mask


def audit_bank_resets(env,bank):
    """Check partial resets preserve unrelated envs and restore commands exactly."""
    import torch
    env.reset();before=capture_states(env,torch.arange(env.num_envs,device=env.device))
    ids=torch.arange(0,env.num_envs,2,device=env.device);other=torch.arange(1,env.num_envs,2,device=env.device)
    env._reset_idx(ids);after=capture_states(env,torch.arange(env.num_envs,device=env.device))
    untouched={k:float((before[k][other]-after[k][other]).abs().max()) for k in before}
    assert max(untouched.values())<2e-5,untouched
    selected=ids[bank.current_entry[ids]>=0];entries=bank.current_entry[selected]
    errors={k:float((after[k][selected]-bank.data[k][entries]).abs().max()) for k in after}
    assert max(errors.values())<2e-5,errors
    assert not env._successes[ids].any() and not env.episode_length_buf[ids].any()
    return dict(partial_reset_untouched_max=untouched,restoration_max=errors,**bank.report())
