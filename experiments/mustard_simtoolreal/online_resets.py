"""Online native-near reset buffer on top of DiverseResetBank (kind 4).

During native-kind episodes (bank kind 0), states where the object rests on the table, is not lifted, and the
nearest fingertip is within `near_m` of the object are captured from the simulator (physically valid by
construction) into a GPU ring buffer. At reset, a fraction `probability` of the envs the bank assigned to
native restart from a uniformly sampled buffered state instead (kind 4). Native metrics and the tolerance
curriculum keep using kind-0 episodes only (bank.last_completed_kind == 0).
"""
import torch
from diverse_resets import capture_states,restore_states,table_top_z


class OnlineNativeNearBuffer:
    def __init__(self,env,bank,probability=.3,capacity=20000,near_m=.06,min_step=20,per_step=64,min_fill=2000):
        assert 0<probability<1 and capacity>=min_fill>0
        self.env=env;self.bank=bank;self.probability=probability;self.capacity=capacity;self.near_m=near_m
        self.min_step=min_step;self.per_step=per_step;self.min_fill=min_fill;self.size=0;self.cursor=0;self.data=None
        self.captured=0;self.restored=0
        reset=env._reset_idx
        def reset_with_buffer(env_ids):
            reset(env_ids)
            if self.size<self.min_fill:return
            ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else torch.as_tensor(env_ids,device=env.device,dtype=torch.long)
            native=ids[bank.current_kind[ids]==0]
            chosen=native[torch.rand(len(native),device=env.device)<self.probability]
            if not len(chosen):return
            entries=torch.randint(self.size,(len(chosen),),device=env.device)
            restore_states(env,chosen,{k:v[entries] for k,v in self.data.items()})
            bank.current_kind[chosen]=4;bank.current_entry[chosen]=-1;self.restored+=len(chosen)
            env.scene.write_data_to_sim();env.sim.forward();env.scene.update(0.)
        env._reset_idx=reset_with_buffer
        rewards=env._get_rewards
        def rewards_and_capture():
            out=rewards();self.capture();return out
        env._get_rewards=rewards_and_capture

    def capture(self):
        env=self.env
        # Same-episode native envs only; the state is read after the physics step, before resets.
        obj=env.object.data.root_pos_w-env.scene.env_origins
        resting=(obj[:,2]-table_top_z(env)<.12)&~env._lifted_object&(env.object.data.root_lin_vel_w.norm(dim=-1)<.3)
        near=env._curr_fingertip_distances.min(-1).values<self.near_m
        ok=(self.bank.current_kind==0)&resting&near&(env.episode_length_buf>=self.min_step)
        ids=ok.nonzero().flatten()
        if not len(ids):return
        ids=ids[torch.randperm(len(ids),device=env.device)[:self.per_step]]
        state=capture_states(env,ids)
        if self.data is None:self.data={k:torch.zeros((self.capacity,*v.shape[1:]),device=env.device,dtype=v.dtype) for k,v in state.items()}
        slots=(self.cursor+torch.arange(len(ids),device=env.device))%self.capacity
        for k,v in state.items():self.data[k][slots]=v
        self.cursor=int((self.cursor+len(ids))%self.capacity);self.size=min(self.capacity,self.size+len(ids));self.captured+=len(ids)

    def report(self):
        return dict(probability=self.probability,capacity=self.capacity,near_m=self.near_m,min_step=self.min_step,
            size=self.size,captured=self.captured,restored=self.restored,min_fill=self.min_fill)
