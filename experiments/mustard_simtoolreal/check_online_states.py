"""Roll out a train_scratch.py checkpoint with the bank + online native-near buffer and report, per reset kind,
episode count, first-goal rate, mean length, lift rate and termination reasons (debug for kind 4)."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parents[1]
sys.path[:0]=[str(ROOT),str(PROJECT/'third_party/simtoolreal'),str(PROJECT/'third_party/simtoolreal/rl_games')]


def main():
    from isaaclab.app import AppLauncher
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--num-envs',type=int,default=600);p.add_argument('--steps',type=int,default=3000)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--filters',nargs='*',default=[])
    AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
    code=0
    try:run(a)
    except BaseException:
        import traceback;traceback.print_exc();code=1
    finally:sys.stdout.flush();sys.stderr.flush();import os;os._exit(code)


def run(a):
    import torch,numpy as np
    import physics_align;physics_align.install()
    if a.filters:
        import self_collision_filters;self_collision_filters.install(a.filters)
    from environment import create_env
    from adept_obs import create_adept_env
    from diverse_resets import DiverseResetBank
    from online_resets import OnlineNativeNearBuffer
    from rl_games.algos_torch import model_builder
    def configure(cfg):cfg.domain_randomization.torque_scale=2.
    env=create_adept_env(create_env,a.num_envs,configure,a.output.parent);dev=env.device;n=a.num_envs;D=env.cfg.observation_space
    bank=DiverseResetBank(env,a.bank);online=OnlineNativeNearBuffer(env,bank,probability=.5,min_fill=300)
    ck=torch.load(a.checkpoint,map_location=dev,weights_only=False);ck=ck.get(0,ck) if 'model' not in ck else ck
    net=json.loads((a.checkpoint.parent/'sapg_config.json').read_text())['params']['network']
    model=model_builder.ModelBuilder().load(dict(network=net,model=dict(name='continuous_a2c_logstd'))).build(dict(actions_num=29,input_shape=(D+1,),
        num_seqs=n,value_size=1,normalize_input=True,normalize_value=True,type='extra_param',coef_ids=torch.linspace(50,0,6,device=dev),coef_id_idx=D)).to(dev)
    model.load_state_dict(ck['model']);model.eval();h=[s.to(dev).zero_() for s in model.get_default_rnn_state()]
    obs,_=env.reset();length=torch.zeros(n,device=dev);lifted=torch.zeros(n,dtype=torch.bool,device=dev);rows=[]
    with torch.no_grad():
        for step in range(a.steps):
            kind=bank.current_kind.clone()
            inp=torch.cat([obs['policy'].clamp(-10,10),torch.full((n,1),50.,device=dev)],-1)
            mu,_,_,hh=model.a2c_network(dict(obs=model.norm_obs(inp),rnn_states=h,seq_length=1));h=list(hh)
            obs,_,term,trunc,_=env.step(mu.clamp(-1,1));length+=1;lifted|=env._lifted_object
            done=(term|trunc).nonzero().flatten()
            reasons=env._termination_reasons
            for i in done.tolist():
                rows.append(dict(kind=int(kind[i]),goals=int(env._prev_episode_successes[i]),length=int(length[i]),lifted=bool(lifted[i]),
                    reason=[k for k,v in reasons.items() if bool(v[i])]))
            length[done]=0;lifted[done]=False
            for s in h:s[:,done]=0
    out={'online':online.report()}
    for k in range(5):
        r=[x for x in rows if x['kind']==k]
        if not r:continue
        out[str(k)]=dict(episodes=len(r),any_goal=float(np.mean([x['goals']>=1 for x in r])),mean_length=float(np.mean([x['length'] for x in r])),
            lifted=float(np.mean([x['lifted'] for x in r])),reasons={q:float(np.mean([q in x['reason'] for x in r])) for q in ('fall','hand_far','timeout','max_successes')})
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=1)+'\n');print('ONLINE_CHECK',json.dumps(out),flush=True)


if __name__=='__main__':main()
