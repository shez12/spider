"""Sampling-based MPC (MPPI) grasp-and-lift trajectories for the mustard bottle in the UNMODIFIED SimToolReal env
(left KUKA+Sharpa, /home/szc/work/newsimtoolreal), from native random initial states.

G start states (vendor native resets, object settled on the table) x S samples = G*S envs. At every replanning step
all S envs of a group are restored to the group's current physical state (diverse_resets.capture/restore_states),
roll out H policy steps of nominal-plus-noise actions in the native 29-D action space, and the nominal sequence is
updated with MPPI weights; the first C actions are then executed. The cost only uses simulator state:
fingertip/palm distance to the object, object lift above its settled height, and a penalty for knocking it off.
Object-state/observation/action DR and wrench impulses are off; automatic resets are disabled.

Writes per group the executed joint/object/goal state sequence (capture_states format) + actions + lift metrics,
for building a left-hand reset bank (left_reset_bank.py). Run from the original repo root (relative asset paths).
"""
import argparse,json,sys,time
from pathlib import Path
PRISTINE=Path('/home/szc/work/newsimtoolreal/simtoolreal');HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(PRISTINE),str(PRISTINE/'rl_games')];sys.path.append(str(HERE))


def main():
    from isaaclab.app import AppLauncher
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--groups',type=int,default=32);p.add_argument('--samples',type=int,default=256)
    p.add_argument('--horizon',type=int,default=40);p.add_argument('--execute',type=int,default=10);p.add_argument('--steps',type=int,default=360)
    p.add_argument('--iterations',type=int,default=2);p.add_argument('--temperature',type=float,default=.05)
    p.add_argument('--arm-sigma',type=float,default=.6);p.add_argument('--hand-sigma',type=float,default=.35);p.add_argument('--knot',type=int,default=5)
    p.add_argument('--settle',type=int,default=60);p.add_argument('--seed',type=int,default=0)
    AppLauncher.add_app_launcher_args(p);a=p.parse_args()
    a.output=a.output.resolve();assert not a.output.exists(),'Use a fresh output directory.';a.output.mkdir(parents=True)
    app=AppLauncher(a).app;code=0
    try:run(a)
    except BaseException:
        import traceback;traceback.print_exc();code=1
    finally:sys.stdout.flush();sys.stderr.flush();import os;os._exit(code)


def run(a):
    import numpy as np,torch,yaml
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env import SimToolRealEnv
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg
    from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import compute_intermediate_values
    from diverse_resets import capture_states,restore_states,table_top_z
    torch.manual_seed(a.seed)
    (a.output/'run_arguments.json').write_text(json.dumps(vars(a),indent=1,default=str)+'\n')
    cal=json.loads((HERE/'assets/object/calibration.json').read_text())
    cfg=SimToolRealEnvCfg();cfg.from_dict(yaml.safe_load((PRISTINE/'isaacsimenvs/cfg/task/SimToolReal.yaml').read_text()))
    G,S=a.groups,a.samples;N=G*S;cfg.seed=a.seed;cfg.scene.num_envs=N
    cfg.assets.object_urdf=str(HERE/'assets/object/mustard_bottle.urdf');cfg.assets.object_scale=tuple(cal['policy_object_scale']);cfg.reward.fixed_size=tuple(cal['bbox_dims_m'])
    dr=cfg.domain_randomization;dr.use_action_delay=False;dr.use_obs_delay=False;dr.use_object_state_delay_noise=False
    dr.joint_velocity_obs_noise_std=0.;dr.force_scale=0.;dr.torque_scale=0.
    cfg.episode_length_s=1e5;cfg.termination.episode_length=10**8;cfg.termination.tolerance_curriculum_interval=10**9
    env=SimToolRealEnv(cfg);dev=env.device
    from left_reset_bank import signature
    (a.output/'signature.txt').write_text(signature(PRISTINE/cfg.assets.robot_urdf,cfg.assets.object_urdf,PRISTINE/'isaacsimenvs/cfg/task/SimToolReal.yaml')+'\n')
    from isaacsimenvs.tasks.simtoolreal.utils.termination_utils import compute_terminations
    def no_done():
        # Vendor bookkeeping (termination reasons, goal trackers) for logging; never terminate or reset.
        compute_intermediate_values(env);compute_terminations(env);z=torch.zeros(N,device=dev,dtype=torch.bool);return z,z.clone()
    env._get_dones=no_done
    env.reset();first=torch.arange(G,device=dev)*S;group=torch.arange(N,device=dev)//S
    def broadcast(state):restore_states(env,torch.arange(N,device=dev),{k:v[group] for k,v in state.items()});env.scene.write_data_to_sim();env.sim.forward();env.scene.update(0.)
    # Settle each group's native start (object dropped from 0.25 m) with the arm still and the hand open.
    broadcast(capture_states(env,first))
    hold=torch.zeros(N,29,device=dev);hold[:,7:]=-1.
    for _ in range(a.settle):env.step(hold)
    start=capture_states(env,first);broadcast(start)
    rest_z=(env.object.data.root_pos_w[first,2]-env.scene.env_origins[first,2]).clone();top=table_top_z(env)[first]
    palm_id=env._palm_body_id
    def metrics():
        obj=env.object.data.root_pos_w-env.scene.env_origins;tips=env._curr_fingertip_distances.sort(-1).values[:,:3].mean(-1)
        palm=(env.robot.data.body_state_w[:,palm_id,:3]-env.scene.env_origins-obj).norm(dim=-1)
        lift=(obj[:,2]-rest_z[group]).clamp(min=0.)
        fell=obj[:,2]<top[group]-.05
        return tips,palm,lift,fell
    def step_cost():
        tips,palm,lift,fell=metrics()
        return tips+.3*palm-4.*lift.clamp(max=.2)+5.*fell.float(),lift,tips
    H,C,K=a.horizon,a.execute,a.knot;knots=H//K+1
    sigma=torch.tensor([a.arm_sigma]*7+[a.hand_sigma]*22,device=dev)
    U=torch.zeros(G,H,29,device=dev);U[:,:,7:]=-1.
    def noise():
        k=torch.randn(N,knots,29,device=dev)*sigma
        t=torch.arange(H,device=dev)/K;i=t.floor().long().clamp(max=knots-2);f=(t-i)[None,:,None]
        e=k[:,i]*(1-f)+k[:,i+1]*f;e[first]=0.;return e   # sample 0 of each group keeps the nominal
    executed=[];actions=[];lifts=[];tipd=[];state=start;t0=time.monotonic()
    for t in range(0,a.steps,C):
        for _ in range(a.iterations):
            broadcast(state);eps=noise();seq=(U[group]+eps).clamp(-1,1);cost=torch.zeros(N,device=dev)
            for h in range(H):
                env.step(seq[:,h]);c,_,_=step_cost();cost+=c*(1.+2.*(h==H-1))
            cg=cost.view(G,S);w=torch.softmax(-(cg-cg.min(1,keepdim=True).values)/(a.temperature*cg.std(1,keepdim=True).clamp_min(1e-6)),1)
            U=(w[:,:,None,None]*seq.view(G,S,H,29)).sum(1)
        broadcast(state)
        for h in range(C):
            env.step(U[group][:,h]);executed.append({k:v.cpu() for k,v in capture_states(env,first).items()});actions.append(U[:,h].cpu())
            _,lift,tips=step_cost();lifts.append(lift[first].cpu());tipd.append(tips[first].cpu())
        state=capture_states(env,first);U=torch.cat([U[:,C:],U[:,-1:].repeat(1,C,1)],1)
        print('MPC',t+C,'lift_mean',float(torch.stack(lifts)[-1].mean()),'lifted>5cm',int((torch.stack(lifts)[-1]>.05).sum()),'of',G,
            'tips',round(float(tipd[-1].mean()),3),'s',round(time.monotonic()-t0),flush=True)
    L=torch.stack(lifts).numpy();held=(L>.05).astype(int)
    run_len=np.zeros(G,int);best=np.zeros(G,int)
    for row in held:run_len=(run_len+1)*row;best=np.maximum(best,run_len)
    success=best>=30
    out={k:torch.stack([e[k] for e in executed],1).numpy() for k in executed[0]}
    np.savez_compressed(a.output/'trajectories.npz',**out,actions=torch.stack(actions,1).numpy(),lift=L.T,tip_distance=torch.stack(tipd).numpy().T,
        rest_z=rest_z.cpu().numpy(),table_top_z=top.cpu().numpy(),success=success,longest_lift_run=best)
    report=dict(groups=G,samples=S,steps=a.steps,success_30_steps_above_5cm=int(success.sum()),max_lift_m=L.max(0).round(3).tolist(),
        final_lift_m=L[-1].round(3).tolist(),longest_lift_run=best.tolist(),elapsed_s=time.monotonic()-t0)
    (a.output/'report.json').write_text(json.dumps(report,indent=1)+'\n');print('MPC_DONE',json.dumps(report),flush=True)


if __name__=='__main__':main()
