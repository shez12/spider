"""Original SPIDER sampling update with direct Isaac Lab PhysX rollouts.

Use the existing SimToolReal asset/scene builder, bypassing policy actions,
reward, randomization and episode resets during rollout. No running training
environment or source assets are modified.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
LOCAL=ROOT/'experiments/mustard_simtoolreal'
VENDOR=ROOT/'third_party/simtoolreal'
sys.path[:0]=[str(ROOT),str(LOCAL),str(VENDOR),str(VENDOR/'rl_games')]


def main():
    from isaaclab.app import AppLauncher
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--samples',type=int,default=512)
    p.add_argument('--max-iterations',type=int,default=200)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--smoke',action='store_true')
    AppLauncher.add_app_launcher_args(p)
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();app=AppLauncher(args).app;env=None
    try:
        import numpy as np
        import torch
        from environment import create_env
        from isaaclab.utils.math import quat_apply,quat_apply_inverse,quat_mul
        from spider.config import Config
        from spider.optimizers.sampling import make_optimize_once_fn
        from examples.arm_hand.optimize_kuka_gpu import rotation_error
        contract=json.loads((args.input_dir/'isaac_contract.json').read_text())
        ref=dict(np.load(args.input_dir/'reference.npz'))
        def configure(cfg):
            cfg.seed=args.seed;cfg.sim.dt=.002;cfg.decimation=5;cfg.sim.render_interval=1000000
            cfg.assets.table_urdf=contract['table_urdf']
            cfg.reset.table_reset_z=.53;cfg.reset.table_reset_z_range=0.
            cfg.reset.table_reset_xy_range_m=(0.,0.);cfg.reset.table_reset_yaw_range_deg=0.
            dr=cfg.domain_randomization
            dr.use_action_delay=False;dr.use_obs_delay=False;dr.use_object_state_delay_noise=False
            dr.joint_velocity_obs_noise_std=0.;dr.force_scale=0.;dr.torque_scale=0.
            cfg.assets.robot_friction=contract['robot_friction']
            cfg.assets.finger_tip_friction=contract['robot_friction']
            cfg.assets.object_friction=contract['object_friction'];cfg.assets.table_friction=contract['table_friction']
        env=create_env(args.samples,configure=configure);env.reset()
        dev=env.device;n=args.samples;steps=len(ref['ctrl_nominal']);nu=29;knots=10
        tensor=lambda x:torch.as_tensor(x,device=dev,dtype=torch.float32)
        perm=[env.robot.joint_names.index(name) for name in contract['joint_names']]
        inv=torch.argsort(torch.tensor(perm,device=dev))
        # Canonical -> Lab tensors index with inverse; Lab -> canonical with perm.
        robot=env.robot;obj=env.object
        kp=tensor(contract['kp'])[inv].expand(n,-1).clone()
        kd=tensor(contract['kd_effective'])[inv].expand(n,-1).clone()
        effort=tensor(contract['effort'])[inv].expand(n,-1).clone()
        armature=tensor(contract['armature'])[inv].expand(n,-1).clone()
        for actuator in robot.actuators.values():
            ids=actuator.joint_indices
            actuator.stiffness[:]=kp[:,ids];actuator.damping[:]=kd[:,ids]
            actuator.effort_limit[:]=effort[:,ids]
        robot.write_joint_stiffness_to_sim(kp);robot.write_joint_damping_to_sim(kd)
        robot.write_joint_effort_limit_to_sim(effort);robot.write_joint_armature_to_sim(armature)
        robot.write_joint_friction_coefficient_to_sim(tensor(contract['frictionloss'])[inv].expand(n,-1).clone())
        q0=tensor(contract['initial_qpos']);v0=tensor(contract['initial_qvel'])
        center=tensor(contract['object_center']);shift=tensor([0.,0.,.53])
        initial_object=torch.zeros((n,13),device=dev)
        initial_object[:,:3]=q0[-7:-4]+quat_apply(q0[-4:].expand(n,-1),center.expand(n,-1))+shift+env.scene.env_origins
        initial_object[:,3:7]=q0[-4:]
        # MuJoCo free-joint linear velocity is world-frame; angular velocity is local.
        omega=quat_apply(q0[-4:].expand(n,-1),v0[-3:].expand(n,-1))
        initial_object[:,7:10]=v0[-6:-3]+torch.cross(omega,quat_apply(q0[-4:].expand(n,-1),center.expand(n,-1)),dim=-1)
        initial_object[:,10:]=omega
        body_ids=[robot.body_names.index(site['body']) for site in contract['sites']]
        site_offsets=tensor([site['pos'] for site in contract['sites']])
        site_quats=tensor([site['quat'] for site in contract['sites']])
        nominal=tensor(ref['ctrl_nominal']);low=tensor(contract['ctrl_low']);high=tensor(contract['ctrl_high'])
        interpolation=tensor(np.stack([np.interp(np.arange(steps),np.linspace(0,steps-1,knots).astype(int),row) for row in np.eye(knots)],axis=1))
        wrist=tensor(ref['wrist_pos']);wquat=tensor(ref['wrist_quat']);target_obj=tensor(ref['object_qpos']);joints=tensor(ref['joint_ik'])
        batch_qpos=torch.empty((n,steps,36),device=dev);batch_qvel=torch.empty((n,steps,35),device=dev)
        def sites():
            pose=robot.data.body_link_state_w[:,body_ids,:7]
            pos=pose[:,:,:3]+quat_apply(pose[:,:,3:7],site_offsets.expand(n,-1,-1))-env.scene.env_origins[:,None,:]-shift
            quat=quat_mul(pose[:,:,3:7],site_quats.expand(n,-1,-1))
            return pos,quat
        def reset():
            robot.write_joint_state_to_sim(q0[:29][inv].expand(n,-1),v0[:29][inv].expand(n,-1))
            robot.set_joint_position_target(q0[:29][inv].expand(n,-1))
            obj.write_root_state_to_sim(initial_object)
            env.scene.reset();env.scene.write_data_to_sim();env.sim.forward();env.scene.update(0.)
        reset()
        initial_sites,_=sites()
        fk_error=float((initial_sites[0]-tensor(contract['initial_site_pos'])).abs().max())
        if fk_error>.001:raise RuntimeError(f'Initial source/Isaac FK mismatch: {fk_error} m')
        startup=time.monotonic()-started
        def evaluate(offsets):
            commands=(nominal+torch.einsum('tk,bkj->btj',interpolation,offsets)).clamp(low,high)
            reset();cost=torch.zeros(n,device=dev)
            for t in range(steps):
                robot.set_joint_position_target(commands[:,t][:,inv])
                for sub in range(5):
                    env.scene.write_data_to_sim();env.sim.step(render=False);env.scene.update(.002)
                    if sub==0 and t%3==0:
                        pos,quat=sites()
                        wrist_cost=225*(pos[:,0]-wrist[t]).square().sum(-1)
                        rot_cost=.64*rotation_error(quat[:,0],wquat[t]).square()
                state=obj.data.root_link_state_w
                object_q=state[:,3:7]
                object_p=state[:,:3]-env.scene.env_origins-shift-quat_apply(object_q,center.expand(n,-1))
                joint_p=robot.data.joint_pos[:,perm]
                if t%3==0:
                    cost+=.5*(wrist_cost+rot_cost+3600*(object_p-target_obj[t+1,:3]).square().sum(-1)
                              +.09*rotation_error(object_q,target_obj[t+1,3:]).square()
                              +.0025*(joint_p[:,7:]-joints[t+1,7:]).square().sum(-1))
                batch_qpos[:,t,:29]=joint_p;batch_qpos[:,t,29:32]=object_p;batch_qpos[:,t,32:]=object_q
                batch_qvel[:,t,:29]=robot.data.joint_vel[:,perm]
                omega_world=state[:,10:13]
                batch_qvel[:,t,29:32]=state[:,7:10]-torch.cross(omega_world,quat_apply(object_q,center.expand(n,-1)),dim=-1)
                batch_qvel[:,t,32:]=quat_apply_inverse(object_q,omega_world)
            cost+=.5*(.0009*offsets.square().sum((1,2))+.0004*offsets.diff(dim=1).square().sum((1,2)))
            return torch.where(torch.isfinite(cost),cost,torch.full_like(cost,float('inf'))),commands
        torch.manual_seed(args.seed)
        best_cost=float('inf');best=None;best_rollout=None;history=[]
        config=Config(device=dev,num_samples=n,use_torch_compile=False,temperature=1.,num_trace_uniform_samples=0,num_trace_topk_samples=0)
        config.knot_steps=1;config.noise_scale=tensor([.035]*7+[.12]*22).expand(n,knots,nu).clone();config.noise_scale[0]=0
        def rollout(config,unused,candidates,ref_slice,env_param):
            nonlocal best_cost,best,best_rollout
            candidates=candidates.clamp(-.4,.4)
            costs,commands=evaluate(candidates);index=costs.argmin();current=float(costs[index])
            if current<best_cost:
                best_cost=current;best=candidates[index].clone()
                best_rollout=dict(qpos=batch_qpos[index].clone(),qvel=batch_qvel[index].clone(),ctrl=commands[index].clone())
            return candidates,-costs,torch.zeros(n,dtype=torch.bool,device=dev),{}
        center_offsets=torch.zeros((knots,nu),device=dev)
        with torch.no_grad():
            warm=time.monotonic();warmcost,_=evaluate(center_offsets.expand(n,-1,-1));torch.cuda.synchronize()
            warm_elapsed=time.monotonic()-warm
            np.savez(args.output_dir/'nominal_trajectory.npz',initial_state=ref['initial_state'],qpos=batch_qpos[0].cpu().numpy(),qvel=batch_qvel[0].cpu().numpy(),control_dt=.01,physics_dt=.002)
            (args.output_dir/'startup.json').write_text(json.dumps(dict(startup_s=startup,warmup_rollout_s=warm_elapsed,initial_fk_max_abs_m=fk_error,nominal_cost=float(warmcost[0]),samples=n,steps=steps),indent=2))
            if args.smoke:return
            torch.manual_seed(args.seed);update=make_optimize_once_fn(rollout);training_started=time.monotonic()
            for iteration in range(args.max_iterations):
                tick=time.monotonic();center_offsets,_,_=update(config,None,center_offsets,(),sample_params={'global_noise_scale':.97**iteration})
                torch.cuda.synchronize()
                history.append(dict(iteration=iteration+1,best_cost=best_cost,iteration_s=time.monotonic()-tick,elapsed_s=time.monotonic()-training_started))
                (args.output_dir/'history.json').write_text(json.dumps(history,indent=2))
                checkpoint=args.output_dir/'checkpoint.tmp'
                torch.save(dict(center=center_offsets,best=best,best_cost=best_cost,best_rollout=best_rollout,history=history,rng=torch.cuda.get_rng_state()),checkpoint)
                checkpoint.replace(args.output_dir/'checkpoint.pt')
                print('ISAAC_SPIDER',json.dumps(history[-1]),flush=True)
            training_s=time.monotonic()-training_started
            for name,arrays in [('trajectory',best_rollout)]:
                np.savez(args.output_dir/f'{name}.npz',initial_state=ref['initial_state'],control_dt=.01,physics_dt=.002,**{k:v.cpu().numpy() for k,v in arrays.items()})
            replay_started=time.monotonic();costs,commands=evaluate(best.expand(n,-1,-1));torch.cuda.synchronize()
            np.savez(args.output_dir/'replay_trajectory.npz',initial_state=ref['initial_state'],qpos=batch_qpos[0].cpu().numpy(),qvel=batch_qvel[0].cpu().numpy(),ctrl=commands[0].cpu().numpy(),control_dt=.01,physics_dt=.002)
            result=dict(backend='isaacsim_physx',samples=n,iterations=len(history),seed=args.seed,best_cost=best_cost,
                independent_replay_cost=float(costs[0]),training_s=training_s,independent_replay_s=time.monotonic()-replay_started,
                startup_s=startup,warmup_s=warm_elapsed,total_s=time.monotonic()-started,extra_hold=False,
                optimizer='original spider.make_optimize_once_fn',fixed_iterations=True,
                limitations=['PhysX contact solver differs from MuJoCo','Source PD including passive damping mapped to implicit PhysX drive',
                    'Friction models and collision import can differ','Geometric validation uses identical source MuJoCo shapes on exported states'])
            (args.output_dir/'result.json').write_text(json.dumps(result,indent=2))
    finally:
        if env is not None:env.close()
        app.close()


if __name__=='__main__':main()
