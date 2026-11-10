"""Extract and independently stabilize physical reset states, without trajectory optimization."""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parents[1]
sys.path[:0]=[str(ROOT),str(PROJECT/'third_party/simtoolreal'),str(PROJECT/'third_party/simtoolreal/rl_games')]


def proposals(dataset,waypoints,audit,drop,recovery):
    d=np.load(dataset);x=d['obs'];lengths=d['valid_lengths'];ids=d['candidate_ids']
    pool={t['candidate_id']:t for t in json.loads(waypoints.read_text())['trajectories']}
    records={r['candidate_id']:r for r in json.loads(audit.read_text())['records']};states=[];metadata=[]
    for i,c in enumerate(ids):
        c=int(c);ob=x[i,:lengths[i]];first=records[c]['first_hold_frame'];rel=ob[:,113:125].reshape(-1,4,3).mean(1);pos=rel+ob[:,87:90]
        distance=np.linalg.norm(ob[:,125:137].reshape(-1,4,3),axis=-1).max(1)
        # Distinct temporal slices, not hundreds of adjacent frames.
        near=[max(0,first-75),max(0,first-45)]
        grasp=[min(int(lengths[i])-1,first+15),min(int(lengths[i])-1,first+60)]
        candidates=np.flatnonzero((np.arange(len(ob))>=first)&(distance>.15)&(distance<.30))
        ng=[int(candidates[len(candidates)//2])] if len(candidates) else []
        for kind,frames in [(1,near),(2,grasp),(3,ng)]:
            for f in set(frames):
                o=ob[f];q=o[94:98][[3,0,1,2]].copy();root=np.r_[pos[f],q,np.zeros(6)].astype('float32')
                left=max(0,f-1);right=min(len(ob)-1,f+1)
                if right>left:root[7:10]=(pos[right]-pos[left])*60/(right-left)
                states.append(dict(normalized_q=o[:29],v=o[29:58],previous=o[58:87],current=o[58:87],root=root,
                    goal=np.array(pool[c]['pos'][-1]+pool[c]['quat_wxyz'][-1],dtype='float32')))
                metadata.append(dict(kind=kind,candidate_id=c,frame=f,origin='native_teacher_cache'))
    for path,origin in [(drop,'natural_drop'),(recovery,'bc_failure')]:
        if path is None:continue
        data=np.load(path)
        for i in range(len(data['q'])):
            states.append({k:data[k][i].copy() for k in ('q','v','previous','current','root','goal')})
            metadata.append(dict(kind=1,candidate_id=int(data['candidate_id'][i] if 'candidate_id' in data else data['candidate_ids'][i]),frame=-1,origin=origin,lab_order=True))
    return states,metadata


def main():
    from isaaclab.app import AppLauncher
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('dataset','waypoints','audit','output'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('drop','recovery'):p.add_argument('--'+k,type=Path,help='Optional extra failure states (omit when they are already in a bank being extended).')
    p.add_argument('--batch-size',type=int,default=120);p.add_argument('--validation-steps',type=int,default=30)
    AppLauncher.add_app_launcher_args(p);a=p.parse_args();assert a.validation_steps>=30
    a.output.mkdir(parents=True,exist_ok=False);app=AppLauncher(a).app;env=None;started=time.monotonic()
    try:
        import torch
        from environment import create_env
        from diverse_resets import restore_states,capture_states,hold_actions,clear_trackers,FIELDS,environment_signature
        from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import compute_intermediate_values
        from isaacsimenvs.tasks.simtoolreal.utils.reward_utils import compute_rewards
        raw,metadata=proposals(a.dataset,a.waypoints,a.audit,a.drop,a.recovery)
        def configure(c):
            c.reset.table_reset_z_range=0.;c.termination.tolerance_curriculum_interval=10**9
            dr=c.domain_randomization;dr.use_action_delay=False;dr.use_obs_delay=False;dr.use_object_state_delay_noise=False
            dr.joint_velocity_obs_noise_std=0.;dr.force_scale=0.;dr.torque_scale=0.
        env=create_env(a.batch_size,configure=configure);env.reset();dev=env.device;n=env.num_envs
        tensor=lambda value:torch.as_tensor(np.array(value),device=dev,dtype=torch.float32)
        table0=capture_states(env,torch.arange(n,device=dev))['table'][0]
        def no_done():
            compute_intermediate_values(env);z=torch.zeros(n,device=dev,dtype=torch.bool);return z,z.clone()
        env._get_dones=no_done;env._get_rewards=lambda:compute_rewards(env)
        snaps=[];snap_owner=[];snap_stage=[];saved=[];quality=[]
        def snapshot(state,owners,stage):
            count=len(owners);ix=torch.arange(count,device=dev)
            base=env.robot.data.root_state_w[ix,:7].clone();base[:,:3]-=env.scene.env_origins[ix]
            body=env.robot.data.body_link_state_w[ix,:,:3]-env.scene.env_origins[ix,None,:]
            snaps.append(dict(q=state['q'].cpu().numpy(),root=state['root'].cpu().numpy(),table=state['table'].cpu().numpy(),robot_base=base.cpu().numpy(),body_positions=body.cpu().numpy()))
            snap_owner.extend(owners);snap_stage.extend([stage]*count)
        lo=env._joint_lower_canon;hi=env._joint_upper_canon;p_c=env._perm_lab_to_canon
        for offset in range(0,len(raw),n):
            batch=raw[offset:offset+n];count=len(batch);ix=torch.arange(count,device=dev);owners=list(range(offset,offset+count));kinds=torch.tensor([metadata[i]['kind'] for i in owners],device=dev)
            q=[];v=[];prev=[];cur=[]
            for r,m in zip(batch,metadata[offset:offset+count]):
                q.append((lo+.5*(tensor(r['normalized_q'])+1)*(hi-lo)).cpu().numpy() if 'normalized_q' in r else r['q'][p_c.cpu().numpy()])
                for target,key in [(v,'v'),(prev,'previous'),(cur,'current')]:target.append(r[key][p_c.cpu().numpy()] if m.get('lab_order') else r[key])
            state=dict(q=tensor(q),v=tensor(v),previous=tensor(prev),current=tensor(cur),root=tensor([r['root'] for r in batch]),goal=tensor([r['goal'] for r in batch]),table=table0.expand(count,-1).clone(),table_z=torch.full((count,),.38,device=dev))
            restore_states(env,ix,state);env.scene.write_data_to_sim();env.sim.forward();env.scene.update(0.)
            snapshot(capture_states(env,ix),owners,'proposal')
            # Settle once; cache the actual settled state, then independently
            # restore and validate it for a second full hold window.
            for _ in range(a.validation_steps):env.step(hold_actions(env))
            settled=capture_states(env,ix);snapshot(settled,owners,'settled')
            restore_states(env,ix,settled);env.scene.write_data_to_sim();env.sim.forward();env.scene.update(0.)
            minz=settled['root'][:,2].clone();maxspeed=torch.zeros(count,device=dev);stable_run=torch.zeros(count,device=dev,dtype=torch.long)
            for step in range(a.validation_steps):
                env.step(hold_actions(env));pos=env.object.data.root_pos_w[ix]-env.scene.env_origins[ix]
                minz=torch.minimum(minz,pos[:,2]);speed=env.object.data.root_lin_vel_w[ix].norm(dim=-1);maxspeed=torch.maximum(maxspeed,speed)
                ob=env._get_observations()['policy'][ix];palm=ob[:,87:90]
                held=(pos[:,2]>.53)&((pos-palm).norm(dim=-1)<.18)&(env._curr_fingertip_distances[ix].sort(-1).values[:,:3].mean(-1)<.1)&(speed<.5)
                stable_run=torch.where(held,stable_run+1,0)
            final=capture_states(env,ix);snapshot(final,owners,'replay_end')
            finite=torch.isfinite(torch.cat([final['q'],final['v'],final['root']],-1)).all(-1)
            valid=finite&(final['q']>=lo-1e-4).all(-1)&(final['q']<=hi+1e-4).all(-1)&(final['v'].abs().max(-1).values<8)&(maxspeed<2.)&(minz>.4)
            valid&=(settled['root'][:,:2].abs()<.45).all(-1)
            valid&=env._keypoints_max_dist[ix]>.12
            valid&=torch.where(kinds>=2,(stable_run>=25)&(minz>=settled['root'][:,2]-.03),torch.ones(count,device=dev,dtype=torch.bool))
            valid&=torch.where(kinds==3,env._keypoints_max_dist[ix]<.35,torch.ones(count,device=dev,dtype=torch.bool))
            for i,owner in enumerate(owners):
                quality.append(dict(candidate=owner,physical_valid=bool(valid[i]),stable_hold_steps=int(stable_run[i]),max_speed=float(maxspeed[i])))
                saved.append({k:v[i].cpu().numpy() for k,v in settled.items()})
            print('RESET_BANK_BATCH',offset,count,'physical_pass',int(valid.sum()),flush=True)
        np.savez_compressed(a.output/'geometry_states.npz',**{k:np.concatenate([s[k] for s in snaps]) for k in snaps[0]},body_names=np.array(env.robot.body_names))
        command=['/home/szc/.conda/envs/spider-world-model/bin/python',str(ROOT/'audit_reset_geometry.py'),'--states',str(a.output/'geometry_states.npz'),'--output',str(a.output/'geometry')]
        result=subprocess.run(command,stdout=(a.output/'geometry.log').open('w'),stderr=subprocess.STDOUT);assert result.returncode==0,'Geometry audit failed'
        geometry=np.load(a.output/'geometry/geometry_results.npz')['depths'];max_depth=np.zeros((len(raw),4))
        for row,owner in zip(geometry,snap_owner):max_depth[owner]=np.maximum(max_depth[owner],row)
        accepted=[]
        for i,q in enumerate(quality):
            q.update(max_depths=max_depth[i].tolist(),**metadata[i]);q['accepted']=bool(q['physical_valid'] and max_depth[i,0]<.001 and max_depth[i,1]<.001 and max_depth[i,2]<.003 and max_depth[i,3]<.003)
            if q['accepted']:accepted.append(i)
        counts={str(k):sum(metadata[i]['kind']==k for i in accepted) for k in (1,2,3)}
        report=dict(proposals=len(raw),accepted=len(accepted),counts=counts,batch_size=a.batch_size,validation_steps=a.validation_steps,independent_replay=True,
            proposal_velocities='Cached joint velocity; object linear finite difference; object angular initialized zero. Stored bank velocities are actual settled Isaac velocities.',
            geometry_backend='MuJoCo collision-only native URDF geometry with Isaac FK agreement; PhysX dynamics used for stabilization',
            geometry_thresholds_m=dict(robot_table=.001,robot_self=.001,robot_object=.003,object_table=.003),
            source_sha256=hashlib.sha256(a.dataset.read_bytes()).hexdigest(),elapsed_s=time.monotonic()-started,records=quality)
        (a.output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
        assert min(counts.values())>0,f'Empty validated reset type: {counts}'
        np.savez_compressed(a.output/'reset_bank.npz',format_version=np.array(1),kind=np.array([metadata[i]['kind'] for i in accepted]),
            environment_sha256=np.array(environment_signature()),
            **{k:np.stack([saved[i][k] for i in accepted]) for k in saved[0]})
        (a.output/'completed.json').write_text(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2)+'\n')
        print('RESET_BANK_COMPLETE',counts,flush=True)
    except BaseException as e:
        (a.output/'failed.json').write_text(json.dumps({'error':repr(e)})+'\n');raise
    finally:
        if env is not None:env.close()
        app.close()


if __name__=='__main__':main()
