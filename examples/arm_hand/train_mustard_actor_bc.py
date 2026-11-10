"""Train the vendored SAPG actor architecture to imitate absolute joint targets.

60 Hz offline BC on frozen source-simulator states; no critic or SAPG update.
The output contract is normalized absolute targets, not native SAPG actions.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import sys
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import torch
import yaml

from examples.arm_hand.test_mustard_simtoolreal_bridge import rotation, save, angle


FIELDS = [('joint_pos',29),('joint_vel',29),('prev_action_targets',29),
          ('palm_pos',3),('palm_rot',4),('object_rot',4),
          ('fingertip_pos_rel_palm',15),('keypoints_rel_palm',12),
          ('keypoints_rel_goal',12),('object_scales',3)]
OBS_DIM=sum(size for _,size in FIELDS)
TIPS=['sharpa_index_tip','sharpa_middle_tip','sharpa_ring_tip','sharpa_thumb_tip','right_pinky_fingertip']


def quat_interp(times, quats, sample_times):
    q=quats[:,[1,2,3,0]]
    return Slerp(times,Rotation.from_quat(q))(sample_times).as_quat()[:,[3,0,1,2]]


def make_obs(model, data, prev, goal, calibration):
    joint_ids=model.actuator_trnid[:,0]
    limits=model.jnt_range[joint_ids]
    q=data.qpos[model.jnt_qposadr[joint_ids]]
    qv=data.qvel[model.jnt_dofadr[joint_ids]]
    palm=data.site('sharpa_palm').xpos.copy()
    palm_q=Rotation.from_matrix(data.site('sharpa_palm').xmat.reshape(3,3)).as_quat()
    obj=data.qpos[-7:];r=rotation(obj[3:])
    center=np.array(calibration['source_to_new_root_center_in_source_m'])
    kp=np.array(calibration['keypoints_local_m'])
    obj_p=obj[:3]+r@center
    goal_p=goal[:3]+rotation(goal[3:])@center
    obj_kp=kp@r.T+obj_p
    goal_kp=kp@rotation(goal[3:]).T+goal_p
    tips=np.stack([data.site(name).xpos for name in TIPS])
    fields=[2*(q-limits[:,0])/(limits[:,1]-limits[:,0])-1,qv,prev,palm,palm_q,
            obj[[4,5,6,3]],(tips-palm).ravel(),(obj_kp-palm).ravel(),
            (obj_kp-goal_kp).ravel(),np.array(calibration['policy_object_scale'])]
    obs=np.concatenate(fields)
    assert obs.shape==(OBS_DIM,)
    return np.clip(obs,-10,10).astype(np.float32)


def prepare_dataset(frozen, out):
    manifest=json.loads((frozen/'manifest.json').read_text())
    calibration=json.loads((frozen/'object_asset/calibration.json').read_text())
    obs_all=[];target_all=[];previous_all=[];model=None
    for item in manifest:
        folder=frozen/f"candidate_{item['candidate_id']:05d}"
        model=mujoco.MjModel.from_xml_path(str(folder/'scene.xml'));data=mujoco.MjData(model)
        d=np.load(folder/'gpu/trajectory.npz');ref=np.load(folder/'reference.npz')
        mujoco.mj_setState(model,data,d['initial_state'],mujoco.mjtState.mjSTATE_FULLPHYSICS)
        q0=data.qpos.copy();v0=data.qvel.copy()
        duration=len(d['ctrl'])*float(d['control_dt'])
        sample_t=np.arange(int(np.ceil(duration*60)))/60
        state_t=np.arange(len(d['qpos'])+1)*float(d['control_dt'])
        qsource=np.vstack([q0,d['qpos']]);vsource=np.vstack([v0,d['qvel']])
        qs=np.stack([np.interp(sample_t,state_t,qsource[:,j]) for j in range(model.nq)],axis=1)
        qs[:,-4:]=quat_interp(state_t,qsource[:,-4:],sample_t)
        vs=np.stack([np.interp(sample_t,state_t,vsource[:,j]) for j in range(model.nv)],axis=1)
        cmd_t=np.arange(len(d['ctrl']))*float(d['control_dt'])
        targets=np.stack([np.interp(sample_t,cmd_t,d['ctrl'][:,j]) for j in range(model.nu)],axis=1)
        joint_ids=model.actuator_trnid[:,0]
        prev=np.vstack([q0[model.jnt_qposadr[joint_ids]],targets[:-1]])
        obs=[]
        for q,v,p in zip(qs,vs,prev):
            data.qpos[:]=q;data.qvel[:]=v;mujoco.mj_forward(model,data)
            obs.append(make_obs(model,data,p,ref['object_qpos'][-1],calibration))
        obs_all.append(obs);target_all.append(targets);previous_all.append(prev)
    limits=model.jnt_range[model.actuator_trnid[:,0]].copy()
    result=dict(obs=np.array(obs_all,dtype=np.float32),targets_rad=np.array(target_all,dtype=np.float32),
                previous_targets_rad=np.array(previous_all,dtype=np.float32),limits=limits.astype(np.float32),
                train=np.array([r['split']=='train' for r in manifest]),candidate_ids=np.array([r['candidate_id'] for r in manifest]),time_s=sample_t)
    result['targets']=2*(result['targets_rad']-limits[:,0])/(limits[:,1]-limits[:,0])-1
    result['targets']=result['targets'].astype(np.float32)
    assert all(np.isfinite(v).all() for v in result.values())
    np.savez_compressed(out/'dataset_60hz.npz',**result)
    schema=dict(fields=FIELDS,obs_dim=OBS_DIM,policy_hz=60,joint_names=[model.joint(int(j)).name for j in model.actuator_trnid[:,0]],
                quaternion_order='xyzw in observations; wxyz in MuJoCo states',
                palm='source sharpa_palm site',fingertips=TIPS,object_calibration=calibration,
                target_contract='normalized absolute joint targets: q=lower+0.5*(mu+1)*(upper-lower)',
                native_sapg_actions=False,interpolation='linear joint/state/control; SLERP object quaternion; pre-step observations',
                split='16 train / 4 held-out spatial variants of ONE original episode')
    save(out/'schema.json',schema)
    return result,schema


def build_model(simroot, batch, device):
    sys.path.insert(0,str(simroot/'rl_games'))
    from rl_games.algos_torch import model_builder
    cfg=yaml.safe_load((simroot/'isaacsimenvs/cfg/train/SimToolRealSAPG.yaml').read_text())['params']
    kwargs=dict(actions_num=29,input_shape=(OBS_DIM+1,),num_seqs=batch,value_size=1,
                normalize_input=True,normalize_value=True,type='extra_param',
                coef_ids=torch.linspace(50,0,6,device=device),coef_id_idx=OBS_DIM)
    model=model_builder.ModelBuilder().load(cfg).build(kwargs).to(device)
    model.a2c_network.sigma.requires_grad_(False)
    model.a2c_network.sigma.data.fill_(-2)
    for p in model.a2c_network.value.parameters():p.requires_grad_(False)
    return model,cfg


def forward(model,x,h,ids):
    b,t,_=x.shape
    cond=ids[:,None,None].expand(b,t,1)
    obs=torch.cat([x,cond],-1).reshape(b*t,-1)
    obs=model.norm_obs(obs)
    mu,_,_,states=model.a2c_network(dict(obs=obs,rnn_states=h,seq_length=t))
    mu=mu.reshape(b,t,29)
    if getattr(model,'bc_residual',False):
        limits=model.bc_limits
        previous_norm=2*(x[:,:,58:87]-limits[:,0])/(limits[:,1]-limits[:,0])-1
        mu=previous_norm+.1*mu
    return mu,states


def zero_state(model,b):
    return tuple(s[:,:1,:].expand(-1,b,-1).contiguous().zero_() for s in model.get_default_rnn_state())


def balanced_loss(pred,y):
    return .5*((pred[:,:,:7]-y[:,:,:7]).square().mean()+(pred[:,:,7:]-y[:,:,7:]).square().mean())


def metrics(pred,target,limits):
    error=(pred-target)*.5*(limits[:,1]-limits[:,0])
    return dict(rmse_rad=float(np.sqrt(np.mean(error**2))),arm_rmse_rad=float(np.sqrt(np.mean(error[:,:,:7]**2))),
                hand_rmse_rad=float(np.sqrt(np.mean(error[:,:,7:]**2))),mae_rad=float(np.abs(error).mean()),
                normalized_balanced_mse=float(.5*(np.mean((pred[:,:,:7]-target[:,:,:7])**2)+np.mean((pred[:,:,7:]-target[:,:,7:])**2))))


def audit_parameters(frozen,simroot,out):
    sample=sorted(frozen.glob('candidate_*/scene.xml'))[0];m=mujoco.MjModel.from_xml_path(str(sample))
    tree=ast.parse((simroot/'isaacsimenvs/tasks/simtoolreal/robot_spec.py').read_text())
    spec=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='KUKA_SHARPA_SPEC' for t in n.targets))
    kwargs={k.arg:k.value for k in spec.keywords}
    target={k:ast.literal_eval(kwargs[k]) for k in ['arm_joint_stiffness','arm_joint_damping','hand_joint_stiffness','hand_joint_damping','robot_base_pos','robot_base_rot_wxyz']}
    cfg=yaml.safe_load((simroot/'isaacsimenvs/cfg/task/SimToolReal.yaml').read_text())
    result=dict(source_scene=str(sample),source_control_hz=100,target_control_hz=60,
                source_physics_dt=float(m.opt.timestep),target_physics_dt=cfg['sim']['dt'],
                source_gravity=m.opt.gravity.tolist(),target_gravity=cfg['sim']['gravity'],
                source_stiffness=m.actuator_gainprm[:,0].tolist(),source_damping=(-m.actuator_biasprm[:,2]).tolist(),target_robot_spec=target,
                target_material_frictions={k:v for k,v in cfg['assets'].items() if 'friction' in k},
                source_explicit_pair_frictions=np.unique(m.pair_friction,axis=0).tolist(),
                source_default_geom_frictions=np.unique(m.geom_friction,axis=0).tolist(),
                source_robot='native right Sharpa Wave',target_default_robot='left Sharpa; right spec still required for transfer',
                note='Matching numeric PD and gravity does not guarantee solver/contact or full asset equivalence; this audit is not a target-engine rollout.')
    save(out/'physics_parameter_audit.json',result)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--frozen',type=Path,default=Path('outputs/mustard_sapg_bridge20_v1'))
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--simtoolreal',type=Path,default=Path('/home/szc/work/simtoolreal_origin/simtoolreal'))
    ap.add_argument('--epochs',type=int,default=150)
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--residual',action='store_true',help='Predict a residual around previous joint targets; keep backbone unchanged.')
    args=ap.parse_args();assert not args.output.exists(),'Use a new output directory.'
    args.output.mkdir(parents=True);torch.set_num_threads(args.threads);torch.manual_seed(20261027);np.random.seed(20261027)
    started=time.monotonic();dataset,schema=prepare_dataset(args.frozen,args.output)
    audit_parameters(args.frozen,args.simtoolreal,args.output)
    train=np.flatnonzero(dataset['train']);val=np.flatnonzero(~dataset['train'])
    device=torch.device(args.device)
    model,cfg=build_model(args.simtoolreal,len(train),device)
    model.bc_residual=args.residual
    model.bc_limits=torch.tensor(dataset['limits'],device=device)
    if args.residual:
        torch.nn.init.zeros_(model.a2c_network.mu.weight)
        torch.nn.init.zeros_(model.a2c_network.mu.bias)
    schema['residual_decoder']=args.residual
    schema['joint_limits_rad']=dataset['limits'].tolist()
    if args.residual:
        schema['target_contract']='normalized absolute target = normalized previous target + 0.1 * actor_mu; decode using joint limits'
    save(args.output/'schema.json',schema)
    x=torch.tensor(dataset['obs'],device=device);y=torch.tensor(dataset['targets'],device=device)
    stats=model.running_mean_std
    train_flat=x[train].reshape(-1,OBS_DIM)
    stats.running_mean.copy_(train_flat.mean(0).double());stats.running_var.copy_(train_flat.var(0).clamp_min(1e-6).double());stats.count.fill_(len(train_flat))
    model.eval()  # freeze normalizer; gradients remain enabled for actor
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    coef_ids=torch.linspace(50,0,6,device=device)

    @torch.no_grad()
    def evaluate(indices):
        scores=[];preds=[]
        for cid in coef_ids:
            pred,_=forward(model,x[indices],zero_state(model,len(indices)),cid.repeat(len(indices)))
            scores.append(metrics(pred.cpu().numpy(),y[indices].cpu().numpy(),dataset['limits']))
            preds.append(pred.cpu().numpy())
        return {k:float(np.mean([m[k] for m in scores])) for k in scores[0]},np.array(preds)

    initial,_=evaluate(val)
    best=float('inf');best_epoch=0;history=[]
    # Temporal chunks preserve episode hidden state; each epoch begins at true episode start.
    for epoch in range(1,args.epochs+1):
        h=zero_state(model,len(train));ids=coef_ids[(torch.arange(len(train),device=device)+epoch)%6]
        total=0.;n=0
        for start in range(0,x.shape[1],16):
            pred,h=forward(model,x[train,start:start+16],h,ids)
            loss=balanced_loss(pred,y[train,start:start+16])
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            h=tuple(s.detach() for s in h)
            total+=float(loss.detach())*pred.shape[1];n+=pred.shape[1]
        if epoch==1 or epoch%5==0:
            validation,_=evaluate(val);row=dict(epoch=epoch,train_balanced_mse=total/n,validation=validation,elapsed_s=time.monotonic()-started)
            history.append(row);save(args.output/'history.json',history);print(json.dumps(row),flush=True)
            if validation['normalized_balanced_mse']<best:
                best=validation['normalized_balanced_mse'];best_epoch=epoch
                torch.save(dict(model=model.state_dict(),epoch=epoch,network_config=cfg['network'],schema=schema,
                                coef_ids=coef_ids.cpu(),seed=20261027,output_contract=schema['target_contract'],
                                native_sapg_actions=False,optimizer=optimizer.state_dict()),args.output/'actor_best.pt')
            if epoch-best_epoch>=40:break
    checkpoint=torch.load(args.output/'actor_best.pt',map_location=device,weights_only=False);model.load_state_dict(checkpoint['model']);model.eval()
    train_result,train_preds=evaluate(train);val_result,val_preds=evaluate(val)
    lim=dataset['limits'];prevnorm=2*(dataset['previous_targets_rad']-lim[:,0])/(lim[:,1]-lim[:,0])-1
    baseline=metrics(prevnorm[val],dataset['targets'][val],lim)
    np.savez_compressed(args.output/'offline_predictions.npz',train_predictions=train_preds,validation_predictions=val_preds,train_indices=train,validation_indices=val)
    report=dict(stage='offline BC actor imitation',network='exact local SAPG LSTM1024 + MLP[1024,1024,512,512]; learned coefficient embeddings',
                obs_shape=list(dataset['obs'].shape),training_episodes=len(train),validation_episodes=len(val),
                validation_initial=initial,validation_best=val_result,training_best=train_result,
                persistence_baseline_validation=baseline,best_epoch=checkpoint['epoch'],epochs_executed=epoch,
                wall_time_s=time.monotonic()-started,device=str(device),native_sapg_actions=False,
                output_contract=schema['target_contract'],
                limitations=['Held-out spatial variants share a source episode with train.','Offline teacher-forced observations; closed-loop imitation still needs evaluation.','Target output differs from native SAPG incremental arm action semantics.','Right-hand observation sites require matching right-hand target environment configuration.'])
    save(args.output/'training_report.json',report)
    print('TRAINING_REPORT',json.dumps(report),flush=True)


if __name__=='__main__':main()
