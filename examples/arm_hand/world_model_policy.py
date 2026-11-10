"""Shared reference-conditioned feedback policy, bootstrapped from strictly valid physics."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch
from torch import nn

from examples.arm_hand.world_model import canonical, scene_spec, usable_states


class ReferencePolicy(nn.Module):
    def __init__(self,spec,width=256):
        super().__init__();self.spec=spec;self.width=width
        d=spec['nq']+spec['nv']+spec['nu']+2*spec['nq'];c=spec['context_dim']
        self.register_buffer('mean',torch.zeros(d+c));self.register_buffer('scale',torch.ones(d+c))
        self.net=nn.Sequential(nn.Linear(d+c,width),nn.ELU(),nn.Linear(width,width),nn.ELU(),nn.Linear(width,spec['nu']))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,state,nominal,next_goal,future_goal,context,low,high):
        features=torch.cat((state,nominal,next_goal,future_goal,context),-1)
        residual=.4*torch.tanh(self.net((features-self.mean)/self.scale))
        return (nominal+residual).clamp(low,high)


    def prepare_context(self,context):
        if any(parameter.requires_grad for parameter in self.parameters()):raise ValueError('Cache only a frozen policy')
        dynamic=self.spec['nq']+self.spec['nv']+self.spec['nu']+2*self.spec['nq']
        normalized=(context-self.mean[dynamic:])/self.scale[dynamic:]
        bias=nn.functional.linear(normalized,self.net[0].weight[:,dynamic:],self.net[0].bias)
        return self.net[0].weight[:,:dynamic].detach().contiguous(),bias.detach()

    def with_context(self,state,nominal,next_goal,future_goal,cache,low,high):
        features=torch.cat((state,nominal,next_goal,future_goal),-1);d=features.shape[-1]
        hidden=nn.functional.elu(nn.functional.linear((features-self.mean[:d])/self.scale[:d],cache[0],cache[1]))
        hidden=nn.functional.elu(self.net[2](hidden))
        return (nominal+.4*torch.tanh(self.net[4](hidden))).clamp(low,high)


def load_policy(path,device):
    ck=torch.load(path,map_location=device,weights_only=False)
    net=ReferencePolicy(ck['spec'],ck['width']).to(device);net.load_state_dict(ck['model']);net.eval()
    return net,ck


def collect(args):
    root=args.campaign;groups=json.loads((root/'selected_sources.json').read_text());records=[];spec=None
    args.output.mkdir(parents=True,exist_ok=True)
    for group in groups:
        result=Path(group['result_dir']);manifest=result/'accepted.json'
        if not manifest.exists():continue
        accepted=([dict(folder=str(p.parent.parent)) for p in sorted(result.glob('candidate_*/gpu/trajectory.npz'))] if args.include_failed else json.loads(manifest.read_text()))[:args.max_per_group]
        for item in accepted:
            case=Path(item['folder']);key=hashlib.sha256(str(case.resolve()).encode()).hexdigest()[:20]
            out=args.output/f'policy_{key}.npz'
            if out.exists():continue
            if not args.include_failed:
                from examples.arm_hand.mustard_valid100 import valid
                audit=json.loads((case/'state.json').read_text())
                if not (valid(audit['selected_metrics']) and valid(audit['replay_metrics'])):raise ValueError('Teacher no longer passes both strict audits')
                for required in ['gpu/trajectory.npz','gpu/replay_trajectory.npz','best_gpu.mp4']:
                    if not (case/required).is_file():raise FileNotFoundError(case/required)
            model,case_spec,context=scene_spec(case/'scene.xml')
            if spec is None:spec=case_spec
            if case_spec!=spec:raise ValueError('Policy data topology differs')
            with np.load(case/'reference.npz') as f:ref={k:f[k] for k in f.files}
            with np.load(case/'gpu/trajectory.npz') as f:tr={k:f[k] for k in f.files}
            if not np.allclose(tr['initial_state'],ref['initial_state']):raise ValueError('Teacher initial state differs')
            if not np.isclose(float(tr['control_dt']),.01) or not np.isclose(float(tr['physics_dt']),.002):raise ValueError('Teacher timebase differs')
            data=mujoco.MjData(model);mujoco.mj_setState(model,data,ref['initial_state'],mujoco.mjtState.mjSTATE_FULLPHYSICS)
            q=canonical(np.concatenate((data.qpos[None],tr['qpos'])));v=np.concatenate((data.qvel[None],tr['qvel']))
            state=np.concatenate((q,v),-1).astype(np.float32);T=len(tr['ctrl'])
            goal=canonical(np.concatenate((ref['joint_ik'],ref['object_qpos']),-1)).astype(np.float32)
            if not args.include_failed and not usable_states(state,case_spec).all():raise ValueError('Accepted teacher contains diverged states')
            np.savez_compressed(out,state=state,goal=goal,nominal=ref['ctrl_nominal'].astype(np.float32),
                action=tr['ctrl'].astype(np.float32),context=context,low=model.actuator_ctrlrange[:,0].astype(np.float32),
                high=model.actuator_ctrlrange[:,1].astype(np.float32),
                metadata=json.dumps(dict(spec=case_spec,source_group=group['task'],case=str(case.resolve()),validated_teacher=not args.include_failed)))
            records.append(dict(path=str(out),source_group=group['task'],frames=T))
    (args.output/f'collect_{time.time_ns()}.json').write_text(json.dumps(records,indent=2));print(json.dumps(dict(new_trajectories=len(records))))


def source_frame_weights(groups, lengths, device):
    """Equal probability per original source; uniform frames within that source."""
    totals={group:sum(n for g,n in zip(groups,lengths) if g==group) for group in set(groups)}
    if not totals or any(n<=0 for n in totals.values()):raise ValueError('Empty source frames')
    return torch.cat([torch.full((n,),1./totals[group],device=device) for group,n in zip(groups,lengths)])


def train(args):
    torch.manual_seed(args.seed);rows=[];spec=None
    for p in sorted(args.data.glob('policy_*.npz')):
        with np.load(p) as f:
            meta=json.loads(str(f['metadata']));current=meta['spec']
            if spec is None:spec=current
            if spec!=current:raise ValueError('Mixed policy topology')
            if not meta['validated_teacher']:continue
            if meta['source_group'] in set(args.test_group):continue
            st=f['state'];nom=f['nominal'];goal=f['goal'];T=len(nom)
            future=np.minimum(np.arange(T)+args.lookahead,T)
            features=np.concatenate((st[:-1],nom,goal[1:],goal[future],np.broadcast_to(f['context'],(T,len(f['context'])))),-1)
            rows.append((features,f['action'].copy(),nom.copy(),meta['source_group'],f['low'].copy(),f['high'].copy()))
    if not rows:raise ValueError('No valid policy teachers')
    held=set(args.holdout_group)
    if held & set(args.test_group):raise ValueError('Validation/test overlap')
    # Whole original source split; if only one source is available, do not claim held-out performance.
    train_rows=[r for r in rows if r[3] not in held];val_rows=[r for r in rows if r[3] in held]
    if not train_rows or not val_rows:raise ValueError('Need disjoint teacher training and validation sources')
    tensor=lambda x:torch.as_tensor(np.concatenate(x).copy(),dtype=torch.float32,device=args.device)
    x,y,nom=[tensor([r[i] for r in train_rows]) for i in range(3)]
    vx,vy,vnom=[tensor([r[i] for r in val_rows]) for i in range(3)]
    net=ReferencePolicy(spec,args.width).to(args.device);net.mean.copy_(x.mean(0));net.scale.copy_(x.std(0,unbiased=False).clamp_min(.01))
    low=torch.tensor(rows[0][4],device=args.device);high=torch.tensor(rows[0][5],device=args.device)
    opt=torch.optim.Adam(net.parameters(),lr=args.lr);best=float('inf');history=[];started=time.monotonic();args.output.mkdir(parents=True,exist_ok=True)
    weights=source_frame_weights([r[3] for r in train_rows],[len(r[0]) for r in train_rows],args.device) if args.source_balanced else None
    source_counts={g:sum(len(r[0]) for r in train_rows if r[3]==g) for g in sorted({r[3] for r in train_rows})}
    for update in range(args.updates):
        ids=(torch.multinomial(weights,min(args.batch,len(x)),replacement=True) if weights is not None else
             torch.randint(len(x),(min(args.batch,len(x)),),device=args.device))
        pred=(nom[ids]+.4*torch.tanh(net.net((x[ids]-net.mean)/net.scale))).clamp(low,high)
        loss=(pred-y[ids]).square().mean();opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
        if update%args.eval_every==0 or update==args.updates-1:
            with torch.no_grad():val=((vnom+.4*torch.tanh(net.net((vx-net.mean)/net.scale))).clamp(low,high)-vy).square().mean()
            history.append(dict(update=update+1,train_mse=float(loss.detach()),holdout_mse=float(val)))
            if float(val)<best:
                best=float(val);tmp=args.output/'policy.tmp'
                torch.save(dict(model=net.state_dict(),spec=spec,width=args.width,lookahead=args.lookahead,
                    train_groups=sorted({r[3] for r in train_rows}),holdout_groups=sorted(held),test_groups=args.test_group,
                    stage='validated-physics BC bootstrap, no imagination update yet',source_balanced=args.source_balanced,source_frame_counts=source_counts,update=update+1),tmp);tmp.replace(args.output/'policy.pt')
            print(json.dumps(history[-1]),flush=True)
    (args.output/'training.json').write_text(json.dumps(dict(history=history,elapsed_s=time.monotonic()-started,train_frames=len(x),holdout_frames=len(vx),source_balanced=args.source_balanced,source_frame_counts=source_counts),indent=2))


def improve(args):
    from examples.arm_hand.world_model import load_model, corrected_transition
    from examples.arm_hand.world_model_kinematics import Kinematics
    if args.skip_critic_training and args.critic_weight:raise ValueError('Skipping critic requires zero terminal critic weight')
    torch.manual_seed(args.seed);actor,policy_ck=load_policy(args.policy,args.device);world,world_ck=load_model(args.world,args.device)
    if actor.spec!=world.spec:raise ValueError('Actor/world topology differs')
    for parameter in world.parameters():parameter.requires_grad_(False)
    original,_=load_policy(args.policy,args.device)
    for parameter in original.parameters():parameter.requires_grad_(False)
    held=set(policy_ck['holdout_groups']);test=set(policy_ck['test_groups']);rows=[];validation=[];scene=None;trained_groups=set();validated_groups=set()
    contact_signatures=set();checked_scenes={};contact_models=[];contact_indices={}
    for path in sorted(args.data.glob('policy_*.npz')):
        with np.load(path) as f:
            meta=json.loads(str(f['metadata']));group=meta['source_group']
            if group in test:continue
            if args.collision_weight and meta['case'] not in checked_scenes:
                from examples.arm_hand.world_model_collision import geometry_signature
                contact_model=mujoco.MjModel.from_xml_path(str(Path(meta['case'])/'scene.xml'))
                signature=geometry_signature(contact_model);contact_signatures.add(signature)
                if signature not in contact_indices:
                    contact_indices[signature]=len(contact_models);contact_models.append(contact_model)
                checked_scenes[meta['case']]=contact_indices[signature]
            states=f['state'];T=len(f['action']);goal=f['goal'];nominal=f['nominal']
            from scipy.spatial.transform import Rotation
            with np.load(Path(meta['case'])/'reference.npz') as reference:
                raw_wrist=reference['wrist_pos'].copy();raw_rotation=Rotation.from_quat(reference['wrist_quat'][:,[1,2,3,0]]).as_matrix()
            if scene is None and group not in held:scene=Path(meta['case'])/'scene.xml'
            for start in range(0,T-args.horizon+1,args.horizon):
                segment=states[start:start+args.horizon+1]
                if not usable_states(segment,actor.spec).all():continue
                nu=actor.spec['nu']
                if np.linalg.norm(segment[0,nu:nu+3]-goal[start,nu:nu+3])>args.max_reset_object_error:continue
                nq=actor.spec['nq'];dot=abs(float(np.dot(segment[0,nq-4:nq],goal[start,nq-4:nq])))
                if 2*np.arccos(np.clip(dot,0,1))>np.radians(args.max_reset_rotation_deg):continue
                if np.sqrt(np.mean((segment[0,:nu]-goal[start,:nu])**2))>args.max_reset_joint_rmse:continue
                ids=np.arange(start,start+args.horizon+1)
                row=(segment,nominal[ids.clip(max=T-1)],goal[ids],goal[(ids+policy_ck['lookahead']).clip(max=T)],f['context'].copy(),raw_wrist[ids],raw_rotation[ids],f['action'][ids.clip(max=T-1)].copy(),np.asarray(checked_scenes.get(meta['case'],0),dtype=np.int64))
                (validation if group in held else rows).append(row)
                (validated_groups if group in held else trained_groups).add(group)
    if not rows or not validation:raise ValueError('Need training and whole-source validation windows')
    def tensors(items):return tuple(torch.as_tensor(np.stack([r[i] for r in items]),dtype=torch.float32,device=args.device) for i in range(9))
    data=tensors(rows);val=tensors(validation)
    def calibrate(batch):
        states,nom,goal,future,c,wrist,rotation,actions,geometry_ids=batch;bias=[]
        with torch.no_grad():
            for start in range(0,len(states),128):
                st=states[start:start+128];ac=actions[start:start+128];ct=c[start:start+128]
                if args.anchored_world:
                    pred=world(st[:,:-1].reshape(-1,st.shape[-1]),ac[:,:-1].reshape(-1,ac.shape[-1]),
                        ct[:,None].expand(-1,args.horizon,-1).reshape(-1,ct.shape[-1])).reshape(len(st),args.horizon,-1)
                    bias.append(st[:,1:]-pred)
                else:bias.append(torch.zeros_like(st[:,1:]))
        return batch+(torch.cat(bias),)
    data=calibrate(data);val=calibrate(val)
    model=mujoco.MjModel.from_xml_path(str(scene));nq=model.nq;nu=model.nu
    fk=Kinematics(model,['sharpa_palm'],args.device)
    from examples.arm_hand.world_model_collision import ContactPenalty
    contacts=[ContactPenalty(m,nu) for m in contact_models] if args.collision_weight else []
    low=torch.tensor(model.actuator_ctrlrange[:,0].copy(),dtype=torch.float32,device=args.device)
    high=torch.tensor(model.actuator_ctrlrange[:,1].copy(),dtype=torch.float32,device=args.device)
    d=len(actor.mean);critic=nn.Sequential(nn.Linear(d,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,1)).to(args.device)
    nn.init.zeros_(critic[-1].weight);nn.init.zeros_(critic[-1].bias)
    opt=torch.optim.Adam(actor.parameters(),lr=args.lr);vopt=torch.optim.Adam(critic.parameters(),lr=.0003)
    def value(st,nom,goal,future,c):
        if args.skip_critic_training:return st.new_zeros(st.shape[:-1])
        x=(torch.cat((st,nom,goal,future,c),-1)-actor.mean)/actor.scale
        return critic(x).squeeze(-1)
    def imagine(batch,anchor=True):
        states,nom,goal,future,c,raw_wrist,raw_rotation,teacher_actions,geometry_ids,bias=batch;st=states[:,0];rewards=[];values=[];trajectory=[];anchor_cost=0.
        for t in range(args.horizon):
            values.append(value(st,nom[:,t],goal[:,t+1],future[:,t],c))
            action=actor(st,nom[:,t],goal[:,t+1],future[:,t],c,low,high)
            if anchor:
                with torch.no_grad():base=original(st.detach(),nom[:,t],goal[:,t+1],future[:,t],c,low,high)
                anchor_cost=anchor_cost+(action-base).square().mean()
            st=corrected_transition(world(st,action,c),bias[:,t],nq);trajectory.append(st)
        pred=torch.stack(trajectory,1);target=goal[:,1:]
        palm,palm_rotation=fk(pred[...,:nu])
        relative=raw_rotation[:,1:].transpose(-1,-2)@palm_rotation[...,0,:,:]
        palm_angle=((relative.diagonal(dim1=-2,dim2=-1).sum(-1)-1)*.5).clamp(-1+1e-6,1-1e-6).acos()
        obj=(pred[...,nu:nu+3]-target[...,nu:nu+3]).square().sum(-1)
        dot=(pred[...,nq-4:nq]*target[...,nq-4:nq]).sum(-1)
        rotation=1-dot.square().clamp(max=1)
        posture=(pred[...,7:nu]-target[...,7:nu]).square().sum(-1)
        cost=3600*obj+225*(palm[...,0,:]-raw_wrist[:,1:]).square().sum(-1)+.64*palm_angle.square()+args.rotation_weight*rotation+.0025*posture
        # A smooth failure cost discourages imaginary policies from abandoning the object.
        cost=cost+10*torch.sigmoid((obj.clamp_min(1e-10).sqrt()-.15)/.02)
        if contacts:
            contact_cost=pred.new_zeros(pred.shape[:2])
            for index,contact in enumerate(contacts):
                selected=torch.nonzero(geometry_ids==index).flatten()
                if len(selected):contact_cost=contact_cost.index_copy(0,selected,contact(pred[selected,...,:nq]))
            cost=cost+args.collision_weight*contact_cost
        rewards=-cost
        terminal=value(st,nom[:,-1],goal[:,-1],future[:,-1],c)
        objective=sum((.99**t)*rewards[:,t] for t in range(args.horizon))+args.critic_weight*(.99**args.horizon)*terminal
        return -objective.mean()+args.anchor*anchor_cost/args.horizon,rewards,torch.stack(values,1),terminal
    started=time.monotonic();history=[];best=float('inf');args.output.mkdir(parents=True,exist_ok=True)
    data_hash=hashlib.sha256()
    for path in sorted(args.data.glob('policy_*.npz')):data_hash.update(hashlib.sha256(path.read_bytes()).digest())
    training_spec=dict(data_sha256=data_hash.hexdigest(),world_sha256=hashlib.sha256(args.world.read_bytes()).hexdigest(),
        policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest(),horizon=args.horizon,batch=args.batch,lr=args.lr,anchor=args.anchor,seed=args.seed,objective_version='raw_reference_palm_pose_v5_original_contact_geometry' if args.collision_weight else 'raw_reference_palm_pose_v4_initial_guard',critic_weight=args.critic_weight,max_reset_object_error=args.max_reset_object_error,max_reset_joint_rmse=args.max_reset_joint_rmse,rotation_weight=args.rotation_weight,max_reset_rotation_deg=args.max_reset_rotation_deg,anchored_world=args.anchored_world)
    if args.collision_weight:training_spec.update(collision_weight=args.collision_weight,collision_geometry_sha256=sorted(contact_signatures))
    if args.skip_critic_training:training_spec['skip_critic_training']=True
    start_update=0;previous_elapsed=0.
    if args.resume:
        if args.resume.resolve().parent!=args.output.resolve():raise ValueError('Resume in original output directory')
        saved=torch.load(args.resume,map_location=args.device,weights_only=False)
        if saved['training_spec']!=training_spec:raise ValueError('Imagination resume data/models/settings differ')
        actor.load_state_dict(saved['actor']);critic.load_state_dict(saved['critic']);opt.load_state_dict(saved['optimizer']);vopt.load_state_dict(saved['value_optimizer'])
        torch.set_rng_state(saved['cpu_rng'].cpu())
        if str(args.device).startswith('cuda'):torch.cuda.set_rng_state(saved['cuda_rng'].cpu(),args.device)
        start_update=saved['update'];best=saved['best'];history=saved['history'];previous_elapsed=saved['elapsed_s']
    if not args.resume:
        # Preserve the BC starting point: an imagination update must beat update zero.
        with torch.no_grad():
            subset=tuple(x[:min(args.batch,len(validation))] for x in val)
            _,initial_rewards,_,_=imagine(subset)
            best=float(-initial_rewards.mean())
        history.append(dict(update=0,validation_objective=best,stage='unchanged BC starting point'))
        torch.save(dict(policy_ck,model=actor.state_dict(),stage='unchanged BC fallback; no imagination improvement selected',
            world=str(args.world.resolve()),train_groups=sorted(trained_groups),holdout_groups=sorted(validated_groups),
            bc_train_groups=policy_ck.get('bc_train_groups',policy_ck['train_groups']),training_spec=training_spec,update=0),args.output/'policy.pt')
        print(json.dumps(history[-1]),flush=True)
    for update in range(start_update,args.updates):
        ids=torch.randint(len(rows),(min(args.batch,len(rows)),),device=args.device);batch=tuple(x[ids] for x in data)
        for param in critic.parameters():param.requires_grad_(False)
        loss,rewards,values,terminal=imagine(batch);opt.zero_grad()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite imagination policy objective')
        loss.backward();nn.utils.clip_grad_norm_(actor.parameters(),.1);opt.step()
        with torch.no_grad():
            target=terminal.detach();targets=[]
            for t in reversed(range(args.horizon)):
                nextvalue=terminal.detach() if t==args.horizon-1 else values[:,t+1].detach()
                target=rewards[:,t].detach()+.99*(.05*nextvalue+.95*target);targets.append(target)
            returns=torch.stack(list(reversed(targets)),1)
        for param in critic.parameters():param.requires_grad_(True)
        with torch.no_grad():current=tuple(x.detach() for x in batch)
        # Fit critic on the same imagined states, keeping actor/world frozen for this value update.
        if args.skip_critic_training:
            vl=loss.new_zeros(())
        else:
            vopt.zero_grad();_,_,v,_=imagine(current,anchor=False);vl=(v-returns).square().mean()
            vl.backward();nn.utils.clip_grad_norm_(critic.parameters(),1.);vopt.step()
        if update%args.eval_every==0 or update==args.updates-1:
            with torch.no_grad():
                subset=tuple(x[:min(args.batch,len(validation))] for x in val);_,validation_rewards,_,_=imagine(subset);score=-validation_rewards.mean()
            history.append(dict(update=update+1,actor_objective=float(loss.detach()),critic_loss=float(vl.detach()),validation_objective=float(score)))
            if float(score)<best:
                best=float(score);tmp=args.output/'policy.tmp'
                torch.save(dict(policy_ck,model=actor.state_dict(),stage='world-model BPTT actor-critic' if args.critic_weight else 'world-model BPTT actor, critic ablation',world=str(args.world.resolve()),
                    train_groups=sorted(trained_groups),holdout_groups=sorted(validated_groups),bc_train_groups=policy_ck.get('bc_train_groups',policy_ck['train_groups']),training_spec=training_spec,
                    update=update+1),tmp);tmp.replace(args.output/'policy.pt')
            elapsed=previous_elapsed+time.monotonic()-started
            last=dict(actor=actor.state_dict(),critic=critic.state_dict(),optimizer=opt.state_dict(),value_optimizer=vopt.state_dict(),
                training_spec=training_spec,update=update+1,best=best,history=history,elapsed_s=elapsed,cpu_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(args.device) if str(args.device).startswith('cuda') else None)
            tmp=args.output/'last.tmp';torch.save(last,tmp);tmp.replace(args.output/'last.pt')
            (args.output/'training.json').write_text(json.dumps(dict(history=history,elapsed_s=elapsed,train_windows=len(rows),validation_windows=len(validation),training_spec=training_spec),indent=2))
            print(json.dumps(history[-1]),flush=True)
    torch.save(critic.state_dict(),args.output/'critic.pt')
    (args.output/'training.json').write_text(json.dumps(dict(history=history,elapsed_s=previous_elapsed+time.monotonic()-started,train_windows=len(rows),validation_windows=len(validation),training_spec=training_spec,
        limitations='Model-based objective selection is not physical validation; accepted-data throughput must be measured'),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('collect');c.add_argument('--campaign',type=Path,default=Path('outputs/mustard_multisource_valid100'))
    c.add_argument('--include-failed',action='store_true')
    c.add_argument('--output',type=Path,required=True);c.add_argument('--max-per-group',type=int,default=100);c.set_defaults(func=collect)
    t=sub.add_parser('train');t.add_argument('--data',type=Path,required=True);t.add_argument('--output',type=Path,required=True)
    t.add_argument('--holdout-group',action='append',required=True);t.add_argument('--test-group',action='append',default=[])
    t.add_argument('--lookahead',type=int,default=8);t.add_argument('--updates',type=int,default=4000)
    t.add_argument('--width',type=int,default=256);t.add_argument('--batch',type=int,default=256);t.add_argument('--seed',type=int,default=42)
    t.add_argument('--lr',type=float,default=.0003);t.add_argument('--eval-every',type=int,default=250);t.add_argument('--device',default='cuda:0')
    t.add_argument('--source-balanced',action='store_true')
    t.set_defaults(func=train)
    m=sub.add_parser('improve')
    for key in ['policy','world','data','output']:m.add_argument('--'+key,type=Path,required=True)
    m.add_argument('--anchored-world',action='store_true')
    m.add_argument('--collision-weight',type=float,default=0.,help='Original robot/table and hand-self penetration cost; CPU first-order contact geometry gradients')
    m.add_argument('--rotation-weight',type=float,default=.36)
    m.add_argument('--max-reset-rotation-deg',type=float,default=45.)
    m.add_argument('--max-reset-object-error',type=float,default=.05)
    m.add_argument('--max-reset-joint-rmse',type=float,default=.3)
    m.add_argument('--critic-weight',type=float,default=1.)
    m.add_argument('--skip-critic-training',action='store_true',help='Omit unused value-network pass when critic weight is zero')
    m.add_argument('--resume',type=Path)
    m.add_argument('--updates',type=int,default=1500);m.add_argument('--horizon',type=int,default=8)
    m.add_argument('--batch',type=int,default=128);m.add_argument('--lr',type=float,default=.00003)
    m.add_argument('--anchor',type=float,default=1.);m.add_argument('--eval-every',type=int,default=100)
    m.add_argument('--device',default='cuda:0');m.add_argument('--seed',type=int,default=42);m.set_defaults(func=improve)
    a=p.parse_args()
    if a.command in ['train','improve'] and str(a.device).startswith('cuda'):
        from examples.arm_hand.resource_slots import resource_slot
        with resource_slot(Path(__file__).resolve().parents[2]/'outputs/mustard_multisource_valid100/resource_slots/gpu',10):a.func(a)
    else:a.func(a)


if __name__=='__main__':main()
