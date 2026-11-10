"""Isolated secant-loss variant; frozen original sha256 16b91a6100c9337c7ab8f850a2f9d6e6c88fb75f0a8631f1403cec69028276f5"""
"""Experimental differentiable dynamics for SPIDER; predictions never certify data."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from torch import nn

VERSION = 2


def scene_spec(scene):
    model = mujoco.MjModel.from_xml_path(str(scene))
    if not np.isclose(model.opt.timestep,.002):
        raise ValueError('Pilot requires 2 ms physics / 10 ms control')
    if model.nq != model.nu + 7 or model.nv != model.nu + 6 or model.nmocap:
        raise ValueError('Pilot supports actuated arm/hand + one free object, no mocap')
    # Topology must match; scene physics and sampled mesh geometry are explicit inputs.
    digest = hashlib.sha256()
    for name in ['body_parentid','jnt_type','jnt_bodyid','jnt_axis','geom_type','geom_bodyid',
                 'actuator_trnid','actuator_trntype']:
        digest.update(np.asarray(getattr(model,name)).tobytes())
    context=[]
    for name in ['body_pos','body_quat','body_mass','body_inertia','body_ipos','body_iquat',
                 'geom_pos','geom_quat','geom_size','geom_friction','geom_solref','geom_solimp',
                 'geom_contype','geom_conaffinity','dof_damping','dof_armature','dof_frictionloss',
                 'actuator_gainprm','actuator_biasprm','actuator_ctrlrange','actuator_forcerange',
                 'jnt_range']:
        context.extend(np.asarray(getattr(model,name)).ravel())
    for geom in range(model.ngeom):
        i=int(model.geom_dataid[geom])
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            context.extend(np.zeros(103));continue
        verts=model.mesh_vert[model.mesh_vertadr[i]:model.mesh_vertadr[i]+model.mesh_vertnum[i]]
        context.extend(model.mesh_pos[i]);context.extend(model.mesh_quat[i])
        context.extend(verts[np.linspace(0,len(verts)-1,32).astype(int)].ravel())
    context.extend(model.opt.gravity);context.extend([model.opt.timestep,model.opt.iterations])
    context=np.asarray(context,dtype=np.float32)
    return model, dict(version=VERSION, topology_sha256=digest.hexdigest(), nq=model.nq,
                       nv=model.nv, nu=model.nu, context_dim=len(context),control_dt=.01,physics_dt=.002),context


def canonical(q):
    q = q.copy()
    quat = q[..., -4:]
    quat /= np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-12)
    # Maintain temporal sign continuity rather than impose a w=0 discontinuity.
    for t in range(1, q.shape[-2]):
        flip = (q[..., t-1, -4:] * q[..., t, -4:]).sum(-1) < 0
        q[..., t, -4:] *= np.where(flip, -1., 1.)[..., None]
    return q


def save_shard(path, qpos, qvel, ctrl, initial_qpos, initial_qvel, costs, spec, group, context):
    """Input qpos/qvel are POST-action states; prepend PRE-action state explicitly."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    qpos, qvel, ctrl = [np.asarray(x, dtype=np.float32) for x in (qpos, qvel, ctrl)]
    if qpos.ndim != 3 or qpos.shape[:2] != ctrl.shape[:2] or qvel.shape[:2] != ctrl.shape[:2]:
        raise ValueError('Expected [trajectory,time,feature] arrays')
    q = np.concatenate((np.broadcast_to(initial_qpos, (len(qpos), 1, spec['nq'])), qpos), axis=1)
    v = np.concatenate((np.broadcast_to(initial_qvel, (len(qvel), 1, spec['nv'])), qvel), axis=1)
    finite = np.isfinite(q).all((1,2)) & np.isfinite(v).all((1,2)) & np.isfinite(ctrl).all((1,2))
    if not finite.any():
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, state=np.concatenate((canonical(q[finite]),v[finite]),-1).astype(np.float32),
            action=ctrl[finite], cost=np.asarray(costs)[finite], context=np.asarray(context,dtype=np.float32),
            metadata=json.dumps(dict(spec=spec, source_group=group, post_action_states=True)))
    temp.replace(path)
    return int(finite.sum())


class Dynamics(nn.Module):
    def __init__(self, spec, width=256):
        super().__init__(); self.spec=spec; self.width=width
        d=spec['nq']+spec['nv']; a=spec['nu']+spec['context_dim']
        self.net=nn.Sequential(nn.Linear(d+a,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,d))
        self.register_buffer('xmean',torch.zeros(d+a));self.register_buffer('xscale',torch.ones(d+a))
        self.register_buffer('yscale',torch.ones(d))

    def forward(self, state, action, context):
        delta=self.net((torch.cat((state,action,context),-1)-self.xmean)/self.xscale)*self.yscale
        out=state+delta; nq=self.spec['nq']
        quat=out[...,nq-4:nq]
        quat=quat/quat.norm(dim=-1,keepdim=True).clamp_min(1e-8)
        return torch.cat((out[...,:nq-4],quat,out[...,nq:]),-1)


class ResidualDynamics(Dynamics):
    """Learn corrections to kinematic integration rather than relearning integration."""
    def __init__(self,spec,width=512):
        super().__init__(spec,width)
        d=spec['nq']+spec['nv'];n=d+spec['nu']+spec['context_dim']
        self.net=nn.Sequential(nn.Linear(n,width),nn.ELU(),nn.Linear(width,width),nn.ELU(),
                               nn.Linear(width,width),nn.ELU(),nn.Linear(width,d))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def baseline(self,state):
        nq=self.spec['nq'];nu=self.spec['nu'];q=state[...,:nq];v=state[...,nq:];dt=.01
        # MuJoCo free-joint angular velocity is expressed in the local body frame.
        quat=q[...,-4:];w=v[...,-3:];half=.5*dt*w.norm(dim=-1,keepdim=True)
        dq=torch.cat((half.cos(),w*(.5*dt*torch.sinc(half/torch.pi))),-1)
        aw,av=quat[...,:1],quat[...,1:];bw,bv=dq[...,:1],dq[...,1:]
        rotated=torch.cat((aw*bw-(av*bv).sum(-1,keepdim=True),aw*bv+bw*av+torch.cross(av,bv,dim=-1)),-1)
        newq=torch.cat((q[...,:nu]+dt*v[...,:nu],q[...,nu:nu+3]+dt*v[...,nu:nu+3],rotated),-1)
        return torch.cat((newq,v),-1)

    def forward(self,state,action,context):
        x=(torch.cat((state,action,context),-1)-self.xmean)/self.xscale
        correction=self.net(x)*self.yscale;out=self.baseline(state)+correction;nq=self.spec['nq']
        quat=out[...,nq-4:nq];quat=quat/quat.norm(dim=-1,keepdim=True).clamp_min(1e-8)
        return torch.cat((out[...,:nq-4],quat,out[...,nq:]),-1)


class CompactResidualDynamics(ResidualDynamics):
    """Discard only context coordinates constant in training; keep all dynamic inputs."""
    def __init__(self,spec,width=1024,input_indices=None):
        super().__init__(spec,width)
        full=spec['nq']+spec['nv']+spec['nu']+spec['context_dim']
        ids=torch.arange(full) if input_indices is None else torch.as_tensor(input_indices,dtype=torch.long)
        self.register_buffer('input_indices',ids)
        d=spec['nq']+spec['nv']
        self.net=nn.Sequential(nn.Linear(len(ids),width),nn.ELU(),nn.Linear(width,width),nn.ELU(),
                               nn.Linear(width,width),nn.ELU(),nn.Linear(width,d))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,state,action,context):
        x=(torch.cat((state,action,context),-1)-self.xmean)/self.xscale
        out=self.baseline(state)+self.net(x[...,self.input_indices])*self.yscale;nq=self.spec['nq']
        quat=out[...,nq-4:nq];quat=quat/quat.norm(dim=-1,keepdim=True).clamp_min(1e-8)
        return torch.cat((out[...,:nq-4],quat,out[...,nq:]),-1)


def corrected_transition(prediction,bias,nq):
    """Cancel logged transition bias without detaching the learned state/action Jacobians."""
    out=prediction+bias;quat=out[...,nq-4:nq]
    quat=quat/quat.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    return torch.cat((out[...,:nq-4],quat,out[...,nq:]),-1)


def load_model(path, device):
    ck=torch.load(path,map_location=device,weights_only=False)
    if ck.get('architecture')=='compact':net=CompactResidualDynamics(ck['spec'],ck['width'],ck['model']['input_indices'].cpu())
    else:net=(ResidualDynamics if ck.get('architecture')=='residual' else Dynamics)(ck['spec'],ck['width'])
    net=net.to(device);net.load_state_dict(ck['model']);net.eval()
    return net,ck


def read_data(root):
    shards=[];spec=None
    for path in sorted(Path(root).rglob('transitions_*.npz')):
        with np.load(path) as f:
            meta=json.loads(str(f['metadata']))
            if spec is None:spec=meta['spec']
            if spec != meta['spec']:raise ValueError(f'Incompatible dynamics in {path}; use separate dataset roots')
            shards.append((f['state'].copy(),f['action'].copy(),meta['source_group'],f['context'].copy()))
    if not shards:raise ValueError('No transition shards')
    return spec,shards


def usable_states(state,spec):
    """Reject numerical explosions, not ordinary failed manipulation transitions."""
    nu=spec['nu'];nq=spec['nq'];v=state[...,nq:]
    return (np.isfinite(state).all(-1) & (np.abs(state[...,:nu]).max(-1)<10)
        & (np.linalg.norm(state[...,nu:nu+3],axis=-1)<2)
        & (np.abs(v[...,:nu]).max(-1)<100)
        & (np.linalg.norm(v[...,nu:nu+3],axis=-1)<10)
        & (np.linalg.norm(v[...,-3:],axis=-1)<100))


def train(args):
    torch.manual_seed(args.seed);spec,shards=read_data(args.data)
    for extra in args.extra_data:
        es,rows=read_data(extra)
        if es!=spec:raise ValueError('Extra dataset topology differs')
        shards.extend(rows)
    groups=sorted({s[2] for s in shards}); held=set(args.holdout_group);test=set(args.test_group)
    if held & test or not (held|test)<set(groups):raise ValueError('Disjoint train, validation and test groups required')
    if not held or not held < set(groups):raise ValueError('Specify held-out ORIGINAL source group(s), with separate training groups')
    window_counts={}
    def windows(items,split):
        states=[];actions=[];contexts=[];kept=0;dropped=0
        for s,a,g,c in items:
            for start in range(0,a.shape[1]-args.window+1,args.window):
                segment=s[:,start:start+args.window+1];control=a[:,start:start+args.window]
                good=usable_states(segment,spec).all(-1)&np.isfinite(control).all((1,2))
                kept+=int(good.sum());dropped+=int((~good).sum())
                if good.any():
                    states.append(segment[good]);actions.append(control[good]);contexts.append(np.broadcast_to(c,(int(good.sum()),len(c))))
        window_counts[split]=dict(kept=kept,dropped_numerically_diverged=dropped)
        if not states:raise ValueError('Insufficient physically bounded multi-step windows')
        return tuple(torch.tensor(np.concatenate(v).copy(),dtype=torch.float32,device=args.device) for v in (states,actions,contexts))
    tr,ta,tc=windows([s for s in shards if s[2] not in held|test],'train');va,vact,vc=windows([s for s in shards if s[2] in held],'holdout')
    x=torch.cat((tr[:,:-1],ta),-1).reshape(-1,spec['nq']+spec['nv']+spec['nu'])
    full_std=torch.cat((x.std(0,unbiased=False),tc.std(0,unbiased=False)))
    dynamic_count=x.shape[-1]
    if args.architecture=='compact':
        ids=torch.cat((torch.arange(dynamic_count,device=args.device),dynamic_count+torch.nonzero(full_std[dynamic_count:]>1e-6).flatten()))
        if args.initialize:
            initialized,_=load_model(args.initialize,args.device)
            if isinstance(initialized,CompactResidualDynamics):ids=initialized.input_indices
        net=CompactResidualDynamics(spec,args.width,ids.cpu()).to(args.device)
    else:net=(ResidualDynamics if args.architecture=='residual' else Dynamics)(spec,args.width).to(args.device)
    net.xmean.copy_(torch.cat((x.mean(0),tc.mean(0))))
    net.xscale.copy_(full_std.clamp_min(.01))
    with torch.no_grad():
        base=net.baseline(tr[:,:-1]) if args.architecture in ['residual','compact'] else tr[:,:-1]
        net.yscale.copy_((tr[:,1:]-base).reshape(-1,tr.shape[-1]).std(0,unbiased=False).clamp_min(.001))
    if args.initialize:
        source,source_ck=load_model(args.initialize,args.device)
        if source_ck['spec']!=spec or source_ck['width']!=args.width or source_ck.get('architecture','dense')!=args.architecture:raise ValueError('Initialization architecture/spec differs')
        net.load_state_dict(source.state_dict())
    opt=torch.optim.Adam(net.parameters(),lr=args.lr)
    weights=torch.ones(spec['nq']+spec['nv'],device=args.device)
    weights[spec['nu']:spec['nu']+3]=args.object_position_weight
    weights[spec['nq']-4:spec['nq']]=args.object_rotation_weight
    weights[spec['nq']:spec['nq']+spec['nu']]=.2
    def loss(s,a,c):
        pred=s[:,0];terms=[]
        for t in range(a.shape[1]):
            pred=net(pred,a[:,t],c);target=s[:,t+1]
            # Quaternion sign equivalent loss.
            nq=spec['nq']; target=target.clone()
            sign=torch.where((pred[:,nq-4:nq]*target[:,nq-4:nq]).sum(-1,keepdim=True)<0,-1.,1.)
            target[:,nq-4:nq]*=sign
            terms.append((((pred-target)/net.xscale[:pred.shape[-1]]).square()*weights).sum(-1).mean()/weights.sum())
        return torch.stack(terms).mean()
    paired_train=None;paired_val=None
    if args.counterfactual_data:
        cs,blocks=read_data(args.counterfactual_data)
        if cs!=spec:raise ValueError('Counterfactual topology differs')
        def pair_arrays(keep):
            arrays=[[],[],[],[],[]]
            for states,actions,group,context in blocks:
                if not keep(group) or len(actions)<4 or len(actions)%2:continue
                half=(len(actions)-2)//2;window=args.window
                if actions.shape[1]<window:continue
                if not np.allclose(states[:,0],states[0,0],atol=1e-6):raise ValueError('Counterfactual starts must match')
                plus=states[:half,:window+1];minus=states[half:2*half,:window+1]
                good=usable_states(plus,spec).all(-1)&usable_states(minus,spec).all(-1)
                if not good.any():continue
                values=(plus[good],minus[good],actions[:half,:window][good],actions[half:2*half,:window][good],np.broadcast_to(context,(int(good.sum()),len(context))))
                for target,value in zip(arrays,values):target.append(value)
            if not arrays[0]:raise ValueError('No counterfactual pairs in required source split')
            return tuple(torch.as_tensor(np.concatenate(v).copy(),dtype=torch.float32,device=args.device) for v in arrays)
        paired_train=pair_arrays(lambda group:group not in held|test);paired_val=pair_arrays(lambda group:group in held)
    def paired_loss(batch):
        plus,minus,ap,am,c=batch;k=len(plus);pred=torch.cat((plus[:,0],minus[:,0]))
        actions=torch.cat((ap,am));context=torch.cat((c,c));terms=[]
        for t in range(actions.shape[1]):
            pred=net(pred,actions[:,t],context)
            difference=(pred[:k]-pred[k:])-(plus[:,t+1]-minus[:,t+1])
            per_pair=((difference/net.xscale[:difference.shape[-1]]).square()*weights).sum(-1)/weights.sum()
            if args.normalize_counterfactual:
                # Normalize squared secant error by squared normalized action radius.
                scales=ap.new_tensor([.003]*(spec['nu']-22)+[.01]*22)
                radius2=(((ap-am)/(2*scales))**2).mean((1,2)).clamp(.25,100.)
                per_pair=per_pair/radius2
            terms.append(per_pair.mean())
        return torch.stack(terms).mean()
    started=time.monotonic();history=[];best=float('inf');args.output.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256()
    for state,action,group,context in shards:
        digest.update(group.encode());digest.update(state.tobytes());digest.update(action.tobytes());digest.update(context.tobytes())
    training_spec=dict(normalize_counterfactual=args.normalize_counterfactual,paired_loss_version='action-radius-normalized-v1' if args.normalize_counterfactual else 'original-squared-difference',dataset_sha256=digest.hexdigest(),window=args.window,width=args.width,architecture=args.architecture,object_position_weight=args.object_position_weight,object_rotation_weight=args.object_rotation_weight,
                       holdout_groups=sorted(held),test_groups=sorted(test),lr=args.lr,batch=args.batch,seed=args.seed,counterfactual_weight=args.counterfactual_weight,counterfactual_batch=args.counterfactual_batch)
    start_update=0;previous_elapsed=0.
    if args.resume:
        if args.resume.resolve().parent!=args.output.resolve():raise ValueError('Resume in the same output directory to preserve best checkpoint')
        saved=torch.load(args.resume,map_location=args.device,weights_only=False)
        if saved['training_spec']!=training_spec:raise ValueError('Resume data/split/training settings differ')
        net.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['cpu_rng'].cpu())
        if str(args.device).startswith('cuda'):torch.cuda.set_rng_state(saved['cuda_rng'].cpu(),args.device)
        start_update=saved['update'];history=saved['history'];best=saved['best'];previous_elapsed=saved['elapsed_s']

    for step in range(start_update,args.updates):
        ids=torch.randint(len(tr),(min(args.batch,len(tr)),),device=args.device)
        opt.zero_grad();l=loss(tr[ids],ta[ids],tc[ids])
        pair_value=0.
        if paired_train is not None:
            pair_ids=torch.randint(len(paired_train[0]),(min(args.counterfactual_batch,len(paired_train[0])),),device=args.device)
            pair_value=paired_loss(tuple(v[pair_ids] for v in paired_train));l=l+args.counterfactual_weight*pair_value
        if not torch.isfinite(l):raise RuntimeError('Nonfinite training loss')
        l.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
        if step%args.eval_every==0 or step==args.updates-1:
            with torch.no_grad():
                total=0.
                for i in range(0,len(va),args.batch):total+=float(loss(va[i:i+args.batch],vact[i:i+args.batch],vc[i:i+args.batch]))*len(va[i:i+args.batch])
                val=total/len(va)
                pair_validation=0.
                if paired_val is not None:
                    for i in range(0,len(paired_val[0]),args.counterfactual_batch):
                        count=len(paired_val[0][i:i+args.counterfactual_batch]);pair_validation+=float(paired_loss(tuple(v[i:i+args.counterfactual_batch] for v in paired_val)))*count
                    pair_validation/=len(paired_val[0]);val=val+args.counterfactual_weight*pair_validation
            history.append(dict(update=step+1,train_loss=float(l.detach()),holdout_loss=val,paired_train_loss=float(pair_value.detach()) if isinstance(pair_value,torch.Tensor) else pair_value,paired_validation_loss=pair_validation))
            print(json.dumps(history[-1]),flush=True)
            if val<best:
                best=val;tmp=args.output/'model.tmp'
                torch.save(dict(model=net.state_dict(),spec=spec,width=args.width,train_groups=sorted(set(groups)-held-test),architecture=args.architecture,test_groups=sorted(test),
                    holdout_groups=sorted(held),window=args.window,seed=args.seed,update=step+1),tmp)
                tmp.replace(args.output/'model.pt')
            elapsed=previous_elapsed+time.monotonic()-started
            last=dict(model=net.state_dict(),optimizer=opt.state_dict(),training_spec=training_spec,update=step+1,
                      history=history,best=best,elapsed_s=elapsed,cpu_rng=torch.get_rng_state(),
                      cuda_rng=torch.cuda.get_rng_state(args.device) if str(args.device).startswith('cuda') else None)
            tmp=args.output/'last.tmp';torch.save(last,tmp);tmp.replace(args.output/'last.pt')
            (args.output/'training.json').write_text(json.dumps(dict(history=history,elapsed_s=elapsed,
                train_windows=len(tr),holdout_windows=len(va),window_counts=window_counts,training_spec=training_spec,
                limitations='No failure head, critic, shared policy, or contact certification'),indent=2))
    (args.output/'training.json').write_text(json.dumps(dict(history=history,elapsed_s=previous_elapsed+time.monotonic()-started,
        train_windows=len(tr),holdout_windows=len(va),window_counts=window_counts,training_spec=training_spec,limitations='Pilot: no failure head, critic, shared policy, or contact certification'),indent=2))


def propose(args):
    started=time.monotonic(); net,ck=load_model(args.model,args.device)
    if args.anchored_world and (args.trajectory is None or args.initial_checkpoint is None):raise ValueError('Physical calibration requires trajectory and its raw optimizer checkpoint')
    if args.collision_weight<0 or args.terminal_weight<1:raise ValueError('Invalid geometry or terminal weight')
    model,spec,context=scene_spec(args.case/'scene.xml')
    if spec!=ck['spec']:raise ValueError('Scene dynamics differ from training dataset')
    for p in net.parameters():p.requires_grad_(False)
    with np.load(args.case/'reference.npz') as f:ref={k:f[k] for k in f.files}
    if 'mocap_pos' in ref:raise ValueError('Mocap not supported')
    data=mujoco.MjData(model);mujoco.mj_setState(model,data,ref['initial_state'],mujoco.mjtState.mjSTATE_FULLPHYSICS)
    tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device=args.device)
    nominal=tensor(ref['ctrl_nominal']);T,nu=nominal.shape;H=min(args.horizon,T)
    # Short shooting windows start from logged physical states; never chain entire imagined episodes.
    if args.trajectory:
        with np.load(args.trajectory) as f:
            if not np.isclose(float(f['control_dt']),.01) or not np.isclose(float(f['physics_dt']),.002):
                raise ValueError('Seed trajectory timebase differs')
            if not np.allclose(f['initial_state'],ref['initial_state']):raise ValueError('Seed initial state differs')
            logged_q=f['qpos'].copy();logged_v=f['qvel'].copy();logged_controls=f['ctrl'].copy()
        if len(logged_q)!=T or logged_q.shape[1]!=model.nq:raise ValueError('Seed trajectory length/dimensions differ')
        signature={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.case/'scene.xml',args.case/'reference.npz']}
        provenance=args.trajectory.parent/'result.json'
        if provenance.exists():
            if json.loads(provenance.read_text())['signature']!=signature:raise ValueError('Seed trajectory signature differs')
        elif args.trajectory.parent.parent.resolve()!=args.case.resolve():raise ValueError('External seed trajectory lacks signature provenance')
        seed_kind='logged_physical'
    else:
        # Cold-start multiple shooting: reference states seed windows; no optimized physics trajectory is required.
        reference_q=np.concatenate((ref['joint_ik'],ref['object_qpos']),-1)
        logged_q=reference_q[1:];logged_v=np.empty((T,model.nv))
        for t in range(T):mujoco.mj_differentiatePos(model,logged_v[t],.01,reference_q[t],reference_q[t+1])
        seed_kind='reference_reset'
    states=tensor(np.concatenate((canonical(np.concatenate((data.qpos[None],logged_q))),np.concatenate((data.qvel[None],logged_v))),-1))
    starts=torch.arange(0,T-H+1,H,device=args.device)
    if starts[-1]!=T-H:starts=torch.cat((starts,starts.new_tensor([T-H])))
    interp=tensor(np.stack([np.interp(np.arange(T),np.linspace(0,T-1,10).astype(int),row) for row in np.eye(10)],1))
    z=nn.Parameter(torch.zeros(10,nu,device=args.device))
    if args.initial_checkpoint:
        source=torch.load(args.initial_checkpoint,map_location=args.device,weights_only=False)
        if source['signature']!={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.case/'scene.xml',args.case/'reference.npz']}:
            raise ValueError('Initial checkpoint candidate signature differs')
        with torch.no_grad():z.copy_(source['best'])
    origin=z.detach().clone()
    low=tensor(model.actuator_ctrlrange[:,0]);high=tensor(model.actuator_ctrlrange[:,1])
    obj=tensor(ref['object_qpos']);joint=tensor(ref['joint_ik']);nq=model.nq
    from examples.arm_hand.world_model_kinematics import Kinematics
    from scipy.spatial.transform import Rotation
    fk=Kinematics(model,['sharpa_palm','sharpa_thumb_tip','sharpa_index_tip','sharpa_middle_tip','sharpa_ring_tip'],args.device)
    wrist_target=tensor(ref['wrist_pos']);tip_target=tensor(ref['fingertips'])
    wrist_rotation=tensor(Rotation.from_quat(ref['wrist_quat'][:,[1,2,3,0]]).as_matrix())
    all_ids=starts[:,None]+torch.arange(H,device=args.device)[None,:]+1
    arm_dofs=nu-22
    context_batch=tensor(context).expand(len(starts),-1)
    calibration=None;initial_control_error=None
    if args.anchored_world:
        original_commands=(nominal+interp@origin).clamp(low,high)
        initial_control_error=float((original_commands-tensor(logged_controls)).abs().max())
        if initial_control_error>1e-5:raise ValueError('Raw checkpoint does not reproduce executed seed controls; feedback seeds need another parameterization')
        with torch.no_grad():
            calibration=torch.stack([states[starts+t+1]-net(states[starts+t],tensor(logged_controls)[starts+t],context_batch) for t in range(H)],1)
    from examples.arm_hand.world_model_collision import ContactPenalty
    contact=ContactPenalty(model,nu) if args.collision_weight else None

    opt=torch.optim.Adam([z],lr=args.lr);history=[];best_loss=float('inf');best=None
    for update in range(args.updates):
        opt.zero_grad();commands=(nominal+interp@z).clamp(low,high);s=states[starts];loss=0.;predicted=[]
        for t in range(H):
            ids=starts+t;s=net(s,commands[ids],context_batch)
            if calibration is not None:s=corrected_transition(s,calibration[:,t],nq)
            predicted.append(s);target=obj[ids+1]
            pos=(s[:,nq-7:nq-4]-target[:,:3]).square().sum(-1)
            dot=(s[:,nq-4:nq]*target[:,3:]).sum(-1)
            rot=1-dot.square().clamp(max=1)
            joint_loss=(s[:,arm_dofs:nu]-joint[ids+1,arm_dofs:nu]).square().sum(-1)
            terminal=torch.where(ids+1==T,args.terminal_weight,1.)
            loss=loss+((3600*pos+args.rotation_weight*rot+.0025*joint_loss)*terminal).mean()/H
        prediction=torch.stack(predicted,dim=1)
        if contact is not None:loss=loss+args.collision_weight*contact(prediction[...,:nq]).mean()
        positions,rotations=fk(prediction[...,:nu])
        wrist_error=(positions[...,0,:]-wrist_target[all_ids]).square().sum(-1)
        relative=wrist_rotation[all_ids].transpose(-1,-2)@rotations[...,0,:,:]
        angle=((relative.diagonal(dim1=-2,dim2=-1).sum(-1)-1)*.5).clamp(-1+1e-6,1-1e-6).acos()
        finger_error=(positions[...,1:,:]-tip_target[all_ids]).square().sum((-1,-2))
        loss=loss+(args.wrist_weight**2*wrist_error+.64*angle.square()+args.fingertip_weight**2*finger_error).mean()
        loss=loss+.0009*z.square().sum()/max(1,T)+.0004*z.diff(dim=0).square().sum()/max(1,T)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite imagination loss')
        value=float(loss.detach())
        if value<best_loss:best_loss=value;best=z.detach().clone()
        loss.backward();nn.utils.clip_grad_norm_([z],1.);opt.step()
        with torch.no_grad():
            z.copy_(torch.maximum(torch.minimum(z,origin+args.trust_radius),origin-args.trust_radius))
            z.clamp_(-args.bound,args.bound)
        history.append(value)
    args.output.mkdir(parents=True,exist_ok=True)
    torch.save(dict(best=best.cpu(),signature={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.case/'scene.xml',args.case/'reference.npz']},
        experimental=True,physically_validated=False,model=str(args.model.resolve())),args.output/'proposal.pt')
    (args.output/'proposal.json').write_text(json.dumps(dict(loss=history,best_loss=best_loss,elapsed_s=time.monotonic()-started,
        seed_kind=seed_kind,horizon=H,control_dt=.01,trust_radius=args.trust_radius,shooting_windows=len(starts),physically_validated=False,anchored_world=args.anchored_world,collision_weight=args.collision_weight,rotation_weight=args.rotation_weight,terminal_weight=args.terminal_weight,initial_control_max_error=initial_control_error,seed_trajectory_sha256=hashlib.sha256(args.trajectory.read_bytes()).hexdigest() if args.trajectory else None,initial_checkpoint_sha256=hashlib.sha256(args.initial_checkpoint.read_bytes()).hexdigest() if args.initial_checkpoint else None,
        objective='object pose + exact analytic palm/fingertip FK + hand joint prior + residual regularity; no hard collision certification',
        limitations='Teacher-seeded short windows; full physical rollout and two audits required'),indent=2))


def evaluate(args):
    net,ck=load_model(args.model,args.device);spec,shards=read_data(args.data)
    for extra in args.extra_data:
        es,rows=read_data(extra)
        if es!=spec:raise ValueError('Extra dataset topology differs')
        shards.extend(rows)
    if spec!=ck['spec']:raise ValueError('Dataset/model specs differ')
    nq=spec['nq'];held=set(ck['holdout_groups']);test=set(ck.get('test_groups',[]));started=time.monotonic()
    result=dict(horizon=args.horizon,seconds=args.horizon*.01,model=str(args.model.resolve()),checkpoint_update=ck.get('update'),metrics={},
        limitations='Bounded prediction windows against saved physics, not task acceptance; independent test groups excluded from training and checkpoint selection')
    for split in ['train','holdout','test']:
        ss=[];aa=[];cc=[];dropped=0
        for states,actions,group,context in shards:
            label='test' if group in test else ('holdout' if group in held else 'train')
            if label!=split:continue
            for start in range(0,actions.shape[1]-args.horizon+1,args.horizon):
                segment=states[:,start:start+args.horizon+1];act=actions[:,start:start+args.horizon]
                good=usable_states(segment,spec).all(-1)&np.isfinite(act).all((1,2));dropped+=int((~good).sum())
                if not good.any():continue
                ss.append(segment[good]);aa.append(act[good]);cc.append(np.broadcast_to(context,(int(good.sum()),len(context))))
        if not ss:continue
        st,ac,ct=[np.concatenate(x).astype(np.float32) for x in [ss,aa,cc]]
        count=0;pos=0.;rot=0.;joint=0.
        with torch.no_grad():
            for i in range(0,len(st),args.batch):
                states=torch.as_tensor(st[i:i+args.batch],device=args.device)
                actions=torch.as_tensor(ac[i:i+args.batch],device=args.device)
                context=torch.as_tensor(ct[i:i+args.batch],device=args.device);pred=states[:,0]
                for t in range(args.horizon):
                    pred=net(pred,actions[:,t],context);target=states[:,t+1]
                    dot=(pred[:,nq-4:nq]*target[:,nq-4:nq]).sum(-1).abs().clamp(0,1)
                    angle=2*torch.atan2((1-dot.square()).clamp_min(0).sqrt(),dot)
                    count+=len(pred);pos+=float((pred[:,nq-7:nq-4]-target[:,nq-7:nq-4]).square().sum())
                    rot+=float(angle.square().sum());joint+=float((pred[:,:spec['nu']]-target[:,:spec['nu']]).square().sum())
        result['metrics'][split]=dict(predicted_steps=count,windows=len(st),dropped_numerically_diverged=dropped,
            object_position_rmse_m=(pos/count)**.5,object_rotation_rmse_deg=np.degrees((rot/count)**.5),
            joint_rmse_rad=(joint/count/spec['nu'])**.5)
    result['elapsed_s']=time.monotonic()-started
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


def import_existing(args):
    """Bootstrap from actual saved trajectories, including failed audits when present."""
    records=[]
    groups=[args.single_group] if args.single_group else sorted(args.groups_root.iterdir())
    for group in groups:
        if not group.is_dir():continue
        count=0
        for trajectory in sorted(group.glob('candidate_*/gpu/*trajectory.npz')):
            if count>=args.max_per_group:break
            case=trajectory.parent.parent
            model,spec,context=scene_spec(case/'scene.xml')
            data=mujoco.MjData(model)
            with np.load(case/'reference.npz') as ref:
                mujoco.mj_setState(model,data,ref['initial_state'],mujoco.mjtState.mjSTATE_FULLPHYSICS)
            with np.load(trajectory) as tr:
                if not np.isclose(float(tr['control_dt']),.01) or not np.isclose(float(tr['physics_dt']),.002):
                    raise ValueError(f'Unsupported timing: {trajectory}')
                key=hashlib.sha256(str(trajectory.resolve()).encode()).hexdigest()[:20]
                path=args.output/f'transitions_{key}.npz'
                count+=1
                if path.exists():continue
                n=save_shard(path,tr['qpos'][None],tr['qvel'][None],tr['ctrl'][None],data.qpos,data.qvel,
                    [np.nan],spec,args.source_group or group.name,context)
            records.append(dict(path=str(path),source=str(trajectory.resolve()),source_group=args.source_group or group.name,trajectories=n))
    args.output.mkdir(parents=True,exist_ok=True)
    # Immutable per-import manifest avoids overwriting previous provenance.
    manifest=args.output/f'import_{time.time_ns()}.json'
    manifest.write_text(json.dumps(records,indent=2));print(json.dumps(dict(shards=len(records),manifest=str(manifest))))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train');t.add_argument('--data',type=Path,required=True);t.add_argument('--output',type=Path,required=True)
    t.add_argument('--holdout-group',action='append',required=True);t.add_argument('--window',type=int,default=8)
    t.add_argument('--counterfactual-data',type=Path)
    t.add_argument('--counterfactual-weight',type=float,default=50.)
    t.add_argument('--normalize-counterfactual',action='store_true')
    t.add_argument('--counterfactual-batch',type=int,default=32)
    t.add_argument('--object-position-weight',type=float,default=1.)
    t.add_argument('--object-rotation-weight',type=float,default=1.)
    t.add_argument('--initialize',type=Path)
    t.add_argument('--resume',type=Path)
    t.add_argument('--extra-data',action='append',type=Path,default=[])
    t.add_argument('--test-group',action='append',default=[])
    t.add_argument('--architecture',choices=['dense','residual','compact'],default='residual')
    t.add_argument('--width',type=int,default=256);t.add_argument('--batch',type=int,default=256);t.add_argument('--seed',type=int,default=42)
    t.add_argument('--eval-every',type=int,default=50);t.add_argument('--updates',type=int,default=2000);t.add_argument('--lr',type=float,default=3e-4)
    t.add_argument('--device',default='cuda:0');t.set_defaults(func=train)
    q=sub.add_parser('propose');q.add_argument('--model',type=Path,required=True);q.add_argument('--case',type=Path,required=True)
    q.add_argument('--trajectory',type=Path);q.add_argument('--initial-checkpoint',type=Path)
    q.add_argument('--anchored-world',action='store_true');q.add_argument('--collision-weight',type=float,default=0.)
    q.add_argument('--rotation-weight',type=float,default=.36);q.add_argument('--terminal-weight',type=float,default=1.)
    q.add_argument('--output',type=Path,required=True);q.add_argument('--horizon',type=int,default=32)
    q.add_argument('--updates',type=int,default=100);q.add_argument('--lr',type=float,default=.005)
    q.add_argument('--wrist-weight',type=float,default=15.)
    q.add_argument('--fingertip-weight',type=float,default=0.)
    q.add_argument('--trust-radius',type=float,default=.02)
    q.add_argument('--bound',type=float,default=.4);q.add_argument('--device',default='cuda:0');q.set_defaults(func=propose)
    i=sub.add_parser('import-existing');i.add_argument('--groups-root',type=Path,required=True)
    i.add_argument('--output',type=Path,required=True);i.add_argument('--max-per-group',type=int,default=4)
    i.add_argument('--single-group',type=Path);i.add_argument('--source-group')
    i.set_defaults(func=import_existing)
    e=sub.add_parser('evaluate');e.add_argument('--model',type=Path,required=True);e.add_argument('--data',type=Path,required=True)
    e.add_argument('--extra-data',action='append',type=Path,default=[]);e.add_argument('--batch',type=int,default=256)
    e.add_argument('--output',type=Path,required=True);e.add_argument('--horizon',type=int,default=8)
    e.add_argument('--device',default='cuda:0');e.set_defaults(func=evaluate)
    for gpu_parser in (t,q,e):
        gpu_parser.add_argument('--gpu-slots-dir',type=Path,default=Path(__file__).resolve().parents[2]/'outputs/mustard_multisource_valid100/resource_slots/gpu')
        gpu_parser.add_argument('--gpu-slots',type=int,default=10)
    a=p.parse_args()
    for name in ['updates','horizon','window','batch','width','eval_every']:
        if hasattr(a,name) and getattr(a,name)<1:p.error(f'{name} must be positive')
    if getattr(a,'initialize',None) and getattr(a,'resume',None):p.error('initialize and resume are mutually exclusive')
    if getattr(a,'source_group',None) and not getattr(a,'single_group',None):p.error('source-group requires single-group')
    if hasattr(a,'trust_radius') and a.trust_radius<=0:p.error('trust-radius must be positive')
    if hasattr(a,'bound') and a.bound<=0:p.error('bound must be positive')
    if hasattr(a,'lr') and a.lr<=0:p.error('lr must be positive')
    if hasattr(a,'max_per_group') and a.max_per_group<1:p.error('max-per-group must be positive')
    if str(getattr(a,'device','cpu')).startswith('cuda'):
        torch.set_float32_matmul_precision('high')
        if a.gpu_slots<1:p.error('gpu-slots must be positive')
        from examples.arm_hand.resource_slots import resource_slot
        with resource_slot(a.gpu_slots_dir,a.gpu_slots):a.func(a)
    else:a.func(a)


if __name__=='__main__':main()
