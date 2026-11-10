"""Actual MuJoCo Warp trust-radius probe; short segments do not certify accepted data."""
import argparse,json,time,hashlib
from pathlib import Path
import mujoco,numpy as np,torch
from examples.arm_hand.world_model import load_model,scene_spec
from examples.arm_hand.resource_slots import resource_slot

def main():
 p=argparse.ArgumentParser();p.add_argument('--world',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True);torch.set_num_threads(2);start=time.time();root=Path(__file__).resolve().parents[2];net,ck=load_model(a.world,'cpu');nq=ck['spec']['nq']
 for param in net.parameters():param.requires_grad_(False)
 recs=json.loads((root/'outputs/world_model_pilot/counterfactual_v1/collection.json').read_text())['records'];recs=[x for x in recs if x['source_index'] in [2,3,18]];selected=[]
 for idx in [2,3,18]:
  rr=[x for x in recs if x['source_index']==idx];selected.extend(rr[i] for i in [0,len(rr)//2,len(rr)-1])
 plan=dict(world=str(a.world),world_sha256=hashlib.sha256(a.world.read_bytes()).hexdigest(),cases=selected,radii=[.25,.5,1.,2.],horizons=[8,16],seed=20261103,objective='3600 * mean squared object position error, anchored gradients',limitation='development states, finite-horizon position+collision diagnostic, not full task or dual validation')
 (a.output/'plan.json').write_text(json.dumps(plan,indent=2));results=[]
 with resource_slot(root/'outputs/mustard_multisource_valid100/resource_slots/gpu',10,block=True):
  import mujoco_warp as mw,warp as wp
  wp.init();wp.set_device('cuda:0');torch.cuda.set_device(0);stream=torch.cuda.Stream();torch.cuda.set_stream(stream);wp.set_stream(wp.stream_from_torch(stream));rng=np.random.default_rng(20261103)
  for rec in selected:
   case=Path(rec['case']);model,spec,context=scene_spec(case/'scene.xml');assert spec==ck['spec'];cpu=mujoco.MjData(model);tr=np.load(case/'gpu/trajectory.npz');ref=np.load(case/'reference.npz');frame=rec['frame'];mujoco.mj_setState(model,cpu,tr['initial_state'],mujoco.mjtState.mjSTATE_FULLPHYSICS);q=cpu.qpos.copy() if frame==0 else tr['qpos'][frame-1].copy();v=cpu.qvel.copy() if frame==0 else tr['qvel'][frame-1].copy();scales=np.array([.003]*(model.nu-22)+[.01]*22,dtype=np.float32);ct=torch.tensor(context)[None]
   for H in [8,16]:
    baseline=tr['ctrl'][frame:frame+H];target=ref['object_qpos'][frame+1:frame+H+1,:3];ac=torch.tensor(baseline,requires_grad=True);st=torch.tensor(np.r_[q,v],dtype=torch.float32)[None];loss=0.
    for t in range(H):
     anchor=np.r_[q,v] if t==0 else np.r_[tr['qpos'][frame+t-1],tr['qvel'][frame+t-1]];obs=np.r_[tr['qpos'][frame+t],tr['qvel'][frame+t]]
     with torch.no_grad():bias=torch.tensor(obs,dtype=torch.float32)[None]-net(torch.tensor(anchor,dtype=torch.float32)[None],torch.tensor(baseline[t])[None],ct)
     st=net(st,ac[t:t+1],ct)+bias;quat=st[:,nq-4:nq];st=torch.cat((st[:,:nq-4],quat/quat.norm(dim=-1,keepdim=True).clamp_min(1e-8),st[:,nq:]),-1);loss=loss+3600*((st[:,nq-7:nq-4]-torch.tensor(target[t]))**2).sum()/H
    grad=torch.autograd.grad(loss,ac)[0].numpy()*scales;direction=-grad/max(float(np.sqrt(np.mean(grad**2))),1e-12);random=rng.normal(size=direction.shape);random/=np.sqrt(np.mean(random**2));offsets=[np.zeros_like(baseline),np.zeros_like(baseline)];labels=['baseline','baseline_duplicate']
    for radius in plan['radii']:
     for label,d in [('descent',direction),('ascent',-direction),('random',random)]:labels.append(f'{label}_{radius}');offsets.append(radius*scales*d)
    controls=np.clip(baseline+np.array(offsets),model.actuator_ctrlrange[:,0],model.actuator_ctrlrange[:,1]).astype(np.float32);N=len(controls);gm=mw.put_model(model);gd=mw.put_data(model,cpu,nworld=N,nconmax=256,njmax=1024);qp,qv,ctrl=[wp.to_torch(getattr(gd,k)) for k in ['qpos','qvel','ctrl']];mw.step(gm,gd);wp.synchronize()
    with wp.ScopedCapture() as capture:mw.step(gm,gd)
    qp.copy_(torch.tensor(q,device='cuda:0',dtype=torch.float32).expand(N,-1));qv.copy_(torch.tensor(v,device='cuda:0',dtype=torch.float32).expand(N,-1))
    for name in ['time','qacc_warmstart','qacc','qfrc_applied','xfrc_applied']:
     val=getattr(gd,name,None)
     if val is not None:wp.to_torch(val).zero_()
    states=[]
    for t in range(H):
     ctrl.copy_(torch.tensor(controls[:,t],device='cuda:0'))
     for _ in range(5):wp.capture_launch(capture.graph)
     states.append(qp.clone())
    states=torch.stack(states,1).cpu().numpy();cost=3600*((states[:,:,nq-7:nq-4]-target)**2).sum(-1).mean(-1);rows=[]
    for i,label in enumerate(labels):
     worst=0.;finite=bool(np.isfinite(states[i]).all())
     if finite:
      for pose in states[i]:
       cpu.qpos[:]=pose;mujoco.mj_fwdPosition(model,cpu)
       for contact in cpu.contact:
        na=model.geom(int(contact.geom1)).name or '';nb=model.geom(int(contact.geom2)).name or ''
        if (na.startswith('collision_sharpa_') and nb.startswith('collision_sharpa_')) or ('floor' in [na,nb] and any(x.startswith(('collision_sharpa_','collision_kuka_')) for x in [na,nb])):worst=max(worst,-float(contact.dist))
     rows.append(dict(label=label,cost=float(cost[i]) if finite else None,finite=finite,max_robot_collision_penetration_m=worst if finite else None,collision_pass=finite and worst<.001))
    key=f"source{rec['source_index']:03d}_frame{frame:03d}_h{H}";np.savez_compressed(a.output/(key+'.npz'),qpos=states,ctrl=controls,initial_qpos=q,initial_qvel=v,labels=labels,reference_object_position=target);results.append(dict(key=key,case=str(case),rows=rows,gradient_rms=float(np.sqrt(np.mean(grad**2)))));(a.output/'report.json').write_text(json.dumps(dict(status='running',results=results,wall_s=time.time()-start),indent=2));del gm,gd;torch.cuda.empty_cache()
 (a.output/'report.json').write_text(json.dumps(dict(status='complete',results=results,wall_s=time.time()-start),indent=2));print(json.dumps(dict(blocks=len(results),wall_s=time.time()-start)))
if __name__=='__main__':main()
