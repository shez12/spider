"""Matched-state paired control perturbations in MuJoCo Warp for causal dynamics fitting."""
import argparse
import json
from pathlib import Path
import time
import mujoco
import numpy as np
import torch
import torch.nn.functional as F
from examples.arm_hand.resource_slots import resource_slot
from examples.arm_hand.world_model import scene_spec,save_shard,usable_states


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--source-indices',nargs='+',type=int,default=[1,2,3,10,11,18])
    p.add_argument('--worlds',type=int,default=128);p.add_argument('--horizon',type=int,default=16)
    p.add_argument('--starts',type=int,default=6);p.add_argument('--seed',type=int,default=20261030);a=p.parse_args()
    if a.worlds<4 or a.worlds%2:raise ValueError('Use an even world count >=4')
    root=Path(__file__).resolve().parents[2];campaign=root/'outputs/mustard_multisource_valid100';a.output.mkdir(parents=True,exist_ok=True)
    selected=[g for g in json.loads((campaign/'selected_sources.json').read_text()) if g['source_index'] in a.source_indices]
    records=[];started=time.monotonic()
    with resource_slot(campaign/'resource_slots/gpu',10):
        import mujoco_warp as mw
        import warp as wp
        wp.init();wp.set_device('cuda:0');torch.cuda.set_device('cuda:0')
        stream=torch.cuda.Stream(device='cuda:0');torch.cuda.set_stream(stream);wp.set_stream(wp.stream_from_torch(stream))
        for group in selected:
            folder=Path(group['result_dir']);manifest=folder/'accepted.json'
            accepted=json.loads(manifest.read_text()) if manifest.exists() else []
            candidates=[Path(item['folder']) for item in accepted]
            candidates.extend(p.parent.parent for p in sorted(folder.glob('candidate_*/gpu/trajectory.npz')) if p.parent.parent not in candidates)
            if not candidates:raise ValueError(f'No physics seed for source {group["source_index"]}')
            case=candidates[0];model,spec,context=scene_spec(case/'scene.xml')
            with np.load(case/'gpu/trajectory.npz') as f:tr={k:f[k] for k in f.files}
            cpu=mujoco.MjData(model);mujoco.mj_setState(model,cpu,tr['initial_state'],mujoco.mjtState.mjSTATE_FULLPHYSICS);mujoco.mj_forward(model,cpu)
            initial_q=cpu.qpos.copy();initial_v=cpu.qvel.copy();n=a.worlds;H=a.horizon;T=len(tr['ctrl'])
            if T<H:continue
            gpu_model=mw.put_model(model);gpu_data=mw.put_data(model,cpu,nworld=n,nconmax=256,njmax=1024)
            qpos,qvel,ctrl=[wp.to_torch(getattr(gpu_data,name)) for name in ['qpos','qvel','ctrl']]
            mw.step(gpu_model,gpu_data);wp.synchronize()
            with wp.ScopedCapture() as capture:mw.step(gpu_model,gpu_data)
            graph=capture.graph
            low=torch.tensor(model.actuator_ctrlrange[:,0].copy(),dtype=torch.float32,device='cuda:0')
            high=torch.tensor(model.actuator_ctrlrange[:,1].copy(),dtype=torch.float32,device='cuda:0')
            key=f"{group['source_index']:03d}_{group['task']}"
            for frame in np.unique(np.linspace(0,T-H,a.starts).astype(int)):
                path=a.output/key/f'transitions_frame_{frame:05d}.npz'
                if path.exists():continue
                q=initial_q if frame==0 else tr['qpos'][frame-1];v=initial_v if frame==0 else tr['qvel'][frame-1]
                if not usable_states(np.r_[q,v],spec):continue
                torch.manual_seed(a.seed+group['source_index']*1000+int(frame));half=(n-2)//2
                coefficients=torch.randn(half,model.nu,4,device='cuda:0')
                scale=torch.tensor([.003]*(model.nu-22)+[.01]*22,device='cuda:0').expand(half,-1).clone()
                scale[int(.75*half):]*=10
                noise=F.interpolate(coefficients,size=H,mode='linear',align_corners=True).transpose(1,2)*scale[:,None]
                noise=torch.cat((noise,-noise,torch.zeros(2,H,model.nu,device='cuda:0')))
                baseline=torch.tensor(tr['ctrl'][frame:frame+H],dtype=torch.float32,device='cuda:0')
                controls=(baseline[None]+noise).clamp(low,high)
                qpos.copy_(torch.tensor(q,dtype=torch.float32,device='cuda:0').expand(n,-1));qvel.copy_(torch.tensor(v,dtype=torch.float32,device='cuda:0').expand(n,-1))
                for name in ['time','qacc_warmstart','qacc','qfrc_applied','xfrc_applied']:
                    value=getattr(gpu_data,name,None)
                    if value is not None:wp.to_torch(value).zero_()
                positions=[];velocities=[];block=time.monotonic()
                for t in range(H):
                    ctrl.copy_(controls[:,t])
                    for sub in range(5):wp.capture_launch(graph)
                    positions.append(qpos.clone());velocities.append(qvel.clone())
                count=save_shard(path,torch.stack(positions,1).cpu().numpy(),torch.stack(velocities,1).cpu().numpy(),controls.cpu().numpy(),q,v,
                    np.full(n,np.nan),spec,key,context)
                record=dict(source_index=group['source_index'],case=str(case),frame=int(frame),horizon=H,worlds=count,path=str(path),
                    elapsed_s=time.monotonic()-block,paired_control_perturbations=True,small_noise_arm_rad=.003,small_noise_hand_rad=.01,broad_noise_multiplier=10)
                records.append(record);print(json.dumps(record),flush=True)
                (a.output/'collection.json').write_text(json.dumps(dict(records=records,wall_s=time.monotonic()-started),indent=2))
            del gpu_data,gpu_model;torch.cuda.empty_cache()
    (a.output/'collection.json').write_text(json.dumps(dict(records=records,wall_s=time.monotonic()-started),indent=2))


if __name__=='__main__':main()
