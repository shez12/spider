"""Paired physical comparison; isolated outputs, existing strict audits, no production counts."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['case','model','initial-checkpoint','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--device',default='cuda:0')
    p.add_argument('--samples',type=int,default=512);p.add_argument('--iterations',type=int,default=200)
    p.add_argument('--gpu-slots-dir',type=Path,default=Path(__file__).resolve().parents[2]/'outputs/mustard_multisource_valid100/resource_slots/gpu')
    p.add_argument('--gpu-slots',type=int,default=10)
    p.add_argument('--horizon',type=int,default=8)
    p.add_argument('--trust-radius',type=float,default=.02)
    p.add_argument('--model-iterations',type=int,default=1)
    p.add_argument('--proposal-updates',type=int,default=100);a=p.parse_args()
    if min(a.iterations,a.model_iterations,a.proposal_updates)<1 or a.samples<10:p.error('Positive iterations and samples >=10 required')
    check=subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],capture_output=True,text=True,timeout=10)
    if check.returncode:raise RuntimeError('GPU driver unavailable: '+check.stderr.strip()+check.stdout.strip())
    if a.gpu_slots<1:p.error('gpu-slots must be positive')
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; paired physical benchmark was not started')
    a.case=a.case.resolve();a.output=a.output.resolve()
    if a.output.exists():raise FileExistsError('Use a fresh isolated benchmark directory')
    a.output.mkdir(parents=True)
    root=Path(__file__).resolve().parents[2]
    def run(module,arguments,log):
        started=time.monotonic()
        with log.open('w') as f:
            from examples.arm_hand.resource_slots import resource_slot
            if module.endswith('world_model'):
                arguments=[*arguments,'--gpu-slots-dir',a.gpu_slots_dir,'--gpu-slots',a.gpu_slots]
            # Dynamics module acquires its own slot; other subprocesses share the campaign pool here.
            with resource_slot(None if module.endswith('world_model') else a.gpu_slots_dir,a.gpu_slots):
                subprocess.run([sys.executable,'-u','-m',module,*map(str,arguments)],cwd=root,stdout=f,stderr=subprocess.STDOUT,check=True)
        return time.monotonic()-started
    from examples.arm_hand.mustard_valid100 import valid
    report=dict(case=str(a.case),seed=a.seed,samples=a.samples,baseline_max_iterations=a.iterations,model_max_iterations=a.model_iterations,
                experiment='same initial checkpoint, model refinement versus original SPIDER continuation',arms={})
    def save(): (a.output/'comparison.json').write_text(json.dumps(report,indent=2))
    save()
    for arm in ['baseline','model']:
        out=a.output/arm;out.mkdir();start=time.monotonic();warm=a.initial_checkpoint.resolve()
        entry=dict(status='running');report['arms'][arm]=entry;save()
        try:
            if arm=='model':
                entry['proposal_s']=run('examples.arm_hand.world_model',['propose','--model',a.model.resolve(),
                    '--case',a.case,'--trajectory',a.case/'gpu/trajectory.npz','--initial-checkpoint',warm,
                    '--output',out/'imagination','--updates',a.proposal_updates,'--horizon',a.horizon,'--trust-radius',a.trust_radius,'--device',a.device],out/'proposal.log')
                warm=out/'imagination/proposal.pt'
            fallback=['--fallback-start',a.initial_checkpoint.resolve()] if arm=='model' else []
            entry['optimizer_s']=run('examples.arm_hand.optimize_kuka_gpu',['--input-dir',a.case,'--output-dir',out/'gpu',
                '--warm-start',warm,'--device',a.device,'--samples',a.samples,'--max-iterations',a.iterations if arm=='baseline' else a.model_iterations,
                '--seed',a.seed,'--patience',30,'--relative-tol',.001,'--absolute-tol',.01,
                '--disable-target-stop','--active','all','--table-collision-weight',0,*fallback],out/'optimizer.log')
            for label,trajectory in [('selected','trajectory.npz'),('replay','replay_trajectory.npz')]:
                audit=out/'gpu' if label=='selected' else out/'replay_audit'
                audit.mkdir(exist_ok=True)
                if label=='replay':(audit/'trajectory.npz').symlink_to(out/'gpu'/trajectory)
                run('examples.arm_hand.audit_right_trajectory',['--case-dir',a.case,'--gpu-dir',audit],out/f'{label}_audit.log')
                entry[label]=json.loads((audit/'tracking_collision_report.json').read_text())
            entry['accepted']=valid(entry['selected']) and valid(entry['replay'])
            entry['status']='complete';entry['result']=json.loads((out/'gpu/result.json').read_text())
        except Exception as error:
            entry.update(status='failed',error=str(error),accepted=False)
        entry['wall_s']=time.monotonic()-start;save()
    report['limitations']='Paired refinement benchmark, not cold-start dataset throughput. Model training/import costs must also be amortized. No production counts modified.'
    save();print(json.dumps(report,indent=2))
    if any(x['status']=='failed' for x in report['arms'].values()):raise SystemExit(1)


if __name__=='__main__':main()
