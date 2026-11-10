"""Frozen, resumable raw/BC/imagination cohort with identical MPS and physical gates."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,data):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(data,indent=2));temporary.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['cohort','bc','model','output','mps-pipe-dir']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--max-runs',type=int,default=3);a=p.parse_args()
    root=Path(__file__).resolve().parents[2];out=a.output.resolve();plan_path=out/'plan.json';status_path=out/'status.json'
    if a.max_runs<1:p.error('max-runs must be positive')
    if not (a.mps_pipe_dir/'nvidia-cuda-mps-control.pid').is_file():raise RuntimeError('Existing production MPS daemon is unavailable')
    code_paths=['examples/arm_hand/optimize_kuka_contact_sampling.py','examples/arm_hand/execute_contact_sampling.py','examples/arm_hand/contact_sampling_cost.py','examples/arm_hand/warmup_candidate_bank.py','examples/arm_hand/strict_candidate_bank.py','examples/arm_hand/warmup_acceptance.py','examples/arm_hand/audit_warmup_trajectory.py','examples/arm_hand/world_model_policy.py','spider/optimizers/sampling.py','spider/simulators/mjwp.py']
    code={name:digest(root/name) for name in code_paths}
    if not a.resume:
        out.mkdir(parents=True,exist_ok=False);models={}
        for name,path in [('bc',a.bc),('imagination',a.model)]:
            target=out/(name+'.pt');shutil.copy2(path,target);models[name]=dict(path=str(target),sha256=digest(target))
        cohort=json.loads(a.cohort.read_text());cases=[s for s in cohort['cases'] if s.get('ik_pass')]
        if len(cases)!=cohort['valid_cases']*len(cohort['source_indices']):raise ValueError('Prepared cohort is incomplete')
        # Fix the candidate order and rotate each three-arm sequence before observing outcomes.
        cases.sort(key=lambda x:(x['candidate_id'],x['source_index']))
        jobs=[];modes=['raw','bc','imagination']
        for i,state in enumerate(cases):
            case=Path(state['folder']);signature={name:digest(case/name) for name in ['scene.xml','reference.npz','augmentation.json']}
            for mode in modes[i%3:]+modes[:i%3]:jobs.append(dict(case=str(case.resolve()),source_index=state['source_index'],candidate_id=state['candidate_id'],mode=mode,signature=signature))
        plan=dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),models=models,jobs=jobs,code_sha256=code,mps_pipe_dir=str(a.mps_pipe_dir.resolve()),
            protocol=dict(experimental_objective='original_plus_hand_self_contacts_v1',hand_self_collision_weight=1e7,acceptance_version='warmup_grasp_v2',samples=512,max_iterations=200,seed=0,strict_check_every=10,strict_candidate_pool=16,disable_target_stop=True),
            complete_only_when_all_planned_jobs_terminal=True,limitations='Development sources 015/018; source 012 is independently evaluated separately. Charge model setup, all IK rejections and all failed physics before any end-to-end claim.')
        save(plan_path,plan);status=dict(status='ready',runs=[],wall_s=0.)
    else:
        plan=json.loads(plan_path.read_text());status=json.loads(status_path.read_text())
        if plan['code_sha256']!=code or plan['mps_pipe_dir']!=str(a.mps_pipe_dir.resolve()):raise ValueError('Frozen code/MPS protocol differs')
    for m in plan['models'].values():
        if digest(m['path'])!=m['sha256']:raise ValueError('Frozen model changed')
    previous=status.get('wall_s',0.);started=time.monotonic();status.update(status='running',pid=os.getpid());save(status_path,status)
    done={r['job_index'] for r in status['runs']};new_runs=0
    for index,job in enumerate(plan['jobs']):
        if index in done:continue
        if new_runs>=a.max_runs:break
        if {name:digest(root/name) for name in code_paths}!=plan['code_sha256']:raise ValueError('Code changed during the frozen cohort')
        case=Path(job['case'])
        if any(digest(case/name)!=sha for name,sha in job['signature'].items()):raise ValueError('Frozen candidate changed')
        target=out/f"{job['source_index']:03d}_{job['candidate_id']}_{job['mode']}"
        resume_execution=False
        if target.exists():
            report_path=target/'execution.json'
            if not a.resume or not report_path.exists():
                raise RuntimeError('Unaccounted execution exists without a resumable execution record')
            previous_report=json.loads(report_path.read_text())
            if previous_report.get('case')!=str(case.resolve()) or previous_report.get('samples')!=512 or previous_report.get('seed')!=0:
                raise ValueError('Interrupted execution identity differs from frozen job')
            # Refuse to overlap a live execution/optimizer even when the
            # coordinator itself is absent. A reboot releases all processes.
            for process in Path('/proc').iterdir():
                if not process.name.isdigit():continue
                try:arguments=(process/'cmdline').read_bytes().split(b'\0')
                except (OSError,ProcessLookupError):continue
                if str(target).encode()in arguments or str(target/'gpu').encode()in arguments:
                    if any(module in arguments for module in [b'examples.arm_hand.execute_contact_sampling',b'examples.arm_hand.optimize_kuka_contact_sampling']):
                        raise RuntimeError(f'Interrupted job still has a live process: {process.name}')
            if previous_report.get('status')=='complete':
                status['runs'].append(dict(job_index=index,execution=str(report_path),returncode=0,
                    accepted=previous_report.get('accepted',False),mode=job['mode'],source_index=job['source_index']))
                save(status_path,status);continue
            if not (target/'gpu/checkpoint.pt').is_file():
                raise RuntimeError('Interrupted execution has no checkpoint; preserve it for inspection')
            resume_execution=True
            backup=target/'pre_reboot_logs';backup.mkdir(exist_ok=True)
            for log in target.glob('*.log'):
                if not (backup/log.name).exists():shutil.copy2(log,backup/log.name)
            history=json.loads((target/'gpu/history.json').read_text())
            save(target/'reboot_recovery.json',dict(resumed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                saved_iterations=len(history),recorded_prior_iteration_compute_s=sum(row.get('iteration_s',0)for row in history),
                timing_limit='Interrupted process wall time was not committed; retain this lower-bound compute record separately.'))
        command=[sys.executable,'-u','-m','examples.arm_hand.execute_contact_sampling','--case',str(case),'--output',str(target),'--iterations','200','--strict-check-every','10','--strict-candidate-pool','16']
        command+=['--baseline'] if job['mode']=='raw' else ['--policy',plan['models'][job['mode']]['path'],'--cache-policy-context']
        if resume_execution:command.append('--resume')
        env=dict(os.environ,CUDA_MPS_PIPE_DIRECTORY=plan['mps_pipe_dir'])
        with (out/f'job_{index:02d}.log').open('a' if resume_execution else 'w') as log:result=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
        report_path=target/'execution.json';report=json.loads(report_path.read_text()) if report_path.exists() else {}
        status['runs'].append(dict(job_index=index,execution=str(report_path),returncode=result.returncode,accepted=report.get('accepted',False),mode=job['mode'],source_index=job['source_index']))
        new_runs+=1;status['wall_s']=previous+time.monotonic()-started;save(status_path,status)
        if result.returncode:status.update(status='infrastructure_failed');save(status_path,status);raise RuntimeError('Execution infrastructure failed; preserve partial outputs')
    status.update(status='complete' if len(status['runs'])==len(plan['jobs']) else 'partial',wall_s=previous+time.monotonic()-started);save(status_path,status)
    print(json.dumps(dict(status=status['status'],completed=len(status['runs']),planned=len(plan['jobs']))),flush=True)


if __name__=='__main__':main()
