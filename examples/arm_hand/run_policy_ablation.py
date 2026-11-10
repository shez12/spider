"""Run raw/BC/imagination physics ablations after a training run actually completes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['case','bc','improved','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--training-pid',type=int,required=True);p.add_argument('--updates',type=int,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);report=dict(status='waiting_for_training',case=str(a.case.resolve()),development_ablation=True,runs=[])
    def save():(a.output/'ablation.json').write_text(json.dumps(report,indent=2))
    save();waiting=time.monotonic();training=a.improved.parent/'training.json'
    while True:
        if training.exists():
            d=json.loads(training.read_text())
            if d.get('history') and d['history'][-1]['update']==a.updates and 'limitations' in d:break
        proc=Path(f'/proc/{a.training_pid}/cmdline')
        if not proc.exists() or b'world_model_policy' not in proc.read_bytes() or str(a.improved.parent.resolve()).encode() not in proc.read_bytes():
            report.update(status='failed',error='Training process ended before final completion metadata');save();raise RuntimeError(report['error'])
        time.sleep(10)
    report.update(status='running',training_wait_s=time.monotonic()-waiting,training_report=str(training.resolve()));save()
    for mode,policy in [('raw',None),('bc',a.bc),('imagination',a.improved)]:
        output=a.output/mode
        command=[sys.executable,'-u','-m','examples.arm_hand.execute_proposal','--case',str(a.case.resolve()),'--output',str(output.resolve()),'--iterations','200','--strict-check-every','10']
        command+=['--policy',str(policy.resolve()),'--cache-policy-context'] if policy else ['--baseline']
        started=time.monotonic()
        with (a.output/(mode+'.log')).open('w') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        execution=output/'execution.json';record=json.loads(execution.read_text()) if execution.exists() else {}
        report['runs'].append(dict(mode=mode,returncode=result.returncode,process_wall_s=time.monotonic()-started,execution=str(execution.resolve()),accepted=record.get('accepted',False)))
        save()
        if result.returncode:report.update(status='failed',error=f'{mode} execution failed');save();raise RuntimeError(report['error'])
    report.update(status='complete',all_three_accepted=all(x['accepted'] for x in report['runs']));save()


if __name__=='__main__':main()
