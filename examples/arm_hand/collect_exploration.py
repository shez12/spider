"""Bounded, resumable exploration campaign; never updates production counts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time
from examples.arm_hand.resource_slots import resource_slot


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('outputs/mustard_multisource_valid100/groups'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--rounds',type=int,default=6);p.add_argument('--samples',type=int,default=64)
    p.add_argument('--worlds',type=int,default=32);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    prefixes=['002_','003_','010_','011_','015_','018_']
    cases=[]
    for prefix in prefixes:
        group=next(g for g in sorted(a.root.iterdir()) if g.name.startswith(prefix))
        states=[(s.parent,json.loads(s.read_text())) for s in sorted(group.glob('candidate_*/state.json')) if (s.parent/'gpu/checkpoint.pt').exists()]
        # Prefer successful demonstrations, preserving a difficult group as well.
        states.sort(key=lambda v:v[1].get('status')!='accepted')
        cases.append(states[0][0].resolve())
    root=Path(__file__).resolve().parents[2]
    def run(case):
        out=a.output.resolve()/case.parent.name/case.name;out.mkdir(parents=True,exist_ok=True)
        result=out/'gpu/result.json';start=time.monotonic()
        if result.exists():return dict(case=str(case),status='already_complete')
        command=[sys.executable,'-u','-m','examples.arm_hand.optimize_kuka_gpu','--input-dir',str(case),
            '--output-dir',str(out/'gpu'),'--samples',str(a.samples),'--max-iterations',str(a.rounds),
            '--disable-target-stop','--disable-plateau-stop','--transition-dir',str(out/'transitions'),
            '--source-group',case.parent.name,'--transition-every','1','--transition-worlds',str(a.worlds),
            '--transition-max-shards',str(a.rounds)]
        if (out/'gpu/checkpoint.pt').exists():command+=['--resume']
        else:command+=['--warm-start',str(case/'gpu/checkpoint.pt')]
        with resource_slot(root/'outputs/mustard_multisource_valid100/resource_slots/gpu',10):
            with (out/'capture.log').open('a') as log:
                proc=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
        record=dict(case=str(case),status='complete' if proc.returncode==0 else 'failed',elapsed_s=time.monotonic()-start,
                    returncode=proc.returncode,shards=len(list((out/'transitions').glob('*.npz'))))
        (out/'capture.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True);return record
    with ThreadPoolExecutor(max_workers=a.workers) as pool:records=list(pool.map(run,cases))
    (a.output/'collection.json').write_text(json.dumps(records,indent=2))
    if any(r['status']=='failed' for r in records):raise SystemExit(1)


if __name__=='__main__':main()
