"""Synchronized, isolated multi-process rollout throughput test (optionally MPS)."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from examples.arm_hand.batch_kuka_gpu import ROOT,save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=10)
    parser.add_argument('--repeats',type=int,default=20)
    parser.add_argument('--controls',type=int,default=6)
    parser.add_argument('--mps-pipe-dir',type=Path)
    args=parser.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    barrier=out/'start.signal'
    if barrier.exists():barrier.unlink()
    source=ROOT/'outputs/mustard_spatial_valid100/candidate_00004'
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    if args.mps_pipe_dir:env['CUDA_MPS_PIPE_DIRECTORY']=str(args.mps_pipe_dir)
    jobs=[];logs=[];start=time.monotonic()
    try:
        for i in range(args.workers):
            folder=out/f'worker_{i:02d}';folder.mkdir(exist_ok=True)
            ready=folder/'profile_ready.json'
            if ready.exists():ready.unlink()
            log=(folder/'run.log').open('w');logs.append(log)
            command=[sys.executable,'-u','-m','examples.arm_hand.optimize_kuka_gpu',
                '--input-dir',str(source),'--output-dir',str(folder),'--samples','512',
                '--profile-rollout','--profile-controls',str(args.controls),'--profile-repeats',str(args.repeats),
                '--profile-barrier',str(barrier),'--warm-start',str(source/'gpu/checkpoint.pt')]
            jobs.append(subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT))
        while not all((out/f'worker_{i:02d}/profile_ready.json').exists() for i in range(args.workers)):
            if any(job.poll() is not None for job in jobs):raise RuntimeError('A worker exited before readiness; inspect logs')
            if time.monotonic()-start>240:raise TimeoutError('Workers did not initialize within 240 seconds')
            time.sleep(.2)
        initialized=time.monotonic();barrier.write_text('start\n')
        for job in jobs:
            if job.wait(timeout=240)!=0:raise RuntimeError('Rollout worker failed')
        elapsed=time.monotonic()-initialized
        reports=[json.loads((out/f'worker_{i:02d}/profile.json').read_text()) for i in range(args.workers)]
        summary=dict(workers=args.workers,repeats=args.repeats,controls=args.controls,
            samples_per_worker=512,mps_pipe_dir=str(args.mps_pipe_dir) if args.mps_pipe_dir else None,
            initialization_wall_s=initialized-start,measurement_wall_s=elapsed,
            aggregate_control_frames_per_s=args.workers*args.repeats*args.controls/elapsed,
            all_finite=all(r['cost_finite'] for r in reports),workers_results=reports,
            completed_at_utc=datetime.now(timezone.utc).isoformat())
        save(out/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
    finally:
        for job in jobs:
            if job.poll() is None:job.terminate()
        for log in logs:log.close()


if __name__=='__main__':main()
