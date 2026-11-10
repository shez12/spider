"""Recover MPS startup failures without treating them as failed physics tests."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from examples.arm_hand.batch_kuka_gpu import ROOT,save


def requeue(root):
    repaired=[]
    for path in root.glob('**/candidate_*/state.json'):
        state=json.loads(path.read_text())
        if state.get('status')!='physics_failed' or state.get('infrastructure_retries',0)>=2:continue
        log=path.parent/'training.log'
        if 'SIGSEGV'in state.get('error',''):continue
        if not log.exists()or 'Error 807'not in log.read_text()[-4000:]:continue
        state.setdefault('infrastructure_failure_history',[]).append(dict(
            recovered_at_utc=datetime.now(timezone.utc).isoformat(),error=state.get('error'),type='mps_startup_807'))
        state['infrastructure_retries']=state.get('infrastructure_retries',0)+1
        state.pop('validated_success',None);state.pop('error',None)
        state['status']='ik_passed'
        save(path,state);repaired.append(str(path.parent))
    return repaired


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--watch-pilot',action='store_true')
    args=p.parse_args();root=args.root.resolve()
    lock=(root/'infra_recovery.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not args.watch_pilot:
        repaired=requeue(root);save(root/'mps807_recovery_20261030.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),requeued=repaired));print(len(repaired));return
    events=[]
    for attempt in range(2):
        launcher=json.loads((root/'launcher.json').read_text());pid=launcher['pid']
        while True:
            try:
                command=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
                raw=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1]
                if b'examples.arm_hand.run_adaptive_spatial_pilot'not in command or raw.startswith('Z'):break
            except (OSError,ProcessLookupError):break
            time.sleep(30)
        # The original pilot must release its run lock before any requeue/start.
        with (root/'pilot.lock').open('a+')as pilot_lock:
            fcntl.flock(pilot_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            repaired=requeue(root)
        if not repaired:break
        env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl')
        with (root/'pilot.log').open('a')as log:
            process=subprocess.Popen(launcher['command'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        launcher.update(pid=process.pid,infrastructure_resume_at_utc=datetime.now(timezone.utc).isoformat());save(root/'launcher.json',launcher)
        events.append(dict(at_utc=datetime.now(timezone.utc).isoformat(),pid=process.pid,requeued=repaired))
        save(root/'infrastructure_recovery_events.json',events)
    print('Infrastructure recovery watcher complete',flush=True)


if __name__=='__main__':main()
