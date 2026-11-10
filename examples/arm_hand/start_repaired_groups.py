"""Start repaired missing groups without restarting/adversely touching active coordinators."""
import json
import os
from pathlib import Path
import subprocess
import sys
from examples.arm_hand.batch_kuka_gpu import save


def main():
    root=Path(__file__).resolve().parents[2];campaign=root/'outputs/mustard_multisource_valid100'
    groups=json.loads((campaign/'selected_sources.json').read_text());launched=[]
    for g in groups:
        if g['source_index'] not in [7,12,14,20,22]:continue
        source=Path(g['source_dir']);folder=Path(g['result_dir'])
        if not (source/'preparation.json').exists():raise RuntimeError(f'Source not prepared: {source}')
        folder.mkdir(parents=True,exist_ok=True)
        launch=folder/'repair_launcher.json'
        if launch.exists():
            pid=json.loads(launch.read_text())['pid']
            try:
                command=Path(f'/proc/{pid}/cmdline').read_bytes()
                if b'examples.arm_hand.mustard_valid100' in command and str(folder).encode() in command:
                    launched.append(dict(index=g['source_index'],pid=pid,status='already_running'));continue
            except FileNotFoundError:pass
        # The coordinator's flock is authoritative even if another process races this launcher.
        control=dict(physics_workers=1,ik_workers=1,global_gpu_slots=10,global_gpu_slots_dir=str(campaign/'resource_slots/gpu'),
            global_ik_slots=8,global_ik_slots_dir=str(campaign/'resource_slots/ik'),mps_pipe_dir='/tmp/spider-mps-mustard-1005')
        save(folder/'parallelism.json',control)
        command=[sys.executable,'-u','-m','examples.arm_hand.mustard_valid100','--output-dir',str(folder),
            '--source-dir',str(source),'--task',g['task'],'--source-index',str(g['source_index']),
            '--augmentation-seed',str(20261029+g['source_index']),'--ik-workers','1','--physics-workers','1']
        with (folder/'repair_coordinator.log').open('a') as log:
            child=subprocess.Popen(command,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        record=dict(index=g['source_index'],pid=child.pid,command=command,status='launched');save(launch,record);launched.append(record)
    save(campaign/'repaired_group_launches.json',launched);print(json.dumps(launched,indent=2))


if __name__=='__main__':main()
