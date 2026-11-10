"""Export the same twenty native-right KUKA/Sharpa IK trajectories, without dynamics."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

from examples.arm_hand.batch_kuka_gpu import prepare, save, ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/kuka_sharpa_right_batch20_ik')
    args=parser.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    selection=ROOT/'outputs/kuka_sharpa_batch20/selected.json'
    rows=json.loads(selection.read_text());save(out/'selected.json',rows)
    dataset=ROOT/'example_datasets/retarget_full';statuses=[]
    for index,row in enumerate(rows,1):
        folder=out/f'{index:02d}_{row["task"]}';folder.mkdir(exist_ok=True)
        state=dict(index=index,task=row['task'],status='running',folder=str(folder))
        statuses.append(state);save(out/'status.json',statuses)
        print(f'IK CASE {index}/{len(rows)} {row["task"]}',flush=True)
        try:
            # Reuse only completed IK exports; never depend on a live preparation.
            cached=ROOT/'outputs/kuka_sharpa_right_batch20_spider_300'/folder.name
            if not (folder/'reference.npz').exists() and (cached/'retargeting.json').exists():
                for name in ['scene.xml','reference.npz','source_reference.npz','retargeting.json','geometry_report.json']:
                    if (cached/name).exists():shutil.copy2(cached/name,folder/name)
            if not (folder/'reference.npz').exists():
                try:prepare(row,dataset,folder,2)
                except RuntimeError:
                    if not (folder/'retargeting.json').exists():raise
                    # A geometric mismatch must remain visible in the report;
                    # it does not prevent inspecting an IK-only visualization.
            report=json.loads((folder/'retargeting.json').read_text())
            model=mujoco.MjModel.from_xml_path(str(folder/'scene.xml'))
            with np.load(folder/'reference.npz') as ref:
                qpos=np.c_[ref['joint_ik'],ref['object_qpos']]
            qvel=np.zeros((len(qpos),model.nv))
            for i in range(len(qpos)-1):
                mujoco.mj_differentiatePos(model,qvel[i],.01,qpos[i],qpos[i+1])
            qvel[-1]=qvel[-2]
            np.savez(folder/'trajectory_ik.npz',qpos=qpos,qvel=qvel,control_dt=.01,
                     physics_simulated=False,object_from_dataset=True)
            video=folder/'source_vs_right_ik.mp4'
            if not video.exists():
                with (folder/'render.log').open('w') as log:
                    subprocess.run([sys.executable,'-m','examples.arm_hand.record_right_alignment',
                        '--input-dir',str(folder),'--source-scene',str(dataset/row['scene']),
                        '--source-trajectory',str(dataset/row['path']/'trajectory_mjwp.npz')],
                        cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            old=ROOT/'outputs/kuka_sharpa_batch20'/folder.name/'source_trajectory.mp4'
            if old.exists() and not (folder/old.name).exists():shutil.copy2(old,folder/old.name)
            report.update(physics_simulated=False,object_from_dataset=True,reference_mirrored=False,
                          frames=len(qpos),extra_hold=False)
            save(folder/'ik_report.json',report)
            state.update(status='complete',video=str(video),
                         initial_alignment_pass=report['initial_alignment_pass'],
                         table_clearance_pass=report['table_clearance_pass'])
        except Exception as error:state.update(status='failed',error=str(error))
        save(out/'status.json',statuses);print(json.dumps(state),flush=True)
    save(out/'summary.json',dict(completed=sum(s['status']=='complete' for s in statuses),
        failed=sum(s['status']=='failed' for s in statuses),cases=statuses))


if __name__=='__main__':main()
