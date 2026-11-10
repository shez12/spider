"""Resolve classified executions into immutable, globally identified BC inputs."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/mustard_simtoolreal'))

def prepare(source,output):
    output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((source/'manifest.json').read_text())
    calibration=json.loads((ROOT/'experiments/mustard_simtoolreal/assets/object/calibration.json').read_text())
    center=np.array(calibration['source_to_new_root_center_in_source_m'])
    items=[];goals=[];excluded=[];seen=set()
    for row in manifest:
        if row['category'] not in ('01_strict_and_grasp','02_grasp_warmup'):continue
        # Learned executions can differ from the original candidate: use the
        # execution that was actually accepted whenever one is recorded.
        folder=Path(row['execution'] or row['case']);path=folder/'gpu/trajectory.npz'
        if not path.is_file():raise FileNotFoundError(path)
        if str(path.resolve()) in seen:continue
        seen.add(str(path.resolve()))
        with np.load(path) as d:
            assert d['ctrl'].shape[1]==29 and d['qpos'].shape[1]==36
            assert d['initial_state'].shape[0]>=37
            assert all(np.isfinite(d[k]).all() for k in ('ctrl','qpos','initial_state'))
            raw=np.concatenate([d['initial_state'][30:37][None],d['qpos'][:,-7:]])
        rotation=Rotation.from_quat(raw[:,[4,5,6,3]])
        pos=raw[:,:3]+rotation.apply(np.broadcast_to(center,(len(raw),3)));pos[:,2]+=.53
        candidate=len(items)
        item=dict(candidate_id=candidate,folder=str(folder),trajectory=str(path.resolve()),
            source_index=row['source_index'],category=row['category'],original_case=row['case'],
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        items.append(item)
        # PairedTrajectoryResets expects two slots; formal reaching uses final slot.
        goals.append(dict(candidate_id=candidate,initial_pose=pos[0].tolist()+raw[0,3:].tolist(),
            pos=pos[[-1,-1]].tolist(),quat_wxyz=raw[[-1,-1],3:].tolist(),source_sha256=item['source_sha256']))
    def save(name,value):(output/name).write_text(json.dumps(value,indent=2)+'\n')
    save('source_manifest.json',items)
    save('waypoints.json',dict(format='simtoolreal_object_waypoints_v1',trajectories=goals,source=str(source)))
    save('preparation.json',dict(count=len(items),classified_manifest_sha256=hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),excluded_category='03_strict_without_verified_grasp'))
    print(json.dumps(dict(trajectories=len(items),output=str(output))),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();prepare(a.source.resolve(),a.output.resolve())
