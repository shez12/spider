"""Extract optimized object poses, recenter them, and map into native coordinates."""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT.parents[1]/'outputs/mustard_sapg_bridge20_v1')
    parser.add_argument('--output',type=Path,default=ROOT/'waypoints.json')
    parser.add_argument('--gap-m',type=float,default=.13)
    parser.add_argument('--expected-count',type=int,default=None)
    args=parser.parse_args()
    calibration=json.loads((ROOT/'assets/object/calibration.json').read_text())
    center=np.asarray(calibration['source_to_new_root_center_in_source_m'])
    corners=np.asarray(calibration['keypoints_local_m'])
    manifest_path=args.source/('accepted.json' if (args.source/'accepted.json').exists() else 'manifest.json')
    manifest=json.loads(manifest_path.read_text());trajectories=[]
    if args.expected_count is not None:assert len(manifest)==args.expected_count,(len(manifest),args.expected_count)
    assert len({i['candidate_id'] for i in manifest})==len(manifest)
    for item in manifest:
        folder=args.source/f"candidate_{item['candidate_id']:05d}"
        path=folder/'gpu/trajectory.npz';before=hashlib.sha256(path.read_bytes()).hexdigest()
        data=np.load(path);raw=data['qpos'][:,-7:]
        nq=data['qpos'].shape[1]
        assert data['initial_state'].shape==(1+nq+data['qvel'].shape[1],)
        initial=data['initial_state'][1+nq-7:1+nq]
        raw=np.concatenate([initial[None],raw])
        rotation=Rotation.from_quat(raw[:,[4,5,6,3]])
        pos=raw[:,:3]+rotation.apply(np.broadcast_to(center,(len(raw),3)))
        pos[:,2]+=.53  # copied native robot/table frame, rather than BC z=0 tabletop
        assert np.isfinite(pos).all() and np.isfinite(raw).all()
        assert np.max(np.abs(np.linalg.norm(raw[:,3:],axis=1)-1))<1e-5
        kp=pos[:,None]+np.einsum('nij,kj->nki',rotation.as_matrix(),corners)
        def distance(a,b):return np.linalg.norm(kp[a]-kp[b],axis=-1).max()
        selected=[];anchor=0
        for index in range(1,len(raw)):
            if distance(anchor,index)>=args.gap_m:
                selected.append(index);anchor=index
        final=len(raw)-1
        if not selected:selected=[final]
        elif selected[-1]!=final:
            if distance(selected[-1],final)>=args.gap_m:selected.append(final)
            else:
                previous=selected[-2] if len(selected)>1 else 0
                if distance(previous,final)>=args.gap_m:selected[-1]=final
        trajectories.append(dict(candidate_id=item['candidate_id'],source_indices=selected,
            initial_pose=pos[0].tolist()+raw[0,3:].tolist(),
            pos=pos[selected].tolist(),quat_wxyz=raw[selected,3:].tolist(),
            final_pose_included=selected[-1]==final,
            source_sha256=before,
            adjacent_max_corner_distances_m=[float(distance(a,b)) for a,b in zip([0]+selected[:-1],selected)]))
        assert hashlib.sha256(path.read_bytes()).hexdigest()==before,'Input changed while extracting poses.'
    payload=dict(format='simtoolreal_object_waypoints_v1',object_id='006_mustard_bottle',
        source=str(args.source.resolve()),poses='optimized object qpos; object root recentered; native +0.53m z',
        extraction_gap_m=args.gap_m,online_skip_margin_m=.01,trajectories=trajectories)
    payload['source_manifest_sha256']=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output),trajectories=len(trajectories),
        counts=[len(t['pos']) for t in trajectories])))


if __name__=='__main__':main()
