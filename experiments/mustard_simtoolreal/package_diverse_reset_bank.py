"""Additional airborne/goal-boundary checks before publishing the training bank."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent


def rotate(q,points):
    w=q[:,0:1];v=q[:,1:]
    points=np.broadcast_to(points,(len(q),)+points.shape)
    t=2*np.cross(v[:,None,:],points)
    return points+w[:,None,:]*t+np.cross(v[:,None,:],t)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with np.load(a.source/'reset_bank.npz') as z:data={k:z[k] for k in z.files}
    report=json.loads((a.source/'validation.json').read_text());accepted=[r for r in report['records'] if r['accepted']]
    assert len(accepted)==len(data['kind'])
    with np.load(a.source/'geometry_states.npz') as z:snap={k:z[k] for k in z.files}
    vertices=[]
    for path in (ROOT/'assets/object').glob('right_object_*.obj'):
        if path.name=='right_object_visual.obj':continue
        vertices.extend([list(map(float,line.split()[1:4])) for line in path.read_text().splitlines() if line.startswith('v ')])
    vertices=np.array(vertices);corners=np.array(json.loads((ROOT/'assets/object/calibration.json').read_text())['keypoints_local_m'])
    def bottom(root):return (rotate(root[:,3:7],vertices)[:,:,2]+root[:,None,2]).min(1)
    # Builder snapshots each chunk in proposal/settled/replay_end order.
    replay_indices=[];total=report['proposals'];batch_size=report.get('batch_size',120)
    for row in accepted:
        owner=row['candidate'];chunk=owner//batch_size;local=owner%batch_size;count=min(batch_size,total-chunk*batch_size)
        replay_indices.append(chunk*3*batch_size+2*count+local)
    replay=snap['root'][replay_indices]
    # table_z and table rigid-body Z are box-center heights, not the top face.
    tabletop=data['table'][:,2]+.15
    clearance_start=bottom(data['root'])-tabletop
    clearance_end=bottom(replay)-tabletop
    clearance=np.minimum(clearance_start,clearance_end)
    object_kp=rotate(data['root'][:,3:7],corners)+data['root'][:,None,:3]
    goal_kp=rotate(data['goal'][:,3:7],corners)+data['goal'][:,None,:3]
    distance=np.linalg.norm(object_kp-goal_kp,axis=-1).max(1)
    keep=((data['kind']==1)|(clearance>.02))&(distance>.125)
    keep&=(data['kind']!=1)|(np.maximum(clearance_start,clearance_end)<=.02)
    keep&=(data['kind']!=3)|(distance<.35)
    counts={str(k):int(((data['kind']==k)&keep).sum()) for k in (1,2,3)};assert min(counts.values())>0,counts
    for i,row in enumerate(accepted):
        row['airborne_clearance_m']=float(clearance[i]);row['restored_goal_distance_m']=float(distance[i])
        row['table_top_z_m']=float(tabletop[i])
        if not keep[i]:row['accepted']=False;row['additional_rejection']='airborne_or_goal_boundary'
    for key,value in data.items():
        if value.ndim>0 and len(value)==len(keep):data[key]=value[keep]
    a.output.mkdir(exist_ok=True)
    np.savez_compressed(a.output/'reset_bank.npz',**data)
    report.update(accepted=int(keep.sum()),counts=counts,additional_checks=dict(min_held_mesh_bottom_above_table_m=.02,
        min_reset_goal_keypoint_distance_m=.125,max_near_goal_keypoint_distance_m=.35,removed=int((~keep).sum()),
        max_near_object_mesh_bottom_above_table_m=.02,table_center_to_top_m=.15),
        validated_source=str(a.source.resolve()),bank_sha256=hashlib.sha256((a.output/'reset_bank.npz').read_bytes()).hexdigest())
    (a.output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    (a.output/'completed.json').write_text(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('records',)},indent=2))


if __name__=='__main__':main()
