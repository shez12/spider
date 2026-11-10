"""E6: change the approach direction relative to the object (DemoGrasp-style edit, random search + physics filter).

The object start is rotated about its vertical axis by any yaw psi_o and translated; the hand path is
rotated by a separate yaw psi_h inside the arm's feasible window (both about the object's start
centre, same translation). The grasp therefore lands on a different side of the object by
psi_o - psi_h. Arm joints are re-solved by IK (e4 solver); hand joints are unchanged. The goal is the
object carried rigidly by the edited hand after the grasp. Isaac teacher replay decides success.
"""
import argparse,hashlib,json
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
import e4_object_frame_augment as e4
ROOT=Path('/home/szc/work/spider')


def mat(p,r):
    m=np.eye(4);m[:3,:3]=r.as_matrix();m[:3,3]=p;return m


def solve(job):
    import mujoco
    folder,traj,psi_o,psi_h,xy,out_path=job
    tr=np.load(traj);q0=tr['qpos'][0];c=q0[29:31].copy();d=np.asarray(xy)-c
    # hand edit via the e4 solver (it moves object and palm with psi_h); then overwrite the object start with psi_o
    row=e4.solve((folder,traj,psi_h,xy,out_path,.01,5.))
    row.update(psi_o_deg=float(np.degrees(psi_o)),psi_h_deg=float(np.degrees(psi_h)),relative_deg=float(np.degrees((psi_o-psi_h+np.pi)%(2*np.pi)-np.pi)))
    if not row['kinematic_ok']:return row
    tp,tq=e4.transform(psi_o,c,d)
    z=dict(np.load(out_path));init=z['initial_state'];src=tr['initial_state']
    init[30:33]=tp(src[30:33][None])[0];init[33:37]=tq(src[33:37])[0]
    z['qpos'][:,29:32]=init[30:33];z['qpos'][:,32:36]=init[33:37]
    # object must not start inside the robot: check initial contacts in MuJoCo
    model=mujoco.MjModel.from_xml_path(folder+'/scene.xml');data=mujoco.MjData(model);data.qpos[:]=init[1:37];mujoco.mj_forward(model,data)
    objs={i for i in range(model.ngeom) if (model.geom(i).name or '').startswith('right_object')}
    if any((c_.geom1 in objs or c_.geom2 in objs) and c_.dist<-.001 for c_ in data.contact[:data.ncon]):row.update(kinematic_ok=False,reason='object_overlaps_robot_at_start');return row
    # carried-object goal: final palm pose relative to start, applied to the edited object start
    def palm(qpos):
        data.qpos[:]=qpos;mujoco.mj_kinematics(model,data);s=model.site('sharpa_palm').id
        return mat(data.site_xpos[s].copy(),R.from_matrix(data.site_xmat[s].reshape(3,3)))
    H0=palm(z['qpos'][0]);HT=palm(z['qpos'][-1])
    O0=mat(init[30:33],R.from_quat(init[33:37][[1,2,3,0]]));OT=HT@np.linalg.inv(H0)@O0
    row['goal']=list(OT[:3,3])+list(R.from_matrix(OT[:3,:3]).as_quat()[[3,0,1,2]])
    np.savez(out_path,**z);return row


def main(a):
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'trajectories').mkdir()
    man=json.loads((a.inputs/'source_manifest.json').read_text());pool={t['candidate_id']:t for t in json.loads((a.inputs/'waypoints.json').read_text())['trajectories']}
    held=set(json.loads((a.teacher/'native_selection.json').read_text())['candidate_ids'])
    rng=np.random.default_rng(a.seed);jobs=[];meta=[];cid=a.id_offset
    for s in a.sources:
        bases=[m for m in man if m['source_index']==s and m['candidate_id'] in held and Path(m['folder'],'scene.xml').exists()]
        for m in [bases[i] for i in rng.permutation(len(bases))[:a.bases_per_source]]:
            for _ in range(a.edits_per_base):
                psi_o=rng.uniform(-np.pi,np.pi);psi_h=np.radians(rng.uniform(a.hand_yaw_min,a.hand_yaw_max));xy=rng.uniform(-a.xy_range,a.xy_range,2)
                path=a.output/'trajectories'/f'cand_{cid:06d}.npz';jobs.append((m['folder'],m['trajectory'],psi_o,psi_h,xy,str(path)));meta.append((cid,m,path));cid+=1
    with Pool(a.workers) as p:rows=p.map(solve,jobs,chunksize=4)
    manifest=[];waypoints=[];report=[]
    for (cid,m,path),row in zip(meta,rows):
        row.update(candidate_id=cid,base_candidate_id=m['candidate_id'],source_index=m['source_index'])
        if row['kinematic_ok']:
            z=np.load(path);obj=z['initial_state'][30:37];goal=row.pop('goal')
            waypoints.append(dict(candidate_id=cid,initial_pose=list(map(float,obj)),pos=[list(map(float,obj[:3])),goal[:3]],quat_wxyz=[list(map(float,obj[3:])),goal[3:]],
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            manifest.append(dict(m,candidate_id=cid,trajectory=str(path),category='e6_approach_edit',base_candidate_id=m['candidate_id'],
                psi_o_deg=row['psi_o_deg'],psi_h_deg=row['psi_h_deg'],relative_deg=row['relative_deg']))
        report.append(row)
    (a.output/'source_manifest.json').write_text(json.dumps(manifest,indent=1))
    (a.output/'waypoints.json').write_text(json.dumps(dict(format='e6_approach_edit_v1',trajectories=waypoints,source=str(a.inputs)),indent=1))
    (a.output/'kinematic_report.json').write_text(json.dumps(report,indent=1))
    print(json.dumps(dict(candidates=len(report),kinematic_ok=len(manifest))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,default=ROOT/'outputs/mustard_classified_native_20261102_inputs')
    p.add_argument('--teacher',type=Path,default=ROOT/'outputs/mustard_classified_20261104_truncated_z2mm_v1')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sources',type=int,nargs='+',default=[2,5,9,13,16,10,15])
    p.add_argument('--bases-per-source',type=int,default=6);p.add_argument('--edits-per-base',type=int,default=20)
    p.add_argument('--hand-yaw-min',type=float,default=-45.);p.add_argument('--hand-yaw-max',type=float,default=0.)
    p.add_argument('--xy-range',type=float,default=.08)
    p.add_argument('--id-offset',type=int,default=500000);p.add_argument('--seed',type=int,default=0);p.add_argument('--workers',type=int,default=16)
    main(p.parse_args())
