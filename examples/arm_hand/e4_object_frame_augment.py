"""E4 pilot: move validated demonstrations onto new object start poses with object-frame rigid edits.

For each base trajectory and sampled target, the whole hand-object interaction is rotated about the
object's vertical axis (any yaw) and translated so the object starts at a target XY in the native
range. Hand joints are kept; arm joints are re-solved per frame by damped-least-squares IK so the palm
follows the transformed path. Candidates with IK error or arm/hand-table penetration are rejected
here; physical success is decided afterwards by Isaac teacher replay (train_bc --collect-only).
"""
import argparse,hashlib,json
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
ROOT=Path('/home/szc/work/spider')


def rz(psi):return R.from_euler('z',psi)


def transform(psi,c,d):
    """x -> Rz(psi)(x-c)+c+d on positions; q -> qz(psi) q on orientations (xy-plane edit, z unchanged)."""
    r=rz(psi)
    def pos(p):
        p=np.asarray(p,float).copy();p[...,:2]=r.apply(np.c_[p[...,:2]-c,np.zeros(len(np.atleast_2d(p)))].reshape(-1,3))[:,:2].reshape(p[...,:2].shape)+c+d;return p
    def quat(q):q=np.atleast_2d(q);return (r*R.from_quat(q[:,[1,2,3,0]])).as_quat()[:,[3,0,1,2]]
    return pos,quat


def solve(job):
    import mujoco
    folder,traj_path,psi,target_xy,out_path,max_pos,max_rot=job
    model=mujoco.MjModel.from_xml_path(folder+'/scene.xml');data=mujoco.MjData(model)
    tr=np.load(traj_path);q=tr['qpos'].copy();ctrl=tr['ctrl'].copy();T=len(q)
    site=model.site('sharpa_palm').id;floor=model.geom('floor').id
    c=q[0,29:31].copy();d=np.asarray(target_xy)-c;tp,tq=transform(psi,c,d)
    lo=model.jnt_range[:7,0];hi=model.jnt_range[:7,1]
    # target palm poses from source forward kinematics
    targets=[]
    for t in range(T):
        data.qpos[:]=q[t];mujoco.mj_kinematics(model,data)
        targets.append((tp(data.site_xpos[site][None])[0],R.from_matrix(data.site_xmat[site].reshape(3,3))))
    jacp=np.zeros((3,model.nv));jacr=np.zeros((3,model.nv));arm=q[0,:7].copy();out=q.copy();worst_p=worst_r=pen=0.
    def err(r_t,p_t):
        mujoco.mj_kinematics(model,data)
        ep=p_t-data.site_xpos[site];er=(r_t*R.from_matrix(data.site_xmat[site].reshape(3,3)).inv()).as_rotvec()
        return np.r_[ep,.2*er]  # 0.2 m per rad weighting
    for t,(p_t,r_src) in enumerate(targets):
        r_t=rz(psi)*r_src;data.qpos[:]=q[t];data.qpos[:7]=arm;lam=1e-2
        e=err(r_t,p_t)
        for _ in range(150 if t==0 else 40):
            if np.linalg.norm(e[:3])<5e-4 and np.linalg.norm(e[3:])<.2*2e-3:break
            mujoco.mj_comPos(model,data);mujoco.mj_jacSite(model,data,jacp,jacr,site);J=np.r_[jacp[:,:7],.2*jacr[:,:7]]
            Jp=J.T@np.linalg.inv(J@J.T+lam*np.eye(6));dq=Jp@e
            # 7-DoF redundancy: move toward joint-range centre in the task null space to stay off limits
            dq+=(np.eye(7)-Jp@J)@(.2*((lo+hi)/2-data.qpos[:7])/((hi-lo)/2)**2);n=np.linalg.norm(dq)
            if n>.2:dq*=.2/n
            prev=data.qpos[:7].copy();data.qpos[:7]=np.clip(prev+dq,lo,hi);e_new=err(r_t,p_t)
            # Levenberg-Marquardt: keep only improving steps; adapt damping near the elbow singularity
            if np.linalg.norm(e_new)<np.linalg.norm(e):e=e_new;lam=max(lam*.5,1e-4)
            else:data.qpos[:7]=prev;mujoco.mj_kinematics(model,data);lam=min(lam*4,10.)
        mujoco.mj_kinematics(model,data)
        worst_p=max(worst_p,float(np.linalg.norm(p_t-data.site_xpos[site])))
        worst_r=max(worst_r,float(np.degrees(np.linalg.norm((r_t*R.from_matrix(data.site_xmat[site].reshape(3,3)).inv()).as_rotvec()))))
        arm=data.qpos[:7].copy();out[t,:7]=arm
        out[t,29:32]=tp(q[t,29:32][None])[0];out[t,32:36]=tq(q[t,32:36])[0]
        data.qpos[:]=out[t];mujoco.mj_forward(model,data)
        for k in range(data.ncon):
            con=data.contact[k];g=(con.geom1,con.geom2)
            if floor in g:
                other=model.geom(g[0] if g[1]==floor else g[1]).name or ''
                if other.startswith(('collision_kuka_','collision_sharpa_')):pen=max(pen,-float(con.dist))
    ok=worst_p<=max_pos and worst_r<=max_rot and pen<.001
    row=dict(psi_deg=float(np.degrees(psi)),target_xy=list(map(float,target_xy)),ik_pos_err_m=worst_p,ik_rot_err_deg=worst_r,table_pen_m=pen,kinematic_ok=bool(ok))
    if ok:
        ctrl[:,:7]=out[:,:7];init=tr['initial_state'].copy();init[1:37]=out[0];init[37:]=0.
        # Object start comes from the source initial state (qpos[0] is after one step and sits lower).
        init[30:33]=tp(tr['initial_state'][30:33][None])[0];init[33:37]=tq(tr['initial_state'][33:37])[0]
        np.savez(out_path,initial_state=init,ctrl=ctrl,qpos=out,qvel=np.zeros_like(tr['qvel']),control_dt=tr['control_dt'],physics_dt=tr['physics_dt'])
    return row


def main(a):
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'trajectories').mkdir()
    man=json.loads((a.inputs/'source_manifest.json').read_text());pool={t['candidate_id']:t for t in json.loads((a.inputs/'waypoints.json').read_text())['trajectories']}
    held=set(json.loads((a.teacher/'native_selection.json').read_text())['candidate_ids'])  # bases that hold in Isaac
    rng=np.random.default_rng(a.seed);jobs=[];meta=[];next_id=a.id_offset
    for s in a.sources:
        bases=[m for m in man if m['source_index']==s and m['candidate_id'] in held]
        for m in [bases[i] for i in rng.permutation(len(bases))[:a.bases_per_source]]:
            for _ in range(a.targets_per_base):
                psi=np.radians(rng.uniform(a.yaw_min,a.yaw_max));xy=rng.uniform(-a.xy_range,a.xy_range,2)
                path=a.output/'trajectories'/f'cand_{next_id:06d}.npz'
                jobs.append((m['folder'],m['trajectory'],psi,xy,str(path),a.max_pos,a.max_rot));meta.append((next_id,m,psi,xy,path));next_id+=1
    with Pool(a.workers) as p:rows=p.map(solve,jobs,chunksize=4)
    manifest=[];waypoints=[];report=[]
    for (cid,m,psi,xy,path),row in zip(meta,rows):
        row.update(candidate_id=cid,base_candidate_id=m['candidate_id'],source_index=m['source_index']);report.append(row)
        if not row['kinematic_ok']:continue
        w=pool[m['candidate_id']];tr=np.load(m['trajectory']);c=tr['qpos'][0,29:31];tp,tq=transform(psi,c,np.asarray(xy)-c)
        waypoints.append(dict(candidate_id=cid,initial_pose=list(tp(np.array(w['initial_pose'][:3])[None])[0])+list(tq(w['initial_pose'][3:])[0]),
            pos=tp(np.array(w['pos'])).tolist(),quat_wxyz=tq(np.array(w['quat_wxyz'])).tolist(),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        manifest.append(dict(m,candidate_id=cid,trajectory=str(path),category='e4_object_frame_edit',base_candidate_id=m['candidate_id'],psi_deg=row['psi_deg'],target_xy=row['target_xy']))
    (a.output/'source_manifest.json').write_text(json.dumps(manifest,indent=1))
    (a.output/'waypoints.json').write_text(json.dumps(dict(format='e4_object_frame_edit_v1',trajectories=waypoints,source=str(a.inputs)),indent=1))
    (a.output/'kinematic_report.json').write_text(json.dumps(report,indent=1))
    by={s:(sum(r['source_index']==s and r['kinematic_ok'] for r in report),sum(r['source_index']==s for r in report)) for s in a.sources}
    print(json.dumps(dict(candidates=len(report),kinematic_ok=len(manifest),per_source=by)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,default=ROOT/'outputs/mustard_classified_native_20261102_inputs')
    p.add_argument('--teacher',type=Path,default=ROOT/'outputs/mustard_classified_20261104_truncated_z2mm_v1',help='Selection whose candidates hold in Isaac (bases).')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sources',type=int,nargs='+',default=[2,5,9,13,16,10,15])
    p.add_argument('--bases-per-source',type=int,default=4);p.add_argument('--targets-per-base',type=int,default=24)
    p.add_argument('--yaw-min',type=float,default=-180.);p.add_argument('--yaw-max',type=float,default=180.,help='Yaw window (deg) about the object vertical axis.')
    p.add_argument('--xy-range',type=float,default=.10);p.add_argument('--max-pos',type=float,default=.01);p.add_argument('--max-rot',type=float,default=5.)
    p.add_argument('--id-offset',type=int,default=100000);p.add_argument('--seed',type=int,default=0);p.add_argument('--workers',type=int,default=16)
    main(p.parse_args())
