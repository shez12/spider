"""E5: start demonstrations from native-reset robot configurations.

Native resets sample q0 = 0.9*default + 0.1*U(lower, upper) for every arm and hand joint. Each
candidate prepends a minimum-jerk joint-space transit from such a sample to the demonstration's
start (object untouched), rejecting transits that touch the table or the object in MuJoCo.
Physical success of the full trajectory is then checked by Isaac teacher replay.
"""
import argparse,hashlib,json
from multiprocessing import Pool
from pathlib import Path
import numpy as np
ROOT=Path('/home/szc/work/spider')


def build(job):
    import mujoco
    folder,traj_path,q_start,steps,out_path=job
    model=mujoco.MjModel.from_xml_path(folder+'/scene.xml');data=mujoco.MjData(model)
    tr=np.load(traj_path);init=tr['initial_state'];q0=init[1:30];obj0=init[30:37]
    s=np.linspace(0,1,steps)[:,None];s=10*s**3-15*s**4+6*s**5   # minimum jerk
    transit=q_start+(q0-q_start)*s
    floor=model.geom('floor').id;objs={i for i in range(model.ngeom) if (model.geom(i).name or '').startswith('right_object')}
    worst=0.;touch=False
    for q in transit:
        data.qpos[:29]=q;data.qpos[29:36]=obj0;mujoco.mj_forward(model,data)
        for k in range(data.ncon):
            c=data.contact[k];g=(c.geom1,c.geom2);names=[model.geom(x).name or '' for x in g]
            robot=[n.startswith(('collision_kuka_','collision_sharpa_')) for n in names]
            if floor in g and any(robot):worst=max(worst,-float(c.dist))
            if any(x in objs for x in g) and any(robot) and c.dist<.005:touch=True
    ok=worst<.001 and not touch
    if ok:
        qpos=np.r_[np.c_[transit,np.repeat(obj0[None],steps,0)],tr['qpos']]
        ctrl=np.r_[transit,tr['ctrl']];qvel=np.r_[np.zeros((steps,tr['qvel'].shape[1])),tr['qvel']]
        new=init.copy();new[1:30]=q_start;new[37:]=0.
        np.savez(out_path,initial_state=new,ctrl=ctrl,qpos=qpos,qvel=qvel,control_dt=tr['control_dt'],physics_dt=tr['physics_dt'])
    return dict(table_pen_m=worst,touches_object=bool(touch),ok=bool(ok))


def main(a):
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'trajectories').mkdir()
    nat=json.loads(a.native_default.read_text());default=np.array(nat['default']);lim=np.array(nat['limits'])
    man=json.loads((a.inputs/'source_manifest.json').read_text());pool={t['candidate_id']:t for t in json.loads((a.inputs/'waypoints.json').read_text())['trajectories']}
    held=set(json.loads((a.teacher/'native_selection.json').read_text())['candidate_ids'])
    rng=np.random.default_rng(a.seed);jobs=[];meta=[];cid=a.id_offset
    for s in a.sources:
        bases=[m for m in man if m['source_index']==s and m['candidate_id'] in held]
        for m in [bases[i] for i in rng.permutation(len(bases))[:a.bases_per_source]]:
            for _ in range(a.starts_per_base):
                q=np.clip(.9*default+.1*rng.uniform(lim[:,0],lim[:,1]),lim[:,0],lim[:,1])
                path=a.output/'trajectories'/f'cand_{cid:06d}.npz';jobs.append((m['folder'],m['trajectory'],q,a.transit_steps,str(path)));meta.append((cid,m,path));cid+=1
    with Pool(a.workers) as p:rows=p.map(build,jobs,chunksize=4)
    manifest=[];waypoints=[];report=[]
    for (cid,m,path),row in zip(meta,rows):
        row.update(candidate_id=cid,base_candidate_id=m['candidate_id'],source_index=m['source_index']);report.append(row)
        if not row['ok']:continue
        waypoints.append(dict(pool[m['candidate_id']],candidate_id=cid,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        manifest.append(dict(m,candidate_id=cid,trajectory=str(path),category='e5_native_robot_start',base_candidate_id=m['candidate_id']))
    (a.output/'source_manifest.json').write_text(json.dumps(manifest,indent=1))
    (a.output/'waypoints.json').write_text(json.dumps(dict(format='e5_native_robot_start_v1',trajectories=waypoints,source=str(a.inputs)),indent=1))
    (a.output/'kinematic_report.json').write_text(json.dumps(report,indent=1))
    print(json.dumps(dict(candidates=len(report),ok=len(manifest),per_source={s:(sum(r['source_index']==s and r['ok'] for r in report),sum(r['source_index']==s for r in report)) for s in a.sources})))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,default=ROOT/'outputs/mustard_classified_native_20261102_inputs')
    p.add_argument('--teacher',type=Path,default=ROOT/'outputs/mustard_classified_20261104_truncated_z2mm_v1')
    p.add_argument('--native-default',type=Path,default=ROOT/'outputs/native_robot_default.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sources',type=int,nargs='+',default=[2,5,9,13,16,10,15,1,18])
    p.add_argument('--bases-per-source',type=int,default=6);p.add_argument('--starts-per-base',type=int,default=8)
    p.add_argument('--transit-steps',type=int,default=150,help='Source-time frames (10 ms) of the transit before retiming.')
    p.add_argument('--id-offset',type=int,default=300000);p.add_argument('--seed',type=int,default=0);p.add_argument('--workers',type=int,default=16)
    main(p.parse_args())
