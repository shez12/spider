"""Freeze accepted examples and test geometry + the real SimToolReal action code on CPU.

This is a bridge acceptance test, not an Isaac Sim rollout or SAPG training run.
Source data is read-only. Output must be a new directory.
"""
from pathlib import Path
from types import SimpleNamespace as NS
import argparse
import hashlib
import importlib.util
import json
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch
import trimesh


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def rotation(q):
    r = np.empty(9)
    mujoco.mju_quat2Mat(r, np.asarray(q, dtype=np.float64))
    return r.reshape(3, 3)


def angle(a, b):
    return float(np.degrees(2*np.arccos(np.clip(abs(np.dot(a, b)), 0, 1))))


def extract_object(model):
    bid = model.body('right_object').id
    meshes = []
    for gid in np.flatnonzero(model.geom_bodyid == bid):
        mid = model.geom_dataid[gid]
        assert mid >= 0
        v = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid]+model.mesh_vertnum[mid]].copy()
        v = v @ rotation(model.geom_quat[gid]).T + model.geom_pos[gid]
        f = model.mesh_face[model.mesh_faceadr[mid]:model.mesh_faceadr[mid]+model.mesh_facenum[mid]].copy()
        meshes.append((model.geom(gid).name, trimesh.Trimesh(v, f, process=False)))
    return bid, meshes


def export_object(model, out):
    out.mkdir()
    bid, meshes = extract_object(model)
    visual = next(mesh for name, mesh in meshes if name.endswith('_visual'))
    bounds = visual.bounds
    center = bounds.mean(0)
    dims = bounds[1]-bounds[0]
    corners = np.array([[1,1,1],[1,1,-1],[-1,-1,1],[-1,-1,-1]])
    offsets = corners * dims * 1.5 / 2
    inertial_r = rotation(model.body_iquat[bid])
    inertia = inertial_r @ np.diag(model.body_inertia[bid]) @ inertial_r.T
    root = ET.Element('robot', name='mustard_bottle')
    link = ET.SubElement(root, 'link', name='object_root')
    inertial = ET.SubElement(link, 'inertial')
    ET.SubElement(inertial, 'origin', xyz=' '.join(map(str, model.body_ipos[bid]-center)), rpy='0 0 0')
    ET.SubElement(inertial, 'mass', value=str(model.body_mass[bid]))
    ET.SubElement(inertial, 'inertia', **{key: str(inertia[i,j]) for key,i,j in [('ixx',0,0),('iyy',1,1),('izz',2,2),('ixy',0,1),('ixz',0,2),('iyz',1,2)]})
    for name, mesh in meshes:
        shifted = mesh.copy()
        shifted.vertices -= center
        shifted.export(out/f'{name}.obj')
        tag = 'visual' if name.endswith('_visual') else 'collision'
        elem = ET.SubElement(link, tag, name=name)
        geo = ET.SubElement(elem, 'geometry')
        ET.SubElement(geo, 'mesh', filename=f'{name}.obj')
    ET.indent(root)
    ET.ElementTree(root).write(out/'mustard_bottle.urdf', encoding='unicode')
    meta = dict(object_id='006_mustard_bottle', bbox_choice='whole visual mesh AABB, source axes; provisional grasp box',
                bbox_dims_m=dims.tolist(), source_to_new_root_center_in_source_m=center.tolist(),
                policy_object_scale=(dims/.04).tolist(), keypoint_scale=1.5,
                keypoints_local_m=offsets.tolist(), mass_kg=float(model.body_mass[bid]),
                inertia_eigenvalues=np.linalg.eigvalsh(inertia).tolist(),
                collision_mesh_count=len(meshes)-1, usd_converted=False)
    save(out/'calibration.json', meta)
    # Self-contained orthographic SVG: source origin vs proposed new root and virtual points.
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1050" height="420" viewBox="0 0 1050 420">',
         '<rect width="1050" height="420" fill="white"/>',
         '<text x="25" y="27" font-family="sans-serif" font-size="18">Mustard bottle: whole-object bbox candidate / SimToolReal virtual keypoints</text>']
    vertices=np.asarray(visual.vertices)-center
    for panel,(i,j,label) in enumerate([(0,1,'X-Y'),(0,2,'X-Z'),(1,2,'Y-Z')]):
        scale=260/float(max(dims)*1.7)
        cx,cy=175+350*panel,210
        xy=lambda p:(cx+p[i]*scale,cy-p[j]*scale)
        for v in vertices[::max(1,len(vertices)//1800)]:
            x,y=xy(v);svg.append(f'<circle cx="{x}" cy="{y}" r="0.7" fill="#84929d"/>')
        x,y=xy(np.array([-dims[0]/2,dims[1]/2,dims[2]/2]))
        svg.append(f'<rect x="{cx-dims[i]*scale/2}" y="{cy-dims[j]*scale/2}" width="{dims[i]*scale}" height="{dims[j]*scale}" fill="none" stroke="#287d65"/>')
        for k,v in enumerate(offsets):
            x,y=xy(v);svg.append(f'<circle cx="{x}" cy="{y}" r="4" fill="#bd493c"/><text x="{x+5}" y="{y-5}" font-size="12">{k}</text>')
        ox,oy=xy(-center)
        svg.append(f'<circle cx="{ox}" cy="{oy}" r="4" fill="#222"/><path d="M {cx-5} {cy} h 10 M {cx} {cy-5} v 10" stroke="#287d65"/><text x="{cx-35}" y="365" font-family="sans-serif" font-size="15">{label} (m)</text>')
    svg.append('<text x="25" y="400" font-family="sans-serif" font-size="13">Gray: visual vertices; green: bbox/new root; black: old root; red: 4 virtual points (scale 1.5, projections may overlap).</text></svg>')
    (out/'calibration.svg').write_text(''.join(svg))
    return center, offsets, meta


def action_env(model, initial_q):
    limits=model.jnt_range[model.actuator_trnid[:,0]]
    t=lambda x:torch.tensor(x,dtype=torch.float64)
    return NS(cfg=NS(domain_randomization=NS(use_action_delay=False),
                     action=NS(dof_speed_scale=1.5,arm_moving_average=.1,hand_moving_average=.1)),
              step_dt=1/60,device='cpu',_perm_canon_to_lab=torch.arange(model.nu),
              _arm_joint_ids=torch.arange(7),_hand_joint_ids=torch.arange(7,model.nu),
              _arm_lower=t(limits[:7,0]),_arm_upper=t(limits[:7,1]),
              _hand_lower=t(limits[7:,0]),_hand_upper=t(limits[7:,1]),
              _prev_targets=t(initial_q[None]),_cur_targets=t(initial_q[None]))


def inverse(env, desired):
    prev=env._prev_targets.numpy()[0]
    a=np.empty_like(desired)
    a[:7]=(desired[:7]-prev[:7])/(.1*1.5/60)
    raw=(desired[7:]-.9*prev[7:])/.1
    lo,hi=env._hand_lower.numpy(),env._hand_upper.numpy()
    a[7:]=2*(raw-lo)/(hi-lo)-1
    return a


def rollout(model, initial, ctrl, dt):
    data=mujoco.MjData(model)
    mujoco.mj_setState(model,data,initial,mujoco.mjtState.mjSTATE_FULLPHYSICS)
    mujoco.mj_forward(model,data)
    # Retain source physics timestep (500 Hz); exact ZOH control timing via floor.
    steps=int(round(len(ctrl)*dt/model.opt.timestep))
    for step in range(steps):
        idx=min(int((step*model.opt.timestep+1e-10)/dt),len(ctrl)-1)
        data.ctrl[:]=ctrl[idx]
        mujoco.mj_step(model,data)
    return data.qpos[-7:].copy()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=Path('outputs/mustard_spatial_valid100'))
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--simtoolreal',type=Path,default=Path('/home/szc/work/simtoolreal_origin/simtoolreal'))
    args=ap.parse_args()
    assert not args.output.exists(), 'Use a new output path to preserve the frozen run.'
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    mod_path=args.simtoolreal/'isaacsimenvs/tasks/simtoolreal/utils/action_utils.py'
    spec=importlib.util.spec_from_file_location('simtoolreal_action_pipeline',mod_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    accepted=json.loads((args.source/'accepted.json').read_text())
    selected=sorted(accepted,key=lambda x:x['candidate_id'])[:20]
    assert len(selected)==20
    manifest=[];rows=[];object_hashes=set();task_ids=set();lineages=set()
    center=offsets=None
    for k,item in enumerate(selected):
        src=Path(item['folder']);state=json.loads((src/'state.json').read_text())
        assert state['validated_success']
        dest=args.output/src.name;dest.mkdir()
        files=['scene.xml','reference.npz','augmentation.json','state.json','gpu/trajectory.npz','gpu/replay_trajectory.npz','gpu/result.json','gpu/tracking_collision_report.json','replay_audit/tracking_collision_report.json']
        hashes={}
        for rel in files:
            old=sha(src/rel);target=dest/rel;target.parent.mkdir(exist_ok=True)
            shutil.copy2(src/rel,target)
            assert old==sha(target)==sha(src/rel),f'File changed during snapshot: {src/rel}'
            hashes[rel]=old
        model=mujoco.MjModel.from_xml_path(str(dest/'scene.xml'))
        bid,meshes=extract_object(model)
        geom_hash=hashlib.sha256(b''.join(mesh.vertices.tobytes()+mesh.faces.tobytes() for _,mesh in meshes)).hexdigest()
        object_hashes.add(geom_hash)
        if center is None:center,offsets,meta=export_object(model,args.output/'object_asset')
        aug=json.loads((dest/'augmentation.json').read_text());task_ids.add(aug['source_task']);lineages.add(aug['source_reference_sha256'])
        d=np.load(dest/'gpu/trajectory.npz');ref=np.load(dest/'reference.npz')
        ctrl=d['ctrl'];dt=float(d['control_dt']);initial=d['initial_state']
        data=mujoco.MjData(model);mujoco.mj_setState(model,data,initial,mujoco.mjtState.mjSTATE_FULLPHYSICS)
        jointids=model.actuator_trnid[:,0];q0=data.qpos[model.jnt_qposadr[jointids]].copy()
        env=action_env(model,q0)
        ts=np.arange(int(np.floor(len(ctrl)*dt*60)))/60
        desired=np.stack([np.interp(ts,np.arange(len(ctrl))*dt,ctrl[:,j]) for j in range(model.nu)],axis=1)
        actual=[];raw_actions=[];ideal_actions=[]
        for target in desired:
            ideal_actions.append(inverse(env,target))
            # Feasibility measured against exactly desired preceding targets, no clipping drift.
            env._prev_targets=torch.tensor(target[None],dtype=torch.float64)
        env=action_env(model,q0)
        for target in desired:
            a=inverse(env,target);raw_actions.append(a)
            module.apply_action_pipeline(env,torch.tensor(np.clip(a,-1,1)[None],dtype=torch.float64))
            actual.append(env._cur_targets.numpy()[0].copy())
        actual=np.array(actual);ideal_actions=np.array(ideal_actions)
        # Independent forward/inverse consistency test with feasible small arm actions.
        test_env=action_env(model,q0);rng=np.random.default_rng(0)
        probe=rng.uniform(-.1,.1,model.nu);probe[7:]=0
        module.apply_action_pipeline(test_env,torch.tensor(probe[None],dtype=torch.float64))
        recovered=inverse(action_env(model,q0),test_env._cur_targets.numpy()[0])
        roundtrip=float(np.max(np.abs(recovered-probe)))
        assert roundtrip<1e-9
        # Coordinate-frame invariant on every saved object pose, including goal.
        max_corner_error=0.
        for pose in np.concatenate([d['qpos'][:,-7:],ref['object_qpos'][-1:]]):
            r=rotation(pose[3:]);newp=pose[:3]+r@center
            oldworld=(offsets+center)@r.T+pose[:3]
            newworld=offsets@r.T+newp
            max_corner_error=max(max_corner_error,float(np.abs(oldworld-newworld).max()))
        assert max_corner_error<1e-12
        outcomes={}
        goal=ref['object_qpos'][-1]
        for name,commands,period in [('source_100hz',ctrl,dt),('direct_target_60hz',desired,1/60),('simtoolreal_pipeline_60hz',actual,1/60)]:
            final=rollout(model,initial,commands,period)
            outcomes[name]=dict(final_position_error_m=float(np.linalg.norm(final[:3]-goal[:3])),final_orientation_error_deg=angle(final[3:],goal[3:]),final_pose=final.tolist())
        row=dict(candidate_id=item['candidate_id'],source_frames=len(ctrl),target_frames=len(ts),duration_s=len(ctrl)*dt,
                 arm_unreachable_fraction=float((np.abs(ideal_actions[:,:7])>1+1e-7).mean()),
                 hand_unreachable_fraction=float((np.abs(ideal_actions[:,7:])>1+1e-7).mean()),
                 max_arm_action_required=float(np.abs(ideal_actions[:,:7]).max()),
                 target_tracking_rmse_rad=float(np.sqrt(np.mean((actual-desired)**2))),
                 forward_inverse_error=roundtrip,corner_frame_error_m=max_corner_error,cpu_rollouts=outcomes)
        np.savez(dest/'bridge_diagnostic.npz',desired_targets_60hz=desired,clipped_pipeline_targets_60hz=actual,unclipped_inverse_actions=ideal_actions,actual_feedback_actions=np.clip(raw_actions,-1,1),time_s=ts)
        save(dest/'bridge_report.json',row);rows.append(row)
        manifest.append(dict(candidate_id=item['candidate_id'],source=str(src),hashes=hashes,geometry_sha256=geom_hash,split='validation' if k%5==4 else 'train'))
        print(json.dumps(row),flush=True)
    assert len(object_hashes)==len(task_ids)==len(lineages)==1
    report=dict(stage='CPU bridge acceptance; not Isaac Sim or SAPG training',selected_count=20,unique_objects=len(object_hashes),unique_source_episodes=len(lineages),accepted_count_at_snapshot=len(accepted),
                action_pipeline_sha256=sha(mod_path),action_parameters=dict(arm_ema=.1,hand_ema=.1,dof_speed_scale=1.5,policy_hz=60),
                object=meta,rows=rows,training_gate_pass=all(r['arm_unreachable_fraction']==0 and r['hand_unreachable_fraction']==0 for r in rows),
                caveats=['CPU MuJoCo dynamics differ from original Warp and target PhysX.','Action test uses right-hand joint limits and identity order, not left-hand RobotSpec.','Clipped action rollouts are diagnostics, not accepted training labels.','All trajectories share one original episode; split only measures spatial variant interpolation.','Whole-object AABB is a provisional grasp box, not manually reviewed grasp-region calibration.'])
    save(args.output/'manifest.json',manifest);save(args.output/'report.json',report)
    print('REPORT',args.output/'report.json',flush=True)


if __name__=='__main__':
    main()
