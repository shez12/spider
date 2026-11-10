"""Export the compiled source right-arm/hand into a self-contained URDF for Isaac Sim."""
from pathlib import Path
import argparse
import json
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import trimesh
from scipy.spatial.transform import Rotation
from examples.arm_hand.test_mustard_simtoolreal_bridge import rotation,save


def fmt(x):return ' '.join(f'{float(v):.12g}' for v in x)


def mesh_for_geom(m,g):
    typ=int(m.geom_type[g]);size=m.geom_size[g]
    if typ==mujoco.mjtGeom.mjGEOM_MESH:
        mid=m.geom_dataid[g]
        v=m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]].copy()
        f=m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid]+m.mesh_facenum[mid]].copy()
        mesh=trimesh.Trimesh(v,f,process=False)
    elif typ==mujoco.mjtGeom.mjGEOM_BOX:mesh=trimesh.creation.box(2*size)
    elif typ==mujoco.mjtGeom.mjGEOM_CYLINDER:mesh=trimesh.creation.cylinder(size[0],2*size[1],sections=24)
    elif typ==mujoco.mjtGeom.mjGEOM_CAPSULE:
        mesh=trimesh.creation.capsule(radius=size[0],height=2*size[1],count=[12,12])
        mesh.vertices[:,2]-=mesh.bounds[:,2].mean()
    elif typ==mujoco.mjtGeom.mjGEOM_SPHERE:mesh=trimesh.creation.icosphere(subdivisions=2,radius=size[0])
    else:raise ValueError(f'Unhandled geom type {typ}')
    mesh.vertices=mesh.vertices@rotation(m.geom_quat[g]).T+m.geom_pos[g]
    return mesh


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--frozen',type=Path,default=Path('outputs/mustard_sapg_bridge20_v1'))
    ap.add_argument('--output',type=Path,default=Path('outputs/mustard_isaac_assets_v1'))
    args=ap.parse_args();assert not args.output.exists(),'Use a fresh asset version.'
    out=args.output.resolve();out.mkdir(parents=True);(out/'meshes').mkdir()
    manifest=json.loads((args.frozen/'manifest.json').read_text());first=args.frozen/f"candidate_{manifest[0]['candidate_id']:05d}"
    m=mujoco.MjModel.from_xml_path(str(first/'scene.xml'));rootid=m.body('kuka_base').id
    bids=[rootid]
    for i in range(rootid+1,m.nbody):
        if m.body_parentid[i] in bids:bids.append(i)
    robot=ET.Element('robot',name='kuka_sharpa_right')
    for bid in bids:
        name=m.body(bid).name;link=ET.SubElement(robot,'link',name=name)
        inertial=ET.SubElement(link,'inertial')
        ET.SubElement(inertial,'origin',xyz=fmt(m.body_ipos[bid]),rpy=fmt(Rotation.from_matrix(rotation(m.body_iquat[bid])).as_euler('xyz')))
        ET.SubElement(inertial,'mass',value=str(m.body_mass[bid]))
        ix,iy,iz=m.body_inertia[bid]
        ET.SubElement(inertial,'inertia',ixx=str(ix),iyy=str(iy),izz=str(iz),ixy='0',ixz='0',iyz='0')
        if bid!=rootid:
            jids=np.flatnonzero(m.jnt_bodyid==bid)
            assert len(jids)<=1
            jid=int(jids[0]) if len(jids) else None
            if jid is not None:
                assert m.jnt_type[jid]==mujoco.mjtJoint.mjJNT_HINGE and np.max(np.abs(m.jnt_pos[jid]))<1e-9
            joint=ET.SubElement(robot,'joint',name=m.joint(jid).name if jid is not None else f'fixed_{name}',type='revolute' if jid is not None else 'fixed')
            ET.SubElement(joint,'parent',link=m.body(int(m.body_parentid[bid])).name);ET.SubElement(joint,'child',link=name)
            ET.SubElement(joint,'origin',xyz=fmt(m.body_pos[bid]),rpy=fmt(Rotation.from_matrix(rotation(m.body_quat[bid])).as_euler('xyz')))
            if jid is not None:
                aid=int(np.flatnonzero(m.actuator_trnid[:,0]==jid)[0])
                ET.SubElement(joint,'axis',xyz=fmt(m.jnt_axis[jid]))
                effort=float(max(abs(m.actuator_forcerange[aid]))) if m.actuator_forcelimited[aid] else 1e6
                ET.SubElement(joint,'limit',lower=str(m.jnt_range[jid,0]),upper=str(m.jnt_range[jid,1]),effort=str(effort),velocity='10')
                ET.SubElement(joint,'dynamics',damping=str(m.dof_damping[m.jnt_dofadr[jid]]),friction=str(m.dof_frictionloss[m.jnt_dofadr[jid]]))
        for g in np.flatnonzero(m.geom_bodyid==bid):
            mesh=mesh_for_geom(m,g);filename=f'meshes/geom_{g:03d}.obj';mesh.export(out/filename)
            visual=ET.SubElement(link,'visual',name=f'visual_{g}')
            ET.SubElement(ET.SubElement(visual,'geometry'),'mesh',filename=filename)
            ET.SubElement(ET.SubElement(visual,'material',name=f'material_{g}'),'color',rgba=fmt(m.geom_rgba[g]))
            if m.geom(g).name.startswith('collision_') or m.geom_group[g]==3:
                ET.SubElement(ET.SubElement(ET.SubElement(link,'collision',name=f'collision_{g}'),'geometry'),'mesh',filename=filename)
    ET.indent(robot);ET.ElementTree(robot).write(out/'kuka_sharpa_right.urdf',encoding='unicode')
    # Table geometry is the same source box, expressed about a root on its top surface.
    table=ET.Element('robot',name='table');link=ET.SubElement(table,'link',name='object_root');g=m.geom('floor')
    for tag in ['visual','collision']:
        elem=ET.SubElement(link,tag);ET.SubElement(elem,'origin',xyz=fmt(g.pos),rpy='0 0 0');ET.SubElement(ET.SubElement(elem,'geometry'),'box',size=fmt(2*g.size))
    inertial=ET.SubElement(link,'inertial');ET.SubElement(inertial,'mass',value='1');ET.SubElement(inertial,'inertia',ixx='.01',iyy='.01',izz='.01',ixy='0',ixz='0',iyz='0')
    ET.indent(table);ET.ElementTree(table).write(out/'table.urdf',encoding='unicode')
    jointids=m.actuator_trnid[:,0]
    joints=[m.joint(int(j)).name for j in jointids]
    tips=['sharpa_index_tip','sharpa_middle_tip','sharpa_ring_tip','sharpa_thumb_tip','right_pinky_fingertip']
    palm=m.site('sharpa_palm')
    params=dict(robot_urdf=str(out/'kuka_sharpa_right.urdf'),table_urdf=str(out/'table.urdf'),source_scene=str(first/'scene.xml'),
                joint_names=joints,base_pos=m.body_pos[rootid].tolist(),base_quat=m.body_quat[rootid].tolist(),
                kp=m.actuator_gainprm[:,0].tolist(),kd=(-m.actuator_biasprm[:,2]).tolist(),
                joint_armature=m.dof_armature[m.jnt_dofadr[jointids]].tolist(),
                joint_friction=m.dof_frictionloss[m.jnt_dofadr[jointids]].tolist(),
                effort=np.where(m.actuator_forcelimited,np.max(np.abs(m.actuator_forcerange),axis=1),1e6).tolist(),
                source_actuator_force_limited=m.actuator_forcelimited.astype(bool).tolist(),
                unlimited_effort_representation='1e6 Nm cap for MuJoCo actuators whose force limits are disabled; zero range is not zero effort',
                palm_body=m.body(int(palm.bodyid[0])).name,palm_offset=palm.pos.tolist(),
                fingertip_bodies=[m.body(int(m.site(n).bodyid[0])).name for n in tips],
                fingertip_offsets=[m.site(n).pos.tolist() for n in tips],
                source_bodies=[m.body(b).name for b in bids],
                exclusions=[[m.body(int(sig>>16)).name,m.body(int(sig&65535)).name] for sig in m.exclude_signature])
    save(out/'robot_spec.json',params)
    calibration=json.loads((args.frozen/'object_asset/calibration.json').read_text())
    center=np.array(calibration['source_to_new_root_center_in_source_m'])
    initial_joints=[];initial_velocities=[];initial_objects=[];initial_object_velocities=[];goal_objects=[]
    for item in manifest:
        folder=args.frozen/f"candidate_{item['candidate_id']:05d}"
        model=mujoco.MjModel.from_xml_path(str(folder/'scene.xml'));data=mujoco.MjData(model)
        trajectory=np.load(folder/'gpu/trajectory.npz');ref=np.load(folder/'reference.npz')
        mujoco.mj_setState(model,data,trajectory['initial_state'],mujoco.mjtState.mjSTATE_FULLPHYSICS)
        mujoco.mj_forward(model,data)
        jid=model.actuator_trnid[:,0]
        initial_joints.append(data.qpos[model.jnt_qposadr[jid]].copy())
        initial_velocities.append(data.qvel[model.jnt_dofadr[jid]].copy())
        start=data.qpos[-7:].copy();goal=ref['object_qpos'][-1].copy()
        start[:3]+=rotation(start[3:])@center;goal[:3]+=rotation(goal[3:])@center
        jacp=np.zeros((3,model.nv));jacr=np.zeros((3,model.nv))
        mujoco.mj_jacBody(model,data,jacp,jacr,model.body('right_object').id)
        omega=jacr@data.qvel
        velocity=jacp@data.qvel+np.cross(omega,rotation(start[3:])@center)
        initial_object_velocities.append(np.r_[velocity,omega])
        initial_objects.append(start);goal_objects.append(goal)
    np.savez_compressed(out/'demo_initials.npz',joint_pos=initial_joints,joint_vel=initial_velocities,
                        object_pose=initial_objects,goal_pose=goal_objects,
                        object_velocity=initial_object_velocities,
                        candidate_ids=[r['candidate_id'] for r in manifest])
    # Check local joint axes and source/URDF end-link FK over nonzero poses.
    imported=mujoco.MjModel.from_xml_path(str(out/'kuka_sharpa_right.urdf'))
    source_data=mujoco.MjData(m);target_data=mujoco.MjData(imported)
    errors=[]
    for pose in np.load(first/'gpu/trajectory.npz')['qpos'][::20]:
        source_data.qpos[:]=pose
        for name in joints:target_data.qpos[imported.joint(name).qposadr]=pose[m.joint(name).qposadr]
        mujoco.mj_forward(m,source_data);mujoco.mj_forward(imported,target_data)
        for name in params['fingertip_bodies']:
            errors.append(float(np.max(np.abs(source_data.body(name).xpos-(target_data.body(name).xpos+np.array(params['base_pos']))))))
    assert max(errors)<1e-7,max(errors)
    save(out/'export_check.json',dict(joints=imported.nu or imported.njnt,source_robot_body_count=len(bids),fingertip_fk_max_error_m=max(errors),right_hand=True))
    print(json.dumps(dict(output=str(out),fk_max_error_m=max(errors))),flush=True)


if __name__=='__main__':main()
