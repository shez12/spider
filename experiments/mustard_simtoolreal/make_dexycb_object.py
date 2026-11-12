"""Object asset directory (URDF + convex pieces + calibration.json) for the SimToolReal env from a pipeline MuJoCo scene.

Same conventions as assets/object (mustard bottle, examples/arm_hand/test_mustard_simtoolreal_bridge.export_object):
link origin at the centre of the visual mesh AABB in the MuJoCo body axes, mass / inertia from the compiled model,
4 keypoints at 1.5 x the half extents, policy_object_scale = extents / 0.04.  Additionally records the proper
rotations (half turns about z and about the two best in-plane axes) that map the mesh onto itself within --symmetry-tol (p95 vertex
distance); demo_relative_grasps / replay_demo_grasps use them to match a demo to more native resting poses.
Needs mujoco (pipeline python), not Isaac.  Usage:
  python make_dexycb_object.py --scene <campaign>/sources/001_.../scene.xml --name potted_meat_can \
      --object-id 010_potted_meat_can --output assets/object_potted_meat_can
"""
import argparse,json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,mujoco,trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot


def rotation(q):return Rot.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix()


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scene',type=Path,required=True);p.add_argument('--body',default='right_object');p.add_argument('--name',required=True)
    p.add_argument('--object-id',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--symmetry-tol',type=float,default=.003)
    p.add_argument('--overwrite',action='store_true')
    a=p.parse_args();m=mujoco.MjModel.from_xml_path(str(a.scene));bid=m.body(a.body).id;a.output.mkdir(parents=True,exist_ok=a.overwrite)
    meshes=[]
    for g in np.flatnonzero(m.geom_bodyid==bid):
        mid=m.geom_dataid[g];assert mid>=0
        v=m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]]@rotation(m.geom_quat[g]).T+m.geom_pos[g]
        meshes.append((m.geom(g).name,trimesh.Trimesh(v,m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid]+m.mesh_facenum[mid]].copy(),process=False)))
    visual=[x for n,x in meshes if n.endswith('_visual')];assert len(visual)==1;visual=visual[0]
    center=visual.bounds.mean(0);dims=visual.bounds[1]-visual.bounds[0];offsets=np.array([[1,1,1],[1,1,-1],[-1,-1,1],[-1,-1,-1]])*dims*1.5/2
    R=rotation(m.body_iquat[bid]);inertia=R@np.diag(m.body_inertia[bid])@R.T
    root=ET.Element('robot',name=a.name);link=ET.SubElement(root,'link',name='object_root');inertial=ET.SubElement(link,'inertial')
    ET.SubElement(inertial,'origin',xyz=' '.join(map(str,m.body_ipos[bid]-center)),rpy='0 0 0');ET.SubElement(inertial,'mass',value=str(m.body_mass[bid]))
    ET.SubElement(inertial,'inertia',**{k:str(inertia[i,j]) for k,i,j in [('ixx',0,0),('iyy',1,1),('izz',2,2),('ixy',0,1),('ixz',0,2),('iyz',1,2)]})
    for n,x in sorted(meshes,key=lambda t:t[0]):
        x=x.copy();x.vertices-=center;x.export(a.output/f'{n}.obj')
        ET.SubElement(ET.SubElement(ET.SubElement(link,'visual' if n.endswith('_visual') else 'collision',name=n),'geometry'),'mesh',filename=f'{n}.obj')
    ET.indent(root);ET.ElementTree(root).write(a.output/f'{a.name}.urdf',encoding='unicode')
    V=np.asarray(visual.vertices)-center;tree=cKDTree(V);sub=V[np.random.default_rng(0).choice(len(V),min(len(V),4000),replace=False)]
    half=lambda axis:Rot.from_rotvec(np.pi*np.asarray(axis,float)).as_matrix();err=lambda axis:float(np.percentile(tree.query(sub@half(axis).T)[0],95))
    inplane=lambda deg:[np.cos(np.deg2rad(deg)),np.sin(np.deg2rad(deg)),0.]
    best=min(np.arange(0,180,.2),key=lambda g:err(inplane(g)))  # the mesh need not be aligned with the body axes
    axes={'z':[0.,0.,1.],f'xy@{best:.1f}deg':inplane(best),f'xy@{best+90:.1f}deg':inplane(best+90)}
    sym={k:err(v) for k,v in axes.items()};keep=[k for k in axes if sym[k]<a.symmetry_tol]
    T=[]
    for k in keep:t=np.eye(4);t[:3,:3]=half(axes[k]);T.append(t.round(12).tolist())
    cal=dict(object_id=a.object_id,bbox_choice='whole visual mesh AABB, source axes',bbox_dims_m=dims.tolist(),source_to_new_root_center_in_source_m=center.tolist(),
        policy_object_scale=(dims/.04).tolist(),keypoint_scale=1.5,keypoints_local_m=offsets.tolist(),mass_kg=float(m.body_mass[bid]),
        inertia_eigenvalues=np.linalg.eigvalsh(inertia).tolist(),collision_mesh_count=len(meshes)-1,usd_converted=False,source_scene=str(a.scene),
        half_turn_p95_vertex_error_m=sym,symmetry_tol_m=a.symmetry_tol,symmetry_rotations=T)
    (a.output/'calibration.json').write_text(json.dumps(cal,indent=2)+'\n')
    print(json.dumps(dict(output=str(a.output),bbox_dims_m=dims.tolist(),mass_kg=cal['mass_kg'],collision_meshes=len(meshes)-1,half_turn_p95_vertex_error_m=sym,symmetries_kept=keep)))


if __name__=='__main__':main()
