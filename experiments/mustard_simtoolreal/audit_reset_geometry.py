"""Collision-only audit of saved Isaac states using the exported native URDFs.

Run with spider-world-model Python (MuJoCo). This does no dynamics rollout.
FK is checked against saved Isaac link frames before using contact distances.
"""
import argparse,json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parent


def make_model(output):
    robot=ROOT/'assets/robot/kuka_sharpa_right.urdf';urdf=ET.parse(robot)
    for mesh in urdf.getroot().iter('mesh'):mesh.set('filename',str(robot.parent/mesh.attrib['filename']))
    mj=ET.SubElement(urdf.getroot(),'mujoco');ET.SubElement(mj,'compiler',fusestatic='false',discardvisual='true')
    path=output/'robot_geometry.urdf';urdf.write(path)
    model=mujoco.MjModel.from_xml_path(str(path));xml=output/'geometry.xml';mujoco.mj_saveLastXML(str(xml),model)
    tree=ET.parse(xml);root=tree.getroot();assets=root.find('asset');world=root.find('worldbody')
    for mesh in assets.findall('mesh'):
        if not Path(mesh.attrib['file']).is_absolute():mesh.set('file',str(robot.parent/mesh.attrib['file']))
    for label,file in [('object',ROOT/'assets/object/mustard_bottle.urdf')]:
        body=ET.SubElement(world,'body',name='audit_'+label);ET.SubElement(body,'freejoint',name='audit_'+label+'_joint')
        for collision in ET.parse(file).getroot().findall('link/collision'):
            mesh=collision.find('geometry/mesh');name='audit_'+collision.attrib['name']
            ET.SubElement(assets,'mesh',name=name,file=str(file.parent/mesh.attrib['filename']))
            ET.SubElement(body,'geom',name=name,type='mesh',mesh=name,mass='.04')
    table=ET.SubElement(world,'body',name='audit_table');ET.SubElement(table,'freejoint',name='audit_table_joint')
    ET.SubElement(table,'geom',name='audit_table_geom',type='box',size='.2375 .2 .15',mass='500')
    contact=root.find('contact')
    if contact is None:contact=ET.SubElement(root,'contact')
    body_names={b.attrib['name'] for b in root.findall('.//body')}
    spec=json.loads((ROOT/'assets/robot/robot_spec.json').read_text())
    pairs=set(tuple(sorted(p)) for p in spec['exclusions'])
    for joint in urdf.getroot().findall('joint'):pairs.add(tuple(sorted((joint.find('parent').attrib['link'],joint.find('child').attrib['link']))))
    for b1,b2 in pairs:
        if b1 in body_names and b2 in body_names:ET.SubElement(contact,'exclude',body1=b1,body2=b2)
    tree.write(xml);return mujoco.MjModel.from_xml_path(str(xml)),spec


def main():
    p=argparse.ArgumentParser();p.add_argument('--states',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);model,spec=make_model(a.output);data=mujoco.MjData(model)
    with np.load(a.states) as archive:s={k:archive[k] for k in archive.files}
    robot_bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'kuka_base')
    joints=[model.joint(name).qposadr[0] for name in spec['joint_names']]
    obj=model.joint('audit_object_joint').qposadr[0];table=model.joint('audit_table_joint').qposadr[0]
    common=[(i,mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,str(name))) for i,name in enumerate(s['body_names'])]
    common=[(i,b) for i,b in common if b>=0];assert len(common)>=25
    table_bid=model.body('audit_table').id;object_bid=model.body('audit_object').id
    depths=[];fk=[]
    for i,q in enumerate(s['q']):
        model.body_pos[robot_bid]=s['robot_base'][i,:3];model.body_quat[robot_bid]=s['robot_base'][i,3:7]
        data.qpos[joints]=q;data.qpos[obj:obj+7]=s['root'][i,:7];data.qpos[table:table+7]=s['table'][i]
        mujoco.mj_fwdPosition(model,data)
        error=max(np.linalg.norm(data.xpos[b]-s['body_positions'][i,j]) for j,b in common);fk.append(error)
        if error>1e-4:raise RuntimeError(f'Geometry FK mismatch at {i}: {error}')
        row=[0.,0.,0.,0.]
        for c in data.contact:
            if c.dist>=0:continue
            b1,b2=model.geom_bodyid[c.geom1],model.geom_bodyid[c.geom2]
            if table_bid in (b1,b2):kind=3 if object_bid in (b1,b2) else 0
            elif object_bid in (b1,b2):kind=2
            else:kind=1
            row[kind]=max(row[kind],float(-c.dist))
        depths.append(row)
        if i%500==0:print('GEOMETRY_PROGRESS',i,len(s['q']),flush=True)
    depths=np.array(depths)
    np.savez(a.output/'geometry_results.npz',depths=depths,fk_error=np.array(fk))
    (a.output/'report.json').write_text(json.dumps(dict(states=len(depths),common_bodies=len(common),fk_max_m=max(fk),
        depth_columns=['robot_table','robot_self','robot_object','object_table'],max_depths=depths.max(0).tolist(),
        scope='Collision geometry/FK check of saved Isaac states using exported native URDFs. No MuJoCo dynamics; not a claim of identical PhysX contact distances.'),indent=2)+'\n')


if __name__=='__main__':main()
