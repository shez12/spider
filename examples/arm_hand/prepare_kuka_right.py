"""Build KUKA + native Sharpa right hand with unchanged published references."""

import argparse
import copy
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from examples.arm_hand.run_sharpa_complex import fit, ROOT
from examples.arm_hand.validate_kuka_sharpa import (
    DEFAULT_DESCRIPTION, URDF_NAME, PLAY2PERFECT_ARM_POSE,
    PLAY2PERFECT_ARM_KP, PLAY2PERFECT_ARM_KV,
)


def build(source, out, base_position):
    root = ET.parse(source).getroot()
    root.set('model', 'kuka_sharpa_right_dataset')
    # Keep source object assets verbatim while isolating their names from the
    # newly attached native Sharpa assets (some published sources already use Sharpa).
    mappings={tag:{node.get('name'):'source_'+node.get('name') for node in root.findall('asset/'+tag) if node.get('name')}
              for tag in ['mesh','material','texture','hfield']}
    classes={node.get('class'):'source_'+node.get('class') for node in root.findall('.//default') if node.get('class')}
    for tag,names in mappings.items():
        for node in root.findall('asset/'+tag):
            if node.get('name') in names:node.set('name',names[node.get('name')])
    for node in root.iter():
        for attr,names in [*mappings.items(),('class',classes),('childclass',classes)]:
            if node.get(attr) in names:node.set(attr,names[node.get(attr)])
    meshdir = (source.parent/root.find('compiler').get('meshdir', '')).resolve()
    root.find('compiler').attrib.pop('meshdir', None)
    for mesh in root.findall('asset/mesh'):
        mesh.set('file', str(meshdir/mesh.get('file')))
    world = root.find('worldbody')
    for node in list(world):
        if node.tag == 'body' and node.get('name') != 'right_object':
            world.remove(node)
    for tag in ['sensor', 'actuator', 'contact', 'equality', 'keyframe']:
        node = root.find(tag)
        if node is not None:
            root.remove(node)
    option = root.find('option')
    if option is None:option=ET.SubElement(root,'option')
    option.set('timestep', '.002')
    option.set('integrator', 'implicitfast')
    option.set('iterations', '40')
    floor = world.find("geom[@name='floor']")
    floor.set('type', 'box'); floor.set('size', '.2375 .2 .15'); floor.set('pos', '0 0 -.15')

    # Import only arm links so the merged final-link inertia does not include
    # the removed left hand or its mount.
    urdf = ET.parse(DEFAULT_DESCRIPTION/URDF_NAME).getroot()
    links = {x.get('name') for x in urdf.findall('link') if x.get('name').startswith('iiwa14_')}
    for node in list(urdf):
        if node.tag == 'link' and node.get('name') not in links:
            urdf.remove(node)
        elif node.tag == 'joint' and (node.find('parent').get('link') not in links or node.find('child').get('link') not in links):
            urdf.remove(node)
    for mesh in urdf.findall('.//mesh'):
        mesh.set('filename', str(DEFAULT_DESCRIPTION/mesh.get('filename')))
    arm_urdf = out/'arm_only.urdf'
    ET.ElementTree(urdf).write(arm_urdf, encoding='unicode')
    arm_model = mujoco.MjModel.from_xml_path(str(arm_urdf))
    mujoco.mj_saveLastXML(str(out/'arm_only.xml'), arm_model)
    arm = ET.parse(out/'arm_only.xml').getroot()
    root.find('asset').extend(copy.deepcopy(list(arm.find('asset'))))
    base = ET.SubElement(world, 'body', name='kuka_base', pos=' '.join(map(str, base_position)))
    world.remove(base)
    world.insert(0, base)
    base.extend(copy.deepcopy(list(arm.find('worldbody'))))
    robot_geoms = []
    for i, geom in enumerate(base.findall('.//geom')):
        geom.set('name', f'collision_kuka_{i:02d}')
        geom.set('contype', '0'); geom.set('conaffinity', '0')
        robot_geoms.append(geom.get('name'))

    hand_path = ROOT/'spider/assets/robots/sharpa_wave_right/right_hand.xml'
    hand = ET.parse(hand_path).getroot()
    defaults=root.find('default')
    if defaults is None:defaults=ET.SubElement(root,'default')
    defaults.extend(copy.deepcopy(list(hand.find('default'))))
    for asset in hand.find('asset'):
        asset = copy.deepcopy(asset)
        if asset.get('file'):
            asset.set('file', str(hand_path.parent/'assets/right'/asset.get('file')))
        root.find('asset').append(asset)
    palm = copy.deepcopy(hand.find('worldbody/body'))
    # The same flange mounting rotation is an explicit fixture assumption;
    # hand/site FK is then checked against the unchanged right-hand reference.
    palm.set('pos', '0 0 .095'); palm.set('quat', '.793352 0 0 -.608763')
    base.find(".//body[@name='iiwa14_link_7']").append(palm)
    ET.SubElement(palm, 'site', name='sharpa_palm', pos='0 .01 .06', size='.003', group='3')
    for finger in ['thumb','index','middle','ring']:
        site = palm.find(f".//site[@name='right_{finger}_fingertip']")
        site.set('name', f'sharpa_{finger}_tip')
    hand_geoms = []
    for i, geom in enumerate(palm.findall('.//geom')):
        if 'collision' in geom.get('class', ''):
            geom.set('name', f'collision_sharpa_{i:02d}')
            geom.set('contype', '1'); geom.set('conaffinity', '1')
            hand_geoms.append(geom.get('name')); robot_geoms.append(geom.get('name'))
    actuator = ET.SubElement(root, 'actuator')
    for i, (kp, kv) in enumerate(zip(PLAY2PERFECT_ARM_KP,PLAY2PERFECT_ARM_KV),1):
        ET.SubElement(actuator, 'position', name=f'iiwa14_target_{i}', joint=f'iiwa14_joint_{i}',
                      kp=str(kp), kv=str(kv), inheritrange='1', forcerange='-300 300')
    actuator.extend(copy.deepcopy(list(hand.find('actuator'))))
    contact = ET.SubElement(root,'contact')
    contact.extend(copy.deepcopy(list(hand.find('contact'))))
    objects = [g.get('name') for g in world.find("body[@name='right_object']").findall('geom') if g.get('name') != 'right_object_visual']
    for name in objects + robot_geoms:
        pair=ET.SubElement(contact,'pair',geom1='floor',geom2=name,condim='4',friction='1 1 .1 0 0')
        if name in robot_geoms:
            pair.set('margin', '.005')
            pair.set('solref', '.004 1')
            pair.set('solimp', '.99 .999 .0001 .5 2')
    for hg in hand_geoms:
        for og in objects:
            ET.SubElement(contact,'pair',geom1=hg,geom2=og,condim='4',friction='1 1 .1 0 0')
    sensor=ET.SubElement(root,'sensor')
    for tag,typ,name in [('framepos','site','sharpa_palm'),('framequat','site','sharpa_palm'),
                         ('framepos','xbody','right_object'),('framequat','xbody','right_object')]:
        ET.SubElement(sensor,tag,objtype=typ,objname=name)
    for finger in ['thumb','index','middle','ring']:
        ET.SubElement(sensor,'framepos',objtype='site',objname=f'sharpa_{finger}_tip')
    ET.indent(root)
    ET.ElementTree(root).write(out/'scene.xml',encoding='unicode')
    model=mujoco.MjModel.from_xml_path(str(out/'scene.xml'))
    assert (model.nq,model.nu)==(36,29)
    assert model.joint(0).name == 'iiwa14_joint_1'
    assert model.jnt_qposadr[model.body('right_object').jntadr[0]] == 29
    return model


def reference(scene, trajectory, out):
    model=mujoco.MjModel.from_xml_path(str(scene)); data=mujoco.MjData(model)
    with np.load(trajectory) as archive: q=archive['qpos'].reshape(-1,model.nq)
    positions=[]; quats=[]; tips=[]
    tip_sites=[]
    for finger in ['thumb','index','middle','ring']:
        candidates=[f'track_hand_right_{finger}_tip',f'right_{finger}_tip']
        tip_sites.append(next(name for name in candidates if mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,name)>=0))
    for pose in q:
        data.qpos[:]=pose; mujoco.mj_forward(model,data)
        site=data.site('right_palm'); quat=np.empty(4);mujoco.mju_mat2Quat(quat,site.xmat)
        positions.append(site.xpos.copy());quats.append(quat)
        tips.append([data.site(name).xpos.copy() for name in tip_sites])
    ref=dict(wrist_pos=np.array(positions),wrist_quat=np.array(quats),fingertips=np.array(tips),object_qpos=q[:,-7:].copy())
    np.savez(out/'source_reference.npz',**ref)
    return ref


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/kuka_sharpa_right_foam')
    parser.add_argument('--task',default='dexycb_01_right_061_foam_brick_20200709_153241_00')
    parser.add_argument('--base-position',nargs=3,type=float,default=[0,.8,-.53])
    parser.add_argument('--restarts',type=int,default=8)
    args=parser.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    source=ROOT/'example_datasets/retarget_full/processed/dexycb_lifted/allegro/right'/args.task/'scene.xml'
    trajectory=source.parent/'0/trajectory_mjwp.npz'
    ref=reference(source,trajectory,out);frozen={k:v.copy() for k,v in ref.items()}
    model=build(source,out,args.base_position)
    fit(model,ref,out,2,initial_arm_pose=PLAY2PERFECT_ARM_POSE,first_frame_restarts=args.restarts)
    for k,v in frozen.items():np.testing.assert_array_equal(ref[k],v)
    report=json.loads((out/'retargeting.json').read_text())
    report.update(hand='native right Sharpa Wave',reference_mirrored=False,
                  source_sha256=hashlib.sha256(trajectory.read_bytes()).hexdigest(),
                  base_position_m=args.base_position,ready_for_gpu=report['initial_alignment_pass'] and report['table_clearance_pass'])
    (out/'geometry_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
