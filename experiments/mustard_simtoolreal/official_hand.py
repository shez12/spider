"""Right KUKA + official Sharpa Wave hand (sharpa-urdf-usd-xml, wave_01/right_sharpa_wave).

Arm links/joints and the flange mount are taken from our kuka_sharpa_right.urdf; every hand link/joint, mesh and
inertial comes from the official right_sharpa_wave.urdf (incl. per-finger elastomer pads). compare_hand_kinematics.py
shows both hands have identical link positions for identical joint values (<1 um), only link-frame conventions differ
(180 deg), so joint-space demos/reset states transfer; fingertip offsets are re-expressed in the official DP frames.

create_env() mirrors environment.create_env (which is hashed into the reset-bank signature and stays untouched).
"""
import dataclasses,json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as Rot

HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[1];VENDOR=PROJECT/'third_party/simtoolreal'
OFFICIAL=PROJECT/'outputs/robot_assets/wave_01/right_sharpa_wave'
OURS=HERE/'assets/robot/kuka_sharpa_right.urdf'
OUT=PROJECT/'outputs/robot_assets/kuka_sharpa_right_official'


def build_urdf(variant='right_sharpa_wave.urdf'):
    ours=ET.parse(OURS).getroot();off=ET.parse(OFFICIAL/variant).getroot()
    robot=ET.Element('robot',name='kuka_sharpa_right_official')
    for link in ours.findall('link'):
        if not link.get('name').startswith('right_'):robot.append(link)
    for joint in ours.findall('joint'):
        if not joint.find('child').get('link').startswith('right_') or joint.get('name')=='fixed_right_hand_C_MC':robot.append(joint)
    for elem in off:
        if elem.tag in ('link','joint'):robot.append(elem)
    for mesh in robot.findall('.//mesh'):
        f=mesh.get('filename')
        if f.startswith('package://right_sharpa_wave/'):mesh.set('filename',str(OFFICIAL/f[len('package://right_sharpa_wave/'):]))
        elif not Path(f).is_absolute():mesh.set('filename',str((OURS.parent/f).resolve()))
    OUT.mkdir(parents=True,exist_ok=True);path=OUT/'kuka_sharpa_right_official.urdf'
    ET.indent(robot);ET.ElementTree(robot).write(path,encoding='unicode');return path


def frame_rotation(link):
    """R_ours^T R_official for a hand link at q=0 (from the palm)."""
    from compare_hand_kinematics import chain,fk
    a=fk(chain(OURS),{})[link][1];b=fk(chain(OFFICIAL/'right_sharpa_wave.urdf'),{})[link][1]
    return a.T@b


def create_env(num_envs=120,render_mode=None,configure=None):
    import yaml
    import isaacsimenvs.tasks.simtoolreal.simtoolreal_env as module
    import isaacsimenvs.tasks.simtoolreal.utils.scene_utils as scene
    from isaacsimenvs.tasks.simtoolreal.robot_spec import KUKA_SHARPA_SPEC
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg
    from isaaclab.sim.converters import UrdfConverter,UrdfConverterCfg
    params=json.loads((HERE/'assets/robot/robot_spec.json').read_text())
    robot=build_urdf();table=VENDOR/'assets/urdf/table_narrow.urdf'
    obj=HERE/'assets/object/mustard_bottle.urdf';calibration=json.loads((HERE/'assets/object/calibration.json').read_text())
    names=params['joint_names'];left=list(KUKA_SHARPA_SPEC.joint_names_canonical)[7:]
    def mapped(field):
        values=getattr(KUKA_SHARPA_SPEC,field);return {new:values[old] for old,new in zip(left,names[7:]) if old in values}
    reference=ET.parse(VENDOR/'assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf').getroot()
    effort=[float(j.find('limit').attrib['effort']) for j in reference.findall('joint') if j.attrib['type']!='fixed'][7:]
    assert len(effort)==22
    # Fingertip contact points: same physical point, expressed in the official DP frames.
    tips=tuple(tuple((frame_rotation(b).T@np.array(o)).tolist()) for b,o in zip(params['fingertip_bodies'],params['fingertip_offsets']))
    assert np.allclose(frame_rotation(params['palm_body']),np.eye(3),atol=1e-6)
    right=dataclasses.replace(KUKA_SHARPA_SPEC,name='kuka_sharpa_right_official',
        hand_joint_regex='right_.*',joint_names_canonical=tuple(names),
        palm_body_name=params['palm_body'],palm_center_offset=tuple(params['palm_offset']),
        fingertip_link_names=tuple(params['fingertip_bodies']),fingertip_offsets=tips,
        hand_joint_stiffness=mapped('hand_joint_stiffness'),hand_joint_damping=mapped('hand_joint_damping'),
        hand_joint_armature=mapped('hand_joint_armature'),hand_joint_friction=mapped('hand_joint_friction'),
        hand_joint_effort=dict(zip(names[7:],effort)))
    original=module.resolve_robot_spec
    def resolve(path):return right if str(path)==str(robot) else original(path)
    module.resolve_robot_spec=resolve;scene.resolve_robot_spec=resolve
    convert_original=scene._convert_urdf_to_usd
    def convert(path,work,**kw):
        if str(path)!=str(robot):return convert_original(path,work,**kw)
        prepared=scene._prepare_urdf_for_isaacsim(path,work)
        return UrdfConverter(UrdfConverterCfg(asset_path=prepared,usd_dir=str(work/'kuka_sharpa_right_official'),
            force_usd_conversion=True,fix_base=True,merge_fixed_joints=False,make_instanceable=False,
            self_collision=True,joint_drive=kw.get('joint_drive'))).usd_path
    scene._convert_urdf_to_usd=convert
    adjacency_original=scene._load_adjacent_links_map
    def adjacency():
        result=adjacency_original()
        pairs=list(params['exclusions'])
        # Pads are rigidly attached to their DP; PhysX only auto-filters jointed pairs, so also filter pad vs MP.
        pairs+=[(f'right_{f}_elastomer',f'right_{f}_{s}') for f in ('index','middle','ring','pinky') for s in ('MP','DP')]
        pairs+=[('right_thumb_elastomer','right_thumb_PP'),('right_thumb_elastomer','right_thumb_DP')]
        for a,b in pairs:result.setdefault(a,[]).append(b);result.setdefault(b,[]).append(a)
        return result
    scene._load_adjacent_links_map=adjacency
    cfg=SimToolRealEnvCfg();cfg.from_dict(yaml.safe_load((VENDOR/'isaacsimenvs/cfg/task/SimToolReal.yaml').read_text()))
    cfg.seed=42;cfg.scene.num_envs=num_envs
    cfg.assets.robot_urdf=str(robot);cfg.assets.table_urdf=str(table)
    cfg.assets.object_urdf=str(obj);cfg.assets.object_scale=tuple(calibration['policy_object_scale'])
    cfg.reward.fixed_size=tuple(calibration['bbox_dims_m'])
    cfg.domain_randomization.torque_scale=0.
    if render_mode:cfg.viewer.origin_type='env';cfg.viewer.env_index=num_envs-1
    if configure is not None:configure(cfg)
    env=module.SimToolRealEnv(cfg,render_mode=render_mode)
    assert env.cfg.observation_space==140 and env.cfg.state_space==162
    return env
