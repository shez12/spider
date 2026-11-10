"""Asset-only adaptation of the copied native SimToolReal environment.

No overrides of action, reset, reward, observation, done, or curriculum hooks.
Import only after AppLauncher starts Isaac Sim.
"""
from pathlib import Path
import dataclasses
import json
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parent
VENDOR=ROOT.parents[1]/'third_party/simtoolreal'


def create_env(num_envs=120,render_mode=None,configure=None):
    import yaml
    import isaacsimenvs.tasks.simtoolreal.simtoolreal_env as module
    import isaacsimenvs.tasks.simtoolreal.utils.scene_utils as scene
    from isaacsimenvs.tasks.simtoolreal.robot_spec import KUKA_SHARPA_SPEC
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg
    from isaaclab.sim.converters import UrdfConverter,UrdfConverterCfg
    params=json.loads((ROOT/'assets/robot/robot_spec.json').read_text())
    robot=ROOT/'assets/robot/kuka_sharpa_right.urdf'
    table=VENDOR/'assets/urdf/table_narrow.urdf'
    obj=ROOT/'assets/object/mustard_bottle.urdf'
    calibration=json.loads((ROOT/'assets/object/calibration.json').read_text())
    names=params['joint_names'];left=list(KUKA_SHARPA_SPEC.joint_names_canonical)[7:]
    def mapped(field):
        values=getattr(KUKA_SHARPA_SPEC,field)
        return {new:values[old] for old,new in zip(left,names[7:]) if old in values}
    # Match the original Sharpa reference's finite effort limits by joint role.
    reference=ET.parse(VENDOR/'assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf').getroot()
    effort=[float(j.find('limit').attrib['effort']) for j in reference.findall('joint') if j.attrib['type']!='fixed'][7:]
    assert len(effort)==22
    right=dataclasses.replace(KUKA_SHARPA_SPEC,name='kuka_sharpa_right',
        hand_joint_regex='right_.*',joint_names_canonical=tuple(names),
        palm_body_name=params['palm_body'],palm_center_offset=tuple(params['palm_offset']),
        fingertip_link_names=tuple(params['fingertip_bodies']),
        fingertip_offsets=tuple(tuple(v) for v in params['fingertip_offsets']),
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
        return UrdfConverter(UrdfConverterCfg(asset_path=prepared,usd_dir=str(work/'kuka_sharpa_right'),
            force_usd_conversion=True,fix_base=True,merge_fixed_joints=False,make_instanceable=False,
            self_collision=True,joint_drive=kw.get('joint_drive'))).usd_path
    scene._convert_urdf_to_usd=convert
    adjacency_original=scene._load_adjacent_links_map
    def adjacency():
        result=adjacency_original()
        for a,b in params['exclusions']:
            result.setdefault(a,[]).append(b);result.setdefault(b,[]).append(a)
        return result
    scene._load_adjacent_links_map=adjacency
    cfg=SimToolRealEnvCfg()
    cfg.from_dict(yaml.safe_load((VENDOR/'isaacsimenvs/cfg/task/SimToolReal.yaml').read_text()))
    cfg.seed=42;cfg.scene.num_envs=num_envs
    cfg.assets.robot_urdf=str(robot);cfg.assets.table_urdf=str(table)
    cfg.assets.object_urdf=str(obj);cfg.assets.object_scale=tuple(calibration['policy_object_scale'])
    cfg.reward.fixed_size=tuple(calibration['bbox_dims_m'])
    # Align this option to the specifically requested September experiment.
    cfg.domain_randomization.torque_scale=0.
    if render_mode:
        cfg.viewer.origin_type='env';cfg.viewer.env_index=num_envs-1
    if configure is not None:configure(cfg)
    env=module.SimToolRealEnv(cfg,render_mode=render_mode)
    assert env.cfg.observation_space==140 and env.cfg.state_space==162
    return env
