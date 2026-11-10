"""Align right-hand environment settings with the original SimToolReal (left hand) before the scene is built.

1. max_depenetration_velocity 10 -> 1000 for robot and object (the 10 came from our 2026-09-07 Yimu commit
   9e9acd70 in simtoolreal_origin; upstream always used 1000).
2. Hand joint friction -> 0 (original left hand has none; ours copied MuJoCo frictionloss values, a torque in N*m,
   into PhysX's dimensionless joint friction coefficient).
3. Optional (restrict_abduction): finger MCP_AA limits +-20 deg -> +-2 deg and thumb CMC_AA upper 0.349 -> 0.131,
   the values of iiwa14_left_sharpa_adjusted_restricted.urdf. Right-hand joint axes are mirrored, so the same
   numeric limits give the mirrored motion (all other right-hand limits already equal the left values).
   Hand action scaling and joint observation normalisation read limits from the simulator, so they follow.

Monkeypatches third_party scene_utils only; environment.py (hashed into the reset-bank signature) is unchanged.
Call after AppLauncher started and before create_env().
"""
DEPENETRATION=1000.
RESTRICTED_LIMITS={**{f'right_{f}_MCP_AA':(-0.0349066,0.0349066) for f in ('index','middle','ring','pinky')},
    'right_thumb_CMC_AA':(-0.3490659,0.1308997)}


def install(restrict_abduction=False):
    import xml.etree.ElementTree as ET
    import isaacsimenvs.tasks.simtoolreal.utils.scene_utils as scene
    bake,extras,prepare=scene._bake_usd,scene.hand_actuator_extras,scene._prepare_urdf_for_isaacsim
    def aligned_bake(*args,props=None,**kwargs):
        if props and 'max_depenetration_velocity' in props:props=dict(props,max_depenetration_velocity=DEPENETRATION)
        return bake(*args,props=props,**kwargs)
    def aligned_extras(spec):
        return dict(extras(spec),friction={name:0. for name in spec.hand_joint_stiffness})
    def restricted_prepare(asset_path,usd_work_dir):
        from pathlib import Path
        source=Path(asset_path);tree=ET.parse(source);root=tree.getroot()
        joints={j.get('name'):j for j in root.findall('joint')}
        if not set(RESTRICTED_LIMITS)<=set(joints):return prepare(asset_path,usd_work_dir)
        for name,(lo,hi) in RESTRICTED_LIMITS.items():
            limit=joints[name].find('limit');limit.set('lower',repr(lo));limit.set('upper',repr(hi))
        for mesh in root.findall('.//mesh'):
            if mesh.get('filename'):mesh.set('filename',str(scene._resolve_urdf_mesh_path(source,mesh.get('filename'))))
        out=Path(usd_work_dir)/'_abduction_restricted';out.mkdir(parents=True,exist_ok=True)
        # Distinct stem -> distinct converted USD, so concurrent unrestricted runs never share the file.
        path=out/(source.stem+'_aa2deg.urdf');tree.write(path,encoding='utf-8',xml_declaration=True)
        return prepare(str(path),usd_work_dir)
    scene._bake_usd,scene.hand_actuator_extras=aligned_bake,aligned_extras
    if restrict_abduction:scene._prepare_urdf_for_isaacsim=restricted_prepare
    return dict(max_depenetration_velocity=DEPENETRATION,hand_joint_friction=0.,
        restricted_abduction_limits=RESTRICTED_LIMITS if restrict_abduction else None)
