"""Verify and atomically admit a genuinely new, doubly validated model-assisted candidate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import mujoco
import numpy as np
from examples.arm_hand.audit_right_trajectory import audit_states
from examples.arm_hand.batch_kuka_gpu import digest,save
from examples.arm_hand.mustard_valid100 import valid
from examples.arm_hand.spatial_augmentation import augment,KEYS


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,required=True);p.add_argument('--execution',type=Path,required=True)
    p.add_argument('--publish',action='store_true');a=p.parse_args();case=a.case.resolve();execution=a.execution.resolve()
    root=Path(__file__).resolve().parents[2];campaign=root/'outputs/mustard_multisource_valid100'
    original_state=json.loads((case/'state.json').read_text());groups=json.loads((campaign/'selected_sources.json').read_text())
    group=next(g for g in groups if g['source_index']==original_state['source_index']);destination=Path(group['result_dir'])/case.name
    report=json.loads((execution/'execution.json').read_text())
    if report['status']!='complete' or not report['accepted']:raise ValueError('Execution is not strictly accepted')
    if report['case']!=str(case) or not original_state.get('ik_pass'):raise ValueError('Case/IK identity differs')
    geometry=json.loads((case/'geometry_report.json').read_text())
    if not geometry.get('ready_for_gpu'):raise ValueError('Original geometric gate did not pass')
    metadata=json.loads((case/'augmentation.json').read_text());source=Path(group['source_dir'])
    if metadata['source_task']!=group['task'] or metadata['source_reference_sha256']!=digest(source/'source_reference.npz'):raise ValueError('Source provenance differs')
    if abs(metadata['yaw_deg'])>10 or np.any(np.abs(metadata['sampled_translation_xy_m'])>.05):raise ValueError('Augmentation outside original bounds')
    model=mujoco.MjModel.from_xml_path(str(case/'scene.xml'));floor=model.geom('floor').id;base=model.body('kuka_base').id
    if model.geom_type[floor]!=mujoco.mjtGeom.mjGEOM_BOX or model.body_parentid[base]!=0:raise ValueError('Expected real tabletop box and fixed robot base')
    height=model.geom_pos[floor,2]+model.geom_size[floor,2]-model.body_pos[base,2]
    if not np.isclose(height,.53,atol=1e-10):raise ValueError('53 cm table/base height difference changed')
    for b in range(1,model.nbody):
        ancestor=b
        while ancestor and ancestor!=base:ancestor=int(model.body_parentid[ancestor])
        if ancestor==base and not np.isclose(model.body_gravcomp[b],1):raise ValueError('Robot gravity compensation changed')
    if model.body_gravcomp[model.body('right_object').id]!=0:raise ValueError('Object must retain gravity')
    with np.load(source/'source_reference.npz') as f:raw={k:f[k] for k in f.files}
    expected,_=augment(raw,metadata['yaw_deg'],np.asarray(metadata['sampled_translation_xy_m']),model.body_pos[base])
    with np.load(case/'source_reference.npz') as f, np.load(case/'reference.npz') as ref:
        for key in KEYS:
            np.testing.assert_allclose(f[key],expected[key],atol=1e-12,rtol=0)
            np.testing.assert_array_equal(f[key],ref[key])
        T=len(ref['ctrl_nominal'])
        if len(raw['object_qpos'])!=T+1:raise ValueError('Source timeline/extra hold changed')
        with np.load(execution/'gpu/trajectory.npz') as selected,np.load(execution/'gpu/replay_trajectory.npz') as replay:
            for tr in [selected,replay]:
                if len(tr['ctrl'])!=T or len(tr['qpos'])!=T:raise ValueError('Trajectory time axis differs')
                if not np.isclose(float(tr['control_dt']),.01) or not np.isclose(float(tr['physics_dt']),.002):raise ValueError('Timebase differs')
                np.testing.assert_array_equal(tr['initial_state'],ref['initial_state'])
            np.testing.assert_array_equal(selected['ctrl'],replay['ctrl'])
            selected_metrics=audit_states(model,ref,selected);replay_metrics=audit_states(model,ref,replay)
    if not valid(selected_metrics) or not valid(replay_metrics):raise ValueError('Fresh verification failed original thresholds')
    signature={p.name:digest(p) for p in [case/'scene.xml',case/'reference.npz']}
    result=json.loads((execution/'gpu/result.json').read_text())
    if result['signature']!=signature or result['extra_hold']:raise ValueError('Physics provenance differs')
    if not (execution/'best_gpu.mp4').is_file():raise FileNotFoundError('Missing saved actual-state video')
    key=(round(metadata['yaw_deg'],12),tuple(np.round(metadata['sampled_translation_xy_m'],12)))
    for path in Path(group['result_dir']).glob('candidate_*/augmentation.json'):
        existing=json.loads(path.read_text());other=(round(existing['yaw_deg'],12),tuple(np.round(existing['sampled_translation_xy_m'],12)))
        if key==other:raise ValueError(f'Duplicate augmentation already exists: {path.parent}')
    accepted=sum(json.loads(p.read_text()).get('validated_success',False) for p in Path(group['result_dir']).glob('candidate_*/state.json'))
    if accepted>=90:raise ValueError('Pilot admission stops before final count boundary; use integrated coordinator for remaining slots')
    controller=report.get('controller')
    if controller not in ['original_spider','reference_feedback_policy']:raise ValueError('Unsupported admission controller')
    policy_hash=None;immutable=None;controller_stage='original SPIDER sampling';controller_kind='original SPIDER paired baseline'
    if controller=='original_spider':
        if result['arguments'].get('reference_policy') or result['arguments'].get('warm_start'):raise ValueError('Baseline contains a controller warm start')
    else:
        policy=execution/'controller_snapshot.pt';policy_hash=digest(policy)
        if policy_hash!=report['controller_sha256']:raise ValueError('Controller snapshot hash differs')
        if not result['arguments'].get('reference_policy'):raise ValueError('Feedback policy missing from actual optimizer provenance')
        import torch
        controller_checkpoint=torch.load(policy,map_location='cpu',weights_only=False)
        controller_stage=controller_checkpoint.get('stage','unspecified feedback training')
        if controller_stage.startswith('world-model BPTT') and controller_checkpoint.get('world') and controller_checkpoint.get('update',0)>0:
            controller_kind='world-model-trained shared feedback policy'
        elif 'BC' in controller_stage:controller_kind='validated-physics BC feedback policy'
        else:controller_kind='reference-conditioned feedback policy; training attribution unspecified'
    verification=dict(source_index=group['source_index'],destination=str(destination),selected=selected_metrics,replay=replay_metrics,
                      source_checks_passed=True,accepted=True,published=False,controller=controller,controller_stage=controller_stage)
    if a.publish:
        if destination.exists():raise FileExistsError(destination)
        stage=destination.parent/('.model_admission_'+uuid.uuid4().hex);stage.mkdir()
        for name in ['scene.xml','reference.npz','source_reference.npz','retargeting.json','geometry_report.json','augmentation.json']:
            shutil.copy2(case/name,stage/name)
        shutil.copytree(execution/'gpu',stage/'gpu');shutil.copytree(execution/'replay_audit',stage/'replay_audit',symlinks=True)
        link=stage/'replay_audit/trajectory.npz';link.unlink();link.symlink_to('../gpu/replay_trajectory.npz')
        shutil.copy2(execution/'best_gpu.mp4',stage/'best_gpu.mp4');shutil.copy2(execution/'execution.json',stage/'acceleration_execution.json')
        if controller=='reference_feedback_policy':
            policies=campaign/'policies';policies.mkdir(exist_ok=True);immutable=policies/(policy_hash+'.pt')
            if not immutable.exists():
                temporary=policies/(policy_hash+'.'+uuid.uuid4().hex+'.tmp');shutil.copy2(policy,temporary);temporary.replace(immutable)
            if digest(immutable)!=policy_hash:raise ValueError('Shared immutable policy corrupted')
            (stage/'controller_snapshot.pt').symlink_to(os.path.relpath(immutable,destination))
        timing=report.get('timing_breakdown',{})
        state=dict(original_state,folder=str(destination),status='accepted',validated_success=True,selected_metrics=selected_metrics,replay_metrics=replay_metrics,
            physics_elapsed_s=timing.get('optimizer',{}).get('process_s',report['optimizer_s']),
            validation_elapsed_s=sum(v['process_s'] for k,v in timing.items() if k in ['selected','replay','render']),
            acceleration=dict(kind=controller_kind,controller_stage=controller_stage,experiment=str(execution),policy_sha256=policy_hash,
                immutable_policy=str(immutable) if immutable else None,pilot_admission=True,optimizer_iterations=result['iterations']))
        save(stage/'state.json',state);verification['published']=True;save(stage/'admission.json',verification);stage.rename(destination)
    save(execution/'admission_verification.json',verification);print(json.dumps(verification,indent=2))


if __name__=='__main__':main()
