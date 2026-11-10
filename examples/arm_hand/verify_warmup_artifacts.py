"""Verify and index complete warmup artifacts, deduplicated with strict data by source."""
import argparse,hashlib,json,time
from pathlib import Path
import mujoco,numpy as np,torch
from examples.arm_hand.spatial_augmentation import augment,KEYS
from examples.arm_hand.warmup_acceptance import VERSION,grasp_retention


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(report_path,groups):
    report=json.loads(report_path.read_text())
    if report['acceptance_version']!=VERSION or not report['warmup_ready']:raise ValueError('Not warmup ready')
    case=Path(report['case']);execution=Path(report['execution']);state=json.loads((case/'state.json').read_text());group=groups[state['source_index']]
    for name,expected in report['input_sha256'].items():
        if digest(Path(name))!=expected:raise ValueError('Artifact changed: '+name)
    for name in ['source_reference.npz','reference.npz','scene.xml','geometry_report.json','augmentation.json']:
        if not (case/name).is_file():raise ValueError('Missing '+name)
    for name in ['checkpoint.pt','history.json','trajectory.npz','replay_trajectory.npz','result.json']:
        if not (execution/'gpu'/name).is_file():raise ValueError('Missing GPU '+name)
    if not Path(report['video']).is_file():raise ValueError('Missing saved-state video')
    if not all(report[key]['warmup_accepted'] for key in ('selected','replay')):raise ValueError('Dual acceptance differs')
    model=mujoco.MjModel.from_xml_path(str(case/'scene.xml'));floor=model.geom('floor').id;base=model.body('kuka_base').id
    if model.nu!=29 or model.nq!=36 or model.body_parentid[base]!=0 or model.geom_type[floor]!=mujoco.mjtGeom.mjGEOM_BOX:raise ValueError('Native fixed KUKA/Sharpa and table expected')
    if not np.isclose(model.geom_pos[floor,2]+model.geom_size[floor,2]-model.body_pos[base,2],.53,atol=1e-10):raise ValueError('Table/base height differs')
    for b in range(1,model.nbody):
        ancestor=b
        while ancestor and ancestor!=base:ancestor=int(model.body_parentid[ancestor])
        if ancestor==base and model.body_gravcomp[b]!=1:raise ValueError('Robot gravity compensation differs')
    if model.body_gravcomp[model.body('right_object').id]!=0:raise ValueError('Object gravity differs')
    metadata=json.loads((case/'augmentation.json').read_text());source=Path(group['source_dir'])/'source_reference.npz'
    if metadata['source_task']!=group['task'] or metadata['source_reference_sha256']!=digest(source):raise ValueError('Original demonstration identity differs')
    if abs(metadata['yaw_deg'])>10 or np.any(np.abs(metadata['sampled_translation_xy_m'])>.05):raise ValueError('Spatial bounds differ')
    with np.load(source) as f:raw={k:f[k] for k in f.files}
    expected,_=augment(raw,metadata['yaw_deg'],np.asarray(metadata['sampled_translation_xy_m']),model.body_pos[base])
    checkpoint=torch.load(execution/'gpu/checkpoint.pt',map_location='cpu',weights_only=False)
    result=json.loads((execution/'gpu/result.json').read_text());signature={n:digest(case/n) for n in ('scene.xml','reference.npz')}
    if checkpoint['signature']!=signature or result['signature']!=signature or result.get('export_policy')!='actual_selected_rollout_v1':raise ValueError('Actual rollout provenance differs')
    with np.load(case/'source_reference.npz') as src,np.load(case/'reference.npz') as ref,np.load(execution/'gpu/trajectory.npz') as tr:
        for key in KEYS:
            np.testing.assert_allclose(src[key],expected[key],atol=1e-12,rtol=0);np.testing.assert_array_equal(src[key],ref[key])
        for key in ('qpos','qvel','ctrl'):np.testing.assert_array_equal(checkpoint['best_rollout'][key].numpy(),tr[key])
    return dict(source_index=group['source_index'],task=group['task'],augmentation_sha256=digest(case/'source_reference.npz'),
                case=str(case),execution=str(execution),report=str(report_path.resolve()),video=report['video'],acceptance_version=VERSION)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,default=Path('outputs/mustard_multisource_valid100'));p.add_argument('--warmup',type=Path,default=Path('outputs/mustard_warmup_grasp_v2'));a=p.parse_args()
    groups={g['source_index']:g for g in json.loads((a.campaign/'selected_sources.json').read_text())};seen={i:set() for i in groups};strict=[];warmup=[];failures=[];strict_not_retained=[];original_strict_count=0;started=time.monotonic()
    for i,g in groups.items():
        path=Path(g['result_dir'])/'accepted.json'
        if not path.exists():continue
        for row in json.loads(path.read_text()):
            case=Path(row['folder']);key=digest(case/'source_reference.npz')
            if key in seen[i]:continue
            original_strict_count+=1
            model=mujoco.MjModel.from_xml_path(str(case/'scene.xml'))
            grasp=[]
            for name in ('trajectory.npz','replay_trajectory.npz'):
                with np.load(case/'gpu'/name) as trace:grasp.append(grasp_retention(model,trace))
            if not all(g['retained'] for g in grasp):
                strict_not_retained.append(dict(case=str(case),source_index=i,grasp=grasp));continue
            seen[i].add(key);strict.append(dict(source_index=i,task=g['task'],case=str(case),augmentation_sha256=key,acceptance_version='original_strict'))
    for path in sorted(a.warmup.glob('*/validation.json')):
        report=json.loads(path.read_text())
        if not report.get('warmup_ready'):continue
        try:
            row=verify(path,groups);i=row['source_index'];key=row['augmentation_sha256']
            if key in seen[i]:row['already_counted']=True
            else:seen[i].add(key);row['already_counted']=False
            warmup.append(row)
        except Exception as exc:failures.append(dict(report=str(path),error=str(exc)))
    per_source=[dict(source_index=i,unique_available=len(keys),count_toward_target=min(100,len(keys))) for i,keys in seen.items()]
    report=dict(acceptance_version=VERSION,target_total=2200,target_per_source=100,strict_count=len(strict),original_strict_count=original_strict_count,strict_not_retained=strict_not_retained,
                verified_warmup_runs=len(warmup),additional_unique_warmup=sum(not r['already_counted'] for r in warmup),
                combined_count_toward_target=sum(r['count_toward_target'] for r in per_source),per_source=per_source,
                strict=strict,warmup=warmup,failures=failures,verification_wall_s=time.monotonic()-started,
                note='Combined dataset uses original strict trajectories plus verified rotation-diverse retained-grasp warmup trajectories. Original strict campaign and frozen benchmark remain unchanged.')
    target=a.warmup/'verified_dataset.json';temporary=target.with_suffix('.tmp');temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(target)
    print(json.dumps({k:v for k,v in report.items() if k not in ('per_source','strict','warmup','strict_not_retained')}),flush=True)


if __name__=='__main__':main()
