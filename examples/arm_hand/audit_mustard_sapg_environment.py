"""Save an exhaustive leaf comparison and authored asset metadata, read-only to runs."""
import argparse
import hashlib
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import xml.etree.ElementTree as ET


def flatten(value,prefix=''):
    if not isinstance(value,dict):return {prefix:value}
    result={}
    for key,item in value.items():result.update(flatten(item,prefix+'.'+key if prefix else key))
    return result


def asset(path):
    root=ET.parse(path).getroot()
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        mass_authored_kg=sum(float(e.attrib['value']) for e in root.findall('./link/inertial/mass')),
        links=len(root.findall('link')),collisions=len(root.findall('./link/collision')),
        joints=[dict(name=j.attrib['name'],type=j.attrib['type'],
            limit=j.find('limit').attrib if j.find('limit') is not None else None,
            dynamics=j.find('dynamics').attrib if j.find('dynamics') is not None else None)
            for j in root.findall('joint')],
        note='Authored URDF values, not a runtime PhysX inertia dump; historic asset content is not archived in Hydra.')


def main():
    import yaml
    import numpy as np
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,default=Path('/home/szc/work/simtoolreal_origin/simtoolreal/outputs/2026-09-14/14-35-39'))
    parser.add_argument('--current',type=Path,default=Path('/home/szc/work/spider/outputs/mustard_sapg_targetpose_v1'))
    parser.add_argument('--output',type=Path,default=Path('/home/szc/work/spider/outputs/mustard_sapg_environment_audit_v1'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    old=yaml.safe_load((args.baseline/'.hydra/config.yaml').read_text())
    new=dict(env=json.loads((args.current/'environment_config.json').read_text()),
        agent=json.loads((args.current/'sapg_config.json').read_text()))
    rows=[]
    for group in ('env','agent'):
        a,b=flatten(old[group]),flatten(new[group])
        for key in sorted(a.keys()|b.keys()):
            rows.append(dict(path=group+'.'+key,old_present=key in a,new_present=key in b,
                old=a.get(key),new=b.get(key),equal=key in a and key in b and a[key]==b[key]))
    params=json.loads(Path('/home/szc/work/spider/outputs/mustard_isaac_assets_v4/robot_spec.json').read_text())
    repo=Path('/home/szc/work/simtoolreal_origin/simtoolreal')
    assets={name:asset(path) for name,path in dict(
        historical_robot=repo/old['env']['assets']['robot_urdf'],current_robot=Path(params['robot_urdf']),
        historical_table=repo/old['env']['assets']['table_urdf'],current_table=Path(params['table_urdf']),
        current_object=Path(new['env']['assets']['object_urdf'])).items()}
    initial=np.load('/home/szc/work/spider/outputs/mustard_isaac_assets_v4/demo_initials.npz')
    a,b=initial['object_pose'],initial['goal_pose']
    position=np.linalg.norm(a[:,:3]-b[:,:3],axis=1)
    angle=np.rad2deg(2*np.arccos(np.abs((a[:,3:]*b[:,3:]).sum(1)).clip(0,1)))
    initial_summary={k:dict(min=v.min(0).tolist(),max=v.max(0).tolist()) for k,v in initial.items() if k!='candidate_ids'}
    initial_summary.update(relative_goal_distance_m=[float(position.min()),float(position.max())],
        relative_goal_angle_deg=[float(angle.min()),float(angle.max())])
    files=[args.baseline/'.hydra/config.yaml',args.current/'environment_config.json',args.current/'sapg_config.json',
        Path(__file__).with_name('train_mustard_actor_isaac.py'),Path(__file__).with_name('run_mustard_sapg.py')]
    audit=dict(captured_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        baseline=str(args.baseline),current=str(args.current),
        field_counts={g:dict(equal=sum(r['equal'] for r in rows if r['path'].startswith(g+'.')),
            different=sum(not r['equal'] for r in rows if r['path'].startswith(g+'.'))) for g in ('env','agent')},
        leaves=rows,assets=assets,current_robot_spec=params,initial_states=initial_summary,
        current_progress=json.loads((args.current/'sapg_progress.jsonl').read_text().splitlines()[-1]),
        evidence_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        limitations=['Hydra dimensions precede runtime robot dimension recomputation.',
            'Historical launch command, package versions, frozen source and generated USDs are not present in the run directory.',
            'Current repository and URDFs may differ from their September contents. Checkpoint shapes are stronger evidence for historic dimensions.',
            'Raw differences include dormant options and serialization/numeric-type differences; runtime overrides require code review.'])
    (args.output/'comparison.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(audit['field_counts']))


if __name__=='__main__':main()
