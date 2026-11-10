"""Enable strict quality gates and a bounded, evidenced model-policy production pilot."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from examples.arm_hand.batch_kuka_gpu import save
from examples.arm_hand.mustard_valid100 import valid


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model-sources',nargs='+',type=int,default=[18]);a=p.parse_args()
    root=Path(__file__).resolve().parents[2];campaign=root/'outputs/mustard_multisource_valid100';evidence=root/'outputs/world_model_pilot/fresh_source018_model'
    if set(a.model_sources)!={18}:raise ValueError('This pilot installer is evidenced for source 18 only; validate each additional source before extending it')
    report=json.loads((evidence/'execution.json').read_text())
    if not report['accepted'] or not valid(report['selected']) or not valid(report['replay']):raise ValueError('Pilot evidence does not pass original double validation')
    original=json.loads((root/'outputs/world_model_pilot/fresh_source018_raw/execution.json').read_text())
    if original['status']!='complete' or not original['accepted']:raise ValueError('Matched original baseline incomplete')
    if report['wall_s']>=original['wall_s']:raise ValueError('Pilot evidence does not improve measured end-to-end execution time')
    policy_hash=report['controller_sha256'];policy=campaign/'policies'/(policy_hash+'.pt')
    if not policy.exists() or hashlib.sha256(policy.read_bytes()).hexdigest()!=policy_hash:raise ValueError('Immutable policy missing/corrupt')
    groups=json.loads((campaign/'selected_sources.json').read_text());deployed=[]
    for group in groups:
        if group['reused_completed_group']:continue
        folder=Path(group['result_dir']);folder.mkdir(exist_ok=True)
        config=dict(version=1,enabled=True,strict_check_every=10,
            transition_root=str(campaign/'model_transitions'),transition_every=10,transition_worlds=8,transition_max_shards=20,
            selection_criteria='Original strict selected + exact-control independent GPU replay, no cost threshold stop')
        if group['source_index'] in a.model_sources:
            config.update(policy=str(policy),policy_sha256=policy_hash,cache_policy_context=True,
                evidence=str(evidence/'execution.json'),pilot=True)
        path=folder/'acceleration.json'
        if path.exists() and json.loads(path.read_text())!=config:save(folder/f'acceleration_before_{time.time_ns()}.json',json.loads(path.read_text()))
        save(path,config);deployed.append(dict(index=group['source_index'],model_enabled=bool(config.get('policy')),config=str(path)))
    save(campaign/'acceleration_deployment.json',dict(version=1,model_sources=a.model_sources,policy_sha256=policy_hash,
        policy=str(policy),evidence_model=str(evidence/'execution.json'),evidence_original=str(root/'outputs/world_model_pilot/fresh_source018_raw/execution.json'),
        matched_execution_speed_ratio=original['wall_s']/report['wall_s'],groups=deployed,
        note='Pilot only in evidenced source. Other sources retain original sampling with exact strict quality early exit. Existing optimizer processes are not interrupted; legacy checkpoints preserve their controller.'))
    print(json.dumps(dict(configured_groups=len(deployed),model_sources=a.model_sources,matched_speed_ratio=original['wall_s']/report['wall_s']),indent=2))


if __name__=='__main__':main()
