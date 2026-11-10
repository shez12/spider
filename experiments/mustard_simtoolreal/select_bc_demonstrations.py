"""Select native demonstrations that grasp and reach within the original horizon.

Cached observation t is the state AFTER action t-1. The successful terminal
observation is used for validation, while BC keeps actions 0..t-1 only.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent


def select(run,output,waypoints,horizon=600,tolerance=.1125):
    assert not output.exists(),'Use a new selection directory.'
    d=np.load(run/'native_bc_dataset.npz')
    assert str(d['action_contract'])=='simtoolreal_native_v1' and not bool(d['smoke_test'])
    x=d['obs'];a=d['actions'];ids=d['candidate_ids'];lengths=d['valid_lengths']
    cfg=json.loads((run/'collection_environment.json').read_text())
    dr=cfg['domain_randomization']
    assert not dr['use_obs_delay'] and not dr['use_object_state_delay_noise']
    assert cfg['termination']['episode_length']==horizon
    assert np.isclose(tolerance,cfg['termination']['success_tolerance']*cfg['reward']['keypoint_scale'])
    payload=json.loads(waypoints.read_text());pool={p['candidate_id']:p for p in payload['trajectories']}
    manifest=json.loads((run/'source_manifest.json').read_text());source_by_id={m['candidate_id']:m for m in manifest}
    selected=[];rows=[];cutoffs=[]
    for i,c in enumerate(ids):
        c=int(c);obs=x[i];palm=obs[:,87:90]
        object_position=obs[:,113:125].reshape(-1,4,3).mean(1)+palm
        fingertips=obs[:,98:113].reshape(-1,5,3)+palm[:,None]
        speed=np.r_[0,np.linalg.norm(np.diff(object_position,axis=0),axis=-1)/(.016666666666666666)]
        height=object_position[:,2]-pool[c]['initial_pose'][2]
        hold=(height>=.05)&(np.linalg.norm(object_position-palm,axis=-1)<.18)
        hold&=(np.sort(np.linalg.norm(fingertips-object_position[:,None],axis=-1),axis=-1)[:,:3].mean(-1)<.10)
        # Conservative finite-difference speed gate; source native rollout used2m/s.
        hold&=speed<1.5
        distance=np.linalg.norm(obs[:,125:137].reshape(-1,4,3),axis=-1).max(-1)
        hits=0;hold_run=0;max_hold=0;terminal=None
        for step in range(1,min(horizon,int(lengths[i]),len(obs)-1)+1):
            hold_run=hold_run+1 if hold[step] else 0;max_hold=max(max_hold,hold_run)
            hits+=int(distance[step]<=tolerance)
            if hits>=cfg['termination']['success_steps']:
                terminal=step;break
        eligible=terminal is not None and max_hold>=30
        rows.append(dict(candidate_id=c,eligible=eligible,terminal_step=terminal,
            held_steps_before_success=max_hold,near_goal_steps=hits))
        if eligible:selected.append(i);cutoffs.append(terminal)
    assert selected,'No demonstrations passed both grasp and reaching criteria.'
    output.mkdir(parents=True)
    shutil.copy2(run/'collection_environment.json',output/'collection_environment.json')
    total=max(cutoffs)
    values=dict(obs=x[selected,:total].copy(),actions=a[selected,:total].copy(),
        candidate_ids=ids[selected],valid_lengths=np.array(cutoffs),
        action_contract=d['action_contract'],smoke_test=np.array(False))
    indices=d['source_indices']
    values['source_indices']=indices[:total,selected] if indices.ndim==2 else indices[:total]
    for i,length in enumerate(cutoffs):
        values['obs'][i,length:]=values['obs'][i,length-1]
        values['actions'][i,length:]=0 # padding never contributes to BC
    np.savez_compressed(output/'native_bc_dataset.npz',**values)
    accepted=[source_by_id[int(ids[i])] for i in selected]
    chosen_pool=dict(payload);chosen_pool['trajectories']=[pool[int(ids[i])] for i in selected]
    def save(name,value):(output/name).write_text(json.dumps(value,indent=2)+'\n')
    save('source_manifest.json',accepted);save('waypoints.json',chosen_pool)
    save('grasp_validation.json',dict(teacher_control='native',trajectories=len(selected),
        original_horizon_result=dict(policy_steps=horizon,held_lift_proxy_30_steps=len(selected),
            reached_initial_tolerance_10_steps=len(selected)),
        selection_method='Cached native post-action states; hold proxy before first goal success; finite-difference speed<1.5m/s',
        task='fixed final pose reaching at original initial success tolerance',tolerance_m=tolerance,
        candidate_ids=ids[selected].tolist()))
    audit=json.loads((run/'retiming_audit.json').read_text())
    audit.update(parent_run=str(run),candidate_ids=ids[selected].tolist(),valid_lengths=cutoffs,
        bc_truncated_at_first_success=True)
    save('retiming_audit.json',audit)
    report=dict(parent_run=str(run),original_count=len(ids),selected_count=len(selected),
        candidate_ids=ids[selected].tolist(),valid_lengths=cutoffs,supervised_frames=sum(cutoffs),
        horizon_steps=horizon,tolerance_m=tolerance,required_hold_steps=30,
        cache_sha256=hashlib.sha256((output/'native_bc_dataset.npz').read_bytes()).hexdigest(),rows=rows)
    save('selection.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--waypoints',type=Path,default=ROOT/'waypoints100.json')
    args=parser.parse_args();select(args.run.resolve(),args.output.resolve(),args.waypoints)
