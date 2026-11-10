"""Reset bank v4.1: v3 + new states from source 5 (z+2mm), E4 v2 object edits and E5 robot starts, deduplicated.

Only trajectories that hold in Isaac (teacher hold proxy) are used, at full length (no 960-step cap).
New states pass the same build/package audits as v2/v3, are merged after v3 (v3 keeps priority),
then near-duplicates are removed with dedup_reset_bank.py.
"""
import json,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path('/home/szc/work/spider');O=ROOT/'outputs';BASE=O/'mustard_diverse_reset_bank_v41'
EXP=ROOT/'experiments/mustard_simtoolreal';PY='/home/szc/.conda/envs/simtoolreal_origin_isaacsim/bin/python'
sys.path.insert(0,str(EXP))
from truncated_selection import first_hold_end,hold_mask
from bc_balance import audit


def load(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}


def main():
    inputs=BASE/'inputs';inputs.mkdir(parents=True,exist_ok=False)
    parts=[]
    a=load(O/'mustard_classified_20261104_truncated_z2mm_v1/selected/native_bc_dataset.npz')
    man={m['candidate_id']:m for m in json.loads((O/'mustard_classified_20261104_truncated_z2mm_v1/selected/source_manifest.json').read_text())}
    parts.append((a,np.flatnonzero([man[int(c)]['source_index']==5 for c in a['candidate_ids']]),O/'mustard_classified_20261104_truncated_z2mm_v1/selected/waypoints.json'))
    for name in ('mustard_e4_window_v2','mustard_e5_robot_start_v1'):
        d=load(O/name/'teacher/native_bc_dataset.npz')
        held=np.array([i for i in range(len(d['obs'])) if first_hold_end(hold_mask(d['obs'][i]),d['valid_lengths'][i])[0] is not None])
        parts.append((d,held,O/name/'waypoints.json'))
    T=max(int(d['valid_lengths'][i].max()) for d,i,_ in parts)
    def take(d,idx):
        o=d['obs'][idx,:T];act=d['actions'][idx,:T];si=d['source_indices'][:T,idx]
        if o.shape[1]<T:
            pad=T-o.shape[1];o=np.concatenate([o,np.repeat(o[:,-1:],pad,1)],1);act=np.concatenate([act,np.zeros((len(idx),pad,29),act.dtype)],1);si=np.concatenate([si,np.repeat(si[-1:],pad,0)],0)
        return o,act,si
    obs,act,si,ids,lengths,way=[],[],[],[],[],{}
    for d,idx,wp in parts:
        o,ac,s=take(d,idx);obs.append(o);act.append(ac);si.append(s);ids.append(d['candidate_ids'][idx]);lengths.append(d['valid_lengths'][idx])
        way.update({t['candidate_id']:t for t in json.loads(wp.read_text())['trajectories']})
    ids=np.concatenate(ids)
    np.savez_compressed(inputs/'new_states_dataset.npz',obs=np.concatenate(obs),actions=np.concatenate(act),source_indices=np.concatenate(si,1),
        candidate_ids=ids,valid_lengths=np.concatenate(lengths),action_contract=np.array('simtoolreal_native_v1'),smoke_test=np.array(False))
    (inputs/'waypoints.json').write_text(json.dumps(dict(format='bank_v41_new_states',trajectories=[way[int(c)] for c in ids])))
    print('new trajectories',{p[2].parent.name if p[2].parent.name!='selected' else 'source5':len(p[1]) for p in parts},flush=True)
    audit(inputs/'new_states_dataset.npz',inputs/'audit')
    def run(name,cmd):
        with (BASE/(name+'.log')).open('w') as log:assert subprocess.run(list(map(str,cmd)),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode==0,name
    run('new_build',['bash',ROOT/'examples/arm_hand/run_local_simtoolreal.sh','build-reset-bank','--headless','--dataset',inputs/'new_states_dataset.npz',
        '--waypoints',inputs/'waypoints.json','--audit',inputs/'audit/audit.json','--output',BASE/'new_build'])
    run('package',[PY,EXP/'package_diverse_reset_bank.py','--source',BASE/'new_build','--output',BASE/'new_packaged'])
    run('merge',[PY,EXP/'merge_reset_banks.py','--sources',O/'mustard_diverse_reset_bank_v3/merged',BASE/'new_packaged','--output',BASE/'merged_raw'])
    run('dedup',[PY,EXP/'dedup_reset_bank.py','--source',BASE/'merged_raw','--output',BASE/'merged'])
    print(open(BASE/'merged/completed.json').read())


if __name__=='__main__':main()
