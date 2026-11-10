"""Rebuild the native BC set, truncating long held demonstrations instead of discarding them.

Same teacher cache and hold proxy as outputs/mustard_classified_20261102_v2; the only
change is the duration rule (see experiments/mustard_simtoolreal/truncated_selection.py).
Writes a fresh directory; the v2 selection is untouched.
"""
import argparse,hashlib,json,shutil,sys
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/mustard_simtoolreal'))
from truncated_selection import VERSION,HOLD_STEPS,cutoff,first_hold_end,hold_mask

LYING={2,4,5,8,9,13,16,19,21,22} # source object starts at 80-100 deg tilt


def main(a):
    teacher=a.teacher;out=a.output/'selected'
    assert not a.output.exists(),'Use a fresh output directory.'
    quality=json.loads((teacher/'grasp_validation.json').read_text())
    manifest=json.loads((a.inputs/'source_manifest.json').read_text());pool=json.loads((a.inputs/'waypoints.json').read_text())
    with np.load(teacher/'native_bc_dataset.npz') as d:
        assert str(d['action_contract'])=='simtoolreal_native_v1' and not bool(d['smoke_test'])
        obs=d['obs'];actions=d['actions'];ids=d['candidate_ids'];lengths=d['valid_lengths'];indices=d['source_indices']
    assert [m['candidate_id'] for m in manifest]==ids.tolist()==quality['candidate_ids']
    saved=np.array(quality['longest_hold_proxy_steps']);rows=[];keep=[];cuts=[]
    for i,c in enumerate(ids.tolist()):
        end,longest=first_hold_end(hold_mask(obs[i]),lengths[i])
        # Offline finite-difference speed can disagree with the online proxy at the 30-step boundary; reject those.
        mismatch=(longest>=HOLD_STEPS)!=(saved[i]>=HOLD_STEPS)
        cut=None if mismatch else cutoff(int(lengths[i]),end,a.max_steps,a.margin)
        rule='reconstruction_mismatch' if mismatch else 'full' if cut==int(lengths[i]) else 'truncated' if cut else 'never_held' if end is None else 'hold_too_late'
        src=manifest[i]['source_index']
        rows.append(dict(candidate_id=c,source_index=src,lying_source=src in LYING,valid_length=int(lengths[i]),
            hold_end_step=end,longest_hold_offline=int(longest),longest_hold_teacher=int(saved[i]),cutoff=cut,rule=rule))
        if cut:keep.append(i);cuts.append(cut)
    out.mkdir(parents=True);total=max(cuts)
    o=obs[keep,:total].copy();act=actions[keep,:total].copy()
    for j,cut in enumerate(cuts):
        o[j,cut:]=o[j,cut-1];act[j,cut:]=0 # padding never contributes to BC or normalization
    np.savez_compressed(out/'native_bc_dataset.npz',obs=o,actions=act,candidate_ids=ids[keep],valid_lengths=np.array(cuts),
        source_indices=indices[:total,keep] if indices.ndim==2 else indices[:total],
        action_contract=np.array('simtoolreal_native_v1'),smoke_test=np.array(False))
    chosen=set(ids[keep].tolist())
    def save(path,value):path.write_text(json.dumps(value,indent=2,default=str)+'\n')
    save(out/'source_manifest.json',[m for m in manifest if m['candidate_id'] in chosen])
    save(out/'waypoints.json',dict(pool,trajectories=[t for t in pool['trajectories'] if t['candidate_id'] in chosen]))
    save(out/'grasp_validation.json',dict(teacher_control='native',held_lift_proxy_30_steps=len(keep),trajectories=len(keep),
        native_parent=str(teacher),selection_version=VERSION))
    save(out/'retiming_audit.json',dict(valid_lengths=cuts,candidate_ids=sorted(chosen),collection_envs=1032,excluded_padding=True,
        truncated_after_hold=sum(r['rule']=='truncated' for r in rows)))
    shutil.copy2(teacher/'collection_environment.json',out/'collection_environment.json')
    count=lambda rule,lying=None:sum(r['rule']==rule and (lying is None or r['lying_source']==lying) for r in rows)
    summary=dict(selection_version=VERSION,teacher=str(teacher),inputs=str(a.inputs),max_steps=a.max_steps,margin_steps=a.margin,
        hold_proxy='teacher proxy on cached post-action observations; finite-difference speed<2m/s; 30 consecutive steps',
        rule='keep full if valid length<=max_steps; else cut at first-hold end+1+margin if <=max_steps; else reject',
        input_count=len(rows),selected_count=len(keep),full=count('full'),truncated=count('truncated'),
        rejected_never_held=count('never_held'),rejected_hold_too_late=count('hold_too_late'),rejected_reconstruction_mismatch=count('reconstruction_mismatch'),
        lying=dict(selected=count('full',True)+count('truncated',True),full=count('full',True),truncated=count('truncated',True),total=sum(r['lying_source'] for r in rows)),
        per_source={str(s):dict(input=n,selected=sum(r['cutoff'] is not None for r in rows if r['source_index']==s),
            truncated=sum(r['rule']=='truncated' for r in rows if r['source_index']==s))
            for s,n in sorted(Counter(r['source_index'] for r in rows).items())},
        supervised_frames=int(sum(cuts)),cache_sha256=hashlib.sha256((out/'native_bc_dataset.npz').read_bytes()).hexdigest())
    save(out/'selection.json',dict(summary,rows=rows))
    save(a.output/'native_selection.json',dict(summary,candidate_ids=sorted(chosen),valid_lengths=cuts))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--teacher',type=Path,default=ROOT/'outputs/mustard_classified_20261102_v2/teacher')
    p.add_argument('--inputs',type=Path,default=ROOT/'outputs/mustard_classified_native_20261102_inputs')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/mustard_classified_20261104_truncated_v1')
    p.add_argument('--max-steps',type=int,default=960);p.add_argument('--margin',type=int,default=60)
    main(p.parse_args())
