"""Prepare independent new augmentations for cold-start throughput comparisons."""
import json
import argparse
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import numpy as np
from examples.arm_hand.batch_kuka_gpu import save,digest
from examples.arm_hand.spatial_augmentation import augment,solve_ik_candidate
from examples.arm_hand.resource_slots import resource_slot


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--max-trials',type=int,default=20);parser.add_argument('--source-indices',nargs='+',type=int,default=[10,15,18]);parser.add_argument('--valid-cases',type=int,default=1);parser.add_argument('--candidate-base',type=int,default=100000);args=parser.parse_args()
    if args.valid_cases<1 or args.max_trials<1:parser.error('Case and trial counts must be positive')
    root=Path(__file__).resolve().parents[2];campaign=root/'outputs/mustard_multisource_valid100'
    selected=[g for g in json.loads((campaign/'selected_sources.json').read_text()) if g['source_index'] in args.source_indices]
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    def prepare(group):
        source=Path(group['source_dir']);group_out=output/f"{group['source_index']:03d}_{group['task']}";group_out.mkdir(exist_ok=True)
        passed=[]
        for trial in range(1,args.max_trials+1):
            index=args.candidate_base+trial;folder=group_out/f'candidate_{index:05d}';folder.mkdir(exist_ok=True)
            if (folder/'state.json').exists():
                state=json.loads((folder/'state.json').read_text())
                if state.get('ik_pass'):
                    passed.append(state)
                    if len(passed)>=args.valid_cases:return passed
                    continue
                if state.get('status')=='ik_rejected':continue
            seed=2026103000+group['source_index'];rng=np.random.default_rng(np.random.SeedSequence([seed,index]))
            yaw=rng.uniform(-10,10);delta=rng.uniform(-.05,.05,2)
            with np.load(source/'source_reference.npz') as f:ref={k:f[k] for k in f.files}
            tree=ET.parse(source/'scene.xml');base=tree.getroot().find("worldbody/body[@name='kuka_base']")
            pos=np.fromstring(base.get('pos'),sep=' ')
            for body in [base,*base.findall('.//body')]:body.set('gravcomp','1')
            tree.write(folder/'scene.xml',encoding='unicode');reference,metadata=augment(ref,yaw,delta,pos)
            np.savez(folder/'source_reference.npz',**reference)
            metadata.update(source_task=group['task'],source_reference_sha256=digest(source/'source_reference.npz'),seed=seed,
                candidate_id=index,independent_experimental_augmentation=True)
            save(folder/'augmentation.json',metadata)
            state=dict(status='generated',candidate_id=index,source_index=group['source_index'],variant=index,task=group['task'],folder=str(folder))
            save(folder/'state.json',state);start=time.monotonic()
            from examples.arm_hand import run_sharpa_complex as isolated_ik
            original_filter=isolated_ik.gaussian_filter1d
            try:
                if group['source_index']==14:isolated_ik.gaussian_filter1d=lambda values,*a,**kw:np.array(values,copy=True)
                with resource_slot(campaign/'resource_slots/ik',8,block=True):state=solve_ik_candidate(state)
            finally:isolated_ik.gaussian_filter1d=original_filter
            state['ik_elapsed_s']=time.monotonic()-start;save(folder/'state.json',state)
            if state.get('ik_pass'):
                passed.append(state);print(json.dumps(dict(source=group['source_index'],case=str(folder),ik_pass=True,passed=len(passed),requested=args.valid_cases)),flush=True)
                if len(passed)>=args.valid_cases:return passed
        return passed+[dict(source_index=group['source_index'],ik_pass=False,status='candidate_budget_rejected',candidate_base=args.candidate_base,requested=args.valid_cases,passed=len(passed))]
    # solve_ik_candidate redirects process-wide stdout/stderr: never share it across threads.
    # Existing completed trials are reused, so serial preparation also resumes interrupted cases.
    results=[state for group in selected for state in prepare(group)]
    manifest=output/'cases.json'
    previous=json.loads(manifest.read_text()) if manifest.exists() else []
    merged={str(state.get('folder',f"budget:{state.get('source_index')}:{state.get('candidate_base',100000)}")):state for state in previous+results}
    save(manifest,list(merged.values()))
    save(output/f'cohort_{args.candidate_base}.json',dict(source_indices=args.source_indices,valid_cases=args.valid_cases,max_trials=args.max_trials,candidate_base=args.candidate_base,cases=results))


if __name__=='__main__':main()
