"""Refresh verified physics, adapt models with source014, then test fresh augmentations.

Acceptance remains warmup_grasp_v2. This is a new training version, never a
mutation of earlier frozen controllers or their data snapshots.
"""
import contextlib,fcntl,hashlib,json,os,re,shutil,subprocess,sys,time
from pathlib import Path
import mujoco,numpy as np
from examples.arm_hand import run_sharpa_complex as ik
from examples.arm_hand.resource_slots import resource_slot
from examples.arm_hand.spatial_augmentation import check_ik,KEYS
from examples.arm_hand.validate_kuka_sharpa import PLAY2PERFECT_ARM_POSE


def save(path,data):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def main():
 root=Path(__file__).resolve().parents[2];pilot=root/'outputs/world_model_pilot';out=pilot/'warmup_v3_source014';out.mkdir(exist_ok=False);started=time.monotonic();status=dict(status='running',stage='verification',steps=[]);save(out/'status.json',status)
 def run(name,cmd):
  status['stage']=name;save(out/'status.json',status);start=time.monotonic()
  with (out/(name+'.log')).open('w') as log:result=subprocess.run([sys.executable,'-u',*map(str,cmd)],cwd=root,stdout=log,stderr=subprocess.STDOUT)
  status['steps'].append(dict(stage=name,returncode=result.returncode,wall_s=time.monotonic()-start));save(out/'status.json',status)
  if result.returncode:raise RuntimeError('Stage failed: '+name)
 try:
  run('refresh',['-m','examples.arm_hand.collect_warmup_candidates','--render','--output',root/'outputs/mustard_warmup_grasp_v2'])
  run('verify',['-m','examples.arm_hand.verify_warmup_artifacts','--warmup',root/'outputs/mustard_warmup_grasp_v2'])
  snapshot=out/'dataset_manifest.json';shutil.copy2(root/'outputs/mustard_warmup_grasp_v2/verified_dataset.json',snapshot);index=json.loads(snapshot.read_text())
  source14=[r for r in index['warmup'] if r['source_index']==14 and not r['already_counted']]
  if len(source14)<4 or index['failures']:raise ValueError('Need all four fully verified source014 teachers, with no provenance failures')
  run('policy_data',['-m','examples.arm_hand.prepare_warmup_policy_data','--manifest',snapshot,'--output',out/'policy_data'])
  groups={g['source_index']:g['task'] for g in json.loads((root/'outputs/mustard_multisource_valid100/selected_sources.json').read_text())}
  # Normalize original-source aliases before splitting; preserve source-file hashes.
  records=[]
  for name,inputs in [('world_data',[pilot/'transitions_v2',pilot/'source014_contact_replication/transitions']),('counterfactual',[pilot/'counterfactual_v1'])]:
   dest=out/name;dest.mkdir()
   for source in inputs:
    for path in sorted(source.rglob('transitions_*.npz')):
     with np.load(path) as f:data={k:f[k].copy() for k in f.files}
     meta=json.loads(str(data['metadata']));original=meta['source_group'];meta['source_group']=re.sub(r'^\d{3}_','',original);meta['original_source_group_alias']=original;meta['source_file_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();data['metadata']=json.dumps(meta)
     target=dest/('transitions_'+hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:24]+'.npz');np.savez_compressed(target,**data);records.append(dict(path=str(target),source=str(path),source_group=meta['source_group'],sha256=meta['source_file_sha256']))
  save(out/'transition_manifest.json',dict(records=records,normalization='strip only numeric source-index prefix, preserving original demonstration identity'))
  run('bc',['-m','examples.arm_hand.world_model_policy','train','--data',out/'policy_data','--output',out/'bc','--holdout-group',groups[18],'--test-group',groups[20],'--test-group',groups[22],'--source-balanced','--updates','1500','--eval-every','100','--device','cpu'])
  run('world',['-m','examples.arm_hand.world_model','train','--data',out/'world_data','--output',out/'world','--holdout-group',groups[3],'--test-group',groups[15],'--architecture','compact','--width','1024','--initialize',pilot/'compact_causal/model.pt','--counterfactual-data',out/'counterfactual','--counterfactual-weight','50','--object-position-weight','25','--object-rotation-weight','5','--window','8','--batch','256','--updates','1000','--eval-every','100','--lr','.00001','--device','cuda:0'])
  run('imagination',['-m','examples.arm_hand.world_model_policy','improve','--data',out/'policy_data','--policy',out/'bc/policy.pt','--world',out/'world/model.pt','--output',out/'imagination','--anchored-world','--collision-weight','1000000','--rotation-weight','0','--max-reset-rotation-deg','180','--critic-weight','0','--skip-critic-training','--anchor','1000','--lr','.00001','--horizon','8','--batch','32','--updates','100','--eval-every','10','--device','cpu'])
  # Hold new source014 augmentations out of this version's model/teacher datasets.
  cases=[];prepstart=time.monotonic();(out/'fresh_cases').mkdir()
  for number in range(300007,300015):
   source=next((pilot/'fresh_eval_cases').glob(f'014_*/candidate_{number}'));case=out/'fresh_cases'/f'candidate_{number}';case.mkdir()
   for name in ['scene.xml','source_reference.npz','augmentation.json']:shutil.copy2(source/name,case/name)
   state=json.loads((source/'state.json').read_text());state.update(folder=str(case.resolve()),status='generated');save(case/'state.json',state)
   with np.load(case/'source_reference.npz') as f:ref={k:f[k].copy() for k in f.files}
   frozen={k:ref[k].copy() for k in KEYS};model=mujoco.MjModel.from_xml_path(str(case/'scene.xml'));original=ik.gaussian_filter1d;trialstart=time.monotonic()
   try:
    with resource_slot(root/'outputs/mustard_multisource_valid100/resource_slots/ik',8,block=True):
     try:
      ik.gaussian_filter1d=lambda values,*args,**kwargs:np.array(values,copy=True)
      with (case/'ik.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):ik.fit(model,ref,case,2,initial_arm_pose=PLAY2PERFECT_ARM_POSE,first_frame_restarts=8,reject_bad_initial=True)
     finally:ik.gaussian_filter1d=original
    for k in KEYS:np.testing.assert_array_equal(ref[k],frozen[k])
    g=check_ik(case);state.update(ik_pass=g['ik_pass'],status='ik_passed' if g['ik_pass'] else 'ik_rejected')
   except Exception as exc:state.update(ik_pass=False,status='ik_rejected',error=str(exc))
   state['ik_elapsed_s']=time.monotonic()-trialstart;save(case/'state.json',state)
   if state['ik_pass']:cases.append(state)
   if len(cases)==2:break
  status['fresh_case_preparation_wall_s']=time.monotonic()-prepstart
  if not cases:status.update(status='no_ik_cases');save(out/'status.json',status);return
  save(out/'cohort_input.json',dict(source_indices=[14],valid_cases=len(cases),cases=cases,limitation='New augmentation evaluation on a now-trained original source; not unseen-original-source evidence'))
  run('physical_cohort',['-m','examples.arm_hand.run_contact_every_step_cohort','--cohort',out/'cohort_input.json','--bc',out/'bc/policy.pt','--model',out/'imagination/policy.pt','--output',out/'cohort','--mps-pipe-dir','/tmp/spider-mps-mustard-1005','--max-runs',str(3*len(cases))])
  run('final_refresh',['-m','examples.arm_hand.collect_warmup_candidates','--render','--output',root/'outputs/mustard_warmup_grasp_v2'])
  run('final_verify',['-m','examples.arm_hand.verify_warmup_artifacts','--warmup',root/'outputs/mustard_warmup_grasp_v2'])
  status.update(status='complete',wall_s=time.monotonic()-started);save(out/'status.json',status)
 except Exception as exc:
  status.update(status='infrastructure_failed',error=str(exc),wall_s=time.monotonic()-started);save(out/'status.json',status);raise
if __name__=='__main__':main()
