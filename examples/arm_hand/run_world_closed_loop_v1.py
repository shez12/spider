"""One bounded real replay -> model/BC fit -> imagination -> fresh physical comparison cycle."""
from pathlib import Path
import os,json,time,subprocess,fcntl,re,hashlib
import numpy as np,mujoco
from examples.arm_hand.world_model import scene_spec,save_shard

def save(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def main():
 r=Path(__file__).resolve().parents[2];pilot=r/'outputs/world_model_pilot';o=pilot/'closed_loop_v1';o.mkdir(exist_ok=False);s=time.time();status=dict(status='waiting_verified_batch',steps=[]);save(o/'status.json',status)
 groups={g['source_index']:g['task'] for g in json.loads((r/'outputs/mustard_multisource_valid100/selected_sources.json').read_text())}
 save(o/'protocol.json',dict(kind='Bounded development closed-loop cycle, not unseen-source proof',dependency_pid=3864715,dependency_identity='source014_contact_replication_v3.recovery.py',excluded_actor_training=[groups[20],groups[22]],world_validation=groups[3],world_test=groups[15],fresh_candidate_base=400000,max_ik_trials_per_source=32,physical_budget=dict(samples=512,max_iterations=200,check_every=1,candidate_pool=16),world_initializer_sha256=hashlib.sha256((pilot/'warmup_v3_source014/world/model.pt').read_bytes()).hexdigest(),old_cohort='shared_seed_trust_v1 frozen',cost_accounting='Record all waiting/preparation/training/failure/dual-validation; no promotion until full comparison'))
 # Do not steal a third world-work slot while the baseline batch is alive.
 while True:
  try:alive=b'source014_contact_replication_v3.recovery.py' in Path('/proc/3864715/cmdline').read_bytes()
  except FileNotFoundError:alive=False
  if not alive:break
  status['wait_s']=time.time()-s;save(o/'status.json',status);time.sleep(10)
 prior=json.loads((pilot/'source014_contact_replication_v3/status.json').read_text())
 if prior['status']!='complete' or prior['valid']<1:raise RuntimeError('Prior physical batch did not finish normally')
 def run(label,module,args):
  status['status']='running';status['stage']=label;save(o/'status.json',status);t=time.time()
  with (o/(label+'.log')).open('w') as f:rc=subprocess.run([os.sys.executable,'-u','-m',module,*map(str,args)],cwd=r,stdout=f,stderr=subprocess.STDOUT).returncode
  status['steps'].append(dict(stage=label,returncode=rc,wall_s=time.time()-t));save(o/'status.json',status)
  if rc:raise RuntimeError(label+' failed')
 # Reuse the global writer lock; preserve every old snapshot.
 with (r/'outputs/mustard_warmup_grasp_v2/heartbeat_refresh.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  run('refresh','examples.arm_hand.collect_warmup_candidates',['--render','--output',r/'outputs/mustard_warmup_grasp_v2'])
  run('verify','examples.arm_hand.verify_warmup_artifacts',['--warmup',r/'outputs/mustard_warmup_grasp_v2'])
 snapshot=r/'outputs/mustard_warmup_grasp_v2/verified_dataset.json';(o/'dataset_manifest.json').write_bytes(snapshot.read_bytes());index=json.loads(snapshot.read_text())
 if index['failures']:raise ValueError('Dataset provenance failures')
 run('policy_data','examples.arm_hand.prepare_warmup_policy_data',['--manifest',o/'dataset_manifest.json','--output',o/'policy_data'])
 data=o/'world_data';data.mkdir();lineage=[]
 for src in [pilot/'warmup_v3_source014/world_data',pilot/'source014_contact_replication_v2/transitions',pilot/'source014_contact_replication_v3/transitions']:
  for p in sorted(src.rglob('transitions_*.npz')):
   with np.load(p) as z:payload={k:z[k].copy() for k in z.files}
   meta=json.loads(str(payload['metadata']));original=meta['source_group'];canonical=re.sub(r'^\d{3}_','',original)
   if canonical in [groups[20],groups[22]]:raise ValueError('Excluded original source in model replay')
   meta.update(source_group=canonical,closed_loop_input_sha256=hashlib.sha256(p.read_bytes()).hexdigest());payload['metadata']=json.dumps(meta);dest=data/('transitions_'+hashlib.sha256(str(p.resolve()).encode()).hexdigest()[:24]+'.npz');np.savez_compressed(dest,**payload);lineage.append(dict(path=str(dest),source=str(p),sha256=meta['closed_loop_input_sha256']))
 # Current-controller real states/actions close the data-distribution loop.
 for arm in ['raw','bc','world']:
  execution=pilot/'shared_seed_trust_v1'/('14_'+arm);rep=json.loads((execution/'execution.json').read_text());case=Path(rep['case']);model,spec,context=scene_spec(case/'scene.xml');cpu=mujoco.MjData(model)
  with np.load(execution/'gpu/trajectory.npz') as tr:
   mujoco.mj_setState(model,cpu,tr['initial_state'],mujoco.mjtState.mjSTATE_FULLPHYSICS);dest=data/('transitions_current_policy_'+arm+'.npz');save_shard(dest,tr['qpos'][None],tr['qvel'][None],tr['ctrl'][None],cpu.qpos,cpu.qvel,[np.nan],spec,groups[14],context)
  lineage.append(dict(path=str(dest),source=str(execution/'gpu/trajectory.npz'),sha256=hashlib.sha256((execution/'gpu/trajectory.npz').read_bytes()).hexdigest(),executed_controller=arm))
 save(o/'transition_manifest.json',dict(records=lineage,policy_data_snapshot_sha256=hashlib.sha256((o/'dataset_manifest.json').read_bytes()).hexdigest()))
 run('bc','examples.arm_hand.world_model_policy',['train','--data',o/'policy_data','--output',o/'bc','--holdout-group',groups[18],'--test-group',groups[20],'--test-group',groups[22],'--source-balanced','--updates','1500','--eval-every','100','--device','cpu'])
 run('world','examples.arm_hand.world_model',['train','--data',data,'--output',o/'world','--holdout-group',groups[3],'--test-group',groups[15],'--architecture','compact','--width','1024','--initialize',pilot/'warmup_v3_source014/world/model.pt','--counterfactual-data',pilot/'warmup_v3_source014/counterfactual','--counterfactual-weight','50','--object-position-weight','25','--object-rotation-weight','5','--window','8','--batch','256','--updates','1000','--eval-every','100','--lr','.00001','--device','cuda:0'])
 run('imagination','examples.arm_hand.world_model_policy',['improve','--data',o/'policy_data','--policy',o/'bc/policy.pt','--world',o/'world/model.pt','--output',o/'imagination','--anchored-world','--collision-weight','1000000','--rotation-weight','0','--max-reset-rotation-deg','180','--critic-weight','0','--skip-critic-training','--anchor','1000','--lr','.00001','--horizon','8','--batch','32','--updates','100','--eval-every','10','--device','cpu'])
 run('prepare','examples.arm_hand.prepare_closed_loop_cases',['--output',o/'fresh_cases','--source-indices','14','20','--valid-cases','1','--max-trials','32','--candidate-base','400000'])
 cases=json.loads((o/'fresh_cases/cohort_400000.json').read_text())['cases'];eligible=[x for x in cases if x.get('ik_pass')];save(o/'cohort_input.json',dict(cases=eligible,source_indices=[14,20],source_gate_outcomes=cases,limitation='Fresh uniform augmentations, source014 trained and source020 training-excluded but previously diagnosed original source'))
 if eligible:run('physical_cohort','examples.arm_hand.run_contact_guarded_cohort',['--cohort',o/'cohort_input.json','--bc',o/'bc/policy.pt','--model',o/'imagination/policy.pt','--output',o/'cohort','--mps-pipe-dir','/tmp/spider-mps-mustard-1005','--max-runs',3*len(eligible)])
 with (r/'outputs/mustard_warmup_grasp_v2/heartbeat_refresh.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);run('final_refresh','examples.arm_hand.collect_warmup_candidates',['--render','--output',r/'outputs/mustard_warmup_grasp_v2']);run('final_verify','examples.arm_hand.verify_warmup_artifacts',['--warmup',r/'outputs/mustard_warmup_grasp_v2'])
 status.update(status='complete' if len({x['source_index'] for x in eligible})==2 else 'complete_with_missing_source_ik',wall_s=time.time()-s,valid_cases=len(eligible),end_to_end_advantage_proven=False);save(o/'status.json',status)
if __name__=='__main__':main()
