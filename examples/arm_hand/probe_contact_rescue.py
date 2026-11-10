"""Bounded raw-SPIDER feasibility probes on previously rejected source candidates.

The added physical self-contact cost is explicit. No world-model acceleration
is inferred, and original production states/configuration are left untouched.
"""
import hashlib,json,subprocess,sys,time,shutil,fcntl
from pathlib import Path


def save(path,data):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def main():
 root=Path(__file__).resolve().parents[2];pilot=root/'outputs/world_model_pilot';out=pilot/'contact_rescue_sources_v1';out.mkdir(exist_ok=False);started=time.monotonic();groups=json.loads((root/'outputs/mustard_multisource_valid100/selected_sources.json').read_text());plans=[]
 for group in groups:
  if group['source_index'] not in [3,6,7,11,12,20]:continue
  choices=[]
  for path in sorted(Path(group['result_dir']).glob('candidate_*/state.json')):
   try:state=json.loads(path.read_text())
   except (OSError,json.JSONDecodeError):continue
   if state.get('status')!='physics_rejected' or not state.get('ik_pass') or not (path.parent/'gpu/result.json').is_file():continue
   m=state.get('selected_metrics',{})
   if not m:continue
   score=(not (m['object_rmse_m']<.03 and m['final_object_position_error_m']<.03),m['max_hand_self_penetration_m']<.001,max(m['max_arm_table_penetration_m'],m['max_hand_table_penetration_m'])>=.001,m['object_rmse_m']+m['final_object_position_error_m'],state['candidate_id'])
   choices.append((score,path,state))
  if not choices:plans.append(dict(source_index=group['source_index'],status='no_prior_geometry_passed_physical_rejection'));continue
  _,path,state=min(choices,key=lambda item:item[0]);plans.append(dict(source_index=group['source_index'],source_case=str(path.parent),state=state,source_reference_sha256=hashlib.sha256((path.parent/'source_reference.npz').read_bytes()).hexdigest()))
 save(out/'plan.json',dict(kind='post-outcome feasibility diagnostic, not blind or original-cost baseline',sources=plans,objective='original plus hand-self contact weight1e7',controller='none',budget_per_case=200,samples=512,acceptance='warmup_grasp_v2',candidate_bank=16,check_every=1))
 status=dict(status='running',records=[],started_at=time.time());save(out/'status.json',status)
 for plan in plans:
  if 'source_case' not in plan:status['records'].append(plan);continue
  source=Path(plan['source_case']);case=out/f"case_{plan['source_index']:03d}";case.mkdir()
  for name in ['scene.xml','source_reference.npz','reference.npz','augmentation.json','geometry_report.json','retargeting.json']:shutil.copy2(source/name,case/name)
  state=dict(plan['state']);state.update(folder=str(case),status='ik_passed');state.pop('validated_success',None);save(case/'state.json',state)
  execution=out/f"execution_{plan['source_index']:03d}"
  command=[sys.executable,'-u','-m','examples.arm_hand.execute_contact_capture','--case',str(case),'--output',str(execution),'--baseline','--iterations','200','--strict-check-every','1','--strict-candidate-pool','16','--transition-dir',str(out/'transitions'/str(plan['source_index'])),'--source-group',state['task']]
  status['current_source']=plan['source_index'];save(out/'status.json',status)
  with (out/f"execution_{plan['source_index']:03d}.log").open('w') as log:result=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
  report=json.loads((execution/'execution.json').read_text()) if (execution/'execution.json').exists() else {}
  status['records'].append(dict(source_index=plan['source_index'],source_case=str(source),execution=str(execution),returncode=result.returncode,accepted=report.get('accepted',False),prior_optimization_elapsed_s=plan['state'].get('physics_elapsed_s'),new_execution_wall_s=report.get('wall_s')));status['wall_s']=time.monotonic()-started;save(out/'status.json',status)
  if result.returncode:status['status']='infrastructure_failed';save(out/'status.json',status);return
 status.update(status='complete',wall_s=time.monotonic()-started);save(out/'status.json',status)
 # Newly successful cases enter the same provenance-verifying index only.
 for name,cmd in [('refresh',['-m','examples.arm_hand.collect_warmup_candidates','--render','--output',root/'outputs/mustard_warmup_grasp_v2']),('verify',['-m','examples.arm_hand.verify_warmup_artifacts','--warmup',root/'outputs/mustard_warmup_grasp_v2'])]:
  with (out/(name+'.log')).open('w') as log:result=subprocess.run([sys.executable,'-u',*map(str,cmd)],cwd=root,stdout=log,stderr=subprocess.STDOUT)
  if result.returncode:status.update(status='infrastructure_failed',failed_stage=name);save(out/'status.json',status);return
 status['final_verification_complete']=True;save(out/'status.json',status)
if __name__=='__main__':main()
