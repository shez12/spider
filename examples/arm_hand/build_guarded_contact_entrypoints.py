from pathlib import Path
import hashlib,json
r=Path('/home/szc/work/spider/examples/arm_hand');hashes={}
for name in ['optimize_kuka_contact_sampling','execute_contact_sampling','execute_contact_capture','audit_warmup_trajectory','run_contact_every_step_cohort']:
 text=(r/(name+'.py')).read_text();hashes[name+'.py']=hashlib.sha256(text.encode()).hexdigest()
 if name=='optimize_kuka_contact_sampling':
  old='from examples.arm_hand.warmup_acceptance import audit_for_optimizer as audit_states';assert text.count(old)==1;text=text.replace(old,'from examples.arm_hand.warmup_validation_guard import audit_for_optimizer as audit_states');target='optimize_kuka_contact_guarded'
 elif name=='audit_warmup_trajectory':text=text.replace('from examples.arm_hand.warmup_acceptance import audit_for_optimizer','from examples.arm_hand.warmup_validation_guard import audit_for_optimizer');target='audit_guarded_warmup_trajectory'
 elif name.startswith('execute_'):
  text=text.replace('examples.arm_hand.optimize_kuka_contact_sampling','examples.arm_hand.optimize_kuka_contact_guarded').replace('examples.arm_hand.audit_warmup_trajectory','examples.arm_hand.audit_guarded_warmup_trajectory');target=name+'_guarded'
 else:
  text=text.replace('examples.arm_hand.execute_contact_sampling','examples.arm_hand.execute_contact_sampling_guarded').replace('examples.arm_hand.optimize_kuka_contact_sampling','examples.arm_hand.optimize_kuka_contact_guarded').replace('examples/arm_hand/optimize_kuka_contact_sampling.py','examples/arm_hand/optimize_kuka_contact_guarded.py').replace("'examples/arm_hand/execute_contact_sampling.py'","'examples/arm_hand/execute_contact_sampling_guarded.py','examples/arm_hand/warmup_validation_guard.py'").replace('examples/arm_hand/audit_warmup_trajectory.py','examples/arm_hand/audit_guarded_warmup_trajectory.py');target='run_contact_guarded_cohort'
 (r/(target+'.py')).write_text('# Numerical-replay rejection wrapper; frozen source SHA256 '+hashes[name+'.py']+'\n'+text)
(r/'guarded_contact_provenance.json').write_text(json.dumps(hashes,indent=2)+'\n')
