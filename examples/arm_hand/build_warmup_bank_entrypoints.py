from pathlib import Path
import hashlib,json
r=Path('/home/szc/work/spider/examples/arm_hand');hashes={}
s=(r/'optimize_kuka_warmup.py').read_text();hashes['optimize_kuka_warmup.py']=hashlib.sha256(s.encode()).hexdigest()
old="    if args.strict_candidate_pool:parser.error('Warmup candidate pool is not enabled; use pool 0')\n";assert s.count(old)==1;s=s.replace(old,'')
s=s.replace('from examples.arm_hand.strict_candidate_bank import StrictCandidateBank','from examples.arm_hand.warmup_candidate_bank import WarmupCandidateBank').replace('strict_bank=StrictCandidateBank(args.strict_candidate_pool)','strict_bank=WarmupCandidateBank(args.strict_candidate_pool)')
s=s.replace("optimizer_settings = dict(acceptance_version='warmup_grasp_v2',", "optimizer_settings = dict(feasible_candidate_pool=args.strict_candidate_pool,acceptance_version='warmup_grasp_v2',")
s=s.replace("selection_policy='strictly_valid_actual_candidate'", "selection_policy='warmup_grasp_valid_actual_candidate'")
s=s.replace("result = dict(backend=","result = dict(candidate_bank_statistics=dict(strict_bank.statistics, checked_shortlists=len(strict_bank.checked)),backend=")
(r/'optimize_kuka_warmup_bank.py').write_text('# Isolated feasible-export variant; source hash '+hashes['optimize_kuka_warmup.py']+'\n'+s)
s=(r/'execute_warmup.py').read_text();hashes['execute_warmup.py']=hashlib.sha256(s.encode()).hexdigest();s=s.replace('examples.arm_hand.optimize_kuka_warmup','examples.arm_hand.optimize_kuka_warmup_bank');(r/'execute_warmup_bank.py').write_text(s)
s=(r/'run_warmup_cohort.py').read_text();hashes['run_warmup_cohort.py']=hashlib.sha256(s.encode()).hexdigest();s=s.replace('examples.arm_hand.execute_warmup','examples.arm_hand.execute_warmup_bank').replace('examples.arm_hand.optimize_kuka_warmup','examples.arm_hand.optimize_kuka_warmup_bank').replace('examples/arm_hand/optimize_kuka_warmup.py','examples/arm_hand/optimize_kuka_warmup_bank.py').replace("'examples/arm_hand/execute_warmup.py'", "'examples/arm_hand/execute_warmup_bank.py','examples/arm_hand/warmup_candidate_bank.py','examples/arm_hand/strict_candidate_bank.py'")
s=s.replace("strict_candidate_pool=0", "strict_candidate_pool=16").replace("'--iterations','200','--strict-check-every','10']", "'--iterations','200','--strict-check-every','10','--strict-candidate-pool','16']")
(r/'run_warmup_bank_cohort.py').write_text(s);(r/'warmup_bank_entrypoint_provenance.json').write_text(json.dumps(hashes,indent=2)+'\n')
