from pathlib import Path
import hashlib,json
r=Path('/home/szc/work/spider/examples/arm_hand');sources={}
s=(r/'optimize_kuka_warmup_bank.py').read_text();sources['optimize_kuka_warmup_bank.py']=hashlib.sha256(s.encode()).hexdigest()
s=s.replace('    args = parser.parse_args()',"    parser.add_argument('--hand-self-collision-weight',type=float,default=1e7)\n    args = parser.parse_args()\n    if args.hand_self_collision_weight<0:parser.error('Negative hand collision weight')")
s=s.replace('    if args.table_collision_weight:', '    if args.table_collision_weight or args.hand_self_collision_weight:',1)
s=s.replace('        contact_dist = wp.to_torch(gpu_data.contact.dist)',"        hand_geom=torch.tensor([(model.geom(g).name or '').startswith('collision_sharpa_') for g in range(model.ngeom)],device=dev)\n        contact_dist = wp.to_torch(gpu_data.contact.dist)")
s=s.replace('            if record:\n                poses.append(qpos[0].clone())',"            if args.hand_self_collision_weight:\n                from examples.arm_hand.contact_sampling_cost import hand_self_penalty\n                cost += args.hand_self_collision_weight*hand_self_penalty(contact_geoms,contact_dist,contact_world,contact_count[0],hand_geom,n)\n            if record:\n                poses.append(qpos[0].clone())")
s=s.replace('optimizer_settings = dict(feasible_candidate_pool=',"optimizer_settings = dict(hand_self_collision_weight=args.hand_self_collision_weight,experimental_objective='original_plus_hand_self_contacts_v1',feasible_candidate_pool=")
assert s.count('cost += args.hand_self_collision_weight')==1
(r/'optimize_kuka_contact_sampling.py').write_text(s)
s=(r/'execute_warmup_bank.py').read_text();sources['execute_warmup_bank.py']=hashlib.sha256(s.encode()).hexdigest();s=s.replace('examples.arm_hand.optimize_kuka_warmup_bank','examples.arm_hand.optimize_kuka_contact_sampling').replace('    p.add_argument(\'--resume\',action=\'store_true\')',"    p.add_argument('--hand-self-collision-weight',type=float,default=1e7)\n    p.add_argument('--resume',action='store_true')")
s=s.replace("'--disable-target-stop','--strict-candidate-pool'", "'--disable-target-stop','--hand-self-collision-weight',a.hand_self_collision_weight,'--strict-candidate-pool'")
s=s.replace('report=dict(case=',"report=dict(experimental_objective='original_plus_hand_self_contacts_v1',hand_self_collision_weight=a.hand_self_collision_weight,case=")
(r/'execute_contact_sampling.py').write_text(s)
s=(r/'run_warmup_bank_cohort.py').read_text();sources['run_warmup_bank_cohort.py']=hashlib.sha256(s.encode()).hexdigest();s=s.replace('examples.arm_hand.execute_warmup_bank','examples.arm_hand.execute_contact_sampling').replace('examples.arm_hand.optimize_kuka_warmup_bank','examples.arm_hand.optimize_kuka_contact_sampling').replace('examples/arm_hand/optimize_kuka_warmup_bank.py','examples/arm_hand/optimize_kuka_contact_sampling.py').replace("'examples/arm_hand/execute_warmup_bank.py'","'examples/arm_hand/execute_contact_sampling.py','examples/arm_hand/contact_sampling_cost.py'")
s=s.replace('protocol=dict(acceptance_version=',"protocol=dict(experimental_objective='original_plus_hand_self_contacts_v1',hand_self_collision_weight=1e7,acceptance_version=")
(r/'run_contact_sampling_cohort.py').write_text(s);(r/'contact_sampling_entrypoint_provenance.json').write_text(json.dumps(sources,indent=2)+'\n')
