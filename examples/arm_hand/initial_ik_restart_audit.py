"""Inspect all original first-frame IK restarts without changing production selection."""
from pathlib import Path
import inspect,json,time,hashlib,contextlib
import numpy as np,mujoco
from examples.arm_hand import run_sharpa_complex as ik
from examples.arm_hand.validate_kuka_sharpa import PLAY2PERFECT_ARM_POSE
from examples.arm_hand.resource_slots import resource_slot
class Done(Exception):pass

def main():
 root=Path(__file__).resolve().parents[2];o=root/'outputs/world_model_pilot/initial_ik_restart_audit_v1';o.mkdir(exist_ok=False);source=inspect.getsource(ik.fit)
 needle='max_nfev=300 if i == 0 and first_frame_restarts > 1 else 80)'
 assert source.count(needle)==1
 source=source.replace(needle,needle+'''
            residual(result.x)
            palm=float(np.linalg.norm(data.sensordata[:3]-ref['wrist_pos'][0]))
            tips=np.linalg.norm(data.sensordata[14:26].reshape(4,3)-ref['fingertips'][0],axis=1)
            table=max([0.]+[-float(c.dist) for c in data.contact if (int(c.geom1)==floor_id and int(c.geom2) in robot_index) or (int(c.geom2)==floor_id and int(c.geom1) in robot_index)])
            self_depth=max([0.]+[-float(c.dist) for c in data.contact if int(c.geom1) in hand_index and int(c.geom2) in hand_index])
            restart_records.append(dict(cost=float(result.cost),palm_error_m=palm,max_tip_error_m=float(tips.max()),table_depth_m=table,self_depth_m=self_depth,passes_initial_gate=bool(palm<.01 and tips.max()<.03 and table<.001 and self_depth<.001),joints=result.x.tolist()))''')
 assert source.count('        seed = best_fit.x')==1;source=source.replace('        seed = best_fit.x','        raise Done()\n        seed = best_fit.x')
 ns=dict(ik.__dict__);ns.update(Done=Done);exec(compile(source,'<instrumented-original-fit>','exec'),ns);groups=json.loads((root/'outputs/mustard_multisource_valid100/selected_sources.json').read_text());rows=[];started=time.time()
 (o/'protocol.json').write_text(json.dumps(dict(source_code_sha256=hashlib.sha256(Path(ik.__file__).read_bytes()).hexdigest(),source_indices=[4,8,19,21,22],kind='same original8 starts,300 evaluations each; observe all candidates then stop before trajectory fitting',changes='instrumentation only, no production change'),indent=2))
 for idx in [4,8,19,21,22]:
  g=next(g for g in groups if g['source_index']==idx);case=next(p.parent for p in sorted(Path(g['result_dir']).glob('candidate_*/source_reference.npz')) if (p.parent/'geometry_report.json').exists());model=mujoco.MjModel.from_xml_path(str(case/'scene.xml'));ref=dict(np.load(case/'source_reference.npz'));records=[];ns['restart_records']=records;t=time.time()
  with resource_slot(root/'outputs/mustard_multisource_valid100/resource_slots/ik',8,block=True):
   try:ns['fit'](model,ref,o,2,initial_arm_pose=PLAY2PERFECT_ARM_POSE,first_frame_restarts=8,reject_bad_initial=True)
   except Done:pass
  best=min(records,key=lambda x:x['cost']);row=dict(source_index=idx,case=str(case),restarts=records,original_best_passes=best['passes_initial_gate'],any_restart_passes=any(x['passes_initial_gate'] for x in records),wall_s=time.time()-t);rows.append(row);(o/'report.json').write_text(json.dumps(dict(status='running',rows=rows,wall_s=time.time()-started),indent=2));print(json.dumps({k:v for k,v in row.items() if k!='restarts'}),flush=True)
 (o/'report.json').write_text(json.dumps(dict(status='complete',rows=rows,wall_s=time.time()-started),indent=2))
if __name__=='__main__':main()
