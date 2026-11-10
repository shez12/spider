"""Isolated full-path original IK with initial alignment hinge, hard gates unchanged."""
import argparse,inspect,json,time,hashlib,shutil,contextlib
from pathlib import Path
import numpy as np,mujoco
from examples.arm_hand import run_sharpa_complex as ik
from examples.arm_hand.resource_slots import resource_slot
from examples.arm_hand.spatial_augmentation import check_ik,KEYS
from examples.arm_hand.validate_kuka_sharpa import PLAY2PERFECT_ARM_POSE

def main():
 p=argparse.ArgumentParser();p.add_argument('--source-case',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True);started=time.time()
 for name in ['scene.xml','source_reference.npz','augmentation.json']:shutil.copy2(a.source_case/name,a.output/name)
 state=json.loads((a.source_case/'state.json').read_text());state.update(status='generated',folder=str(a.output.resolve()));(a.output/'state.json').write_text(json.dumps(state,indent=2))
 ref=dict(np.load(a.output/'source_reference.npz'));frozen={k:ref[k].copy() for k in KEYS};s=inspect.getsource(ik.fit);needle='0.015 * (joints - previous),';assert s.count(needle)==1;s=s.replace(needle,needle+"\n                1000 * np.maximum(np.linalg.norm(data.sensordata[14:26].reshape(4,3)-ref['fingertips'][i],axis=1)-.029,0) if i==0 else np.zeros(4),\n                1000 * np.maximum(np.linalg.norm(data.sensordata[:3]-ref['wrist_pos'][i])-.009,0) if i==0 else 0.,")
 assert s.count('500 * table_penetration,')==1;s=s.replace('500 * table_penetration,','500 * table_penetration, 1000 * np.maximum(table_penetration-.0003,0),');ns=dict(ik.__dict__);ns['gaussian_filter1d']=lambda values,*args,**kwargs:np.array(values,copy=True);exec(compile(s,'<isolated-initial-gate-objective-unsmoothed>','exec'),ns);(a.output/'protocol.json').write_text(json.dumps(dict(parent_code_sha256=hashlib.sha256(Path(ik.__file__).read_bytes()).hexdigest(),instrumented_code_sha256=hashlib.sha256(s.encode()).hexdigest(),source_case=str(a.source_case.resolve()),change='initial alignment hinge plus identity filter and extra table hinge beyond0.3mm weight1000; stride2,eight starts and all geometry thresholds unchanged'),indent=2));root=Path(__file__).resolve().parents[2];model=mujoco.MjModel.from_xml_path(str(a.output/'scene.xml'))
 try:
  with resource_slot(root/'outputs/mustard_multisource_valid100/resource_slots/ik',8,block=True):
   with (a.output/'ik.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):ns['fit'](model,ref,a.output,2,initial_arm_pose=PLAY2PERFECT_ARM_POSE,first_frame_restarts=8,reject_bad_initial=True)
  for k in KEYS:np.testing.assert_array_equal(ref[k],frozen[k])
  g=check_ik(a.output);state.update(status='ik_passed' if g['ik_pass'] else 'ik_rejected',ik_pass=g['ik_pass'])
 except ik.InitialGeometryRejected as e:
  (a.output/'geometry_report.json').write_text(json.dumps(e.report,indent=2));state.update(status='ik_rejected',ik_pass=False,error=str(e))
 except Exception as e:state.update(status='ik_failed',ik_pass=False,error=str(e));raise
 finally:
  state['ik_elapsed_s']=time.time()-started;(a.output/'state.json').write_text(json.dumps(state,indent=2));print(json.dumps(state),flush=True)
if __name__=='__main__':main()
