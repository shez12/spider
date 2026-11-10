"""Isolated IK post-filter diagnostic; keeps original references and geometry gates.

The original fit solver and random starts are unchanged. In this process only,
replace its final Gaussian joint filter by identity to test filter-introduced
collisions. No production file or policy is modified.
"""
import contextlib,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
import mujoco,numpy as np
from examples.arm_hand import run_sharpa_complex as ik
from examples.arm_hand.spatial_augmentation import check_ik,KEYS
from examples.arm_hand.validate_kuka_sharpa import PLAY2PERFECT_ARM_POSE
from examples.arm_hand.resource_slots import resource_slot


def main():
 root=Path(__file__).resolve().parents[2];pilot=root/'outputs/world_model_pilot';out=pilot/'ik_smoothing_source014';out.mkdir(exist_ok=False);started=time.monotonic()
 source=next((pilot/'fresh_eval_cases').glob('014_*/candidate_300001'));case=out/'case';case.mkdir()
 for name in ['scene.xml','source_reference.npz','augmentation.json']:shutil.copy2(source/name,case/name)
 baseline=json.loads((source/'geometry_report.json').read_text());state=json.loads((source/'state.json').read_text());state.update(folder=str(case.resolve()),status='generated');(case/'state.json').write_text(json.dumps(state,indent=2))
 protocol=dict(kind='isolated_post_smoothing_diagnostic',source_case=str(source.resolve()),source_index=14,
               original_fit_sha256=hashlib.sha256(Path(ik.__file__).read_bytes()).hexdigest(),solver='original fit stride2, eight initial restarts, unchanged residuals/bounds',
               only_change='identity replaces Gaussian sigma3 joint smoothing in this process only',baseline_geometry=baseline,status='preparing',source_reference_sha256=hashlib.sha256((case/'source_reference.npz').read_bytes()).hexdigest())
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
 try:
  with np.load(case/'source_reference.npz') as f:ref={k:f[k].copy() for k in f.files}
  frozen={k:ref[k].copy() for k in KEYS};model=mujoco.MjModel.from_xml_path(str(case/'scene.xml'))
  original_filter=ik.gaussian_filter1d
  with resource_slot(root/'outputs/mustard_multisource_valid100/resource_slots/ik',8,block=True):
   try:
    ik.gaussian_filter1d=lambda values,*args,**kwargs:np.array(values,copy=True)
    with (out/'ik.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
     ik.fit(model,ref,case,2,initial_arm_pose=PLAY2PERFECT_ARM_POSE,first_frame_restarts=8,reject_bad_initial=True)
   finally:ik.gaussian_filter1d=original_filter
  for key in KEYS:np.testing.assert_array_equal(ref[key],frozen[key])
  report=check_ik(case);protocol.update(status='geometry_passed' if report['ik_pass'] else 'geometry_rejected',geometry=report,ik_elapsed_s=time.monotonic()-started)
  state.update(status='ik_passed' if report['ik_pass'] else 'ik_rejected',ik_pass=report['ik_pass'],ik_elapsed_s=protocol['ik_elapsed_s']);(case/'state.json').write_text(json.dumps(state,indent=2));(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
  if report['ik_pass']:
   manifest=out/'cohort_input.json';manifest.write_text(json.dumps(dict(source_indices=[14],valid_cases=1,cases=[state]),indent=2))
   frozen_models=json.loads((pilot/'blind_warmup_v2_300000_intent.json').read_text())['models']
   for m in frozen_models.values():assert hashlib.sha256(Path(m['path']).read_bytes()).hexdigest()==m['sha256']
   command=[sys.executable,'-u','-m','examples.arm_hand.run_warmup_cohort','--cohort',str(manifest),'--bc',frozen_models['bc']['path'],'--model',frozen_models['imagination']['path'],'--output',str(out/'cohort'),'--mps-pipe-dir','/tmp/spider-mps-mustard-1005','--max-runs','3']
   with (out/'physics.log').open('w') as log:result=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
   protocol.update(status='physics_complete' if result.returncode==0 else 'infrastructure_failed',returncode=result.returncode)
 except Exception as exc:protocol.update(status='failed',error=str(exc))
 protocol['wall_s']=time.monotonic()-started;(out/'protocol.json').write_text(json.dumps(protocol,indent=2));print(json.dumps(protocol),flush=True)
if __name__=='__main__':main()
