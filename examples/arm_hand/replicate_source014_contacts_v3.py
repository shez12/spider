# Parent SHA db0d3015759a84b6e6141b924a0542cb5a49f86c29cb5d73e5718217f9f78891; next untouched uniform augmentations
# Frozen parent SHA 2c8ad3da3e71215da117323dac3b9f41d0daacaeb65bfb4fbed08229f2ea8751; isolated baseline expansion,not world-model benefit
"""Bounded raw-SPIDER contact-cost replication and real-transition collection.

Uses eight previously uniformly generated augmentations, unchanged references,
original IK solver with explicitly unsmoothed output, and unchanged v2 audits.
This is baseline data collection, not a world-model benefit claim.
"""
import contextlib,fcntl,hashlib,json,subprocess,sys,time
from pathlib import Path
import shutil
import mujoco,numpy as np
from examples.arm_hand import run_sharpa_complex as ik
from examples.arm_hand.spatial_augmentation import check_ik,KEYS
from examples.arm_hand.validate_kuka_sharpa import PLAY2PERFECT_ARM_POSE
from examples.arm_hand.resource_slots import resource_slot


def save(path,data):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def main():
 root=Path(__file__).resolve().parents[2];pilot=root/'outputs/world_model_pilot';out=pilot/'source014_contact_replication_v3';out.mkdir(exist_ok=True)
 lock=(out/'coordinator.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 protocol=dict(source_index=14,target_valid=10,trials=list(range(300015,300033)),ik_stride=2,first_frame_restarts=8,
               smoothing='identity in this process only',samples=512,max_iterations=200,strict_check_every=1,candidate_pool=16,
               hand_self_collision_weight=1e7,acceptance='warmup_grasp_v2',controller='raw SPIDER with experimental self-contact cost; no learned controller',
               original_fit_sha256=hashlib.sha256(Path(ik.__file__).read_bytes()).hexdigest(),capture_every=1,capture_worlds=8,capture_max_shards=20)
 if (out/'protocol.json').exists():
  if json.loads((out/'protocol.json').read_text())!=protocol:raise ValueError('Frozen replication protocol differs')
 else:save(out/'protocol.json',protocol)
 records=[];started=time.monotonic()
 for number in protocol['trials']:
  source=next((pilot/'fresh_eval_cases').glob(f'014_*/candidate_{number}'));case=out/f'candidate_{number}';execution=out/f'execution_{number}';case.mkdir(exist_ok=True)
  if not (case/'reference.npz').exists():
   for name in ['scene.xml','source_reference.npz','augmentation.json']:shutil.copy2(source/name,case/name)
   state=json.loads((source/'state.json').read_text());state.update(folder=str(case.resolve()),status='generated');save(case/'state.json',state)
   with np.load(case/'source_reference.npz') as f:ref={k:f[k].copy() for k in f.files}
   frozen={k:ref[k].copy() for k in KEYS};model=mujoco.MjModel.from_xml_path(str(case/'scene.xml'));original=ik.gaussian_filter1d;start=time.monotonic()
   try:
    with resource_slot(root/'outputs/mustard_multisource_valid100/resource_slots/ik',8,block=True):
     try:
      ik.gaussian_filter1d=lambda values,*args,**kwargs:np.array(values,copy=True)
      with (case/'ik.log').open('a') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):ik.fit(model,ref,case,2,initial_arm_pose=PLAY2PERFECT_ARM_POSE,first_frame_restarts=8,reject_bad_initial=True)
     finally:ik.gaussian_filter1d=original
    for key in KEYS:np.testing.assert_array_equal(ref[key],frozen[key])
    geometry=check_ik(case);state.update(ik_pass=geometry['ik_pass'],status='ik_passed' if geometry['ik_pass'] else 'ik_rejected',ik_elapsed_s=time.monotonic()-start)
   except Exception as error:state.update(status='ik_rejected',ik_pass=False,error=str(error),ik_elapsed_s=time.monotonic()-start)
   save(case/'state.json',state)
  state=json.loads((case/'state.json').read_text())
  if not state.get('ik_pass'):records.append(dict(candidate_id=number,ik_pass=False));continue
  complete=False
  if (execution/'execution.json').exists():complete=json.loads((execution/'execution.json').read_text()).get('status')=='complete'
  if not complete:
   for process in Path('/proc').iterdir():
    if not process.name.isdigit():continue
    try:args=(process/'cmdline').read_bytes().split(b'\0')
    except OSError:continue
    if (str(execution).encode() in args or str(execution/'gpu').encode() in args) and any(m in args for m in [b'examples.arm_hand.execute_contact_capture_guarded',b'examples.arm_hand.optimize_kuka_contact_guarded']):raise RuntimeError('Live matching execution exists')
   cmd=[sys.executable,'-u','-m','examples.arm_hand.execute_contact_capture_guarded','--case',str(case),'--output',str(execution),'--baseline','--iterations','200','--strict-check-every','1','--strict-candidate-pool','16','--transition-dir',str(out/'transitions'/str(number)),'--source-group',state['task']]
   if (execution/'gpu/checkpoint.pt').is_file():cmd.append('--resume')
   with (out/f'execution_{number}.log').open('a') as log:result=subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT)
   if result.returncode:save(out/'status.json',dict(status='infrastructure_failed',candidate_id=number,records=records));raise RuntimeError('Stop on infrastructure failure; preserve checkpoint')
  report=json.loads((execution/'execution.json').read_text());records.append(dict(candidate_id=number,ik_pass=True,accepted=report['accepted'],case=str(case),execution=str(execution),execution_wall_s=report['wall_s']))
  save(out/'status.json',dict(status='running',records=records,valid=sum(x.get('accepted',False) for x in records),wall_s=time.monotonic()-started))
  if sum(x.get('accepted',False) for x in records)>=protocol['target_valid']:break
 save(out/'status.json',dict(status='complete',records=records,valid=sum(x.get('accepted',False) for x in records),wall_s=time.monotonic()-started,budget_exhausted=sum(x.get('accepted',False) for x in records)<protocol['target_valid']))
if __name__=='__main__':main()
