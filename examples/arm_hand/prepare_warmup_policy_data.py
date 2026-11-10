"""Export only independently verified retained-grasp teachers for policy warmup."""
import argparse,hashlib,json,time
from pathlib import Path
import mujoco,numpy as np
from examples.arm_hand.world_model import canonical,scene_spec,usable_states
from examples.arm_hand.warmup_acceptance import VERSION


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();started=time.monotonic()
 manifest=json.loads(a.manifest.read_text())
 if manifest['acceptance_version']!=VERSION or manifest['failures']:raise ValueError('Require complete verified current-version artifact index')
 a.output.mkdir(parents=True,exist_ok=True);rows=manifest['strict']+[r for r in manifest['warmup'] if not r['already_counted']];records=[];spec=None
 for row in rows:
  case=Path(row['case']);execution=Path(row.get('execution',row['case']));key=row['augmentation_sha256'];out=a.output/f'policy_{key[:24]}.npz'
  model,current,context=scene_spec(case/'scene.xml')
  if spec is None:spec=current
  if spec!=current:raise ValueError('Topology differs')
  with np.load(case/'reference.npz') as f:ref={k:f[k] for k in f.files}
  with np.load(execution/'gpu/trajectory.npz') as f:tr={k:f[k] for k in f.files}
  np.testing.assert_array_equal(tr['initial_state'],ref['initial_state'])
  data=mujoco.MjData(model);mujoco.mj_setState(model,data,ref['initial_state'],mujoco.mjtState.mjSTATE_FULLPHYSICS)
  q=canonical(np.concatenate((data.qpos[None],tr['qpos'])));v=np.concatenate((data.qvel[None],tr['qvel']));state=np.concatenate((q,v),-1).astype(np.float32)
  if not usable_states(state,current).all():raise ValueError('Teacher contains diverged states')
  goal=canonical(np.concatenate((ref['joint_ik'],ref['object_qpos']),-1)).astype(np.float32)
  np.savez_compressed(out,state=state,goal=goal,nominal=ref['ctrl_nominal'].astype(np.float32),action=tr['ctrl'].astype(np.float32),context=context,low=model.actuator_ctrlrange[:,0].astype(np.float32),high=model.actuator_ctrlrange[:,1].astype(np.float32),metadata=json.dumps(dict(spec=current,source_group=row['task'],case=str(case.resolve()),execution=str(execution.resolve()),validated_teacher=True,acceptance_version=VERSION,teacher_validation=row.get('report'),dataset_manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest())))
  records.append(dict(path=str(out),source_group=row['task'],frames=len(tr['ctrl']),acceptance_version=VERSION))
 (a.output/'collection.json').write_text(json.dumps(dict(records=records,elapsed_s=time.monotonic()-started,manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest()),indent=2)+'\n');print(json.dumps(dict(trajectories=len(records),elapsed_s=time.monotonic()-started)),flush=True)
if __name__=='__main__':main()
