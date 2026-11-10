"""Sequential native Isaac teacher validation, BC, critic and fixed-50 SAPG."""
import json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/mustard_classified_20261102_v2'
INPUT=ROOT/'outputs/mustard_classified_native_20261102_inputs'
LAUNCH=ROOT/'examples/arm_hand/run_local_simtoolreal.sh'

def save(name,value):
    (BASE/name).write_text(json.dumps(value,indent=2,default=str)+'\n')

def run(stage,args):
    save('pipeline_state.json',dict(stage=stage,status='running',fixed_target_cap=50))
    with (BASE/(stage+'.log')).open('w') as log:
        child=subprocess.Popen(['bash',str(LAUNCH),*map(str,args)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        save('child.json',dict(stage=stage,pid=child.pid,command=args))
        result=child.wait()
    if result:raise RuntimeError(f'{stage} failed ({result}); see {BASE/stage}.log')

def select():
    teacher=BASE/'teacher';out=BASE/'selected';out.mkdir()
    with np.load(teacher/'native_bc_dataset.npz') as d:
        quality=json.loads((teacher/'grasp_validation.json').read_text())
        keep=np.flatnonzero((np.array(quality['longest_hold_proxy_steps'])>=30)&(d['valid_lengths']<=960))
        assert len(keep)>0,'No native demonstrations meet grasp and 16s requirements.'
        lengths=d['valid_lengths'][keep];total=int(lengths.max())
        values={k:d[k] for k in ('action_contract','smoke_test')}
        values.update(obs=d['obs'][keep,:total],actions=d['actions'][keep,:total],candidate_ids=d['candidate_ids'][keep],valid_lengths=lengths,source_indices=d['source_indices'][:total,keep])
        np.savez_compressed(out/'native_bc_dataset.npz',**values)
        ids=set(map(int,values['candidate_ids']))
    manifest=json.loads((INPUT/'source_manifest.json').read_text());pool=json.loads((INPUT/'waypoints.json').read_text())
    pool['trajectories']=[x for x in pool['trajectories'] if x['candidate_id'] in ids]
    for name,value in [('source_manifest.json',[x for x in manifest if x['candidate_id'] in ids]),('waypoints.json',pool),('grasp_validation.json',dict(teacher_control='native',held_lift_proxy_30_steps=len(keep),trajectories=len(keep),native_parent=str(teacher))),('retiming_audit.json',dict(valid_lengths=lengths.tolist(),candidate_ids=sorted(ids),collection_envs=1032,excluded_padding=True))]:
        (out/name).write_text(json.dumps(value,indent=2,default=str)+'\n')
    import shutil
    shutil.copy2(teacher/'collection_environment.json',out/'collection_environment.json')
    save('native_selection.json',dict(input_count=len(manifest),selected_count=len(keep),rejected_count=len(manifest)-len(keep),criterion='native hold>=30 steps within valid trajectory; valid duration<=16 seconds',valid_lengths=lengths.tolist(),candidate_ids=sorted(ids)))
    return len(keep)

def main():
    resume_bc=os.environ.get('MUSTARD_RESUME_BC')
    bc_output=BASE/('bc4000_resume1000' if resume_bc else 'bc4000')
    adopt=int(os.environ.get('MUSTARD_ADOPT_TEACHER_PID','0'))
    BASE.mkdir(exist_ok=bool(adopt or resume_bc))
    if adopt:
        (BASE/'pipeline_failed.json').unlink(missing_ok=True)
        save('pipeline_state.json',dict(stage='teacher',status='running',fixed_target_cap=50))
    n=len(json.loads((INPUT/'source_manifest.json').read_text()))
    if not adopt and not resume_bc:run('teacher',['bc','--headless','--num-envs',1032,'--block-size',172,'--expected-trajectories',n,'--source',INPUT/'source_manifest.json','--waypoints',INPUT/'waypoints.json','--collect-only','--output',BASE/'teacher'])
    if adopt:
        while not (BASE/'teacher/completed.json').exists():
            if (BASE/'teacher/failed.json').exists():raise RuntimeError((BASE/'teacher/failed.json').read_text())
            os.kill(adopt,0);time.sleep(10)
        # Wait for Kit to release its context before starting the learner.
        for _ in range(12):
            try:os.kill(adopt,0)
            except ProcessLookupError:break
            time.sleep(5)
    selected=BASE/'selected'
    n=len(json.loads((selected/'source_manifest.json').read_text())) if resume_bc else select()
    common=['--headless','--num-envs',8184,'--block-size',1364,'--expected-trajectories',n,'--qualification','full-path-grasp','--demo-max-seconds',16,'--dataset',selected/'native_bc_dataset.npz','--source',selected/'source_manifest.json','--waypoints',selected/'waypoints.json','--learner-device','cuda:0','--batch-sequences',128,'--sequence-length',16,'--burn-in',16,'--checkpoint-every',100,'--video-every',100,'--video-length',960,'--eval-reset-modes','demo,native','--wandb-project','simtoolreal-warmstart']
    run('bc4000_resume1000' if resume_bc else 'bc4000',['bc',*common,'--epochs',4000,'--learning-rate',.0001,'--output',bc_output,*(['--resume-bc',resume_bc] if resume_bc else [])])
    run('bc_carry200',['bc',*common,'--epochs',200,'--learning-rate',.00001,'--rnn-context','carry','--init-checkpoint',bc_output/'actor_init.pt','--output',BASE/'bc_carry200'])
    run('sapg50',['critic','--headless','--num-envs',8184,'--block-size',1364,'--checkpoint',BASE/'bc_carry200/actor_init.pt','--dataset',selected/'native_bc_dataset.npz','--waypoints',selected/'waypoints.json','--episode-seconds',16,'--sigma-scale',.25,'--critic-epochs',683,'--critic-lr',.00005,'--joint-epochs',200000,'--actor-lr',.00001,'--actor-clip',.05,'--multi-target','--multi-target-level',4,'--restore-native-curriculum','--curriculum-interval',500,'--curriculum-min-episodes',1000,'--checkpoint-every',1000,'--video-every',1000,'--video-length',960,'--wandb-project','simtoolreal-warmstart','--output',BASE/'sapg50'])
    save('pipeline_state.json',dict(stage='complete',status='complete',fixed_target_cap=50))

if __name__=='__main__':
    try:main()
    except BaseException as error:
        if BASE.exists():save('pipeline_failed.json',dict(type=type(error).__name__,message=str(error)))
        raise
