"""BC v3 on the truncated 782-demo set: validated window BC, carry fine-tune, paired closed-loop check.

Changes vs mustard_classified_20261102_v2: truncated long demonstrations (+279), stratified
validation with best-checkpoint selection and early stopping, pre-grasp goal relabeling, and a
closed-loop comparison against the previous BC actor before any critic/SAPG stage.
"""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/mustard_bc_v3'
DATA=ROOT/'outputs/mustard_classified_20261104_truncated_v1/selected'
OLD=ROOT/'outputs/mustard_classified_20261102_v2/bc_carry200/actor_init.pt'
DROP=ROOT/'outputs/mustard_classified_20261102_v2/drop_bc_coverage_check_v1/drop_bank.npz'
LAUNCH=ROOT/'examples/arm_hand/run_local_simtoolreal.sh'
PY='/home/szc/.conda/envs/simtoolreal_origin_isaacsim/bin/python'


def save(name,value):(BASE/name).write_text(json.dumps(value,indent=2,default=str)+'\n')


def run(stage,command):
    save('pipeline_state.json',dict(stage=stage,status='running'))
    with (BASE/(stage+'.log')).open('w') as log:result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{stage} failed ({result.returncode}); see {BASE/stage}.log')


def main():
    BASE.mkdir(exist_ok=False)
    common=['--headless','--num-envs',8184,'--block-size',1364,'--expected-trajectories',782,'--qualification','full-path-grasp',
        '--demo-max-seconds',16,'--dataset',DATA/'native_bc_dataset.npz','--source',DATA/'source_manifest.json','--waypoints',DATA/'waypoints.json',
        '--learner-device','cuda:0','--batch-sequences',128,'--sequence-length',16,'--burn-in',16,'--video-every',0,
        '--val-fraction',.1,'--split-seed',0,'--goal-relabel-prob',.5,'--wandb-project','simtoolreal-warmstart']
    run('bc_window',['bash',LAUNCH,'bc',*map(str,common),'--epochs','4000','--learning-rate','0.0001','--checkpoint-every','100',
        '--val-every','50','--early-stop-evals','10','--output',str(BASE/'bc_window')])
    run('bc_carry',['bash',LAUNCH,'bc',*map(str,common),'--epochs','200','--learning-rate','0.00001','--rnn-context','carry',
        '--init-checkpoint',str(BASE/'bc_window/actor_best.pt'),'--checkpoint-every','50','--val-every','10','--output',str(BASE/'bc_carry')])
    sys.path.insert(0,str(ROOT/'experiments/mustard_simtoolreal'))
    from bc_balance import audit
    audit(DATA/'native_bc_dataset.npz',BASE/'audit')
    run('closed_loop_eval',['bash',LAUNCH,'balance-evaluate','--headless','--checkpoints',str(OLD),str(BASE/'bc_carry/actor_best.pt'),
        '--names','bc_v2_carry200','bc_v3_best','--dataset',str(DATA/'native_bc_dataset.npz'),'--waypoints',str(DATA/'waypoints.json'),
        '--audit',str(BASE/'audit/audit.json'),'--drop-bank',str(DROP),'--output',str(BASE/'closed_loop_eval')])
    save('pipeline_state.json',dict(stage='closed_loop_eval',status='complete',next='review before critic calibration and SAPG'))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        if BASE.exists():save('pipeline_failed.json',dict(type=type(error).__name__,message=str(error)))
        raise
