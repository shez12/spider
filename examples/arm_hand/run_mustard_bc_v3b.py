"""BC v3b: memorizing BC (v2 recipe: all data, no goal relabel, 4000+200 epochs) on the 782-demo set.

v3 (validated, goal-relabeled) lost closed-loop success on training demos without generalizing
better; the warm-start curriculum relies on reproducing demonstrations, so v3b restores the v2
recipe on the larger set and keeps periodic checkpoints for closed-loop selection.
"""
import json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/mustard_bc_v3b'
DATA=ROOT/'outputs/mustard_classified_20261104_truncated_v1/selected'
OLD=ROOT/'outputs/mustard_classified_20261102_v2/bc_carry200/actor_init.pt'
DROP=ROOT/'outputs/mustard_classified_20261102_v2/drop_bc_coverage_check_v1/drop_bank.npz'
AUDIT=ROOT/'outputs/mustard_bc_v3/audit/audit.json'  # bc_balance audit of the same 782-demo cache
LAUNCH=ROOT/'examples/arm_hand/run_local_simtoolreal.sh'


def save(name,value):(BASE/name).write_text(json.dumps(value,indent=2,default=str)+'\n')


def run(stage,command):
    save('pipeline_state.json',dict(stage=stage,status='running'))
    with (BASE/(stage+'.log')).open('w') as log:result=subprocess.run(list(map(str,command)),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{stage} failed ({result.returncode}); see {BASE/stage}.log')


def main():
    BASE.mkdir(exist_ok=False)
    common=['--headless','--num-envs',8184,'--block-size',1364,'--expected-trajectories',782,'--qualification','full-path-grasp',
        '--demo-max-seconds',16,'--dataset',DATA/'native_bc_dataset.npz','--source',DATA/'source_manifest.json','--waypoints',DATA/'waypoints.json',
        '--learner-device','cuda:0','--batch-sequences',128,'--sequence-length',16,'--burn-in',16,'--video-every',0,
        '--keep-checkpoints','--wandb-project','simtoolreal-warmstart']
    run('bc4000',['bash',LAUNCH,'bc',*common,'--epochs',4000,'--learning-rate',.0001,'--checkpoint-every',1000,'--output',BASE/'bc4000'])
    run('bc_carry200',['bash',LAUNCH,'bc',*common,'--epochs',200,'--learning-rate',.00001,'--rnn-context','carry',
        '--init-checkpoint',BASE/'bc4000/actor_init.pt','--checkpoint-every',100,'--output',BASE/'bc_carry200'])
    run('closed_loop_eval',['bash',LAUNCH,'balance-evaluate','--headless','--checkpoints',OLD,BASE/'bc4000/actor_e002000.pt',
        BASE/'bc4000/actor_e004000.pt',BASE/'bc_carry200/actor_init.pt','--names','bc_v2_carry200','bc_v3b_e2000','bc_v3b_e4000','bc_v3b_carry200',
        '--dataset',DATA/'native_bc_dataset.npz','--waypoints',DATA/'waypoints.json','--audit',AUDIT,'--drop-bank',DROP,'--output',BASE/'closed_loop_eval'])
    save('pipeline_state.json',dict(stage='closed_loop_eval',status='complete',next='review before critic calibration and SAPG'))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        if BASE.exists():save('pipeline_failed.json',dict(type=type(error).__name__,message=str(error)))
        raise
