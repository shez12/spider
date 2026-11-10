"""Step-1 arm D (ADEPT observations): wait for the running ADEPT BC (window 4000), run the carry-200 fine-tune
(same recipe as the native BC v2: lr 1e-5, carried LSTM state), then start arm D.

Arm D = arm S (right hand + original task settings + --align-left-physics + ADEPT obs + 40% native / 60% state bank v4.1) initialised
from the BC actor, sigma 0.4, BC input normaliser frozen, actor frozen for the first 683 epochs (critic-only, as v5).
Progress: outputs/mustard_bc_adept_v1/arm_d_pipeline_state.json.
"""
import json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs'
BC=OUT/'mustard_bc_adept_v1';DATA=OUT/'mustard_classified_20261102_v2/selected'
BANK=OUT/'mustard_diverse_reset_bank_v41/merged/reset_bank.npz';ARM_D=OUT/'right_mustard_pristine_adept_aligned_D_s42'
LAUNCH=ROOT/'examples/arm_hand/run_local_simtoolreal.sh';STATE=BC/'arm_d_pipeline_state.json'


def save(**value):STATE.write_text(json.dumps(dict(value,updated=time.strftime('%Y-%m-%d %H:%M:%S')),indent=2)+'\n')


def run(stage,command,log):
    save(stage=stage,status='running',log=str(log))
    with log.open('w') as f:result=subprocess.run(list(map(str,command)),cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{stage} failed ({result.returncode}); see {log}')


def main():
    while not (BC/'bc4000/completed.json').exists():
        if (BC/'bc4000/failed.json').exists():raise RuntimeError('bc4000 failed')
        save(stage='wait_bc4000',status='waiting');time.sleep(300)
    common=['--obs','adept','--source',DATA/'source_manifest.json','--waypoints',DATA/'waypoints.json','--qualification','full-path-grasp',
        '--expected-trajectories',503,'--dataset',DATA/'native_bc_dataset.npz','--video-every',0,'--wandb-project','simtoolreal-warmstart']
    if not (BC/'bc_carry200/completed.json').exists():
        run('bc_carry200',['bash',LAUNCH,'bc','--headless',*common,'--epochs',200,'--learning-rate',1e-5,'--rnn-context','carry',
            '--init-checkpoint',BC/'bc4000/actor_init.pt','--checkpoint-every',100,'--output',BC/'bc_carry200'],BC/'bc_carry200.log')
    run('arm_d',['bash',LAUNCH,'scratch','--headless','--pristine-settings','--obs','adept','--align-left-physics','--diverse-reset-bank',BANK,
        '--actor-init',BC/'bc_carry200/actor_init.pt','--init-sigma',.4,'--critic-epochs',683,'--epochs',12000,'--output',ARM_D],
        OUT/'logs/right_mustard_pristine_adept_aligned_D_s42.log')
    save(stage='arm_d',status='complete')


if __name__=='__main__':
    try:main()
    except BaseException as error:
        save(stage='failed',status=type(error).__name__,message=str(error));raise
