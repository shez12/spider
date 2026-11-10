"""v5: critic calibration and joint SAPG from the BC v3b actor (782 demos), reset bank v3 and pose+goal curriculum.

Stage 1 (critic only): frozen BC actor, demo resets/targets, log-std -1 (sigma-scale e), 683 epochs.
Stage 2 (joint): start unfrozen from the calibrated critic; native object/robot resets mixed with bank v3,
pose+goal curriculum (gate 0.15, non-overlapping windows), native tolerance curriculum, actor lr 1e-4
adaptive (KL 0.016), clip 0.1. A stage-eval monitor uses the same fixed cases as v3/v4b/v4d.
"""
import argparse,json,math,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'outputs/mustard_classified_20261104_truncated_z2mm_v1/selected'
BANK=ROOT/'outputs/mustard_diverse_reset_bank_v41/merged/reset_bank.npz'
LAUNCH=ROOT/'examples/arm_hand/run_local_simtoolreal.sh'
PY='/home/szc/.conda/envs/simtoolreal_origin_isaacsim/bin/python'


def main(a):
    base=a.output;base.mkdir(exist_ok=False)
    def save(name,value):(base/name).write_text(json.dumps(value,indent=2,default=str)+'\n')
    def run(stage,command,background_monitor=None):
        save('pipeline_state.json',dict(stage=stage,status='running'))
        with (base/(stage+'.log')).open('w') as log:result=subprocess.run(list(map(str,command)),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'{stage} failed ({result.returncode}); see {base/stage}.log')
    common=['--headless','--num-envs',8184,'--block-size',1364,'--checkpoint',a.actor,'--dataset',DATA/'native_bc_dataset.npz',
        '--waypoints',DATA/'waypoints.json','--episode-seconds',16,'--sigma-scale',math.e,'--critic-epochs',683,'--critic-lr',.00005,
        '--multi-target','--multi-target-level',4,'--wandb-project','simtoolreal-warmstart']
    run('critic',['bash',LAUNCH,'critic',*common,'--joint-epochs',0,'--checkpoint-every',100,'--video-every',0,'--output',base/'critic'])
    joint=base/'joint'
    save('pipeline_state.json',dict(stage='joint',status='running'))
    command=['bash',LAUNCH,'critic',*common,'--start-unfrozen','--critic-initialization',base/'critic/critic_calibrated.pth',
        '--joint-epochs',200000,'--run-epochs-limit',a.joint_epochs,'--native-object-reset','--native-robot-reset',
        '--diverse-reset-bank',BANK,'--native-tolerance','--pose-curriculum','--pose-goal-curriculum','--pose-threshold',.15,
        '--actor-lr',.0001,'--adaptive-kl',.016,'--actor-clip',.1,'--checkpoint-every',500,'--video-every',500,'--video-length',960,'--output',joint]
    with (base/'joint.log').open('w') as log:
        child=subprocess.Popen(list(map(str,command)),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        # The trainer requires a fresh output directory; attach the monitor only after it exists.
        while child.poll() is None and not (joint/'run_arguments.json').exists():time.sleep(5)
        if child.poll() is None:
            monitor=subprocess.Popen([PY,str(ROOT/'examples/arm_hand/monitor_mustard_stage_eval.py'),'--run-dir',str(joint),'--label','v5',
                '--start-epoch','1000'],cwd=ROOT,stdout=(base/'monitor.log').open('w'),stderr=subprocess.STDOUT,start_new_session=True)
            save('monitor.json',dict(pid=monitor.pid))
        result=child.wait()
    if result:raise RuntimeError(f'joint failed ({result}); see {base}/joint.log')
    save('pipeline_state.json',dict(stage='joint',status='complete'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--actor',type=Path,required=True,help='Selected BC v3b checkpoint.')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/mustard_sapg_v5')
    p.add_argument('--joint-epochs',type=int,default=6000)
    a=p.parse_args()
    try:main(a)
    except BaseException as error:
        if a.output.exists():(a.output/'pipeline_failed.json').write_text(json.dumps(dict(type=type(error).__name__,message=str(error)))+'\n')
        raise
