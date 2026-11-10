"""Durable sequential matched-budget BC experiment; leaves formal SAPG running."""
import json
import subprocess
import time
from pathlib import Path

ROOT=Path('/home/szc/work/spider')
OUT=ROOT/'outputs/mustard_balance_ab_v1'
DATA=ROOT/'outputs/mustard_classified_20261102_v2/selected'
INIT=ROOT/'outputs/mustard_classified_20261102_v2/bc_carry200/actor_init.pt'


def main():
    status={}
    try:
        for name,epochs,weighted in [('smoke_weighted',2,True),('baseline',200,False),('balanced',200,True)]:
            dest=OUT/name
            command=['bash',str(ROOT/'examples/arm_hand/run_local_simtoolreal.sh'),'bc','--headless',
                '--dataset',str(DATA/'native_bc_dataset.npz'),'--waypoints',str(DATA/'waypoints.json'),
                '--qualification','full-path-grasp','--expected-trajectories','503',
                '--init-checkpoint',str(INIT),'--rnn-context','carry','--epochs',str(epochs),
                '--learning-rate','0.00001','--sequence-length','16','--batch-sequences','128',
                '--checkpoint-every','100','--video-every','0','--seed','42','--output',str(dest)]
            if weighted:command+=['--sample-weights',str(OUT/'audit/weights.npz')]
            # Local exact records avoid tying learner lifetime to remote network health.
            if not (dest/'completed.json').exists():
                assert not dest.exists(),f'Incomplete output requires investigation: {dest}'
                with (OUT/(name+'.launch.log')).open('x') as log:
                    result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                assert result.returncode==0,(name,result.returncode)
            rows=[json.loads(s) for s in (dest/'bc_progress.jsonl').read_text().splitlines()]
            assert len(rows)==epochs and all(r['supervised_frames']==377309 for r in rows)
            assert rows[-1]['amp_skipped_updates']/(rows[-1]['optimizer_steps']+rows[-1]['amp_skipped_updates'])<.001,'Excessive AMP overflow requires investigation'
            status[name]=dict(completed=True,epochs=epochs,last=rows[-1])
            (OUT/'pipeline_status.json').write_text(json.dumps(status,indent=2)+'\n')
        attempts=lambda name:status[name]['last']['optimizer_steps']+status[name]['last']['amp_skipped_updates']
        assert attempts('baseline')==attempts('balanced'),'Optimizer attempt budgets differ'
        (OUT/'training_completed.json').write_text(json.dumps(dict(status=status,finished=time.time()),indent=2)+'\n')
    except BaseException as e:
        (OUT/'pipeline_failed.json').write_text(json.dumps(dict(error=repr(e),status=status),indent=2)+'\n')
        raise


if __name__=='__main__':main()
