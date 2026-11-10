"""Controlled finite SAPG ablation; preserves the existing long native run."""
import json,subprocess,time,os
from pathlib import Path
ROOT=Path('/home/szc/work/spider');OUT=ROOT/'outputs/mustard_diverse_reset_ab_v1'
BANK=ROOT/'outputs/mustard_diverse_reset_bank_v1'


def command(name,epochs,small=False,mixed=False):
    data=ROOT/'outputs/mustard_classified_20261102_v2/selected'
    cmd=['bash',str(ROOT/'examples/arm_hand/run_local_simtoolreal.sh'),'critic','--headless',
        '--num-envs','600' if small else '8184','--block-size','100' if small else '1364',
        '--checkpoint',str(ROOT/'outputs/mustard_classified_20261102_v2/bc_carry200/actor_init.pt'),
        '--resume-training',str(ROOT/'outputs/mustard_native_object_from_checkpoint_20261103/sapg_latest.pth'),
        '--dataset',str(data/'native_bc_dataset.npz'),'--waypoints',str(data/'waypoints.json'),
        '--critic-epochs','683','--joint-epochs','200000','--actor-lr','0.00001','--actor-clip','0.05','--critic-lr','0.00005',
        '--multi-target','--multi-target-level','4','--native-object-reset','--fixed-difficulty','--episode-seconds','16',
        '--run-epochs-limit',str(epochs),'--checkpoint-every','100','--video-every','0' if small else '100000',
        '--video-length','960','--wandb-project','' if small else 'simtoolreal-warmstart','--output',str(OUT/name)]
    if mixed:cmd+=['--diverse-reset-bank',str(BANK/'reset_bank.npz')]
    return cmd


def run(name,epochs,small=False,mixed=False):
    if (OUT/name).exists():
        while not (OUT/name/'completed.json').exists():
            assert not (OUT/name/'failed.json').exists(),f'{name} failed'
            proof=json.loads((OUT/(name+'.process.json')).read_text())
            process=Path('/proc')/str(proof['pid'])
            assert process.exists() and b'train_critic.py' in (process/'cmdline').read_bytes(),f'{name} process exited before completion'
            time.sleep(10)
        assert not (OUT/name/'failed.json').exists();return
    with (OUT/(name+'.launch.log')).open('x') as f:
        p=subprocess.Popen(command(name,epochs,small,mixed),cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
        (OUT/'active_child.json').write_text(json.dumps(dict(name=name,pid=p.pid))+'\n')
        status=p.wait()
    assert status==0 and (OUT/name/'completed.json').exists() and not (OUT/name/'failed.json').exists(),(name,status)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    try:
        run('native_only',300)
        while not (BANK/'completed.json').exists():
            if (BANK/'failed.json').exists():raise RuntimeError((BANK/'failed.json').read_text())
            time.sleep(5)
        if (OUT/'smoke_mixed_early/completed.json').exists():
            proof=json.loads((OUT/'smoke_mixed_early/joint_update_check.json').read_text())
            assert proof['critic_changed'] and proof['actor_changed_keys']
        else:run('smoke_mixed',3,small=True,mixed=True)
        run('mixed_reset',300,mixed=True)
        for name in ('native_only','mixed_reset'):
            rows=[json.loads(s) for s in (OUT/name/'sapg_progress.jsonl').read_text().splitlines()]
            assert len(rows)==300 and rows[0]['epoch']==10001 and rows[-1]['epoch']==10300
            assert all(row['max_targets']==50 and row['episode_seconds']==16 and row['success_tolerance']==.075 for row in rows)
        (OUT/'training_completed.json').write_text(json.dumps({'epochs_per_branch':300,'new_frames_per_branch':39283200})+'\n')
    except BaseException as e:
        (OUT/'pipeline_failed.json').write_text(json.dumps({'error':repr(e)})+'\n');raise


if __name__=='__main__':main()
