"""Evaluate a run's checkpoints on the same fixed stage cases as the v3 monitor.

Same bank, dataset, seed and v2 epoch-12300 baseline as
monitor_mustard_diverse_reset_v3.py, so `after_<label>` rows are directly
comparable with v3's `after_v3` rows at matching epochs.
"""
import argparse,json,subprocess,time
from pathlib import Path
ROOT=Path('/home/szc/work/spider')
BASELINE=ROOT/'outputs/mustard_diverse_reset_v2_native_robot/stage_evaluations/actor_012300.pt'


def main(a):
    import torch
    out=a.run_dir;template=torch.load(BASELINE,map_location='cpu',weights_only=False)
    evaluations=out/'stage_evaluations';evaluations.mkdir(exist_ok=True);done=set()
    state=evaluations/'monitor_state.json'
    if state.exists():done=set(json.loads(state.read_text())['epochs'])
    next_epoch=(max(done)//a.every+1)*a.every if done else a.start_epoch
    after='after_'+a.label
    while True:
        if (out/'failed.json').exists():raise RuntimeError((out/'failed.json').read_text())
        checkpoint=out/'sapg_latest.pth';finished=(out/'completed.json').exists()
        if not checkpoint.exists():time.sleep(10);continue
        try:ck=torch.load(checkpoint,map_location='cpu',weights_only=False)[0]
        except (EOFError,RuntimeError):time.sleep(5);continue
        epoch=int(ck['epoch'])
        if epoch not in done and (epoch>=next_epoch or finished):
            snapshot=evaluations/f'actor_{epoch:06d}.pt';torch.save(dict(template,model=ck['model'],training_epoch=epoch),snapshot)
            destination=evaluations/f'epoch_{epoch:06d}'
            command=['bash',str(ROOT/'examples/arm_hand/run_mustard_reset_stage_eval.sh'),'--headless','--native-robot-reset',
                '--bank',str(ROOT/'outputs/mustard_diverse_reset_bank_v2/reset_bank.npz'),
                '--dataset',str(ROOT/'outputs/mustard_classified_20261102_v2/selected/native_bc_dataset.npz'),
                '--checkpoints',str(BASELINE),str(snapshot),'--names','baseline_v2_012300',after,'--output',str(destination)]
            with (evaluations/f'epoch_{epoch:06d}.launch.log').open('x') as log:result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            assert result.returncode==0 and (destination/'completed.json').exists(),('stage eval failed',epoch)
            report=json.loads((destination/'results.json').read_text());lines=[f'# {a.label}分阶段评估：epoch {epoch}','',
                '与v3监控相同的种子、物理起点和基线（v2 epoch 12300）；原版机器人插值初始化。固定单目标16秒诊断，不是50目标任务成功率。','']
            for name,model in report['models'].items():
                lines.append('## '+name)
                for suite,row in model['suites'].items():lines.append(f"- {suite}: 初始持物{row['held_initial']}/{row['cases']}，持续抓持{row['sustained_hold_before_goal']}/{row['cases']}，目标到达{row['first_goal_reached']}/{row['cases']}，成功前丢失初始抓持{row['lost_initial_hold_before_goal']}。")
            (destination/'FINDINGS.md').write_text('\n'.join(lines)+'\n')
            done.add(epoch);next_epoch=(epoch//a.every+1)*a.every
            state.write_text(json.dumps({'epochs':sorted(done),'latest_report':str(destination/'FINDINGS.md')},indent=2)+'\n')
            scalars={'evaluated_training_epoch':epoch}
            for suite,row in report['models'][after]['suites'].items():
                scalars['stage/'+suite+'/first_goal_rate']=row['first_goal_reached']/row['cases']
                scalars['stage/'+suite+'/sustained_grasp_rate']=row['sustained_hold_before_goal']/row['cases']
            video_dir=out/'videos';video_dir.mkdir(exist_ok=True)
            (video_dir/f'stage_{epoch:06d}.metrics.json').write_text(json.dumps(scalars)+'\n')
        if finished and epoch in done:
            (evaluations/'completed.json').write_text(json.dumps({'epochs':sorted(done)})+'\n');return
        time.sleep(10)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--label',required=True)
    p.add_argument('--start-epoch',type=int,required=True);p.add_argument('--every',type=int,default=500)
    args=p.parse_args()
    try:main(args)
    except BaseException as e:
        (args.run_dir/'stage_evaluation_failed.json').write_text(json.dumps({'error':repr(e)})+'\n');raise
