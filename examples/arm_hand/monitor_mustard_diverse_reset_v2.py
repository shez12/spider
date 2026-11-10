"""Evaluate archived v2 actors at each checkpoint on fixed stage cases."""
import json,subprocess,time
from pathlib import Path
ROOT=Path('/home/szc/work/spider');OUT=ROOT/'outputs/mustard_diverse_reset_v2_native_robot'


def main():
    import torch
    template=torch.load(ROOT/'outputs/mustard_diverse_reset_ab_v1/mixed_reset_actor.pt',map_location='cpu',weights_only=False)
    evaluations=OUT/'stage_evaluations';evaluations.mkdir(exist_ok=True);done=set()
    state=evaluations/'monitor_state.json'
    if state.exists():done=set(json.loads(state.read_text())['epochs'])
    next_epoch=(max(done)//500+1)*500 if done else 10500
    while True:
        if (OUT/'failed.json').exists():raise RuntimeError((OUT/'failed.json').read_text())
        checkpoint=OUT/'sapg_latest.pth';finished=(OUT/'completed.json').exists()
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
                '--checkpoints',str(ROOT/'outputs/mustard_diverse_reset_ab_v1/mixed_reset_actor.pt'),str(snapshot),
                '--names','before_v2','after_v2','--output',str(destination)]
            with (evaluations/f'epoch_{epoch:06d}.launch.log').open('x') as log:result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            assert result.returncode==0 and (destination/'completed.json').exists(),('stage eval failed',epoch)
            report=json.loads((destination/'results.json').read_text());lines=[f'# v2分阶段评估：epoch {epoch}', '',
                '同种子、相同物理起点，原版机器人插值初始化；真实桌面高度=桌体中心+0.15m。固定单目标16秒诊断，不是50目标任务成功率。','']
            for name,model in report['models'].items():
                lines.append('## '+name)
                for suite,row in model['suites'].items():lines.append(f"- {suite}: 初始持物{row['held_initial']}/{row['cases']}，持续抓持{row['sustained_hold_before_goal']}/{row['cases']}，目标到达{row['first_goal_reached']}/{row['cases']}，成功前丢失初始抓持{row['lost_initial_hold_before_goal']}。")
            (destination/'FINDINGS.md').write_text('\n'.join(lines)+'\n')
            done.add(epoch);next_epoch=(epoch//500+1)*500;state.write_text(json.dumps({'epochs':sorted(done),'latest_report':str(destination/'FINDINGS.md')},indent=2)+'\n')
            scalars={'evaluated_training_epoch':epoch}
            for name,model in report['models'].items():
                if name!='after_v2':continue
                for suite,row in model['suites'].items():
                    scalars['stage/'+suite+'/first_goal_rate']=row['first_goal_reached']/row['cases']
                    scalars['stage/'+suite+'/sustained_grasp_rate']=row['sustained_hold_before_goal']/row['cases']
            video_dir=OUT/'videos';video_dir.mkdir(exist_ok=True)
            (video_dir/f'stage_{epoch:06d}.metrics.json').write_text(json.dumps(scalars)+'\n')
        if finished and epoch in done:
            (evaluations/'completed.json').write_text(json.dumps({'epochs':sorted(done)})+'\n');return
        time.sleep(10)


if __name__=='__main__':
    try:main()
    except BaseException as e:
        (OUT/'stage_evaluation_failed.json').write_text(json.dumps({'error':repr(e)})+'\n');raise
