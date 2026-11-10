"""Evaluate both finite SAPG branches only after matching training budgets."""
import json,subprocess,time
from pathlib import Path
ROOT=Path('/home/szc/work/spider');OUT=ROOT/'outputs/mustard_diverse_reset_ab_v1'


def main():
    while not (OUT/'training_completed.json').exists():
        if (OUT/'pipeline_failed.json').exists():raise RuntimeError((OUT/'pipeline_failed.json').read_text())
        time.sleep(10)
    import torch
    template=torch.load(ROOT/'outputs/mustard_classified_20261102_v2/bc_carry200/actor_init.pt',map_location='cpu',weights_only=False)
    sources=[ROOT/'outputs/mustard_native_object_from_checkpoint_20261103/sapg_latest.pth',OUT/'native_only/sapg_latest.pth',OUT/'mixed_reset/sapg_latest.pth']
    names=['initial','native_only','mixed_reset'];paths=[]
    for name,source in zip(names,sources):
        checkpoint=torch.load(source,map_location='cpu',weights_only=False)[0]
        path=OUT/(name+'_actor.pt');torch.save(dict(template,model=checkpoint['model'],training_epoch=checkpoint['epoch']),path);paths.append(path)
    data=ROOT/'outputs/mustard_classified_20261102_v2/selected'
    command=['bash',str(ROOT/'examples/arm_hand/run_local_simtoolreal.sh'),'balance-evaluate','--headless',
        '--checkpoints',*map(str,paths),'--names',*names,'--dataset',str(data/'native_bc_dataset.npz'),'--waypoints',str(data/'waypoints.json'),
        '--audit',str(ROOT/'outputs/mustard_balance_ab_v1/audit/audit.json'),
        '--drop-bank',str(ROOT/'outputs/mustard_classified_20261102_v2/drop_bc_coverage_check_v1/drop_bank.npz'),
        '--output',str(OUT/'evaluation'),'--native-cases','256','--frames','960']
    with (OUT/'evaluation.launch.log').open('x') as log:result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    assert result.returncode==0 and (OUT/'evaluation/completed.json').exists(),'Paired evaluation failed'
    report=json.loads((OUT/'evaluation/results.json').read_text());lines=['# 多样reset：300轮SAPG配对对照','',
        '两分支从同一第10000轮完整actor/critic/Adam快照出发，各新增39,283,200帧；6×1364环境、原生奖励与动作、50目标、固定16秒和tolerance .075。',
        '训练reset混合比例native/near/grasp/near-goal=.40/.25/.20/.15。以下评估不使用混合reset，固定每例16秒、目标不变且关闭自动reset；不是50目标回合成功率。','']
    for name,model in report['models'].items():
        lines.append('## '+name)
        for suite,row in model['suites'].items():lines.append(f"- {suite}: 抓持{row['grasped']}/{row['cases']}，首次目标到达{row['reached']}/{row['cases']}。")
    lines+=['','## 混合减原版：配对差']
    for suite,metrics in report['paired_differences'].items():
        for metric,row in metrics.items():
            lo,hi=row['paired_bootstrap_95'];lines.append(f"- {suite}/{metric}: {row['balanced_minus_baseline']:+.1%}，95%案例bootstrap区间[{lo:+.1%},{hi:+.1%}]。")
    lines+=['','只有一个训练seed和300轮窗口，不能据此断言长期效果。状态库复用已有优化抓持构型，不宣称已覆盖全SO(3)或复现OmniReset灵巧手能力。',
        '几何准入使用原生URDF的碰撞几何并验证Isaac FK一致，动态稳定性在Isaac中测试；不宣称几何接触距离与PhysX完全一致。']
    (OUT/'FINDINGS.md').write_text('\n'.join(lines)+'\n')
    (OUT/'experiment_completed.json').write_text(json.dumps({'models':names,'cases_per_model':773,'report':'FINDINGS.md'})+'\n')


if __name__=='__main__':
    try:main()
    except BaseException as e:
        OUT.mkdir(exist_ok=True);(OUT/'finish_failed.json').write_text(json.dumps({'error':repr(e)})+'\n');raise
