"""Summarize completed Isaac Sim actor training without changing checkpoints."""
from pathlib import Path
import argparse
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,default=Path('outputs/mustard_actor_isaac20_v3'))
    args=ap.parse_args();p=args.run
    done=json.loads((p/'completed.json').read_text());history=json.loads((p/'history.json').read_text())
    runtime=json.loads((p/'asset_runtime_check.json').read_text());config=json.loads((p/'config_summary.json').read_text())
    best=next(r['evaluation'] for r in history if r['round']==done['best_round'])
    teacher=done['teacher'];initial=done['initial_actor']
    total=lambda r:r['train_pass']+r['validation_pass']
    text=f'''# Isaac Sim 中的芥末瓶 actor 模仿训练

日期：2026-10-27。该轮已经实际在 Isaac Sim / Isaac Lab / PhysX 中采样，并完成 actor 监督更新。环境由 SimToolReal 的 SimToolRealEnv 派生，载入与示范一致的 KUKA + 原生右 Sharpa 和芥末瓶。

## 实际训练设置

- 固定原有 20 条空间变体，16 条训练、4 条验证；没有把正在生成的数据加入本轮。
- 仿真在 CUDA 上运行，actor 的监督梯度在 CPU 上计算。两者每轮交替：真实环境采样 → 更新网络 → 完整 actor 闭环验证。
- 控制 {config['control_hz']} Hz，PhysX {config['physics_hz']} Hz，20 个并行环境；右手 PD、质量惯量从编译后的源模型导出，机器人重力补偿，物体正常受重力。
- 网络为本地 SAPG 的 LSTM 1024 + MLP [1024,1024,512,512]，保留六组探索条件。该阶段不进行 SAPG policy-gradient 或 critic 更新。
- 输出为归一化绝对关节目标的残差解码，未启用原 SAPG 的 arm 增量/EMA 映射。因此这是 SimToolReal 环境中的 actor BC 阶段，不是未经修改的原生 SAPG action contract。
- 用 PhysX 中示范回放的实际观测训练，再逐步降低示范控制比例，收集 actor 会访问的状态继续监督。标签为同一示范时间的目标关节控制，不是可查询任意偏离状态的专家；不能称为通用 DAgger。
- 仅训练集观测参与归一化拟合与梯度更新；验证变体用于选模。所有变体仍共享一个源 episode，结论限于这一物体和轨迹族。
- 运行 {done['rounds']} 轮，每轮 {done['epochs_per_round']} 次全序列更新，合计 {done['rounds']*done['epochs_per_round']} 次；总耗时 {done['wall_time_s']:.1f} s。

## 闭环结果

以下通过数仅表示末端位置误差 <3 cm 且姿态误差 <20°，未包含额外 hold 或完整碰撞验收，不能当作部署成功率。

- PhysX 示范回放：{total(teacher)}/20，验证 {teacher['validation_pass']}/4；平均末端位置误差 {teacher['mean_position_error_m']:.4f} m。
- 训练前 actor：{total(initial)}/20，验证 {initial['validation_pass']}/4；平均末端位置误差 {initial['mean_position_error_m']:.4f} m。
- 按验证表现选中的 actor（round {done['best_round']}）：{total(best)}/20，训练 {best['train_pass']}/16，验证 {best['validation_pass']}/4；平均末端位置误差 {best['mean_position_error_m']:.4f} m、方向误差 {best['mean_orientation_error_deg']:.2f}°。

不能把混合示范/actor 的 rollin 通过数作为独立 actor 的结果。上面的 actor 数值均来自 teacher_probability=0 的完整闭环。

## 环境接入检查与修复

- 导出机器人在源引擎中的多姿态指尖 FK 误差约 4.3e-12 m。
- 原数据与目标环境初始 140 维观测对齐，已读取的数值差异在约 1e-7 量级。
- Isaac runtime 实际关节数 29、观测维数 {runtime['observation_dim']}；导入关节限位与 BC 解码限位最大差异 {runtime['limits_max_error_rad']:.3g} rad。
- 修复了 MuJoCo actuator_forcelimited=false / forcerange=[0,0] 被误当作零力矩上限的问题；现在按“源模型未限力”导出足够大的仿真上限。之前 v1 的手部无驱动回放不作为有效训练结果。
- 保留固定的掌心 link，使用源 palm/tip 局部偏移；原 SimToolReal 仓库没有被修改，适配仅限本轮进程。
- 全序列 GPU 反向在共享 GPU 上迟迟未返回有效更新，v2 在生成诊断回放后被中止；v3 使用 CPU learner 完成训练。GPU 上原有数据验证进程没有停止。

## 当前结论

已完成“直接在 Isaac Sim 真实 rollout 上训练 actor”的首轮实验。物理迁移后的示范自身尚未稳定通过，因此需要分别改善 PhysX 示范执行一致性与 actor 的闭环跟踪。当前不能仅凭训练 loss 下降进入 critic 热启动，也不能把本轮视为 SAPG 加速实验完成。

## 产物

- actor_best.pt：验证选中的 actor、归一化、schema、网络配置及优化器。
- actor_latest.pt：最后一轮。
- teacher_baseline.npz / actor_initial.npz / actor_round_*.npz：真实 PhysX 观测、物体轨迹和关节轨迹。
- history.json / completed.json：训练过程与完成状态。
- config_summary.json / asset_runtime_check.json：环境、输入输出合同与导入检查。
- training_progress.png：完整 actor 闭环指标变化。

复现：

```bash
bash examples/arm_hand/run_mustard_isaac_actor.sh --output outputs/mustard_actor_isaac20_retest --assets outputs/mustard_isaac_assets_v3 --headless --rounds 6 --epochs-per-round 40 --learner-device cpu
```
'''
    (p/'REPORT.md').write_text(text)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
    image=Image.new('RGB',(1100,460),'white');draw=ImageDraw.Draw(image)
    draw.text((25,15),'Isaac Sim closed-loop actor evaluation (no teacher actions)',fill='black',font=font)
    evals=[initial]+[r['evaluation'] for r in history]
    for panel,key,title,unit in [(0,'mean_position_error_m','Mean final position error','m'),(1,'validation_pass','Held-out variants passing pose thresholds','episodes / 4')]:
        x0,y0=75+550*panel,85;w,h=420,270
        values=[float(r[key]) for r in evals];lo=0;hi=max(values)*1.15 if key!='validation_pass' else 4
        hi=max(hi,.01)
        draw.text((x0-20,y0-25),title,fill='black',font=small)
        for v in np.linspace(lo,hi,5):
            yy=y0+h-(v-lo)/(hi-lo)*h
            draw.line((x0,yy,x0+w,yy),fill='#eeeeee');draw.text((x0-50,yy-6),f'{v:.2f}',fill='black',font=small)
        draw.line((x0,y0,x0,y0+h,x0+w,y0+h),fill='#444444')
        points=[(x0+i/max(1,len(values)-1)*w,y0+h-(v-lo)/(hi-lo)*h) for i,v in enumerate(values)]
        draw.line(points,fill='#267a67',width=3)
        for i,(xx,yy) in enumerate(points):
            draw.ellipse((xx-3,yy-3,xx+3,yy+3),fill='#267a67');draw.text((xx-8,y0+h+10),'init' if i==0 else str(i-1),fill='black',font=small)
        draw.text((x0,y0+h+35),f'Training round; vertical axis: {unit}',fill='black',font=small)
    draw.text((25,425),'Source: 20 actual PhysX rollouts per evaluation; 16 train / 4 validation spatial variants; thresholds 3 cm and 20 deg.',fill='black',font=small)
    image.save(p/'training_progress.png')
    print(json.dumps(dict(best_round=done['best_round'],actor_pose_pass=total(best),validation_pose_pass=best['validation_pass'],report=str(p/'REPORT.md'))))


if __name__=='__main__':main()
