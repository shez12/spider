# 多样reset v2：原生机器人初态、真实桌面高度与阶段诊断

## 必须修正的定义
- 原版关节采样：q0=0.9*qdefault+0.1*Uniform(lower,upper)，机械臂和手指均如此。0.1是插值系数，不是弧度噪声；速度Uniform(-0.5,0.5)rad/s。
- 直接调用vendor `_randomize_robot_dof_state`。native reset分支应用它；其余缓存分支恢复配套q/v和控制器目标，保持已验证的接触构型。
- `_table_z_per_env`是桌体中心高度，table_narrow厚0.30m，因此实际桌面为table.root_pos_z+0.15m（正常值0.53m）。
- 第一版额外离桌检查和第一轮stage诊断错误地减去了桌体中心。第一轮stage的held/drop指标作废；既有目标到达与碰撞审核没有使用这个误差判据。

## v2状态库
从v1已完成的独立Isaac验证和几何快照重新审核，不重新优化整条轨迹。
- near-object：保存状态及独立重放末态，物体网格最低点距实际桌面不超过2cm；排除已悬空持物的候选。
- grasp/near-goal：两个时刻均必须离实际桌面至少2cm；其余原审核继续成立。
- 目标初始缩放角点距离>0.125m；near-goal另要求<0.35m，避免直接reset成功。
- v2共1511状态：near-object379、grasp770、near-goal362。v1库保留用于追溯。
- native部分在线使用原版物体、桌高、机器人关节与速度采样；混合概率仍为.4/.25/.2/.15，6个block一致。

## 执行顺序
1. 修正高度判据，测试原生采样公式与±0.5速度，用相同随机数验证真实采样结果。
2. 用真实桌面高度和原版机器人reset重做分阶段评估，包含保持指令物理对照；第一轮误判记录在metric_warning.json。
3. 600环境3轮短测native-robot + cached-bank组合，验证恢复、局部reset与两网络更新。
4. 从v1 mixed的10300轮完整快照续训2000轮至12300轮。8184环境、原观测/动作/奖励/50目标保持；LR/clip沿用1e-5线性衰减/.05，critic5e-5。
5. 固定16s和tolerance .075用于此探索窗口；本轮同时修正robot初态与状态类别，不声称隔离两者的因果效果。
6. 每500轮存档和原生reset视频；定期评估真实native机器人+物体初态及阶段状态，单列各类reset的完成回合数、目标均值和任一目标到达率。
7. 当前其他聊天的native-robot/tolerance训练及已暂停进程不由本实验改动。

## 已启动与实测证据
- 启动脚本：examples/arm_hand/run_mustard_diverse_reset_v2.sh；输出outputs/mustard_diverse_reset_v2_native_robot。
- 正式运行native_robot_reset_check.json已核对16368次采样，速度范围[-0.4999992,0.4999995]，控制器previous/current目标与q误差0。
- resume_check.json确认epoch10300、actor/critic优化器和AMP恢复，未重启actor LR日程。
- 周期阶段评估由examples/arm_hand/monitor_mustard_diverse_reset_v2.py执行，首次10500，之后每500轮及最终检查点。
- 修正后的配对诊断：mixed10300在128个真正near-teacher状态中109个到达目标；84个真实失败/恢复状态仍0个到达。原生机器人+物体初态128个也为0，不把熟悉状态表现视为原生任务成功。
