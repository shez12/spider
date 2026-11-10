# 当前BC结果及后续阶段约定

## 2026-11-04：v3原生回合驱动tolerance课程已启动

v2已完成12300。v3从该完整快照续训至16300；原版采样、6×1364、50目标、16s、混合库不变。只用最近完成native回合统计原版tolerance收紧条件；缓存成功不推动收紧，其他难度保持。5项单测通过，600环境3轮actor/critic更新检查通过。正式进程已启动，初始化中；输出outputs/mustard_diverse_reset_v3_native_curriculum。计划docs/development/mustard-omnireset-v3.md。

## 2026-11-03：多样reset v2与原版机器人初始化

已从mixed_reset第10300轮完整快照启动v2续训，计划至12300轮；8184环境（6×1364）。
原生分支直接调用vendor关节采样：q=0.9*default+0.1*Uniform(lower,upper)，速度±0.5rad/s；缓存分支保留经过验证的配套机器人状态。
正式运行16368次初始化审计通过，previous/current目标与采样q完全一致；优化器与AMP恢复，学习率进度不重启。
纠正桌体中心与桌面高度混淆后，v2库1511状态；旧stage诊断的held/drop指标作废，以mustard_reset_stage_diagnosis_v2为准。
新输出：outputs/mustard_diverse_reset_v2_native_robot；详细配置：docs/development/mustard-omnireset-v2.md。
每500轮视频及阶段评估。其他聊天的训练和已暂停进程未改动。下面v1“正在执行”条目为历史记录，v1对照已完成。

## 2026-11-03：多样reset SAPG对照正在执行

计划与边界：`docs/development/mustard-omnireset-plan.md`。
状态库`outputs/mustard_diverse_reset_bank_v1`有2073个验证状态；600环境3轮混合短测通过。
对照目录`outputs/mustard_diverse_reset_ab_v1`，native_only/mixed_reset各8184环境、同10000轮快照新增300轮。
固定16秒、tolerance .075、50目标，混合概率.40/.25/.20/.15，每个SAPG block同分布。
原native-object长训练继续作为在线参考。状态库不是BC动作标签，也没有重启暂停的优化任务。
两分支完成后自动运行原版reset等773案例配对诊断；尚不能声称混合方法提高了成功率。


## 2026-11-03：抓取分布均衡BC对照（不替换正式SAPG）

详细计划：`BALANCED_BC_PLAN.md`；结果：`outputs/mustard_balance_ab_v1/FINDINGS.md`。
两组从同一bc_carry200初始化，各200轮、75,461,800个有效监督帧、47,997次更新尝试；
普通组AMP跳过5次，均衡组6次，成功更新次数相差1。只训练actor，归一化、sigma、critic不变。
分组按初始物体局部Z轴朝向与稳定抓持构型；503条中388条朝上、115条侧向、0条朝下。
8组26～147条；50%轨迹均匀+50%组均匀形成固定逐帧权重，范围0.558～2.302。

每模型503示范+256原版物体reset+14自然掉落，共773例，固定目标16秒诊断：
原始BC示范抓持396/503、到达178/503；再普通BC为377/503、182/503；均衡BC为380/503、172/503。
普通与均衡随机物体抓持均1/256、到达均0/256，掉落恢复均0/14。
均衡相对普通的示范到达差-2.0个百分点，配对bootstrap95%区间[-6.2,+2.2]，没有确认收益。
不据此替换初始化、不以训练MSE下降宣称改善，也不把这些数当原版50目标成功率。

W&B run：mustard_balance_ab_v1_baseline / mustard_balance_ab_v1_balanced，标量已同步。
269个失败后的物理状态导出至`outputs/mustard_balance_ab_v1/recovery_requests.npz`，它们不是成功标签。
额外发现：523条被排除轨迹中296条在16秒内已完成抓持（其中168条侧向），
记录在`audit/prefix_coverage.json`；应独立验证成功前缀扩充，不能混入本次均衡A/B宣称效果。
正式训练仍为下节的mustard_native_object_sapg_v1，没有被BC对照覆盖。


## 2026-11-03：原版物体随机 reset，直接联合训练

新目录：`outputs/mustard_native_object_sapg_v1`。
启动脚本：`examples/arm_hand/run_mustard_native_object_sapg_v1.sh`。
正式PID639361，已确认10001轮起联合更新。旧sapg50训练已停止，日志与checkpoint保留。
W&B filestream已返回200 OK；正式16368次reset采样检查通过。
初始完整权重固定在 `outputs/mustard_native_object_from_checkpoint_20261103`，来自
classified_v2/sapg50 第10000轮；包含 actor、central critic、两套 Adam、AMP 和课程状态。
从10001轮直接联合更新，不重复 BC 或冻结 critic 预热，学习率衰减进度也不重置。

调用原版物体 reset：XY 各 ±0.1m，Z=桌高+0.25m±0.02m，完整三维随机旋转，
初速度为0；桌高0.38m±0.01m。首次目标也用原版 absolute 采样，后续用原版 delta。
机器人关节位置/速度/上一动作目标仍来自随机选取的503条示范初态；没有恢复原版
机器人关节噪声、动作延迟、观测噪声或随机外力。因此不是全部环境参数都恢复原版。

6×1364=8184环境，50目标固定。actor基础LR=1e-5（承接线性衰减），clip=.05，
central critic LR=5e-5。物体随机范围立即固定为原版，不再走位置扰动课程。
16秒无进展时限继续按平均完成目标数≥3的门槛逐步缩短到10秒，之后启用原版tolerance
课程；每次只增加一项难度，表现下降时保持，不回退。每1000轮保存checkpoint并录制
960帧视频，恢复后的首轮也录制。W&B使用独立run名 mustard_native_object_sapg_v1。

600环境3轮短测通过（10001～10003）：两网络均更新、两套优化器恢复，1200次reset
范围/姿态/零速度检查通过。证据：`outputs/mustard_native_object_smoke_20261103`。
以下为历史记录，旧PID及课程不代表当前运行设置。


## 当前版本v4：单向课程，先上传历史再从5000轮重启

用户明确取消所有难度回退。表现下降时保持当前目标数、reset范围、超时和tolerance，
直到表现恢复才增加难度；本节取代下方v3中的回退规则，旧记录仅供追溯。
v3在6326轮停止，其5000轮后数据不混入v4；v4使用原始v2第5000轮完整快照。

原版termination_utils.update_tolerance_curriculum：距离上次收紧至少3000个控制步，
各环境最近完成回合的平均成功目标数≥3时，容差乘.9，下限.01；不满足时保持且不更新
收紧计时点，没有放宽或回退。本版本最终直接启用该原生函数与原生门槛。
目标数/位置/时限过渡每500轮检查，至少积累1000个完成leader回合；推进依据也改成
原生各环境最近回合平均目标数。只有上限1或2的桥接阶段采用其上限的80%（.8/1.6），
从5目标起门槛为平均3个目标。未达标原地保持，达标一次只推进一个难度维度。
依次交替推进目标数与位置范围，然后16→14→12→10秒；全部恢复后启用原生容差课程。
没有回退分支；完整目标上限成功率和回合达成≥min(3,cap)目标的比例仍分别记录，
两者均不再冒充课程推进门槛。课程实际门槛及原生平均目标数另有日志字段。

单元验证：test_curriculum.py直接执行原版tolerance函数，确认低表现保持、恢复后收紧、
以及.01下限；同时验证课程低表现不变、恢复后每次仅一步。三项测试通过。
正式5000轮checkpoint的120环境3轮续训验证通过，目录mustard_monotonic_curriculum_smoke_v4。

正式输出：`/home/szc/work/spider/outputs/mustard_warmstart_sapg_monotonic_v4`，PID2492127。
脚本：`/home/szc/work/spider/examples/arm_hand/run_mustard_monotonic_v4.sh`。
启动之前已向同名W&B run补齐原始v2第1～5000轮的5000条历史记录，含93类TensorBoard
标量及训练/视频评估标量。独立远端查询确认5000轮齐全且不重复，frame=epoch×8184×16，
范围130944～654720000。证明文件history_remote_verification.json在启动前写入，
训练入口必须验证它才能使用预先上传历史的输出目录。训练从5001轮接上。
actor、central critic、两套Adam、AMP和学习率衰减进度保留；物理回合重新初始化。
W&B：https://wandb.ai/shezhangcong-johns-hopkins-university/simtoolreal-warmstart/runs/mustard_warmstart_sapg_monotonic_v4

## 2026-10-29续训：多目标、原版物体位置随机化、16秒渐进到10秒

已从v2最近落盘的第5000轮checkpoint续训（654720000 frames），不是重新BC或critic预热。
v2在第5379轮停止，5000～5379未落盘更新未迁移；其日志和文件全部保留在v2目录。
正式输出：`/home/szc/work/spider/outputs/mustard_warmstart_sapg_nativecurriculum_v3`，PID1449657。
启动脚本：`/home/szc/work/spider/examples/arm_hand/run_mustard_nativecurriculum_v3.sh`。
保留actor、central critic、两个Adam、AMP scaler、输入统计与原lr衰减进度；物理回合重新reset，
没有声称恢复PhysX接触快照。6×1364环境，actor lr实际约9.76e-6，clip=.05，critic lr5e-5。

多目标课程上限1→2→5→10→50；首个目标仍取配套示范最终位姿，后续调用原生delta采样器。
后续目标位置/旋转幅度分别从.05m/30°，经.075m/45°，恢复至原版.1m/90°。
目标到达时沿用原生goal tracker清零和计时归零，不重置机器人、物体或LSTM。
终止为达到当前目标数上限、物体掉落、手过远或无进展超时。

物体reset从当前XY±.005m，逐步到±.01、±.02、±.05、±.1m；Z逐步到±.002、±.005、
±.01、±.02m。同步将采样中心从缓存的实际物体初态插值到原版中心：X/Y=0，
Z=桌面根高度+table_object_z_offset，最终位置分布与原版一致。首个目标共享物体的位姿变换。
物体朝向仍采用示范朝向+现有2°yaw，机器人关节扰动仍±.01rad，不恢复全随机朝向/延迟/外力。

每500轮、至少1000个完成leader回合才检查课程，每次只调一个难度；交替扩展目标数和物体
位置范围，之后才把无进展超时16→14→12→10秒，最后再收紧目标容差到原版配置下限。
课程达标率为“回合至少达到min(3,目标数上限)个目标”的比例，≥.70推进，<.40回退；
完整达到目标数上限的回合成功率另记，不能把达标率当作50目标整回合成功率。
这与原版按若干目标完成情况推进的逻辑相近，避免必须70%回合完成50目标才能继续课程。
原版小数dt在DirectRLEnv向上取整，16/14/12/10秒对应实际最大计数961/841/721/601步；
本实验保留该原生计时行为，没有额外改变控制频率。

验证：mustard_multitarget_resume_smoke_v2检查了原生到达后继续与cap终止；
mustard_native_curriculum_smoke_v4用正式5000轮权重验证两个优化器恢复、完整位置边界、
四档动态时限、后续SAPG更新和当前课程视频。快照保存在mustard_multitarget_from_v2_epoch5000。

## 2026-10-29新训练：小噪声重校准后自动解冻SAPG

启动脚本：`/home/szc/work/spider/examples/arm_hand/run_mustard_warmstart_sapg_v2.sh`。
输出：`/home/szc/work/spider/outputs/mustard_warmstart_sapg_lownoise_v2`，启动PID1758393。
6×1364=8184物理环境，任务16秒；BC actor仍来自4000+200轮的carry模型。
初始动作sigma为exp(-2)×0.25≈0.033834；加载上一轮已校准central critic权重和统计，
central Adam重新初始化，先冻结actor全部权重与缓冲区重校准200轮。
随后自动进行200000轮原生SAPG联合更新（含原生经验共享），actor lr从1e-5线性衰减，
actor clip=.05，central critic lr固定5e-5，两者4个mini epochs。
联合阶段sigma恢复原生可学习状态；actor输入统计全程固定，central统计继续更新。
仅对小sigma下有偏的原生KL诊断作FP32稳定计算修正，没有修改vendor代码或PPO损失。

解冻后至少500轮才考虑初态扰动。每500轮检查最近最多10000个已结束leader回合，
至少1000个回合且成功率≥.70才增加一级，低于.55则回退一级；变更后清空该统计窗口。
四档（xy平移上界米、物体yaw上界度、关节位置上界弧度）：
0=(0,0,0)，1=(.001,.5,.002)，2=(.003,1,.005)，3=(.005,2,.01)。
扰动仅作用于后续reset；物体与最终目标共享平移/yaw，关节位置和上一目标同步偏移并限位，
关节速度保持示范值。没有旋转固定机器人基座；延迟、观测噪声、外力继续关闭。
最大扰动档稳定后，成功率≥.80才逐次将目标容差乘.9收紧至原生配置下限.01。

每1000轮保存最新完整策略/critic及central Adam；阶段切换也保存检查与checkpoint。
第1轮、阶段边界和每1000轮以当时权重录制16秒本地视频，带当前扰动/容差，
W&B同步原生各项奖励、critic损失、actor辅助价值损失、lr/clip/sigma、成功率和课程级别。
视频仅是6环境确定性诊断，不代表训练时随机策略总体成功率。

验证：120环境64轮冻结+4轮joint通过（actor更新60次）；加速课程81轮通过全部扰动档、
容差收紧及当前训练权重视频；最后版本2轮检查区分central/actor辅助价值损失。
对应目录为mustard_sapg_lownoise_smoke68_v1、mustard_sapg_curriculum_smoke81_v1、
mustard_sapg_finalcode_smoke2_v1；这些烟雾测试权重不用于正式训练。

2026-10-29复查：critic683轮已完整结束（89434752 frames，10928次critic更新），
actor及全部缓冲区无变化，joint未启动。最后100轮rollout自身GAE目标的价值解释方差均值
约0.961；这不是独立Monte Carlo验证，不能单凭它断言长期价值已经准确。
正式冻结随机策略的leader回合成功率最后100轮约0.352，大多数失败为超时。
`all_goals_hit=0`存在口径错误：单最终目标回合沿用了max_consecutive_successes=2。
train_critic.py及final-goal-only评估现已改成1；旧结果保留，不回写历史数据。
视频新增eval/completed_episodes、eval/successful_episodes、eval/success_rate，
旧episode_final/*时间平均值不能读作回合成功率。

同100初态新对照（diagnostics/deterministic_vs_stochastic100.json）：确定性持续抓起96、
宽松到达82；原生logstd=-2随机策略54、38；示范99、97。
采样标准差乘0.25的另一次对照（diagnostics/noise_quarter100.json）：确定性96、84，
小噪声86、72，示范99、97。相同权重的确定性对照有少量物理回放波动。
两次诊断均不自动reset、在各自示范有效长度内统计，不能替代正式回合成功率。
当前证据支持先降低动作探索噪声；若改变sigma，需重新短程校准critic后再解冻actor，
最初仍不加初态扰动。该建议尚未启动新训练，现有校准权重仍对应原始sigma。

2026-10-28：基础4000轮已完成。保留基础权重，当前优先用于后续验证的权重为
`/home/szc/work/spider/outputs/mustard_bc100_carry_lr1e5_pilot200_v1/actor_init.pt`。
它由基础4000轮权重以Adam学习率1e-5、连续携带LSTM状态再微调200轮，总计4200轮；
没有训练critic。不能把它标为仅训练4000轮。

同一批100条、相同缓存初态、各自有效轨迹时长内的闭环对照：基础模型持续抓起84/100、
初始宽松目标到达55/100；上述模型96/100、87/100；示范动作回放99/100、97/100。
初始actor观测最大误差2.861e-6。判据不是严格1cm，且这些结果不能代表原版10秒任务成功率。
证据：`outputs/mustard_bc100_carry_lr1e5_pilot200_v1/comparison100.json`。
学习率1e-4的carry试验退化，不作为后续初始化。

用户确认的阶段顺序：

1. BC使用每条示范对应的初态，不额外增加初态扰动；100条之间原有差异仍保留。
2. 冻结actor进行critic校准时，继续使用该示范配套的机器人关节位置/速度、
   上一关节目标、物体初始位姿及最终目标。冻结actor输入归一化统计与策略分布参数，
   防止只冻结网络权重却改变策略；critic按原生target-reaching奖励学习回报。
3. 解冻actor后先维持相同reset，再按闭环任务成功率逐步增加初态扰动；
   先做小幅、分项扰动，成功率明显下降时保持或回退。不能直接恢复全量随机reset。
   物体位置/朝向扰动和目标坐标变换必须明确配对；机器人关节扰动需单独验证接触与关节限位。
   延迟、观测噪声及外力随机化另行验证，不与首次初态扰动同时全量打开。

缓存reset是actor可见状态对齐，物体速度设为0，并非完整PhysX接触状态快照。
2026-10-28用户明确将当前任务时限改为16秒（960个控制步），critic和配套评估同步采用16秒；
原版10秒仅作为历史参考，不能混用两者成功率。
新的critic入口为train_critic.py，启动脚本为
`/home/szc/work/spider/examples/arm_hand/run_mustard_critic100_683.sh`。
正式配置6个block×1364=8184环境，683轮，新central critic Adam学习率5e-5，
每轮16步rollout、4轮critic小批量更新。保留原生奖励、GAE和SAPG系数条件；
仅使用当前冻结策略的on-policy数据训练critic，无actor反向传播或经验重标记。
actor全部权重和缓冲区逐项核对；critic独立归一化可更新，另存central Adam状态。
目标固定为示范最终位姿，到达后结束该目标回合；校准期间保持原生初始成功容差，
关闭容差收紧。位置额外扰动为0，延迟/观测噪声/外力关闭，保持示范采集条件。
第1轮和每200轮单独录制6环境、确定性冻结策略视频，奖励标量同步W&B；
训练rollout仍使用原生随机策略分布，视频不作为随机策略总体成功率。
输出目录`/home/szc/work/spider/outputs/mustard_critic_demo16s_683_v1`。
已启动正式进程PID1599346并通过首轮冻结检查；正式规模初态观测最大误差1.371e-5，
6个block在初次reset时均覆盖全部100条轨迹。W&B运行ID为mustard_critic_demo16s_683_v1。
此前120环境、16秒、64轮短程验证已完成，actor更新0次，critic更新1024次。
683轮后保存critic_calibrated.pth并停止，不自动解冻或加扰动。
旧train_staged.py含waypoint RL和随机reset切换逻辑，不能用于这里的阶段交接。

以下保留基础4000轮训练记录：

用户已明确接受最长约15.8秒示范，取消10秒内完成reaching的筛选。
当前输出：`/home/szc/work/spider/outputs/mustard_bc_native100_4000_v1`。
启动脚本：`/home/szc/work/spider/examples/arm_hand/run_mustard_bc100_4000.sh`。
基础训练已完成。10条旧版已按用户要求停止，文件保留。

使用`outputs/mustard_continuous_teacher100_v2/native_bc_dataset.npz`完整100条。
有效长度按轨迹独立保留，10.4～15.8秒，最长947步，总计75108个监督帧/epoch。
补齐帧不参与输入统计或BC损失。没有沿用10条版本的权重或优化器，从头训练4000轮。

资格模式full-path-grasp：100条均在各自有效长度内通过持续持握几何代理判据；
再次对缓存重建检查，最短连续持握153步。完整回放96/100达到初始宽松reaching，
另4条也按用户要求保留作动作模仿数据，并非宣称100条都完成严格目标。

原版140维观测、29维动作、LSTM/MLP/六系数条件不变。Adam1e-4，seq16、burn-in16，
sigma=-2，输入统计固定。无central critic训练，无PPO或任务奖励优化。
BC读取缓存，不创建8184物理环境；后续SAPG布局元数据为6×1364。

每100轮保存actor_init.pt、bc_optimizer.pt；第1轮、每100轮及最终轮触发16秒本地视频。
已有评估运行时跳过中间触发，最终轮保证另录最终模型。
视频诊断专用时限16秒，目标池100条、最终目标；正式SAPG环境仍为原版10秒，未修改。

W&B：https://wandb.ai/shezhangcong-johns-hopkins-university/simtoolreal-warmstart/runs/fc53m5cy
仅上传BC/eval标量。日志为输出目录同名.launch.log，bc_progress.jsonl、video_evaluator.log。
completed.json确认4000轮完成；不自动进入critic校准和联合SAPG。
