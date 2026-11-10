> 历史奖励训练方案，已被当前纯BC请求替代。请使用 [BC4000.md](BC4000.md) 和 `bc` 入口。

# 100轨迹、6×1364环境：仅进行actor途径点预热

此入口与旧的 `train` 三阶段入口分开，使用 `warmup`。它只运行当前阶段，到指定
epoch后保存权重并退出，不切换到随机目标，不进入后续critic校准或联合后训练。
SAPG为更新actor提供优势估计，所以辅助critic仍与actor一起训练。

## 数据与物体reset

数据为 `outputs/mustard_spatial_valid100/accepted.json` 中完成验证的100条轨迹。
`waypoints100.json` 已冻结100条的初始物体位姿、各2个途径点、输入NPZ SHA256及
accepted清单SHA256。它们是同一mustard源任务的空间变体；运行时不再读取活动数据目录。

- 初始物体pose取优化结果的 `initial_state`，不是第一个动作执行后的qpos。
- 所有物体pose先加旋转后的资产根坐标重定位offset，再整体z+.53到原版桌面坐标。
- 原始物体root的native初始z约.626756m；实际桌高仍保留±.01m随机化，物体和整串目标
  共用同一桌高偏移，不会让物体穿桌或让目标与起点错位。
- 每次reset随机选择一条轨迹并绑定它的起点和目标；xy默认±.005m、world yaw默认±2°。
  yaw围绕物体初始root旋转整条路径。没有独立扰动目标，更没有随机另选一条目标串。
- 物体初速度清零；机器人仍使用原版随机reset，不恢复示范关节角或速度。
- 原版动作、PD/有限effort、观测140维、关键点奖励、延迟/观测噪声/随机力保持本地环境设置。

## 目标、回合与课程的必要调整

只改变实验进程中的物体reset、目标采样以及链完成终止条件，不改vendor源文件。
一条链的两个目标完成后结束该回合，下一回合重新抽轨迹，不fallback到随机目标。
目标已经在成功容差+.01m内时跳过，不额外增加successes或到达奖励。

因为每条只有2个目标：

- `max_consecutive_successes=2`，原版`all_goals_hit`此时表示两个目标都按原成功计数达成。
- `tolerance_curriculum_success_threshold=1.0`，替代原版不可达的3；即平均每回合至少1次成功。
- 容差仍从.075降至.01、keypoint_scale=1.5、间隔3000 policy steps、缩放.9不变。
- 仍需累计10个near-goal steps，沿用原版非连续计数；不是“3cm+20°”独立判据。
- 跳过最后一个点的链可以结束，但不会凭空计作2次成功；另外记录`chains_ended`与
  `chains_successful`，避免把跳过的链当作完整成功。

## SAPG参数核对

并行环境8184，6个block，每block1364；探索系数ID为50/40/30/20/10/0。
每轮真实采样8184×16=130944，名义minibatch=8184×4=32736，
每minibatch含2046条长度16的序列，整除检查通过。

预热训练保留原版SAPG参数：actor LR=1e-4且KL-adaptive(.016)，critic LR=1e-4，
actor clip=.1，actor/critic mini_epochs=2/2，AMP开启，gamma=.99、GAE=.95，
horizon=seq_length=16，梯度裁剪1，reward scale=.01，entropy coefficient=0，
六组条件sigma、learned embedding、lf经验共享与原版normalization/value-head设置。
没有在这一阶段套用后训练的actor LR=1e-5、clip=.05、冻结actor或log_std=-2。
Actor随机初始化为原版动作合同，不加载旧residual checkpoint。

默认10000轮预算=1,309,440,000条真实transition。达到预算不保证学会；根据
`all_goals_hit`、每组successes、容差和独立视频判断学习效果。

## 性能与检查

目标和reset均在GPU批量计算，无逐环境Python循环。主训练不启用camera和视频wrapper。
每1000轮保存一个只读policy快照，启动独立6环境评估进程录300帧；该进程短时仍会
使用同一GPU，但不会渲染8184个训练环境。若前一个评估尚未退出，不重复启动。
评估视频只保存在本地；W&B继续仅同步数值指标。

实测检查目录：`outputs/mustard_actor100_8184_check_v1`。
8184环境跑完4轮，共523776 transitions；更新阶段9.99s，约52,431 transitions/s。
这是短时数据，包含旧实验仍在同卡运行的条件，不包含USD/PhysX初始化耗时，不能承诺
长期固定吞吐。所有6组都采样到了全部100条轨迹。

物体pose绑定、扰动范围、目标共享变换、首次成功推进且不重置物体、第二次成功终止、
下一回合重新reset、跳过无额外成功计数等检查全部通过。模型及训练loss有限。
保留AMP时初期有2次梯度溢出被GradScaler跳过，32次尝试中Adam实际更新30次，
scale自动由65536回退到16384；原版梯度余弦诊断出现NaN，未造成模型/损失NaN。
该诊断与任务成功率无关，未用改动训练参数的方式掩盖。

## 启动

先停止不再需要的旧残差训练，避免同卡抢资源。不要同时运行旧`train`和新`warmup`。

```bash
nohup bash /home/szc/work/spider/examples/arm_hand/run_local_simtoolreal.sh warmup \
  --headless --num-envs 8184 --block-size 1364 \
  --waypoints /home/szc/work/spider/experiments/mustard_simtoolreal/waypoints100.json \
  --epochs 10000 --reset-xy-noise-m 0.005 --reset-yaw-noise-deg 2 \
  --checkpoint-every 100 --video-every 1000 --video-length 300 \
  --wandb-project simtoolreal-warmstart \
  --output /home/szc/work/spider/outputs/mustard_actor_warmup100_8184_v1 \
  > /home/szc/work/spider/outputs/mustard_actor_warmup100_8184_v1.launch.log 2>&1 &
```

最终`actor_init.pt`是可用于后续随机目标任务的native动作actor初始化；
`actor_warmup.pth`是完整训练权重，`sapg_latest.pth`每100轮覆盖保存。
阶段只会是`actor_waypoint_warmup`，不会执行后续阶段。
W&B链接会写入新输出目录`wandb_run.json`。
