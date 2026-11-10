# 本地 SimToolReal：mustard 热启动实验

环境代码已复制到 `/home/szc/work/spider/third_party/simtoolreal`，不再从外部
`simtoolreal_origin` 导入任务或 SAPG。注意安装的 conda runtime 名字仍含
`simtoolreal_origin_isaacsim`，它是运行依赖，不是外部任务代码。

## 基础环境与实验适配

`environment.py` 实例化复制后的原版 `SimToolRealEnv`，没有覆盖其动作、reset、
观测、奖励、done 或 curriculum hook。保留随机初始状态、随机目标、多次成功后
继续采样目标、50 次成功上限、平均成功数门槛 3 及原版控制合同。

只为右手资产适配 RobotSpec、palm/tip、固定 link 与自碰撞名称，并指定 mustard
URDF、标定尺度和 reward corners。使用原版 robot base z=0、原桌面/地面、原
contact offset=.002；torque_scale=0 对齐指定的历史 run。手部 PD 使用原 Sharpa
参考值按关节角色映射；effort 使用原 Sharpa URDF 的有限力矩范围，而不是源
MuJoCo 的近似无限值。

本目录 `assets` 是单独的实际复制。资产中的 `robot_spec.json` 可能保留导出时
的 provenance 路径；构建器明确用本目录本地 URDF，不把该 provenance 当运行路径。
`experiment.yaml` 明确列出热启动、并行规模和记录参数。

## 当前入口

本次使用独立 `bc` 入口：纯动作模仿4000 epoch，critic不训练，详见 [BC4000.md](BC4000.md)。
示范路径与连续时序的最新验证见 [RETIMING.md](RETIMING.md)：完整路径100/100可抬升，
但前10秒内为75/100，仍需筛选或原生优化才能全部进入正式BC。
下面的 `warmup` / `train` 属于历史奖励训练方案。

## 已实现入口

本地原版控制环境短运行检查：

```bash
bash /home/szc/work/spider/examples/arm_hand/run_local_simtoolreal.sh \
  --headless --num-envs 12 --steps 64
```

默认输出 `/home/szc/work/spider/outputs/mustard_local_native_smoke_v1`。
`completed.json` 记录实际模块导入路径、观测/动作尺寸和有限力矩范围。
此入口运行 smoke，不是正式 SAPG，不会冒充已完成原版动作 actor 初始化。

## 历史方案：途径点奖励训练（不用于当前纯BC请求）

运行 `prepare_waypoints.py` 从二十条优化轨迹提取物体 pose，修正物体根坐标并上移
z=.53 到原版坐标系。目前每条得到两个有效目标，共四十个途径点；数据保存在
`waypoints.json`。提取间距按最大角点位移计算，默认 .13 m。

`waypoint_goals.py` 只替换实验进程的目标采样入口，不改 vendor 文件。每次原版
reset 随机抽取一串目标，成功后推进下一点；如果点已在当前物体成功容差+.01m内，
直接跳过，避免静止刷奖励。串结束后继续原版随机目标，不重复奖励最后一个点、
不提前终止回合，因此保留达到三个以上目标后收紧容差的原机制。机器人和物体
依然由原版随机 reset 初始化，不恢复示范中的机器人状态。

`train_staged.py` 实现：

1. `waypoint_actor`：SAPG 用途径点目标训练 actor 和辅助 critic。不是 BC，没有
   关节动作标签或轨迹跟踪损失。默认10000轮是首轮预算，不保证已经学会。
2. `critic_only`：关闭途径点采样，硬重置到原版随机目标任务；保留 actor，重置
   critic 权重和 optimizer，冻结 actor 及其输入归一化，log_std=-2，critic LR=5e-5。
3. `joint_sapg`：解冻 actor 和输入归一化；actor LR=1e-5并按本阶段预算线性衰减，
   clip=.05，actor/critic mini-epochs=4/4，继续随机目标 reaching。

启动入口：

```bash
bash /home/szc/work/spider/examples/arm_hand/run_local_simtoolreal.sh train \
  --headless --num-envs 120 --waypoint-epochs 10000 --critic-epochs 683 \
  --joint-epochs 200000 \
  --output /home/szc/work/spider/outputs/mustard_waypoint_sapg_v1 \
  --capture-video --video-interval 6000 --video-length 300 \
  --wandb-project simtoolreal-warmstart
```

进度保存于 `sapg_progress.jsonl`；阶段权重为 `waypoint_actor.pth`、
`critic_calibrated.pth` 和 `sapg_latest.pth`。`critic_warmup_check.json`
验证冻结期 actor 权重未变、optimizer没有step、critic确实变化。
示范文件改变时需重新生成 waypoint pool；所有阶段使用同一原版 action 合同。
W&B沿用已审查的metrics-only旁路，只上传原版数值指标和阶段进度，不上传配置、
轨迹、权重或视频；视频继续保存在新run的`videos`目录。

实际24环境/6轮短测试见
`/home/szc/work/spider/outputs/mustard_waypoint_staged_smoke_v2`：三阶段均完成；
目标推进/结束后随机采样/跳过已满足点检查通过；冻结期actor无变化，critic有变化。
这些检查验证控制流程，不代表actor已学会抓取或reaching。

## 旧权重约束

现有 `mustard_actor_isaac_single64_v1/actor_best.pt` 是 residual 合同，不能直接作为
native SimToolReal actor。当前新方案从 native action actor 开始，通过示范物体
途径点上的 SAPG 学会 reaching；不再需要把旧残差策略或源关节动作转换为标签。

旧 `mustard_sapg_targetpose_v1` 的进程不受这份复制影响，也不会热切换环境；后续
使用新输出目录，避免把不同 MDP 的曲线混在一起。两组对照应共同使用本地这份
环境及完全相同的资产/动作/目标分布，仅改变明确声明的热启动方案。
