# SPIDER → SimToolReal：KUKA + Sharpa 热启动实验计划

日期：2026-10-26。状态：方案，尚未实施转换或训练。用户指定数据源为 `/home/szc/work/spider/outputs/kuka_sharpa_right_batch20_gravcomp_plateau200/`；部分样本仍在生成，本计划不修改、重启或占用该生成进程。

## Isaac Sim 执行修订（2026-10-27，后续用户指令）

用户已明确要求直接在 Isaac Sim 中训练。当前有效训练应使用 SimToolReal 的 Isaac Lab / PhysX 环境生成真实 rollout 观测，再进行 actor 监督更新；离线 MuJoCo BC 只作为初始化和历史对照。右手机器人、物体惯性/角点、物体初始速度、接触裕量与控制输出合同需要一起记录。采样在 CUDA，梯度可在 CPU 更新以避开共享 GPU 竞争，训练数据仍来自实际 PhysX 环境。

actor 输出当前统一为绝对关节目标（可由残差解码得到）；原生 SAPG 增量/EMA 映射将在后续迁移时明确处理，不能混称。当前阶段不更新 critic，不执行 SAPG policy gradient。所有变体均来自一个源 episode，验证仅覆盖留出的空间变体。

## 当前执行范围修订（2026-10-27）

按用户明确的近期目标，优先训练 actor 模仿现有 20 条轨迹。100 Hz→60 Hz 先通过关节/控制线性插值与物体姿态 SLERP 处理；物理参数一致性单独审计。先前默认动作处理回放失败，仅说明该执行接口下无法直接重现目标，不能据此阻止离线 actor 模仿训练或认定轨迹必须重优化。

第一轮采用实际 SAPG 的 LSTM/MLP/探索条件结构，监督目标为归一化绝对关节控制目标。其输出合同与原 SAPG 手臂增量动作不同，checkpoint 必须显式标明，未来接入时需要适配输出头或统一控制合同。仅进行 BC，不训练 critic。离线验证、CPU 源环境闭环验证和未来 PhysX/SAPG 验证分别报告。

## 1. 目标与结论边界

验证：优化示范 BC 初始化 actor → 固定 actor 在目标环境采样 → 仅训练 critic → 保守 SAPG 联合训练，是否降低 SimToolReal 达到相同成功率所需的环境交互数和总耗时。

这是 **demo-BC warm start**，借鉴 ADEPT 的稳定迁移配方，不等于复现 ADEPT。ADEPT 的源策略由大规模 RL 获得，BC 有可查询的在线 teacher；这里是少量优化轨迹，没有可在任意偏离状态给出正确动作的 teacher。不能假设相同泛化能力，也不能直接照搬 DAgger 或 40,000 次蒸馏更新。

第一轮研究抓取、抬升、物体位姿到达；后续再测试程序化工具、连续目标及工具使用。若数据只覆盖抬升，不将其称为完整工具操作示范。

## 2. 已核实的本地事实

- 指定批次 `settings.json`：20 条计划样本、原生右 Sharpa Wave、机器人重力补偿、每条最多 200 次优化、512 个候选、plateau patience 30、无额外 hold；允许 IK 几何检查失败后继续优化。
- 检查时 `status.json` 有 2 条 complete、1 条 running，其余不能据此认定失败或缺失。此为读取快照，不是实时监控结果。
- 第一条 master_chef_can：抬升 0.332 m、末端位置误差 0.0129 m、原始姿态误差 62.7°；最终保存状态审计的手臂桌面最大穿透约 0.927 mm，低于其 1 mm 门槛。
- 第二条 cracker_box：抬升 0.336 m、末端位置误差 0.00376 m、姿态误差 15.1°；最终保存状态审计未报告手臂桌面穿透。
- 两条最终 arm_table_clearance_pass 均为 true；第一条的 IK 阶段 clearance=false 不能替代最终轨迹验收。现有审计是 CPU 对保存状态做 FK/接触检查，不是 Isaac Sim 中的动作回放，也不是完整碰撞保证。
- 示例 NPZ：`ctrl=(174,29)`、`qpos=(174,36)`、`qvel=(174,35)`、`initial_state=(72,)`；控制周期 0.01 s、物理周期 0.002 s。`qpos[t]` 是执行 `ctrl[t]` **之后**的状态。
- SimToolReal Isaac Sim 默认机器人是 iiwa14 + **左** Sharpa 22 DoF 手，总动作 29 维；默认控制 60 Hz、物理 120 Hz。维数相同不代表关节和动作语义一致。
- Isaac Sim PPO 配置是 LSTM actor + 独立 central-value critic；actor 的 network.separate=False 并不意味着启用 central_value_config 后仍应通过共享 actor trunk 训练真实 critic。
- 当前 actor 是关节、掌心、物体姿态、相对关键点和尺寸输入，并非 ADEPT 的物体点云/视觉学生。首轮不增加点云网络。

## SAPG 修订约束（按用户要求）

本地 SAPG 是 vendored PPO 实现上的扩展，代码中 `ppo=True` 不代表使用普通 PPO 实验。实际 YAML 为 `use_others_experience=lf`、`expl_type=mixed_expl_learn_param`、`fixed_sigma=coef_cond`、32 维探索系数 embedding、`expl_reward_type=entropy`、`expl_reward_coef_scale=0.002`、`expl_coef_block_size=1364`。配置文件头部的 4096 是旧说明，以实际配置及运行时解析结果为准。

- BC 从一开始构建 SAPG 兼容模型：actor observation 和 critic state 都会由 runner 追加探索 embedding；不能漏掉，也不能在环境和 runner 重复追加。同一示范可在多个探索条件下监督相同动作均值，但这不增加独立轨迹数。
- 本地实现主要是共享、由探索系数条件化的网络与分组 sigma，不应误写成必须训练若干完全独立 actor。逐组验证条件输入、sigma 选择和 LSTM 边界。
- WU 保留 SAPG 的组别和采样逻辑，冻结整个条件 actor 及所有组的 sigma。critic 的训练目标沿用 SAPG 对任务奖励、探索奖励和条件重标记的实现，而非只拿外部任务奖励重新算一套不兼容目标；正式成功率仍按任务指标评价。需专门验证共享样本的条件、bootstrap、行为概率和 return 一致性。
- 低噪声初始化是 warm-start 设置，不是原 SAPG 默认值。所有 BC 消融组使用相同的逐组 sigma 初始化；原始从零 SAPG 保留默认设置。若需要隔离 sigma 初始化贡献，另加从零 SAPG + 同样 sigma 的控制组。
- 正式比较始终为 SAPG 对 SAPG。保守更新只改变明确登记的 actor LR / clip，不关掉经验共享或探索条件。正式预算建议以在线 transition 为准；20 iterations 会随并行环境数改变成本。

## 3. 数据量建议

这里“条”指独立完整 episode，不是帧、裁剪窗口、重复回放或随机种子。

- **当前 20 条：适合证明管线可运行、BC 能否闭环、WU 是否防止早期退化。** 如果每个物体仅 1 条，无法证明同物体新初态泛化，更不足以代表通用 SimToolReal 能力。
- 单任务或 1–3 类物体：先争取 **50–100 条通过目标环境验收的独立示范**。
- 5–10 类物体的正式实验：初始预算 **200–500 条**，覆盖抓取方向、初始姿态、目标方向、接触切换和接近失败的恢复。
- 若目标是跨程序化工具形状的广泛泛化，再评估 **1,000 条以上**是否有收益；不预先承诺某个数量足够。

以上是工程起始预算，不是论文给出的样本复杂度。以 5/10/20 条的嵌套子集先做数据规模曲线，有收益再增加 50/100/200 条。验证集固定，不随训练规模扩大而变化。按物体和原始 episode 分组划分；同一 episode 的片段、增强、不同优化版本必须同组。

假设 20 条平均 2–4 秒，在 60 Hz 下只有约 2,400–4,800 个 transition，且高度相关。重复 40k 次更新可能严重过拟合。实际帧数按完成文件统计，不将此估算当成盘点结果。

## 4. 阶段 A：冻结数据版本、对齐机器人与任务

### A1. 数据清单与分级

生成流程继续写原目录。只在 episode 完成、NPZ 可完整读取、result/audit 文件齐全且内容稳定后，复制到独立 versioned 数据目录，保存 SHA256、来源、场景、关节名、控制周期、优化参数、物体 ID、审计结果。一个实验绑定一个 manifest；新完成样本进入下一版本，不热插入正在比较的实验。

分三级：

1. A：目标环境动作回放成功，达到位姿目标并保持；可用于完整任务 BC。
2. B：稳定抓取抬升，但姿态/保持不足；可用于明确标记的前置技能 BC，不能标为完整任务成功。
3. C：严重穿透、非有限数、关节越界、动作不可实现或动态回放失败；隔离修复。

对圆柱等对称物体，先定义任务允许的对称群，再计算 symmetry-aware 姿态误差；不要仅凭“像罐子”就忽略 62.7°。报告原始误差与对称感知误差。抬升数据没有 hold，后续在目标仿真追加 0.5–1 s 保持验收并记录新版本，不伪造原始轨迹。

### A2. 机器人路线

首选：给 SimToolReal 增加显式 `kuka_sharpa_right` RobotSpec/任务配置，使用与这批数据一致的右手资产、法兰安装变换、关节限位/顺序和力矩限制；所有从零/热启动组都运行这个目标机器人。

如最终研究必须使用原有左手，则先把示范重新 retarget/优化到该左手，不能把 right 关节重命名为 left、简单符号翻转或直接载入旧 actor 权重。

核对 arm/hand PD、armature、摩擦、重力补偿、台面尺寸/高度、基座位置、碰撞过滤、掌心/指尖定义。MuJoCo gravcomp 与 PhysX disable_gravity 的设置目的相近，但仍需动态校验。当前轨迹跟踪仅监督四个指尖，而目标环境会观测五指，需补齐真实 FK 的小指观测和审计。

先选 cracker_box 等几何清晰样本在单环境完成对齐，再扩大批量；不要边调机器人边比较训练算法。

### A3. 固定任务合同

第一轮采用“桌面初始化 → 抓取 → 抬升 → 固定目标位姿”的 episode。明确目标来源、初态范围、是否已抓住、抬升高度、位姿误差、保持时间和最大时长。从已抓住状态启动的片段另做子任务，不能代表从桌面开始的成功率。

初始目标可以来自示范终态，但需固定并审计其与任务目标一致性。若为匹配数据新增抬升子任务，所有组共用该任务，报告为子任务结果。连续轨迹跟踪是下一阶段，不混用 YimuTraj 注册项当作 Sharpa 配置。

## 5. 阶段 B：物体进入 SimToolReal 的处理

1. **提取完整资产与变换。** 从每条 `scene.xml` 的 right_object 读取 visual/collision mesh、各 geom 的局部 pose、scale、body pose、质量惯量；解析绝对路径并打包成可迁移资产。不要只导出可视 mesh。
2. **统一单位与坐标。** 米、弧度；显式保存 dataset object frame → asset root → world 的刚体变换。任何重心化/轴重排同步变换轨迹、goal、关键点和惯性系。分别核对 MuJoCo/Isaac Sim 使用的 wxyz 及旧 Gym 工具可能使用的 xyzw。
3. **刚体范围。** 首轮只保留单刚体物体。铰接物体需锁定并说明构型，或另建 articulation 支持；柔性物体单独处理。罐、盒等也需判断是否以刚体近似。
4. **清理几何与碰撞。** 检查退化面、法向、尺度、非流形、薄结构。复用源凸分解前，验证每块在 PhysX 中可正确 cooking；保留凹槽/把手等接触相关结构，不能用单凸包封死。必要时重新凸分解；SDF 作为单独兼容性与性能实验。缩减凸块数/面数需同时检查抓取几何和吞吐。
5. **质量、质心与惯量。** 优先继承优化场景实际参数，写出标准 URDF mass/inertia 和正确 inertial origin；不要只写 density（本地生成器明确说明 Isaac Sim 可能回退到 1 kg）。惯性矩阵需正定且物理合理。非均匀缩放需重算惯量。
6. **URDF → USD。** 单刚体对象使用统一 root link，例如现有 `object_root`；保留多碰撞形状、单个动态刚体、非固定基座。使用现有 `_convert_urdf_to_usd` 和 mesh 名称清理机制，检查转换后碰撞、质量、重心及刚体数量。检查默认 cylinder→capsule 转换是否改变目标几何，必要时对数据物体关闭。
7. **尺寸与关键点。** `assets.object_scale` 是策略归一化尺寸，不是再次缩放 mesh 的系数；本地约定为 grasp-bbox 实际尺寸 / 0.04。完整 bbox、抓取区域 bbox、root offset 分开存储。原环境使用四个角点及可选固定尺寸 reward keypoints，不应将它们误认为点云；首轮沿用相同规则，各组一致。
8. **批量加载。** 现有分支支持单个 `object_urdf + object_scale`，或程序化物体池；新增 dataset manifest 分支，把每个 object 的 URDF/USD、scale、物理参数、初态池和 goal 池绑定。核对 asset_index 与 env_id，不允许视觉是 A 而 scale/轨迹属于 B。保持各 USD 相同 root 层级以兼容 RigidObject view。
9. **缓存与验证。** 按源 mesh/URDF、转换参数与软件版本 hash 缓存 USD，避免每次训练重复 cooking。单物体依次做静置、跌落、接触、抓取和完整动作回放，然后测多环境 FPS、显存与加载耗时。

首轮成功标准建议：无明显初始穿透、接触稳定、回放中不依赖强制写物体位姿、达到预先定义的任务成功阈值。可先用位置 2–3 cm、方向 15–20°、保持 0.5 s 做工程验收；正式评价同时保留框架原生 success，所有阈值必须在看比较结果前固定。

### B 补充：抓取包围盒与角点标定是接入必做项

每个唯一物体资产标定一次抓取包围盒的尺寸、中心和朝向，再通过每帧物体 pose 自动生成角点，不需要逐帧手标。框架实际使用四个有固定顺序的虚拟角点 `(1,1,1),(1,1,-1),(-1,-1,1),(-1,-1,-1)`；它们不是必须落在 mesh 表面的真实特征点。默认 keypoint_scale=1.5，生成的角点甚至会位于抓取盒外，属于策略/误差度量约定。

当前实现只接受以 object root 为中心、沿 root 轴的尺寸，公式为 `corners * object_scale * 0.04 * keypoint_scale / 2`；没有独立 grasp-bbox 中心和旋转输入。源资产若不满足此假设，首选将 root 重新定义为抓取盒坐标系，并同步变换所有 visual/collision/inertial 数据、物体初态、整段轨迹和 goal；另一方案是显式扩展局部关键点变换，但需要统一修改 obs/reward/reset/eval，不能只写元数据而期待现有代码读取。

盒子/罐子可用整体 bbox 作为候选；工具应优先采用可抓取部分（如手柄）的 bbox，不能一律用整物体包围盒。自动估计后逐物体可视化确认轴方向、中心、尺寸和角点编号，保持同一物体跨 episode 完全一致。局部盒坐标 B 到源物体坐标 O 的变换若为 T_OB，则世界新 root 位姿为 T_WB=T_WO*T_OB，mesh 坐标变换为 v_B=inverse(T_OB)*v_O，goal 同理。

每个物体保存 source-to-grasp-frame 变换、grasp_bbox_dims_m、policy_object_scale=dims/0.04、角点顺序及 schema 版本。验收包含 mesh+抓取盒+四角点叠加、物体与 goal 同 pose 时四点误差为零、已知平移/旋转下对应关系正确，以及源/目标引擎在相同物理位姿的世界角点一致。

特别区分：actor 观测采用按物体尺寸生成的角点（还可能施加 scale noise）；默认 `fixed_size_keypoint_reward=true` 时，奖励/终止采用固定尺寸角点。两套定义有意不同，导入时必须分别保留、验证，不能把观测角点直接替换奖励角点。SAPG 与这套物体表示是独立层次，使用 SAPG 不会免除标定。

## 6. 阶段 C：从优化控制转为 actor 监督

### C1. 时间对齐

源控制 100 Hz → 目标 60 Hz。关节目标按时间插值，姿态用连续符号处理后的 SLERP，速度从一致时间基准重建；不要简单每隔几帧抽取。

源保存的是 post-step qpos：第 t 个动作对应的 pre-state 为 initial_state（t=0）或 qpos[t-1]。禁止把 qpos[t] 和 ctrl[t] 当作同一时刻的输入输出标签。使用模型解码 initial_state，不把整个 72 维数组当 qpos。

### C2. 动作语义

SimToolReal arm 是相对“上一步控制目标”的速度增量，hand 是限位范围内绝对归一化目标；二者均有 EMA。监督对象必须是送入 `apply_action_pipeline` 的 canonical policy action，不能直接用 ctrl、实际 qpos 或相邻 qpos 差值。

在关闭动作延迟、未触碰 clamp 的条件下，设 p 为上一步 target，u 为期望本步平滑后 target，α 为 EMA 系数，v 为 dof_speed_scale：

- arm：a = (u - p) / (α_arm · v · Δt)。
- hand：先求 x = [u - (1-α_hand)p] / α_hand，再 a = 2(x-lower)/(upper-lower)-1。

默认 α_arm=0.1、v=1.5、Δt=1/60，因此 arm target 每个策略步最大变化约 **0.0025 rad**，稳态约 0.15 rad/s（饱和附近更受限制）。这可能比源轨迹需求慢得多，是首要检查项。

统计每个关节反算动作越界比例和 target 重建误差。发生越界时，优先时间拉长或按目标控制器重新优化轨迹，再次验证接触任务；不能直接 clip 标签后声称轨迹等价。若改 action smoothing 或 speed_scale，所有对照组使用同样设置，并标记这是修改后的 SimToolReal。

### C3. 用目标仿真生成最终训练数据

先用独立 target replay 做诊断，再使用真实 policy action pipeline 回放，关闭延迟和随机扰动来验证确定性接口。不要使用 `_replay_target_lab_order` 绕过真实 pipeline 的结果作为可执行性证明。

最终记录目标仿真真实产生的 `obs_policy, state_critic, action_policy, reward, terminated, truncated, goal, object_id, episode_id, prev_targets`，以及序列边界、控制配置 hash。obs/state 从同一个环境 builder 生成，确保坐标、关键点、顺序、裁剪和归一化一致。若开环回放偏离明显，需要目标环境重优化/反馈跟踪恢复，不能把源参考状态混入真实状态。

Critic 的 progress、successes、closest distances 等历史量必须沿回放真实更新。初期 BC 可不用 reward，但后续价值训练必须使用目标环境奖励和 reward scale，而非负 SPIDER cost。

## 7. 阶段 D：BC actor 预训练

- 正式实验从第一轮就使用 `Isaacsimenvs-SimToolReal-Direct-v0` 的右手扩展任务与 `rl_games_sapg_cfg_entry_point`。从零基线和所有热启动组均保留 SAPG 的探索条件、分组和 leader–follower 经验共享，不用普通 PPO 替代。
- 保持本地 LSTM actor 结构。序列长度从 16 起步，比较 32；按 episode 训练并正确 mask padding/reset hidden state。中间窗口使用前缀 burn-in 或一致的隐藏状态重建，禁止跨 episode 拼接。
- actor 只接收 obs_list；privileged critic state 不得泄漏给 actor。观测 normalizer 只在训练数据拟合并随 checkpoint 保存；验证/test 不更新。
- 使用归一化动作均值 MSE/Huber，arm 与 hand 分组平均再加权，避免 22 个手关节天然压过 7 个臂关节；按动作阶段平衡采样。没有 teacher 方差，不照搬论文 Mahalanobis teacher-distribution loss。
- 起始 BC LR 1e-4，可试 3e-4；按验证集闭环表现早停，训练上限先设 200 epochs。网络较大时少量数据过拟合很可能，先保留基线结构，缩小网络只能作为双方一致的独立实验。
- 初始 log_std 可固定为 -2（σ≈0.135），但这是从论文借来的试验起点；需验证这种噪声在当前动作映射下是否破坏抓取。BC 以均值为主要监督，完整冻结窗口内 std 不更新。
- 同时评估动作误差、目标环境成功率、掉落、姿态误差与扰动恢复。BC 的离线 loss 下降而闭环失败时，先修数据/控制接口或补偏离状态示范，不进入大规模 WU。

产物：actor checkpoint、normalizer、joint/obs/action schema、split manifest、deterministic 与 stochastic 的闭环报告。

## 8. 阶段 E：冻结 actor，仅 warm up critic

- 新初始化独立 central-value critic；采样来自当前 BC actor 在目标环境的真实 rollout，包含失败、掉落和超时，不只挑成功片段。
- 冻结 actor 的 LSTM/MLP/mu/log_std、输入 normalizer 及所有影响动作的可变 buffer；actor optimizer 完全不 step，不仅仅把学习率置零。禁用 actor bounds/entropy 等间接更新。critic 使用自己的 normalizer 和优化器。
- 固定动作分布和初态/goal/DR 设置，先用小扰动分布；和后续 SAPG 起始分布一致。用固定 probe obs + hidden state 检查前后 mu/std 和 actor 参数 hash 不变；hidden state 在 episode 内仍正常演化。
- 真实 critic 是 `central_value_net`；不能误训练 actor 网络里未被使用的 value head。确认 advantage 的 bootstrap values 来自更新后的 central critic。
- 使用目标奖励的 discounted return/GAE λ-return：γ=0.99、λ=0.95、reward scale=0.01 先沿用现有配置。reward 不重复缩放，value normalizer 与 bootstrap 使用一致尺度。真终止不 bootstrap，time-limit 截断使用终止前 final observation/state，不能用自动 reset 后观测。
- 初始 warm-up **20 个采样—价值更新 iteration**；8184 env × horizon 16 × 20 = **2,618,880 transitions**（6 个探索组，每组 1364 env）。若并行数不同，按总 transition 预算换算，mini_epochs 不计为额外采样。显存不足可用 2728 env（2×1364）调通；正式组间必须固定探索组数和每组环境数。
- critic LR 起点 5e-5，2–4 mini_epochs。随机 critic 初期可评估关闭 value clipping，但单独记录，不把 PPO policy clipping 与 value clipping 混为一谈。
- 监控 held-out rollout 上的 value MSE、explained variance、按成功/失败分组的预测回报偏差；EV 在回报方差很小时不可靠。20 次只是初始预算，可预注册 0/5/20/50 次消融；不以训练 loss 单独判断可解冻，也不要求拍脑袋固定 EV 必须达到某值。

产物：actor 不变证明、critic checkpoint、normalizers、完整阶段/步数/优化器状态与 rollout 统计。

## 9. 阶段 F：解冻并保守 SAPG

- actor LR 从 1e-5 开始，critic LR 5e-5，policy clip 0.05 起步；保留 γ、λ、horizon、网络和奖励定义。
- 当前 SimToolReal 基线 LR 是 1e-4、clip=0.1，所以不是照搬论文“从 1e-3 降 100 倍”。解冻后先使用固定或受上界约束的 actor LR，防止 adaptive scheduler 自动抬高学习率；后续再线性衰减。
- 继承 actor normalizer 并先保持冻结；若需更新统计，单独设置缓慢过渡并评估 policy drift，不在解冻瞬间更换统计。critic normalizer 按独立规则继续。
- 丢弃过期采样，使用当前 SAPG 行为策略采集 rollout，并保留原实现的 leader–follower 经验共享、行为 log-prob、条件重标记及相应 return/advantage 处理；不能把共享经验误当纯 on-policy 数据，也不能把历史 demo 直接当 SAPG rollout。各组 std 逐步允许训练，监控 entropy/饱和率。
- 监控 KL、clip fraction、成功率、掉落率、value error。预先规定 rollback 条件，例如连续 3 次同种子评价比 BC 基线下降超过 10 个百分点时保留失败记录并恢复安全 checkpoint 调低 LR；此规则对相应比较组一致，不能择优隐藏坏种子。
- 第一轮不额外加入 BC/KL regularization，也不改 reward；有明确遗忘证据后将其作为独立消融，避免无法判断 WU 的贡献。
- 主效应验证后再逐步扩展初态/goal、物体与 DR。冻结边界覆盖共享条件 actor、每个探索组的 sigma 参数、归一化及任何影响行为的可变状态；不能只检查一个探索组。

## 10. 对照实验与“更快”的定义

必要组：

1. R：随机初始化 → 原 SAPG，测实际基线。
2. B：BC actor → 原 SAPG，不 WU，测直接初始化效果。
3. C：BC actor → 保守 SAPG，不 WU，测小 LR/clip 效果。
4. D：BC actor → critic WU → 与 C 相同的保守 SAPG，主方案。

资源允许增加随机初始化 → 保守 SAPG，以及 BC + WU → 原 SAPG；BC-only 是零在线更新参考点。C 与 D 必须来自同一 seed 的同一 BC checkpoint、相同 log_std 与 actor normalizer。R/B 等各组网络、数据外的任务设置与评价相同。

先用 3 个配对 seed 筛查，结论性实验至少 5 个。每次固定 200–500 个评价 episode，报告每个 seed 与跨 seed 均值/区间；统计独立单位是训练 seed，不能把大量相关帧当独立重复。小样本下不夸大统计显著性。

评价集分开：训练物体的新初态/goal；完全未见物体；原程序化工具分布。当前每物体一条示范时，用仿真新初态做闭环评价，并另留未见物体；动作监督验证不足要明确标注。

速度指标：

- 达到预定成功率（如 60%、80%，按任务可达性事先选择）所需 env transitions，**包括 WU**。
- 达到同阈值的训练 wall time，包括 BC + WU + SAPG；另报完整 pipeline time，加入轨迹优化、资产转换和数据适配成本。
- 同预算下最终成功率和学习曲线面积；连续至少 3 次评价达到阈值才算到达。未达阈值按未达到/右删失报告，不能只平均成功 seed。
- FPS、启动/cooking 时间、显存、GPU-hours。多物体网格比程序化物体慢时，不把吞吐差异误认为算法效率差异。
- 机器人执行时间只在成功 episode 上单独报告，不能等同训练加速；失败率也必须同时报告。

配对加速比：T_baseline / (T_BC + T_WU + T_SAPG)，并给出含预处理的端到端版本。已有数据成本可单列为沉没成本，但不得省略端到端结果。

数据预算消融先 5/10/20 条；WU 预算消融先 0/5/20/50 次。分两轮做，避免全笛卡尔积。固定 GPU、并行环境数、总在线预算、评价间隔和训练软件版本。实验计时期间避开正在运行的 SPIDER 优化任务，或独占另一 GPU。

## 11. 实施顺序、文件落点与验收

以下名称为拟新增文件，不表示已经实现：

1. SPIDER `examples/arm_hand/export_simtoolreal_dataset.py`：完成样本盘点、hash、分级、坐标/物体元数据与时间序列导出。验收：不读取半成品，不漏 initial_state，不错一帧。
2. SimToolReal `robot_spec.py`、`simtoolreal_env_cfg.py`、任务 YAML：右手机器人支持。验收：关节名/顺序/限位一致，FK 与碰撞验证通过。
3. `utils/scene_utils.py` + 拟新增 `utils/dataset_assets.py`：dataset object pool、转换缓存、scale/frame 绑定。验收：单物体正确，多物体 env-index 正确，刚体数量和参数正确。
4. 拟新增 `isaacsimenvs/tools/replay_spider_dataset.py`：动作反算、100→60 Hz、真实 pipeline 回放、录制 obs/state/action。验收：动作边界和回放成功达到预设门槛。
5. 拟新增 `isaacsimenvs/pretrain_actor_bc.py`：序列 BC、归一化、闭环选模。验收：验证闭环优于随机 actor，并能保持训练示范技能。
6. vendored `rl_games/rl_games/algos_torch/a2c_continuous.py`、`a2c_common.py`、`central_value.py`：显式 warm-up 阶段、actor/normalizer freeze、central critic 更新、独立 LR、可恢复阶段计数。验收：actor probe 与 hash 不变，critic 确实更新，截断 bootstrap 正确。
7. 拟新增独立 warm-start train YAML 和实验 launcher：R/B/C/D、seed、step/time 账本与统一 evaluator。验收：小环境 smoke + 恢复后阶段正确，再扩大到正式规模。

依赖顺序：数据盘点 → 右手/物体/动作接口 → 目标仿真合格回放 → BC 闭环 → WU 冻结验证 → SAPG 小实验 → 多 seed 正式对照。每个环节通过才进入下一个。

先做一个盒子和一个罐子的最小闭环，估计 1–2 个工作日完成接口诊断；资产/右手兼容、动作限速或跨引擎接触若需重优化，另计工期。BC/WU smoke 通常再预留 1–2 个工作日实现与验证；正式 GPU 预算通过实测 FPS 和基线 learning curve 决定，不预估确定的训练完成小时数。

## 12. 推荐的第一轮执行配置

以用户指定批次最终通过验收的样本为准，最多 20 条；先使用右手扩展的 SAPG、固定物体池和较窄的初态分布。BC 只训带探索条件的 actor；WU 20 iterations（8184×16，对应约 2.62M transitions），每个探索组的 actor log_std 可从 -2 起步且冻结，critic LR 5e-5；解冻后 actor LR 1e-5、clip 0.05。首先跑 C/D 配对 smoke 验证冻结与价值适配，再跑完整 R/B/C/D 的 3-seed 筛查。

如果 20 条无法获得闭环稳定 actor，优先修动作/物体/机器人不一致并补恢复数据；如果 C≈D，则说明此设置下 critic WU 收益有限；如果在线 steps 变少但总时间变长，应如实报告为采样效率提升而非端到端加速。

## 正式 SAPG 执行修订（2026-10-27）

用户明确：模仿只用于热启动，不以空间泛化作为进入 RL 的门槛。正式任务仍是 **target pose reaching**，不使用逐帧示范动作监督、轨迹跟踪损失或 progress actor 观测。采用单轨迹检查后的 actor 权重，保留 140 维观测；二十个冻结变体只提供初始状态/目标位姿。此轮不保留验证集，六个探索组使用同一组二十个场景。

入口为 `examples/arm_hand/run_mustard_isaac_actor.sh --sapg`，实际调用原 SimToolReal vendored rl_games SAPG。保留 `use_others_experience=lf`、六组 `mixed_expl_learn_param`、条件 sigma、LSTM/MLP、非对称 critic、原奖励系数、10 秒回合、成功容差课程，以及原观测/动作/物体状态延迟、噪声、随机力矩。核对论文后，正式启动命令采用 120 个并行环境、683 epochs 只更新 critic（683×120×16=1,311,360 transitions，与论文 20×4096×16 的预热样本量近似），随后自动解冻 actor；最大总预算 200,000 epochs。actor 输入归一化固定为模仿时统计量，actor LR=1e-5 并线性衰减，critic LR=5e-5，clip=.05，actor/critic mini-epochs=4/4，预热 log_std=-2。此处保留 SimToolReal 的 gamma=.99 和 SAPG 探索奖励，未声称完整复现论文 FMB 插入任务。

与原默认任务的必要差异必须纳入后续从零对照：右手 KUKA/Sharpa 源资产与 PD、mustard 的质量/碰撞/角点尺寸、桌面位于 z=0、掉落阈值改为桌面下 5 cm、到达一个目标后重置场景、冻结变体的初始状态与目标。动作保留已训练的 `normalized_target = previous_normalized_target + 0.1 * actor_output` 解码，最终裁剪到关节限制；不能把这个权重直接解释为原版 arm velocity-delta / hand absolute 输出。动作延迟和外力扰动仍应用于该接口。

训练保存完整环境配置、SAPG 配置、初始化检查、critic 阶段 actor 不变/critic 已更新的检查、周期权重和进度。录制 Isaac Sim 实际渲染视频：初始阶段以及每 6000 policy steps 录制 300 帧。视频与训练指标必须共同检查，不能把单轮 reward 或终点误差当作稳定抓取证据。

## 依据

- 用户提供论文 `/home/szc/Documents/paper/2608.19182v2.pdf`：§3.3、Table 3、Appendix D。原文 BC 为在线 teacher 蒸馏 40k iterations，WU 20 epochs；附录给出 log_std=-2、critic LR=5e-5、post actor LR=1e-5、clip=.05。Table 3 表明低 LR 是防止崩溃的关键，WU 主要提升该实验的成功率/稳定性，不保证显著缩短所有任务的时间。
- 官方论文页面：https://arxiv.org/abs/2608.19182v2 ，项目页：https://adept-dexterity.github.io/ 。
- SPIDER：`examples/arm_hand/{batch_kuka_gpu,prepare_kuka_right,optimize_kuka_gpu,audit_right_trajectory}.py` 及指定批次的 settings/status/tracking_collision_report。
- SimToolReal 根目录 `/home/szc/work/simtoolreal_origin/simtoolreal` 下：`isaacsimenvs/cfg/train/SimToolRealSAPG.yaml`、`isaacsimenvs/cfg/train/SimToolRealPPO.yaml`、`isaacsimenvs/cfg/task/SimToolReal.yaml`、`isaacsimenvs/tasks/simtoolreal/{robot_spec,simtoolreal_env_cfg}.py`、`utils/{action_utils,obs_utils,reset_utils,scene_utils,generate_objects}.py`、vendored rl_games 的 a2c_continuous.py。
