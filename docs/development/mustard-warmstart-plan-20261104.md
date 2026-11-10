# Mustard 热启动：直到完成目标的执行计划

制定日期：2026-11-04；**2026-11-05 修订（收敛版）**。流程可按决策点改动；每次改动记录原因和证据，不改最终判据。

## 1. 研究问题与完成判据

**问题**：用前处理好的数据（示范动作和/或物理验证过的状态）热启动 RL，能否比从零更快学会原生任务，同时保留对没见过情形（原生随机初态、掉落后重新抓起）的泛化。

- **能力判据（固定评估，同一套 seed/起点）**：原生随机初态首目标到达 ≥ 50%（≥64/128）；掉落恢复 ≥ 30%（≥25/84）；在线原生回合平均完成目标数 ≥ 3。
- **效率判据（相对匹配基线）**：同一环境配置、并行规模和评估下，达到同一里程碑（原生平均完成目标数 ≥1、≥3、≥10）所需的在线帧数和墙钟时间少于从零 SAPG；单独报告离线成本（数据生成 GPU 小时、BC 时间）。
- 至少 2 个种子；只用缓存起点成功或混合奖励不算达成。
- 若匹配任务从零学不会、热启动能学会，结论改为"热启动使原本学不会的任务可学"，同样需要 2 个种子和固定评估支持。
- 预算内达不成时，交付带证据的否定结论。

## 2. 当前事实（决定计划的只有这几条）

1. **匹配任务目前从零学不会**：右手 + 芥末瓶 + 关闭域随机化 + 16 秒，2.09B 帧后原生目标数约 0，σ 各 block 都约 1，奖励停在约 740。没有可学会的匹配任务，效率判据就无法计算，这是当前的第一瓶颈。
2. **与原版的差别只在用法上**（2026-11-05 对比 `/home/szc/work/newsimtoolreal`）：奖励、终止、动作、观测的代码逻辑一致；不同的是机器人（右手、自转换 URDF vs 原版左手）、物体（芥末瓶 vs 600 个程序生成工具）、域随机化（我们关闭外力/延迟/噪声）、时限（16 vs 10 秒）、并行规模（8184 vs 24576）。
3. **BC 不泛化**：数据从 503 条增加到 782 条也没有改善原生和掉落（都是约 0）。BC 学到的是记住示范，继续改 BC 数据或配方收益很低。
4. **热启动 RL 只在缓存起点上有效**：放宽探索并加课程后，抓持和接近目标起点约 127/128，原生和掉落恢复仍为 0；课程在 α=0.3 卡住。
5. **数据已经够用**：状态库 v4.1 有 3684 个去重后的状态（躺倒 1911 个，36 个水平朝向分箱全覆盖），包括机械臂随机起点的接近段。现阶段瓶颈不在数据量。

## 3. 执行顺序（只有三步，按门槛推进）

### 第 0 步：得到一个从零可学会的匹配任务（最高优先级）
目的：确定后面所有比较使用的任务设定。每个实验只改一项。

| 编号 | 实验 | 状态 | 回答的问题 |
|---|---|---|---|
| 0a | 官方预训练策略在原版环境中抓芥末瓶（左手，评估几分钟） | 完成（见进展记录，芥末瓶首目标 0–3%） | 芥末瓶对 Sharpa 手在 Isaac 里是否抓得起 |
| 0b | 原版环境原样，只把物体换成芥末瓶，从零训练，8000 轮约 1.05B 帧 | **已停止**（11-05，第 4197 轮，最后保存第 4000 轮；抬起约 1%）`outputs/pristine_mustard_s42` | 难点是否在物体 |
| 0c | 右手 + 我们的 URDF，其余用原版设置（打开域随机化、10 秒），从零训练 | **已停止**（11-05，第 2702 轮，最后保存第 2500 轮；抬起约 1%）`outputs/right_mustard_pristine_settings_s42` | 难点是否在右手转换或我们关掉的设置 |

| 0b-pc | 与 0b 完全相同，只把观测里物体的 4 个包围盒角点（相对手掌、相对目标）换成网格表面 64 个最远点采样点（点数同 ADEPT 表 6/7）；奖励仍用包围盒关键点，任务定义不变 | **已停止**（第 2648 轮，与包围盒无差别）`outputs/pristine_mustard_pc64_s42` | 点云表示是否比包围盒更易学 |
| 0b-adept | 与 0b 相同，观测换成 ADEPT 式字段：去掉关键点，改为物体位置+朝向（带噪声）、目标位置+四元数、物体缩放、64 点物体点云（环境坐标系）；critic 用干净值加原有特权量；奖励不变 | 运行中 `outputs/pristine_mustard_adept64_s42r`（0.40B 帧抬起约 57%，到达目标约 0.05%） | ADEPT 的观测组织方式是否更易学 |
| 0c-adept | 右手 + 原版设置 + ADEPT 观测，从零，12000 轮 | 运行中 `outputs/right_mustard_pristine_adept_s42`（0.12B 帧抬起约 0–0.6%，同帧数 0b-adept 2.2% 并随后起飞：**右手落后**） | 右手能否复现 0b-adept |
| **0c-adept-aligned** | 与 0c-adept 完全相同，只加 `--align-left-physics`（`physics_align.py`）：depenetration 10→1000、手指关节摩擦→0，与原版左手一致 | 运行中（11-05 启动）`outputs/right_mustard_pristine_adept_aligned_s42` | 右手落后是否来自这两项设置偏差 |
| **0c-adept-aligned-aa2deg** | 与 0c-adept-aligned 完全相同，再加 `--restrict-abduction`：四指 MCP_AA ±20°→±2°、拇指 CMC_AA 上限 0.349→0.131（原版左手限位；审计确认 29 个关节限位全部与左手一致） | 运行中（11-05 启动）`outputs/right_mustard_pristine_adept_aligned_aa2deg_s42` | 外展范围是否是右手落后的原因 |

判定：原生平均完成目标数 ≥1 记为"学会"，预算 1.5B 帧；早期信号看抬起比例和奖励是否持续上升。
- 0b 学会、0c 学会 → 用 0c 的设置作为匹配任务（右手，与已有数据和状态库兼容）。
- 0b 学会、0c 学不会 → 问题在右手模型或配置。逐项对比 URDF 的关节限位、PD 增益、碰撞体、手掌和指尖坐标系，修好后再跑 0c。
- 0b 学不会 → 芥末瓶本身对从零训练太难。保留我们的任务设定，判据改为"热启动使原本学不会的任务可学"，直接进入第 1 步。
- 0a 只作辅助证据，不单独决定分支。
- **右手诊断（11-05 加入，当前最高优先级）**：0b-adept 学会抬起、0c-adept 落后，属于第二种分支。左右手逐项对比（`outputs/left_right_robot_audit_v1/`）结果：
  - 一致：手臂/手指刚度、阻尼、armature、力矩上限（按角色映射正确）；基座位姿、手臂默认关节角；物体质量、接触偏移、摩擦材料；任务 YAML 与 SAPG 等价配置；掌心朝向和弯曲方向一致（拇指左右镜像为正常）。
  - 不一致 1：`max_depenetration_velocity` 右手 10、原版 1000（机器人和物体）。10 来自我们 2026-09-07 在 simtoolreal_origin 的提交 `9e9acd70`（Yimu 支持），上游一直是 1000；`third_party/simtoolreal` 继承了它。
  - 不一致 2：手指关节摩擦右手 0.004–0.132、原版 0。右手值来自 MuJoCo `frictionloss`（N·m 力矩），被直接写进 PhysX 的无量纲关节摩擦系数。
  - 不一致 3：四指外展 MCP_AA 右手 ±20°、原版 ±2°（URDF 名为 `adjusted_restricted`）；拇指 CMC_AA 原版 [-0.349, +0.131]、右手 ±0.349。手部动作直接映射到关节范围，外展探索范围大 10 倍。但示范大量用外展（食指中位 -17°，98% 帧超过 2°），锁死会使现有示范/状态库失效，**需用户决定**。
  - 次要：速度上限（右 10、左 11.6–35 rad/s）、手掌碰撞体（右 32 块凸分解、左 1 个网格）、指骨质量小差异、手掌中心高 2 cm、指尖远约 1 cm。
  - 判定（三组同 seed 并行，相邻两组只差一项）：0c-adept → aligned 看第 1、2 项；aligned → aligned-aa2deg 看第 3 项。按 0b-adept 节奏在约 0.2B 帧（约第 1500 轮）比较抬起比例，之后看首目标。
    - aligned 追上 0b-adept：第 1、2 项是原因；之后所有右手实验加 `--align-left-physics`。
    - 只有 aligned-aa2deg 追上：外展范围是原因；之后所有右手实验加 `--align-left-physics --restrict-abduction`，并按下方"数据重新微调"更新示范和状态库。
    - 两组都落后：再查碰撞体（手掌 32 块凸分解 vs 原版单网格）和速度上限，每组只改一项。
  - **数据重新微调（仅当外展范围确认是原因时执行，用户 11-05 已同意）**：现有 503/892 条示范与状态库 v4.1 用 ±20° 外展（食指中位 -17°），在受限手上不可直接复用。步骤：①把右手 MJCF/URDF 外展限位改为原版值，重新做 IK 与 MJWP 优化（或先把现有轨迹的外展投影到限位内）；②Isaac 中用受限手重新回放验证抓持；③重建 BC 数据集（ADEPT 版）并重训 BC；④按同一审核流程重建状态库（v5）。旧数据和状态库保留，不覆盖。

### 第 1 步：在匹配任务上比较三种初始化（同一任务、同一评估）
| 臂 | 初始化 | 初始状态分布 | 说明 |
|---|---|---|---|
| A | 从零 | 100% 原生 | 基线 = 第 0 步选定的右手 ADEPT 组（0c-adept 或 0c-adept-aligned） |
| S | 从零（只用状态） | 原生 40% + 状态库 v4.1 三类 60%（固定比例 .4/.25/.2/.15，当前实现没有姿态课程） | OmniReset 风格；探索和超参与基线相同。**11-05 启动后已停止**，等右手问题查清 |
| D | ADEPT 版 BC actor + critic 校准 | 同 S | BC：`outputs/mustard_bc_adept_v1/bc4000`（运行中）→ carry200 → D（σ 0.4、critic 预热 683 轮、BC 输入归一化冻结）。自动流水线 `examples/arm_hand/run_mustard_arm_d_adept.py` **已停止**（D 未开始），bc4000 继续跑 |

- **S/D 暂停**：右手在 0c-adept 上学不会抬起，S/D 都建立在同一右手设定上，先查清右手问题再启动；重启时用第 0 步确定的物理设置（如 `--align-left-physics`），A/S/D 三个臂设置必须一致。

- 每个臂都使用匹配任务的完整设定。如果匹配任务打开了随机外力，掉落练习来自任务本身，不需要额外机制。
- 固定评估每 500 轮一次：原生、掉落恢复、抓持、接近目标、示范附近。
- 门槛（1.5B 帧内）：原生首目标到达 >10% 且掉落恢复 >0。S 和 D 中达标、且帧数更少的一个进入第 2 步；都不达标，再只做一项调整：提高原生比例，或提高躺倒状态的采样比例。

### 第 2 步：确认与报告
- 选定的方法和基线各 2 个种子；报告在线帧数、墙钟时间、离线成本；固定评估全套。

### v5（旧任务设定）
- 已于 11-05 按用户决定停止（见进展记录），不再继续。

## 4. 冻结项（第 1 步门槛之前不做）
以下工作有价值，但都不是当前瓶颈。只有在第 1 步两个热启动臂都失败、且失败原因明确指向它们时才重启：
- 数据生成：E6 接近方向 v2、源 13 滚动、E2 管线效率改造（每源预算、60 轮封顶、短片段）、恢复生产战役、状态库 v4.2。
- BC：用 892 条重训、hindsight 目标重标注、留出式评估。
- 抓取专项：DemoGrasp 式编辑参数 RL、单独抓取策略、反向课程。
- 改摩擦等物理参数（会改变任务设定，需要用户批准）。
- 新文献调研。

## 5. 资源
- GPU 85 GB，每个 8184 环境训练约 17–19 GB，评估/录像另算。
- 当前占用（11-05 约 17:50 后）：0b-adept、0c-adept、0c-adept-aligned、0c-adept-aligned-aa2deg 四个 SAPG 训练 + ADEPT BC（约 4.5 GB），约 80/85 GB，**没有余量做评估/录像**；需要时先停一组。0b、0c、0b-pc、S、D 流水线均已停止。

## 6. 进展记录（历史，保留原文）
- 2026-11-04：计划制定。BC v3b 训练中；v5 脚本冒烟通过；下一步启动基线 A。
- 2026-11-04：基线 A 启动：`experiments/mustard_simtoolreal/train_scratch.py`（启动器入口 `scratch`），输出 `outputs/mustard_scratch_baseline_s42`，16000 轮（约 2.1B 帧），阶段评估监控从 500 轮起每 500 轮一次（label `scratch`）。
- 2026-11-04：数据集诊断：782 条只来自 11 个独立示范（有效约 7 个）；同源抓取构型距离仅为跨源的 0.21；原生初态 0%、掉落状态 7% 有相近示范帧；BC 在未训练轨迹上的误差随同源训练条数增加而下降、跨源基本不泛化。掉落/失败状态用现有增强覆盖 1–29%，绕物体竖直轴任意 yaw 后 30° 内覆盖 100%。据此加入 E4（物体坐标系编辑与批量回放）。
- 2026-11-04：E1 完成：源 5 在 Isaac 中失败不是摩擦（摩擦提高到与 MuJoCo 一致无改善），也不是质心/惯量/碰撞网格（三者一致）；原因是 MuJoCo 软接触下的静止位姿略微压入桌面，PhysX 去穿透把躺倒瓶子弹开，手到达前已滚动约 60°。起始高度 +2 mm 后静止测试转动 58°→9°，教师回放抓持 0/24→22/24；源 13 在任何高度都会滚动（另有原因，未解决）。`train_bc.py` 新增 `--object-z-offset`，用 +2 mm 重采教师 `outputs/mustard_teacher_z2mm_v1`（抓持 803→917），截断筛选得到 `outputs/mustard_classified_20261104_truncated_z2mm_v1`：892 条（782→892），躺倒 270→381，独立来源 11→12（源 5 新增 112 条）。
- 2026-11-04：E4 试点（运动学阶段）：`examples/arm_hand/e4_object_frame_augment.py` 把示范绕物体竖直轴旋转并平移，手指关节不变，机械臂逐帧 IK（LM 阻尼 + 关节中心零空间项）。7 个源各 2 条基轨迹的扫描：yaw 负方向多数可达（-8° 到 -30° 基本全部，-60° 约一半），正方向连 +8° 都失败；平移 +y 可达、-y 失败。原因推断：源示范的机械臂关节 6 已接近限位（约 8° 余量），保持相对抓法的刚体编辑只能覆盖很窄的方向。结论：纯刚体编辑做不到 360° 覆盖，需要像 DemoGrasp 那样改变接近方向（让编辑参数改动抓取相对位姿），或在可行窗口内生成后用 Isaac 回放验证。
- 2026-11-04：E4 可行窗口批量生成：yaw ∈[-60°,0°]、物体初始 xy ∈±10 cm，7 源 × 6 基轨迹 × 20 目标 = 840 候选；运动学通过 434；Isaac 回放（+2 mm）抓持 355/434（源 5 的编辑版本 64/64 全部失败，原因未查）。截断筛选（≤960 步）只留 146 条，209 条因放慢后抓稳太晚被拒。覆盖增益有限：掉落状态 30°/5 cm 内 0%→21%，BC 失败状态 5%→10%，15°/3 cm 内仍为 0%；水平朝向分箱 17→24/36；物体 x 扩到约 ±11 cm。输出 `outputs/mustard_e4_window_v1`。
- 2026-11-04：状态库 v4：源 5 的 112 条 + E4 中能抓住的 355 条（不受 960 步限制）经同一套稳定/几何/打包审核后新增 1370 个状态，与 v3 合并为 3490 个（接近物体 780、稳定抓持 1980、接近目标 730；躺倒 1750）。Isaac 加载与恢复检查通过（误差 2.4e-7）。`outputs/mustard_diverse_reset_bank_v4/merged/reset_bank.npz`。注：状态库覆盖率包含已收录的失败/掉落状态本身，不能当作泛化证据。
- 2026-11-04：源 5 编辑版本失败原因：E4 脚本用 qpos[0]（第一步之后、略低 0.6 mm）作为物体起点，叠加原有轻微压入导致仍被弹开滚动。改为从源 initial_state 变换物体起点后，源 5 编辑版本抓持 0/64→61/61（`outputs/mustard_e4_window_src5_fix`）。其余源的 E4 候选也用了旧起点，重新生成后成功率可能更高。
- 2026-11-04：E5 机械臂/手初始状态随机化：`examples/arm_hand/e5_robot_start_augment.py` 按原生重置规则采样起始关节（0.9×默认 + 0.1×限位内均匀），用最小加加速度轨迹在关节空间过渡到示范起点，再接原示范；MuJoCo 中检查过渡段不碰桌、不碰物体。9 源 × 6 基 × 8 起点 = 432 候选，运动学通过 304（源 18 过渡段全部碰到物体）；Isaac 回放抓持 290/304（95%）。原生默认姿态与示范起点差距大（如关节 4：1.38 vs 0.41 rad），加上过渡段后放慢的轨迹中位 1309 步，按 ≤960 步截断只留 64 条；作为状态库来源不受此限制。`build_truncated_native_bc.py` 改为离线/在线持握判定不一致时拒收该条（而非报错）。输出 `outputs/mustard_e5_robot_start_v1`。
- 2026-11-04：E4 v2（修正物体起点后重新生成）：运动学通过 434/840，Isaac 抓持 423/434（97%，v1 为 82%）。
- 2026-11-04：状态库 v4.1（`examples/arm_hand/build_reset_bank_v41.py`）：v3 + 源 5（112 条）+ E4 v2（423 条）+ E5（289 条）的新状态经同一套审核后共 4520 个；新增 `dedup_reset_bank.py` 去除近重复（同类型且物体位置 <1 cm、朝向 <5°、机械臂关节 RMS <0.05 rad、手关节 RMS <0.1 rad、目标位置 <3 cm 视为重复，v3 优先保留），删除 836 个（v3 自身 83 个，新增部分 753 个），剩 3684 个：接近物体 783（躺倒 457）、稳定抓持 2104（躺倒 1134）、接近目标 797（躺倒 320）。Isaac 加载与恢复检查通过。`outputs/mustard_diverse_reset_bank_v41/merged/reset_bank.npz`，v5 改用此版本。
- 2026-11-04：E6 改变接近方向（`examples/arm_hand/e6_approach_edit.py`）：物体绕竖直轴任意旋转 ψ_o，手的路径只在可行窗口内旋转 ψ_h（-45°–0°），抓取落在物体的另一侧（相对角 ψ_o−ψ_h）。840 候选中 485 个通过 IK，但其中 451 个因示范起始时手离物体很近、旋转后的物体与手重叠而被拒，仅 34 个进入 Isaac 回放，抓住 14 个。按相对角：0–30° 6/6，30–90° 1/9，90–150° 0/12，150–180° 7/7。即从瓶子正对面（约 180°）抓同样可行，侧向 90° 左右不可行。下一步：先接 E5 的过渡段（从远离物体的原生起点出发）解决起始重叠，并把相对角采样集中在 0°±30° 与 180°±30°，预计可覆盖约 210°/360° 的物体朝向。
- B0 结果：BC v3b 未通过。示范起点抓持：旧 BC 510/782（旧 503 条 405、新 279 条 105、其中新躺倒 155 条 84）；v3b carry 410/782（268 / 142 / 40）。v3b 只在新 279 条总数上更好，躺倒和旧示范都更差，到达目标也更少。按计划回退：v5 用旧 BC actor（v2 carry200）+ 892 条数据集 + 状态库 v4.1 启动，输出 `outputs/mustard_sapg_v5`。
- 基线 A 结果：匹配的从零 SAPG 跑满 16000 轮（2.09B 帧，约 9.2 小时）后仍学不会原生任务：奖励在约 0.8B 帧后停在约 740，原生回合平均目标数始终约 0，σ 一直约 0.99 没有下降，各阶段固定评估基本全 0。按 A 的决策规则：这套机器人/物体/设置下从零也学不会，下一步要先查原生任务本身的瓶颈（奖励、容差、时限、物体尺寸/质量、σ 不收敛的原因），再判断热启动效率。
- 2026-11-04：参考文献：ADEPT（arXiv 2608.19182，无示范，RL 预训练 + ADR 放宽生成范围到完整 SO(3)/±15 cm，后训练 lr 1e-3→1e-5）支持 C 与课程思路；DemoGrasp 支持 E4；Dex-X 的增强与当前相同。
- 2026-11-05：与原版 SimToolReal 逐项对比，代码逻辑一致，差别在机器人、物体、域随机化、时限和并行规模。新增 `experiments/mustard_simtoolreal/train_pristine_mustard.py`（原版环境只换芥末瓶），启动 0b `outputs/pristine_mustard_s42`；新增 `eval_pristine_pretrained.py`（0a），官方预训练权重已下载。从零基线最终 checkpoint 的 6 个 block σ 均约 0.9–1.1。
- 2026-11-05：计划修订为收敛版：第 0 步先得到可学会的匹配任务，第 1 步比较 A/S/D 三种初始化，其余数据生成、BC 和抓取专项冻结。
- 2026-11-05：用户要求加入点云对比（0b-pc）。`train_pristine_mustard.py --object-repr pointcloud --pc-points 32`：观测 140→308 维、critic 状态 162→330 维；点云为芥末瓶视觉网格表面 32 点最远点采样（URDF 连杆坐标系，保存为 `pointcloud_object_frame.npy`）；奖励与成功判定仍用包围盒关键点。注意：单一刚体物体时点云与包围盒携带的位姿信息相同，差别只在编码方式（表面点更直接给出手与瓶身的几何关系）；ADEPT 的收益主要来自多物体泛化，本实验不能检验这一点。
- 2026-11-05：ADEPT 核对：表 6（Kuka-Allegro）与表 7（Flexiv-Sharpa）中点云均为每个物体固定 64 点 × 3 = 192 维，直接展平进 MLP+LSTM；目标用位置+四元数；奖励位姿误差用固定立方体 8 角点；采样方式与坐标系未说明。据此 0b-pc 改为 64 点（32 点版本运行约 14 分钟后停止，输出保留在 `outputs/pristine_mustard_pc32_s42`），观测 140→500 维。
- 2026-11-05：新增 0b-adept（`--object-repr adept`）：actor 318 维 = 关节位置/速度/上一目标 87 + 手掌位姿 7 + 指尖相对手掌 15 + 物体位置 3 + 朝向 4 + 目标位置 3 + 目标四元数 4 + 物体缩放 3 + 点云 192；critic 340 维。实测 critic 物体位姿与仿真一致（误差 0），actor 位姿带域随机化噪声/延迟（平均偏差约 2 cm），点云点到物体原点 2.8–10.3 cm。与 ADEPT 的剩余差别：无指尖接触力、无 geometric fabric 动作空间、无 PBT。首次启动因 32 点版本的残留进程占显存而 OOM，清理后以 `outputs/pristine_mustard_adept64_s42r` 重新启动。
- 2026-11-05：按用户决定停止 v5（联合训练第 657 轮，sapg_latest 为第 1000 轮保存，输出保留），为 0b 三组对照腾出显存。v5 第 1000 轮固定评估：原生 0/128 首目标、掉落恢复 0/84；从抓持起点到首目标 109→79、接近目标 114→93、示范附近 118→86，即联合训练初期相对初始策略有所退化。
- 2026-11-05：0a 扩展为多物体测试。`build_object_pool.py` 从四个数据集各类别取一个实例，共 197 个物体（按视觉网格包围盒中心重定根，凸分解作碰撞，密度 500 kg/m³ 求质量并夹在 0.03–0.6 kg，芥末瓶因此为 0.23 kg，真实约 0.475 kg）；`eval_pristine_pretrained.py --object pool`，原版环境与奖励不变（左手、域随机化、固定尺寸关键点），每物体 16 环境 × 2 回合，最多 1800 步。官方预训练策略：自带程序生成工具上抬起 83%、首目标 74%、平均 15.6 个目标；197 个真实物体整体抬起 16%、首目标 3%。首目标 ≥20% 的只有 2 个（hrdexdb pingpong 31%、hot3d mouse 22%），≥10% 的 16 个；三个芥末瓶实例首目标 0–3%。每物体只有 32 回合，±15% 量级的抽样误差。候选物体及右手 Sharpa 重定向数：hot3d mouse（抬起 59%、首目标 22%，85 条）、whiteboard_marker（47%/19%，47 条）、puzzle_toy（53%/12%，66 条）、cellphone（50%/16%，28 条）。结果 `outputs/pretrained_eval_v1/`。
- 2026-11-05：启动 0c：`train_scratch.py --pristine-settings`（右手 + 我们的 URDF + 芥末瓶；原版 YAML 的延迟/噪声/外力，力 20、力矩 2，10 秒），8000 轮，`outputs/right_mustard_pristine_settings_s42`。与 0b 的差别只剩机器人（URDF 和机器人配置）。
- 2026-11-05：第 1 步的 S、D 臂已在 `train_scratch.py` 中实现并冒烟通过，任务设定与 A（0c 或最终选定的设定）完全相同：S = `--diverse-reset-bank`（从零初始化，原生 40% + 状态库 v4.1 三类 60%，在线原生指标只统计原生回合；恢复误差 4.8e-7）；D = S + `--actor-init`（旧 BC v2 carry200，σ 设为 0.4，BC 输入归一化冻结）+ `--critic-epochs`（actor 冻结、只训练 critic，期间学习率固定在 1e-4；验证：第 1 轮后 actor 未变，第 2 轮后更新）。GPU 已有 4 个训练（约 71 GB），S/D 等第 0 步有结论或腾出显存后启动。
- 2026-11-05：第 0 步中期结果（同一 seed，原版环境 + 芥末瓶，左手）：ADEPT 式观测明显更易学抬起。约 320M 帧时抬起比例：ADEPT 47%（约 200M 帧开始上升，260M 帧出现零星首目标）；包围盒 0.9%（435M 帧）；64 点点云 0.8%（342M 帧）。点云版与包围盒无差别，说明收益来自 ADEPT 的观测组织（显式物体位置/朝向 + 目标位姿 + 环境坐标系点云，去掉相对手掌/相对目标的关键点），而不是"点云 vs 包围盒"本身。据此调整：停止点云版（第 2648 轮，输出与 checkpoint 保留），新增 0c-adept：右手 + 原版设置 + ADEPT 观测，从零，12000 轮（约 1.57B 帧），`outputs/right_mustard_pristine_adept_s42`。`adept_obs.py` 抽成共享模块，`train_scratch.py --obs adept`（不修改 environment.py，因其计入状态库环境哈希；状态库 v4.1 在 ADEPT 观测下加载与恢复检查通过）。若 0c-adept 能学会，匹配任务改用 ADEPT 观测；S 臂与观测无关可直接用，D 臂需要在新观测下重新做 BC（旧 BC actor 只接受 140 维原生观测）。
- 2026-11-05：D 臂改用 ADEPT 观测：`adept_obs.native_to_adept` 把缓存的干净 140 维原生观测精确换算为 318 维 ADEPT 观测（4 个关键点之和为零，均值即位置；目标朝向由共面角点 Kabsch 求得；点云由物体位姿变换），`check_native_to_adept.py` 在右手环境中对比在线 ADEPT 观测，最大误差 3.6e-7。目标四元数统一取 w≥0（环境与换算一致）；因此重启了 0c-adept（之前约 15 分钟未规范化的版本移到 `outputs/right_mustard_pristine_adept_s42_noncanonical_aborted`）。`train_bc.py --obs adept` 用旧 BC 的同一套 503 条数据与配方（window 4000 轮 lr 1e-4，之后 carry 200 轮 lr 1e-5）训练 ADEPT 版 BC：`outputs/mustard_bc_adept_v1/bc4000`，约 7.4 秒/轮。
- 2026-11-05：0b-adept 第 2000 轮视频（`experiments/mustard_simtoolreal/record_pristine_video.py`，需在 `/home/szc/work/newsimtoolreal/simtoolreal` 目录运行；输出 `outputs/pristine_mustard_adept64_s42r_video/epoch_002000/`）：16 环境 68 回合，抬起 56/68，首目标 0/68。行为：能从原版随机初态（含躺倒）抓起，但抬起后掉落或松手，未学会搬运到目标。
- 2026-11-05：评估脚本 SAPG block 修正：此前 `evaluate_reset_stages.py`、`evaluate_actor_video.py` 在观测末尾补 0（= 第 5 个 follower），官方 `rl_player.py` 用 50（leader）。两脚本新增 `--coefs`/`--coef`，默认 50。用 leader 重评 v2-12300、v3-15500、v4d-16500、v5-1000（`outputs/mustard_reset_stage_leader_eval_v1/`）：与旧口径相差 0–5 个，原生和掉落恢复仍全为 0，历史结论不变。未修改的旧诊断脚本（evaluate_balanced_bc、compare_drop_models、test_drop_recovery、diagnose_bc_rollout、compare_bc_checkpoints）仍补 0。
- 2026-11-05：左右手逐项审计（`experiments/mustard_simtoolreal/compare_left_right_robot.py`，结果 `outputs/left_right_robot_audit_v1/{left,right,right_aligned}.json`），结论见第 0 步"右手诊断"。新增 `physics_align.py` 与 `train_scratch.py --align-left-physics`，审计确认生效后 depenetration=1000、手指摩擦=0，其余参数与几何不变。`train_scratch.py` 同时允许 ADEPT 观测 + BC actor（检查 checkpoint 的 `obs_layout`/`actor_obs_dim`）。
- 2026-11-05 约 16:40：按用户要求停止 0b（PID 1278810）和 0c（PID 1323144）及其 W&B sidecar，各写 `stopped.json`。
- 2026-11-05 约 17:00：启动臂 S `outputs/right_mustard_pristine_adept_S_s42`（右手 + 原版设置 + ADEPT + 状态库 v4.1，600 环境冒烟 `..._S_smoke` 通过）；启动臂 D 自动流水线 `examples/arm_hand/run_mustard_arm_d_adept.py`（600 环境冒烟 `..._D_smoke` 通过，用中间 BC 权重验证加载、σ=0.4、预热期 actor 冻结）。
- 2026-11-05 约 17:15：用户确认 0c-adept 抬起和到达明显不如 0b-adept，**停止臂 S**（第 76 轮，无 checkpoint，`stopped.json`）和**臂 D 流水线**（仍在等待 bc4000，D 未开始；`mustard_bc_adept_v1/arm_d_pipeline_state.json` 标为 stopped）。ADEPT BC（bc4000）继续运行。
- 2026-11-05 约 17:25：启动 0c-adept-aligned：`train_scratch.py --headless --pristine-settings --obs adept --align-left-physics --epochs 12000 --output outputs/right_mustard_pristine_adept_aligned_s42`（PID 1429758，日志 `outputs/logs/right_mustard_pristine_adept_aligned_s42.log`；600 环境冒烟 `..._aligned_smoke` 通过）。与 0c-adept 只差这两项物理设置，同 seed 42。
- 2026-11-05 约 17:50：用户决定外展范围也要检验。`physics_align.install(restrict_abduction=True)` 在 URDF 转换前把右手四指 MCP_AA 改为 ±0.0349、拇指 CMC_AA 改为 [-0.349, 0.131]（写到转换工作目录的 `_abduction_restricted/kuka_sharpa_right_aa2deg.urdf`，源资产不改；不同文件名保证转换出的 USD 与未受限组不共用）。审计 `outputs/left_right_robot_audit_v1/right_aligned_aa2deg.json`：29 个关节限位全部与左手一致，其他参数与默认姿态几何不变。启动 0c-adept-aligned-aa2deg：`train_scratch.py --headless --pristine-settings --obs adept --align-left-physics --restrict-abduction --epochs 12000 --output outputs/right_mustard_pristine_adept_aligned_aa2deg_s42`（PID 1435084，日志 `outputs/logs/right_mustard_pristine_adept_aligned_aa2deg_s42.log`；600 环境冒烟 `..._aligned_aa2deg_smoke` 通过）。若外展确认是原因，按第 0 步"数据重新微调"更新示范与状态库。
- 2026-11-05：`diagnose_pristine_rollout.py` 诊断 ADEPT 左手第 3500 轮（约 460M 帧，确定性动作，256 环境 × 1200 步）：抬起 89% 回合，抬起后中位只保持 111 步，79% 回合提前终止（非超时）；抬起期间离目标最近的关键点距离中位 0.39 m（成功阈值 0.1125 m），只有 5% 进入 2 倍阈值。即已学会抓起抬高，尚未学会搬运到目标，属于 SimToolReal 从零训练的正常阶段顺序；继续训练。包围盒左手同期奖励相同（约 387）但抬起只有约 1%。
- 2026-11-05 约 18:30：另一会话的管理已停止，后续由本会话统一管理。保持其第 0 步右手诊断方案不变：0c-adept / aligned / aligned-aa2deg 三组同 seed，在约 0.2B 帧比较抬起比例（0b-adept 在 202M 帧为 13%，0c-adept 在 200M 帧为 0.9%），之后看首目标。
- 2026-11-05 约 20:15：停止 0c-adept（`right_mustard_pristine_adept_s42`，第 2335 轮/306M 帧，已过 0.2B 帧比较点：抬起 1.2%，0b-adept 同期 13%），为诊断腾出显存；aligned 两组继续。
- 2026-11-05：ADEPT 左手第 4500 轮（约 590M 帧）诊断（`outputs/diagnostics/adept_left_ep4500.json`）：抬起 90%，抬起后中位保持 186 步；76% 回合因物体掉到地面（fall，z<0.1 m）结束，24% 超时，hand_far 0；抬起期间离目标最近关键点距离中位 0.38 m（阈值 0.1125 m）。奖励自约 330M 帧起停在约 388，首目标约 0。结论：左手 + 芥末瓶从零也卡在"抓起—握住—掉落"的局部最优，没有学会搬运/手内转向（首目标为相对当前位姿 0.1 m + 最多 90° 的增量）。这意味着即使右手问题解决，A 臂很可能也学不会到达目标；第 1 步按第三种分支推进："热启动（状态库中的稳定抓持/接近目标状态练习搬运）能否让从零学不会的任务变得可学"。
- 2026-11-05 约 20:50：右手诊断结论（0.2B 帧）：aligned 抬起 0.8%（223M 帧）、aligned-aa2deg 1.1%（206M 帧）、0b-adept 13%（202M 帧）。depenetration/手指摩擦和外展范围都不是右手落后的主因（剩余差异：手掌 32 块凸分解碰撞体 vs 原版单网格、速度上限、手掌中心与指尖偏移）。停止 aa2deg（`stopped.json`），不做外展受限的数据重新微调。
- 2026-11-05 约 20:55：第 1 步按第三种分支启动（左手从零也卡在抓起—掉落，右手从零抬起更弱）：A = `right_mustard_pristine_adept_aligned_s42`（继续运行）；S = `right_mustard_pristine_adept_aligned_S_s42`（右手 + 原版设置 + `--align-left-physics` + ADEPT + 状态库 v4.1，原生 40%，12000 轮；状态库恢复误差 1.9e-6）；D = `examples/arm_hand/run_mustard_arm_d_adept.py` 流水线已重新启动并改为 `--align-left-physics`、输出 `right_mustard_pristine_adept_aligned_D_s42`（等 bc4000 → carry200 → D）。三臂物理、观测、任务设置一致。
- 2026-11-05 约 21:20：`evaluate_reset_stages.py` 新增 `--obs adept`、`--align-left-physics`，并可直接读原始 `sapg_latest.pth`（网络结构取同目录 `sapg_config.json`）；`adept_obs.create_adept_env` 抽出 ADEPT 环境构建。A（aligned）第 1500 轮（196M 帧）固定评估 `outputs/stage_eval_adept/A_aligned_ep1500`：原生 0/128、掉落恢复 0/84、抓持起点 0/128、接近目标 0/128、示范附近 0/128 首目标；hold_command 分支抓持起点 128/128 持续保持（状态库在 aligned 物理下有效）。之后对 A/S/D 每约 2000 轮做一次同口径评估。
- 2026-11-05 约 22:40：ADEPT BC 完成（bc4000 末 MSE 1.1e-4；carry200 末 3.2e-5，`outputs/mustard_bc_adept_v1/bc_carry200`），D 自动启动：`right_mustard_pristine_adept_aligned_D_s42`（`actor_init_check.json`：obs_layout adept、σ 0.4、critic 预热 683 轮、归一化冻结；第 17 轮奖励 520）。
- 2026-11-05 约 22:40：固定评估 `outputs/stage_eval_adept/S500_A3000`（首目标/持续抓持，各 128 例，掉落恢复 84 例）：S 第 500 轮（65M 帧）抓持起点 38/127、接近目标 41/127、示范附近 35/126、原生 0/2、恢复 0/0；A 第 3000 轮（393M 帧）抓持 0/60、接近目标 0/82、示范附近 0/52、原生 0/1、恢复 0/0。状态热启动在 1/6 的帧数下已学会从抓持状态搬运到目标，A 仍完全不会；原生与掉落恢复两者都还是 0。
- 2026-11-06 约 01:30：左手 ADEPT（0b-adept）跑满 8000 轮（1.05B 帧）：奖励自约 330M 帧起停在约 389，首目标约 0.1%，抬起 67%——左手 + 芥末瓶从零在 1B 帧内也没学会原生任务。
- 2026-11-06 约 01:30：固定评估 `outputs/stage_eval_adept/S2000_D1000_A4500`（首目标，128 例/类，恢复 84 例）：S 第 2000 轮（262M 帧）抓持 58、接近目标 74、示范附近 61、原生 0、恢复 0；D 第 1000 轮（131M 帧，联合训练仅约 317 轮）抓持 69、接近目标 93、示范附近 52、原生 0、恢复 0；A 第 4500 轮（590M 帧）抓持 0、接近目标 1、示范附近 0、原生 0、恢复 0。原生起点三者持续抓持都只有 1/128：热启动解决了"抓住后搬运"，原生抓取仍是瓶颈（与 v4 系列相同）。
- 2026-11-06：`diagnose_finger_tracking.py`（不接触物体，按原生动作通道把手指目标设为 -1/0/0.3/0.6/1，120 步后测跟踪误差）：左手全闭合平均误差 16.6%，右手 14.8%（aligned 与否相同）；右手拇指 CMC_AA 全闭合时几乎不跟踪（误差约 100% 量程，左手 66%），小指 CMC 78%（左 51%）。参数按名称核对一致（刚度/阻尼/力矩一致，速度上限右 10、左 11.6–35 rad/s）。未发现能解释右手原生抬起 1% vs 左手 13% 的单一缺陷；另一差异：原版左手每指有独立的 elastomer 指垫碰撞体（摩擦 0.5，DP 1.5），右手只有 DP（1.5）。
- 2026-11-06 约 04:30：按固定评估分析，"恢复"一类（状态库 kind 1 中非示范来源：BC 失败 79 + 自然掉落 5）S/D 都是 0/84 持续抓持；其余 3600 个状态都来自示范接近过程（含 E4/E5 编辑），即训练里几乎没有"物体自由躺着、手不在示范接近路径上"的起点。新增 `online_resets.py`（kind 4）：在原生回合中，物体静止在桌上、未抬起、最近指尖距物体 <6 cm、回合步数 ≥20 时，从仿真截取状态（物理上天然有效）进 GPU 环形缓冲（容量 2 万）；重置时被分配为原生的环境以 30% 概率改从缓冲恢复。原生指标与容差课程仍只用 kind 0。`train_scratch.py` 新增 `--online-near-prob`、`--resume`（rl_games 完整恢复，轮数/帧数延续）和按 kind 的首目标率日志。冒烟（600 环境，从 S 第 3000 轮恢复 60 轮）：缓冲 1.2 万，恢复 112 次。启动 S-online：从 S 第 3000 轮（393M 帧）分支，`outputs/right_mustard_pristine_adept_aligned_S_online_s42`；S 本身继续作对照。
- 2026-11-06 约 05:00：固定评估 `outputs/stage_eval_adept/S3000_D2500`：S 第 3000 轮 抓持 63、接近目标 73、示范附近 65、原生 0、恢复 0；D 第 2500 轮 抓持 67、接近目标 85、示范附近 48、原生 0（持续抓持 1）、恢复 0。与上一轮相比基本持平，缓存起点已进入平台期。
- 2026-11-06 约 06:40：右手自碰撞定位（`diagnose_finger_tracking.py`，不接触物体）：关闭自碰撞后右手全部关节精确跟踪（误差 0），开启时全闭合误差 14.8%。逐组过滤：pinky–ring 过滤后小指完全正常，且**张开时**小指 CMC 被推到 0.089（命令 0，左手为 0），说明右手小指与无名指碰撞体在静止张开姿态就相互穿插——建模缺陷（MuJoCo 导出的碰撞体，原 MJCF 中这些对可能靠 contype/conaffinity 屏蔽，未出现在 exclude 列表里）。拇指单独闭合时左右手表现相同（都在接近满量程处被正常接触挡住），全手同时闭合时右手拇指外展被推到反向限位，过滤 thumb–index 可部分缓解，属于手指之间的真实挤占。新增 `self_collision_filters.py` 与 `train_scratch.py --self-collision-filter`（不改资产与 environment.py）。停止 A（第 8148 轮，1.07B 帧，抬起 1.2%、首目标 0，已超过 1B 参考预算，`stopped.json`），启动 A-pr：aligned + pinky–ring 过滤从零，`outputs/right_mustard_pristine_adept_aligned_pr_s42`，在 0.2B 帧与 0b-adept（13%）比较抬起比例。
- 2026-11-06 约 07:05：`audit_self_penetration.py`（MuJoCo 符号距离，跳过 Isaac 中已过滤的对）：右手在默认和原生重置姿态下，相邻手指（index–middle、middle–ring、ring–pinky 的 PP/MP/DP/MCP_VL）互相穿插 1–3.6 mm；原版左手相同检查只有掌–拇指 MC（Isaac 已过滤），手指间无穿插。即右手（MuJoCo 导出网格）静止时就持续存在指间接触力。新增过滤组 `adjacent-fingers`：过滤后右手默认/原生姿态无穿插；动态跟踪张开 0、中段 0.2%、全闭合 10.4%（左手 14.7%）。pinky–ring 单独版本（A-pr）启动约 15 分钟后改为该超集（`stopped.json`），启动 A-adj：aligned + adjacent-fingers 从零，`outputs/right_mustard_pristine_adept_aligned_adj_s42`，0.2B 帧比较抬起比例（0b-adept 13%，aligned 0.8%）。
- 2026-11-06 约 08:30：S-online 第 4376 轮（573M 帧）在线指标：kind 1/2/3 首目标 70%/85%/91%，kind 0（原生）与 kind 4（在线截取的原生近物体状态）均为 0。`check_online_states.py`（300 环境、1800 步、确定性动作）：kind 0 抬起 1.8%、95% 超时；kind 4 抬起 6.9%、98% 超时、几乎不掉落——策略在原生近物体状态下"停在物体旁边不抓"，而不是抓了又掉；kind 1–3 首目标 69–88%，结束多为后续目标中掉落。缓冲状态本身有效（恢复后无异常终止）。结论：瓶颈是原生构型下"开始抓取"这一行为从未被探索到，状态覆盖本身不足以诱导；需要的是抓取引导（预抓取姿态塑形或抓取课程）或修复右手模型后看从零能否学会抬起（A-adj 进行中）。
- 2026-11-06 约 09:30：A-adj（aligned + adjacent-fingers 过滤）0.2B 帧判定：206M 帧抬起 0.9%（0b-adept 202M 帧 13%）。指间穿插也不是右手原生抬起落后的主因。至此已排除：depenetration、手指关节摩擦、外展范围、小指/相邻手指自碰撞。剩余差异：手掌 32 块凸分解碰撞体 vs 原版单网格、手指速度上限、指垫 elastomer 碰撞体缺失、手掌中心/指尖偏移（影响奖励中的指尖距离项）。
- 当前状态汇总：热启动（S/D）在缓存起点上明显有效（S 第 3000 轮抓持/接近目标/示范附近首目标 63/73/65 of 128，A 同设定 1.07B 帧全 0），但原生起点和掉落恢复三臂都是 0；原生近物体状态（S-online kind 4）下策略停在物体旁不抓。能力判据未达成。
- 2026-11-06 约 09:50：停止 A-adj（判定已记录）。新增 `palm_hull.py` / `train_scratch.py --palm-hull`：在 USD 转换工作目录改写 URDF，把右手掌 32 块凸分解碰撞体换成它们的单个凸包（原版左手掌是单个网格，PhysX 按凸包处理）。启动 A-hull：aligned + palm hull 从零，`outputs/right_mustard_pristine_adept_aligned_hull_s42`，0.2B 帧与 aligned（0.8%）和 0b-adept（13%）比较抬起比例。
- 2026-11-06 约 13:15：A-hull（右手掌单凸包）149M 帧抬起 0.66%，与 aligned 相同，停止（`stopped.json`）。
- 2026-11-06：按用户提供的官方 Sharpa Wave 资产（`/home/szc/work/sharpa-urdf-usd-xml`，该仓库工作区为暂存删除状态且网格在 `origin/main` 的 `wave_01/` 下；用 `git archive origin/main` 导出到 `outputs/robot_assets/wave_01/`，未改动该仓库）。`compare_hand_kinematics.py`：官方右手与我们 MuJoCo 导出的右手在相同关节角下所有连杆位置差 <1 µm，只是连杆坐标系约定差 180°——关节空间的示范与状态库可直接迁移，指尖偏移在官方 DP 坐标系下数值不变，手掌坐标系相同。新增 `official_hand.py`：手臂与法兰安装取自我们的 URDF，手部全部连杆/关节/网格/惯量用官方文件（含每指 elastomer 指垫，手掌为单网格），`create_env` 复刻 environment.create_env（后者计入状态库哈希，未改）；另过滤指垫与 MP/DP（刚性附着）。`train_scratch.py --official-hand`。手指跟踪：张开误差 0.8%、全闭合 14.4%（左手 0/14.7%，旧右手 3%/14.8%）。启动 A-official：`outputs/right_official_pristine_adept_aligned_s42`（aligned + 官方手 + ADEPT，从零），0.2B 帧比较抬起比例（0b-adept 13%）。
- 2026-11-06 约 14:25：停止 S（第 6993 轮，916M 帧，S-online 自其第 3000 轮分支，作用重复；`stopped.json`，最终权重拷贝 `outputs/stage_eval_adept/ck/S_final.pth`）。启动关键对照 L-vendored：原版左手 + 芥末瓶 + ADEPT，但走我们的 `third_party/simtoolreal` 代码路径（`SIMTOOLREAL_PRISTINE=third_party/simtoolreal train_pristine_mustard.py`），3000 轮，`outputs/vendored_left_mustard_adept64_s42`。若它像原版仓库的左手一样在约 0.2B 帧抬起 ~13%，说明代码路径无问题、差异在右手资产；若失败，问题在 vendored 代码。
- 2026-11-06 约 16:20：A-official（官方右手）237M 帧抬起 0.78%，没有追上 0b-adept（202M 帧 13%）。L-vendored（原版左手走 vendored 代码）147M 帧抬起 0.53%，原版仓库同配置 142M 帧为 2.1%——需要看到约 250M 帧再判定是代码路径问题还是单 seed 偶然（0b-adept 是唯一学会抬起的从零运行）。
- 2026-11-07 约 13:50：L-vendored 247M 帧抬起 0.85%（原版仓库同配置 202M 帧 13%）——同一左手在 vendored 代码里没有复现。A-official 336M 帧 0.83%，停止（`stopped.json`）。启动 0b-adept seed 7（原版仓库代码，与 0b-adept 只差 seed），`outputs/pristine_mustard_adept64_s7`：若也在约 0.2B 帧抬起，说明差别在 vendored 代码；若不抬起，说明 0b-adept 的成功是单 seed 偶然。
- 2026-11-07 约 15:20：按用户确认停止 D 与 S-online（均为第 7500 轮，原生首目标率约 1B 帧内一直约 0.02%，`stopped.json`；checkpoint 拷贝到 `outputs/stage_eval_adept/ck/{D,Sonline}_final_ep7500.pth`）。最终固定评估 `outputs/stage_eval_adept/final_D7500_Sonline7500_S7000`（首目标 / 128，恢复 / 84）：D 第 7500 轮 抓持 125、接近目标 121、示范附近 118、原生 0、恢复 0；S-online 第 7500 轮 125 / 123 / 125 / 0 / 0；S 第 7000 轮 122 / 123 / 123 / 0 / 0。A（同设定从零 1.07B 帧）在这些起点上全为 0。原生起点三者持续抓持 0–1/128。
- 2026-11-07 约 16:00：seed 7 复现（原版仓库代码，左手 ADEPT）：297M 帧抬起 21.8%——0b-adept 的成功不是偶然。L-vendored（同左手走 vendored 代码，未加 physics_align）跑满 3000 轮 393M 帧抬起 0.94%。rl_games 两份完全相同；环境代码除机器人配置泛化和默认关闭的功能外，只有两处物理差异且对左手同样生效：①`scene_utils` 中机器人与物体 `max_depenetration_velocity` 10（原版 1000，来自 2026-09-07 Yimu 提交）；②`hand_actuator_extras` 给 KUKA_SHARPA_SPEC 加了手指关节摩擦 0.07–0.13（原版无）。启动 L-vendored-aligned（vendored 代码 + `--align-left-physics`，seed 42，与 0b-adept 同 seed），`outputs/vendored_left_mustard_adept64_aligned_s42`：若在约 0.2B 帧抬起，vendored 代码问题确认是这两项；之后所有实验固定使用 physics_align，并据此重新评估右手（aligned 右手组仍落后则另有右手原因）。
- 2026-11-07 约 17:00：用户要求在原版代码左手环境中用优化得到芥末瓶抓取轨迹并放入随机 reset。新增（均为新文件，原版仓库与既有默认行为不变）：`left_grasp_mpc.py`（原版 SimToolReal 左手 + 芥末瓶，关闭 DR 与自动重置；G 个原生初态 × S 条采样，MPPI：每次重规划把组内所有环境恢复到当前物理状态，H=40 步噪声动作（5 步结点线性插值），代价 = 最近 3 指尖距离 + 0.3×手掌距离 − 4×抬升(≤0.2 m) + 掉落惩罚，执行前 10 步）；`left_reset_bank.py`（从成功轨迹每 5 步取状态：接近 / 抓住 1–6 cm / 抬起 >6 cm，签名为左手 URDF+物体+任务 YAML；`StateResetBank` 与原生 reset 混合）；`train_pristine_mustard.py --state-bank/--bank-probabilities`（默认不启用；按 kind 记录首目标率，原生指标只统计 kind 0）；`run_left_grasp_mpc_batches.sh`。第一批 `outputs/left_grasp_mpc_v1`：16 个原生初态中 6 个成功（抬升 >5 cm 持续 ≥30 步，最高 0.25 m），约 6 分钟。放大：4 批 × 32 初态（seed 1–4）进行中，之后建 `outputs/left_reset_bank_v1`。
- seed 7（原版代码左手 ADEPT）跑满 3000 轮 393M 帧：抬起 47%，首目标 0，与 seed 42 同节奏。L-vendored-aligned 110M 帧抬起 0.3%（原版同期约 0.8%，仍在早期）。
- 2026-11-07 约 18:30：MPPI 放大：4 批 × 32 原生初态（seed 1–4）成功 11/6/11/10，加上第一批 6/16，共 44/144。状态库 `outputs/left_reset_bank_v1/reset_bank.npz`：1850 个状态（接近 139、抓住 464、抬起 1247）。`StateResetBank` 恢复后把抬升参考高度设为原生口径（桌面中心 z + table_object_z_offset），并对已超过抬升阈值的状态置 lifted（不白给一次性抬升奖励、抬起状态直接进入搬运奖励阶段）。启动左手 S-mpc：原版代码 + ADEPT + 状态库（原生 50% / 接近 20% / 抓住 15% / 抬起 15%），seed 42，8000 轮，`outputs/pristine_mustard_adept64_mpcbank_s42`；对照为同 seed 从零的 0b-adept（1.05B 帧首目标约 0）。
- 2026-11-07 约 22:40：S-mpc 第 3301 轮（432M 帧）：原生首目标约 0.03%，状态库各类首目标 0–0.4%，抬起比例停在约 0.15（同 seed 从零 430M 帧为 0.57），奖励 176（从零 390）——状态库反而拖慢。排查：`check_left_bank_hold.py` 恢复状态后保持控制目标 90 步，接近/抓住/抬起三类物体分别 100%/97%/94% 保持原位，恢复本身有效；原因是库内目标为原生首目标（absolute 采样），离物体中位 0.33 m、旋转 130°（右手状态库为示范目标 0.19 m、61°），状态库回合一开始就面对最难的目标。新增 `StateResetBank(delta_first_goal=True)` / `train_pristine_mustard.py --bank-delta-first-goal`：状态库回合的首目标改用原生 delta 采样器（0.1 m、≤90°，即原生任务后续每个目标的采样方式）围绕恢复后的物体位姿生成；原生回合不变。启动 S-mpc-delta：`outputs/pristine_mustard_adept64_mpcbank_delta_s42`（其余同 S-mpc）。L-vendored-aligned 跑满 3000 轮 393M 帧抬起 0.9%：恢复两项物理设置后 vendored 代码的左手仍学不会，代码中还有其他差异待查。
- 2026-11-07 约 23:40：S-mpc-delta 第 810 轮（106M 帧）状态库回合首目标：接近 15.1%、抓住 13.7%、抬起 10.6%；S-mpc 第 4296 轮（562M 帧）为 6.6%/0.9%/0.2%。原生两者都约 0。delta 首目标修正有效，继续观察能否迁移到原生。
- 2026-11-08 约 21:20：按用户确认停止 S-mpc（第 5111 轮，`stopped.json`）。启动长基线 A-left：原版代码 + 左手 + ADEPT 从零，seed 42（与 S-mpc-delta 相同），20000 轮（约 2.6B 帧，此前 0b-adept 只跑 1.05B），`outputs/pristine_mustard_adept64_baseline_long_s42`；第 1 步左手对比改为 A-left vs S-mpc-delta，同口径比较到达原生首目标 ≥1/3/10 的帧数。
- 2026-11-09：S-mpc-delta 跑到 2496 轮（327M 帧）各指标自 65M 帧起持平：状态库回合首目标 12–16%、原生约 0、抬起约 11%（同 seed 从零 300M 帧 46%）。`check_left_bank_hold.py` 核实：对状态库状态 + delta 首目标只保持控制目标不动，就有 12–27% 回合"到达"目标（初始关键点误差已在阈值内），即这 12–16% 是白得的成功，策略没有学到搬运；打开训练时的 DR（外力/延迟/噪声）后抓取仍有 90–96% 保持住（脆弱性不是主因）。修正：`StateResetBank._delta_goal` 对初始误差 ≤1.2×阈值的 delta 目标最多重采样 8 次。停止 S-mpc-delta（`stopped.json`），启动 S-mpc-delta2（同配置 + 去掉平凡目标），20000 轮，与长基线同帧数预算：`outputs/pristine_mustard_adept64_mpcbank_delta2_s42`。
- 2026-11-10：长基线（seed 42，2.62B 帧）全程抬起 0.9%、原生首目标约 0——与同 seed 的 0b-adept（202M 帧抬起 13%）结果不同：GPU 物理不可严格复现，从零学会抬起本身有随机性（3 次中 2 次）；此前所有单次运行的结论（含"vendored 代码有问题"）需要多 seed 确认。S-mpc-delta2（2.62B 帧）：抬起 74%，原生首目标约 0.04%，状态库回合首目标 2–3%（去除平凡目标后）且自 100M 帧起不再上升。
- 2026-11-10：问题 1（搬运为什么学不会）：`diagnose_pristine_rollout.py` 新增 `--state-bank`、按 kind 统计、位置/朝向误差（`outputs/diagnostics/delta2_final_perkind.json`，delta2 最终权重，512 环境，确定性）：原生起点 94% 回合能抓起；各类起点结束原因 74–80% 为掉落（fall），20–26% 超时；离目标最近时的位置误差中位 5–10 cm（已接近阈值），但此时朝向误差中位 66–78°。结论：平移基本能做到，瓶颈是手内转向（目标要求最多 90° 旋转），且转不到位时物体最终掉落。
- 问题 2（目标难度课程）：新增 `goal_curriculum.py`（RotationCurriculum：首目标朝向 = slerp(重置时物体朝向, 原生随机朝向, α)，后续目标 delta_rotation_degrees = 90°×α；原生回合首目标率 50 轮平均 ≥0.2 且距上次 ≥100 轮时 α+0.1，只升不降；α=1 即原任务），`train_pristine_mustard.py --rot-curriculum`（默认关闭）。
- 问题 3（多 seed）：`run_rotcurr_matrix.sh` 启动 4 组（A = 从零，S = + MPPI 状态库；均带旋转课程；seed 42 与 7；12000 轮约 1.57B 帧）：`outputs/pristine_mustard_adept64_rotcurr_{A,S}_s{42,7}`。
- 问题 4（代码路径复核）：GPU 已满（4 组共约 72 GB），排在任一组完成或判定后运行 vendored 代码左手 seed 7 对照。
