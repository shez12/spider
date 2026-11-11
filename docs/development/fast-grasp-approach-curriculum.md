# 用管线示范做起点课程：从完全随机起点快速学会抓起

更新：2026-11-11。环境：原版 SimToolReal（左 KUKA + Sharpa，`/home/szc/work/newsimtoolreal`），原尺寸芥末瓶，奖励、观测、域随机化、干扰力、SAPG 参数都不改。逐条记录见 `mustard-warmstart-plan-20261104.md` 的"进展记录"。

## 1. 一句话

把管线示范里"手相对瓶子"的抓法搬到每个随机起点的瓶子上，用脚本在训练环境里从原生起点实际走一遍，取抬起之前路径上的仿真状态当一部分回合的起点；起点随抓起率逐级退回原生起点，退完后只剩随机起点。

## 2. 结果（seed 42 / 7）

原生随机起点确定性回放（512 环境、1200 步），第 2500 轮存档（327M 帧）：

| 组 | seed 42 | seed 7 |
|---|---|---|
| 起点课程 | 85.3% | 76.5% |
| 从零（同时跑） | 1.3% | 1.7% |

- 训练中随机起点回合的抓起率首次过 50%：课程 s42 在 250M 帧，s7 在 310M 帧；从零两组到 340M 帧都约 1.4%（之后到 393M 帧的日志我还没读）。
- 课程退到只剩随机起点：s42 在 183M 帧，s7 在 254M 帧。
- "抓起"指触发环境的"已抬起"标志（瓶子中心高出参考约 20 cm）。

**还不能说的话**

- 不能说"更快"：历史上学会了的从零（s42r）在 262M 帧训练日志抬起 40%，课程 s42 在 260–280M 帧是 33%。
- 目前的证据是学会的比例：起点课程 2/2，从零 2/5（含历史 3 次）。样本小，seed 1、2、3 在补。
- 抓起后握不住：抬起后保持步数中位约 63 步，首目标率约 0。录像里手举起后约 0.5 秒张开。
- 四处改动（见第 4 节）是一起改的，没有逐项对照。

## 3. 流程

### 第 1 步：提取"手相对瓶子"的抓法（CPU）

`experiments/mustard_simtoolreal/demo_relative_grasps.py extract` → `outputs/demo_relative_v1/demos.npz`

- 输入：管线有效轨迹 `outputs/mustard_multisource_valid100/accepted.json`（478 条、10 个示范源），每源最多取 8 条，共 52 条。
- 关节镜像到左手（手臂取反、第 7 关节减 30°、手指按关节名照搬并夹到左手限位）；瓶子位姿换到 Isaac 资产坐标系并做同样的镜像。
- 用左臂 URDF 正运动学得到每帧手腕位姿。
- 保留：手腕位姿轨迹、每帧手指关节目标、瓶子起始位姿、抬起帧（瓶子升高 >1 cm 的第一帧）。丢弃：示范的手臂关节角和瓶子在桌上的位置。
- 复核：抬起帧拇指到无名指的指尖连杆离瓶面中位 2.0–3.1 cm。

### 第 2 步：搬到随机起点并用脚本重放（Isaac，不训练）

`experiments/mustard_simtoolreal/replay_demo_grasps.py` → `outputs/demo_relative_v1/replay_bank_s{1,2}/replay.npz`

- 原生重置后静置 60 步。
- 对每个示范求刚体变换（示范瓶子起始位姿 → 当前瓶子位姿），只接受倾斜 ≤15° 的（朝下的面一致），允许瓶子绕长轴转 180°。
- 逆解手臂，选离原生手臂构型最近的示范。路径：预抓取位上方 8 cm → 预抓取位（抬起帧前 50 帧）→ 示范之后每隔一帧。
- 用环境的 29 维动作接口驱动，手臂受 0.0025 rad/步（0.15 rad/s）限速；手指用示范同帧目标。域随机化和干扰力关闭。
- 每步保存完整仿真状态。

数字（2 个种子 × 4096 个随机起点）：抓握位有逆解约 33%；整条路径有解并重放 1452 个（18%）；其中 881 个（61%）触发"已抬起"。

### 第 3 步：构建起点状态库（CPU）

`experiments/mustard_simtoolreal/approach_curriculum.py build` → `outputs/approach_bank_v1/approach_bank.npz`

- 只用触发了"已抬起"的路径。
- 把"原生起点（0）→ 脚本开始抬起（1）"等分 10 级，各取一个状态。
- 剔除开始抬起前瓶子已离桌超过 1 cm 的路径（2 条）。
- 结果：879 条路径、8790 个状态；823 条来自第 2 源（瓶子宽面朝下躺着）。各级手臂离原生构型的最大关节差中位 0.25–0.69 rad。

### 第 4 步：带起点课程训练

`train_pristine_mustard.py --approach-bank <bank>`（对照：`--episode-stats`）

- 30% 的回合从库内级别在 (stage−0.2, stage] 的状态起步，70% 始终是原生随机起点。
- stage 从 1 开始；当前 stage 的库内回合满 2000 个且 ≥50% 触发"已抬起"，就减 0.1；到 0 后关掉状态库。
- 库内回合沿用该路径原生起点的目标和抬起参考高度。
- 日志新增 `native_episode_lift_rate`（每个结束的原生回合是否触发过"已抬起"）、`approach_stage`；课程变化写 `approach_changes.jsonl`。

### 第 5 步：评估

只用原生随机起点的确定性回放（`outputs/diagnostics/native_replay_2100/run.sh`），不用训练日志的抬起比例。

## 4. 和之前状态库做法的差别

| | MPPI 库 / 镜像库 delta2 | 起点课程 |
|---|---|---|
| 状态来源 | MPPI 快照；或右手管线状态整体镜像 | 示范的相对抓法 + 训练环境里从原生起点的脚本路径 |
| 取哪一段 | 接近、已抓住、已抬起 | 只取抬起之前 |
| 混入比例 | 固定 50% | 30%，随抓起率退到 0 |
| 库内回合的目标 | 在瓶子附近重采 | 沿用原生起点的目标 |

镜像库的手臂离原生构型约 3.2 rad，随机起点回放抓起率 393M 帧时 1.0%。

## 5. 已知局限

- 覆盖面窄：约 1/5 的随机起点有完整路径，状态几乎都来自一个示范。
- 用到的是示范的抓握位置和手型，不是它的速度（示范手臂峰值约 3.5 rad/s，环境限速 0.15 rad/s）。
- 课程在第 0.8 级卡了 80–110M 帧，是主要的时间开销。
- 第 1.0 级只确认了瓶子还在桌上，没有逐个核对手指是否贴住瓶子。

## 6. 复现命令

从 `/home/szc/work/newsimtoolreal/simtoolreal` 运行，使用 conda Python，并设置 `LD_LIBRARY_PATH`、`OMNI_KIT_ACCEPT_EULA`、`TMPDIR`、`XDG_CACHE_HOME`（见 `outputs/diagnostics/queue_20261111/queue.sh` 开头）。

```bash
S=/home/szc/work/spider/experiments/mustard_simtoolreal; O=/home/szc/work/spider/outputs
python $S/demo_relative_grasps.py extract
python $S/replay_demo_grasps.py --headless --output $O/demo_relative_v1/replay_bank_s1 --envs 4096 --seed 1 --steps 1100 --min-steps 4
python $S/approach_curriculum.py build --replays $O/demo_relative_v1/replay_bank_s*/replay.npz --output $O/approach_bank_v1/approach_bank.npz
python -u $S/train_pristine_mustard.py --headless --output $O/<run> --num-envs 8184 --epochs 3000 --seed 42 \
  --object-repr adept --pc-points 64 --approach-bank $O/approach_bank_v1/approach_bank.npz
```

代码都未提交。
