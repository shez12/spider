# Mustard 多样reset试验 v1

执行日期：2026-11-03。借鉴OmniReset的状态库设计，不声称复现论文的完整算法或灵巧手结果。

## 目标和边界
- 减少新增整条优化轨迹的需要，用现有原生Isaac缓存和失败状态构建可恢复状态库。
- 原有暂停的数据生成进程保持暂停；现有native-object SAPG继续作为在线参考。
- actor原有140维观测、29维动作、LSTM、6×1364环境、原生奖励、50目标上限不变。
- 试验比例：native .40、near-object .25、stable grasp .20、near-goal .15；每个block采用相同概率。
- 不是四类状态的均匀并集采样复现；比例为本项目首轮试验设定。

## 状态库
- 从503条原生缓存提取接近物体、连续抓持30步之后和接近目标但尚未成功的帧。
- 加入已保存14个自然掉落状态和269个BC诊断失败末态作为候选，而非成功标签。
- 保留机器人q/v、object root pose/velocity、previous/current target、table pose/height、goal、来源和类别。
- 候选物体线速度从缓存有限差分估计，角速度以零初始化；最终库保存短时稳定后的真实Isaac线速度/角速度。xyzw/wxyz显式转换；不把候选估计值当传感器真值。
- 首版不添加未经验证的空间扰动。程序化扩充需以本版验证结果为依据。
- Isaac短时动力学保持测试；几何检查使用导出的原生URDF在MuJoCo中只算FK/接触距离，先与Isaac保存的31个连杆位置核对（本轮最大误差2.27e-6m）。保持1mm机器人桌面/自碰限制，机器人物体/物体桌面限制3mm。PhysX直接接触查询探针阻塞，未用于合格结论；不宣称MuJoCo接触距离与PhysX完全一致。
- 持续执行原生保持目标动作验证，排除异常速度、关节越界、物体掉出工作空间；stable grasp必须持续保持，near-goal在有效阈值之外。
- 再次restore并独立重放验证；不宣称恢复了PhysX接触缓存。
- 状态库和环境/数据SHA256绑定，写入审核报告；某类为空则修正候选，不以其他类偷偷补满。

## 训练接入和验证
- reset先执行原版基础清理，再对选中的env恢复库状态；刷新scene与仿真缓存，重新初始化reward/goal trackers和计时。
- 不恢复旧成功数/奖励；新回合LSTM清零；状态库ID不加入actor观测。
- near-goal确保距离大于有效成功阈值；成功后使用原版delta目标采样与结束条件。
- GPU缓存状态，避免训练中CPU逐条传输；记录每block各类采样数及当前回合类别。
- 600环境3轮短测，确认actor/critic更新、reset局部env不污染其余env、采样比例和统计有限。
- 从同一固定checkpoint运行native-only与mixed-reset各300轮，比较同新增帧数；固定16s/tolerance .075以隔离reset效应。
- 每个分支首轮与末轮保存权重；原版reset配对评估、掉落恢复评估和独立视频。训练内混合reset成功率不能冒充原版成功率。
- 有限对照窗口后评估决定是否继续长训，不能仅凭near-goal训练奖励升高判定改进。

## 交付证据
状态库、两次验证报告、短测证明、固定初始化与运行参数、两分支训练日志/W&B、原版reset视频和评估结果。

## 本轮执行记录

- 状态库：`outputs/mustard_diverse_reset_bank_v1/reset_bank.npz`，2073状态：near-object941、grasp770、near-goal362；native在线采样。
- near-object中包含5个自然掉落来源与79个BC失败来源的有效状态；其余来自原生示范缓存。各类别仍偏向已有抓法，未补齐全SO(3)。
- 合格检查包括30步稳定+独立restore后30步保持、31连杆FK核对、三次离散几何快照；不是连续时间无碰撞证明。
- 机器人桌面/自碰上限1mm，物体接触3mm；持物网格最低点距桌面至少2cm；reset成功距离大于0.125m。
- 600环境3轮短测：`outputs/mustard_diverse_reset_ab_v1/smoke_mixed_early`，已通过局部reset不污染、恢复精度与actor/critic更新检查。
- 两分支目录：`outputs/mustard_diverse_reset_ab_v1/native_only`、`mixed_reset`。各8184环境，从相同10000轮快照新增300轮。
- 对照与混合组并行运行，原长训练继续；比较新增帧数与闭环表现，不据共享GPU墙钟比较速度。
- W&B ID前缀为mustard_diverse_reset_ab_v1_；通用native_only临时run已由唯一ID的分支run替代并补齐标量。
- 协调器：`examples/arm_hand/run_mustard_diverse_reset_ab_v1.py`；评估器：`finish_mustard_diverse_reset_ab_v1.py`。完成两分支后自动执行773例/模型的固定原版reset/示范/掉落配对诊断，并生成FINDINGS.md。
- 长训练不等于已经证明有效；300轮结果和原版reset视频仍需等待。
