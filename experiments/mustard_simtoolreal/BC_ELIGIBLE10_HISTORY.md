# 当前BC运行：合格示范10条，4000 epoch

输出目录：`/home/szc/work/spider/outputs/mustard_bc_native_eligible10_4000_v2`。
启动脚本：`/home/szc/work/spider/examples/arm_hand/run_mustard_bc_eligible10_4000.sh`。
2026-10-28启动，持久执行会话28263。查询时以输出目录和实际进程为准。
首次后台v1在第32轮意外退出，没有正常结束记录；原因未确认。当前v2重新训练，
缓存BC仅启动PyTorch，独立评估进程启动Isaac Sim。

选择集：`outputs/mustard_bc_eligible_v1/selection.json`。
ID：62、64、87、111、118、149、159、163、215、278。
每条保留533、600、579、577、591、583、532、542、523、587个动作，共5647个有效帧。
保留首次成功前的动作：缓存观测t是动作t-1之后的状态，成功状态只用于验证，
不把成功后应重新采样目标的动作继续加入BC。其他90条不训练。

资格条件：600步内累计10个near步，角点最大距离使用原版初始容差0.075×1.5=0.1125m；
在首次成功前，物体抬升>=5cm、持续持握几何代理>=30步。手指和掌心距离由原生缓存
重建，有限差分速度要求<1.5m/s，比原回放2m/s门槛更保守。并非最终1cm精度认证。
此前9条保守候选未读取缓存的第600个post-action状态；正确对齐后包含第600步成功
的candidate64，总计10条，与在线post-step统计一致。

纯BC，Adam1e-4，4000完整遍历，seq16/burn-in16，六个原版SAPG系数条件ID轮换。
无PPO/GAE/reward优化，无central critic训练；未使用value head、sigma和输入统计冻结。
每100轮保存actor_init.pt和bc_optimizer.pt。后者不能直接当SAPG优化器恢复。

本轮使用已验证的120环境采集缓存；BC阶段不创建8184物理环境。
8184=6×1364仅保留为后续SAPG配置元数据，不能声称本轮BC中有8184环境在运行。

W&B： https://wandb.ai/shezhangcong-johns-hopkins-university/simtoolreal-warmstart/runs/izou0f3r
仅上传BC和eval标量。第1轮、每100轮、最后一轮尝试录制6环境、600帧本地视频；
已有评估运行时跳过中间触发，最终轮等待并保证另录最终模型。采用所选10条目标池、
最终目标，以及原版随机机器人reset/延迟/噪声；没有把teacher初始化当原版reset。
评估失败不等于BC反向传播失败，也不能凭BC损失认定闭环任务成功。

日志：输出目录同名`.launch.log`；bc_progress.jsonl；video_evaluator.log。
completed.json确认4000轮结束；failed.json记录失败。
不自动进入critic校准或联合SAPG阶段。
