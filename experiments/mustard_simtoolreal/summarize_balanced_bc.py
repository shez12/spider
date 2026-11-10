"""Write an evidence-linked experimental decision and recovery-state requests."""
import argparse,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();root=a.run.resolve()
    report=json.loads((root/'evaluation/results.json').read_text());audit=json.loads((root/'audit/audit.json').read_text())
    progress={name:[json.loads(s) for s in (root/name/'bc_progress.jsonl').read_text().splitlines()] for name in ('baseline','balanced')}
    attempts=[rows[-1]['optimizer_steps']+rows[-1]['amp_skipped_updates'] for rows in progress.values()]
    assert attempts[0]==attempts[1] and all(len(rows)==200 and sum(r['supervised_frames'] for r in rows)==75461800 for rows in progress.values())
    lines=['# BC抓取分布均衡：同预算对照','',
        '同一初始化、seed42、carry LSTM、LR1e-5，各200轮；每组75,461,800个有效监督帧。现有SAPG没有被替换。',
        '数据审计、权重、逐轮日志和逐例配对评估都保存在本目录。聚类是抓持构型代理，不是人工抓法标签。','',
        '## 数据分布',
        '503条：局部Z轴朝上388条，侧向115条，朝下0条；8组构型26～147条。',
        '原始1026条中侧向470条，16秒完整路径筛选后侧向剩115条。排除的轨迹中296条在16秒内已抓持，其中168条侧向；这些前缀尚未混入本次A/B。',
        f"逐帧权重范围 {audit['weight_min']:.3f}～{audit['weight_max']:.3f}；全部有效帧保留，补齐帧排除。",'',
        '## 训练预算和结果']
    for name,rows in progress.items():
        last=rows[-1];lines.append(f"- {name}: 原始MSE={last['action_balanced_mse']:.8g}，加权MSE={last['weighted_action_mse']:.8g}；更新尝试{attempts[0]}，成功更新{last['optimizer_steps']}，AMP跳过{last['amp_skipped_updates']}。")
    lines+=['','## 配对闭环诊断',
        '每例16秒、确定性leader动作、零初始LSTM，目标保持不变且不自动reset。示范503例、原版随机物体256例、掉落14例；不是原版50目标回合成功率。',
        f"初始观测配对最大误差 {report['paired_initial_observation_max_error']:.3g}。抓持使用连续30步几何代理；到达用原版tolerance0.075×1.5={report['effective_keypoint_tolerance_m']:.4f}m缩放角点距离、累计10步。",
        '原版物体从桌上方生成，抓起高度以运行中最低高度为基准；示范和掉落使用各自初始高度。没有恢复PhysX接触缓存。']
    for name,model in report['models'].items():
        lines.append(f'\n### {name}')
        for suite,row in model['suites'].items():lines.append(f"- {suite}: 抓持 {row['grasped']}/{row['cases']} ({row['grasp_rate']:.1%})；到达 {row['reached']}/{row['cases']} ({row['reach_rate']:.1%})。")
        for stratum,row in model['strata'].items():
            lines.append(f"- 示范初态层{stratum}: 抓持{row['grasped']}/{row['cases']}，到达{row['reached']}/{row['cases']}。")
    lines+=['','## 均衡减原始：配对差']
    for suite,metrics in report['paired_differences'].items():
        for metric,row in metrics.items():
            low,high=row['paired_bootstrap_95'];lines.append(f"- {suite}/{metric}: {row['balanced_minus_baseline']:+.1%}，配对案例bootstrap 95%区间[{low:+.1%}, {high:+.1%}]，胜{row['wins']}、负{row['losses']}。")
    lines+=['','置信区间只描述这批配对案例，不包含训练seed不确定性；全零样本的bootstrap不能证明总体成功率恒为零。示范评估复用了训练初态，不代表泛化。',
        '', '## 决策']
    delta=report['paired_differences'];native=delta['native_object']['reached'];demo=delta['demo']['reached']
    if native['paired_bootstrap_95'][0]>0:
        decision='native_gain_requires_second_seed';lines.append('随机物体目标到达改善，需第二训练seed复现后才能推荐替换初始化。')
    elif demo['paired_bootstrap_95'][0]>0:
        decision='demo_gain_only_requires_second_seed';lines.append('示范目标到达改善；随机物体收益尚未证实，先用第二seed确认示范收益，不能直接替换正式SAPG。')
    else:
        decision='no_confirmed_gain_keep_existing';lines.append('本次没有确认均衡带来目标到达改善，保留现有初始化和正在运行的SAPG；不因加权损失下降宣称效果更好。')
    lines.append('若随机物体/掉落仍普遍失败，优先补成功恢复与缺失朝向覆盖；仅重采样已有成功首次抓取无法填补这些状态。16秒内成功抓持前缀是独立的数据扩充候选，需要单独对照。')
    final=np.load(root/'evaluation/evaluation_final_states.npz');names=final['model_names'].tolist();model_index=names.index('balanced');n=int(final['cases_per_model'])
    selected=[i for i,row in enumerate(report['models']['balanced']['cases']) if row['suite'] in ('native_object','drop') and not row['grasped']]
    indices=np.array(selected,dtype=int)+model_index*n
    values={k:final[k][indices] for k in ('candidate_ids','q','v','previous','current','root','goal','obs','h0','h1')}
    values['case_indices']=np.array(selected,dtype=int)
    np.savez_compressed(root/'recovery_requests.npz',**values)
    (root/'decision.json').write_text(json.dumps(dict(decision=decision,recovery_cases=len(selected),evaluation=str(root/'evaluation/results.json')),indent=2)+'\n')
    lines+=['',f'已导出{len(selected)}个均衡模型失败后的物理状态到 recovery_requests.npz，供当前环境优化/恢复示范使用；这些是失败状态请求，不是新增成功训练标签。']
    (root/'FINDINGS.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))


if __name__=='__main__':main()
