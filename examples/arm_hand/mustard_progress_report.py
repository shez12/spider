"""Append a one-screen progress snapshot of the active mustard runs to outputs/mustard_progress_reports.md."""
import json,subprocess,time
from pathlib import Path
OUT=Path('/home/szc/work/spider/outputs')


def tail(path):
    try:return json.loads(path.read_text().splitlines()[-1])
    except Exception:return None


def stage_evals(run,label):
    rows=[]
    for res in sorted((run/'stage_evaluations').glob('epoch_*/results.json')):
        m=json.loads(res.read_text())['models'].get('after_'+label)
        if m:rows.append(f"{res.parent.name}: "+', '.join(f"{k} {v['first_goal_reached']}/{v['cases']}" for k,v in m['suites'].items()))
    return rows[-2:]


def main():
    lines=[f"## {time.strftime('%Y-%m-%d %H:%M')}"]
    gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
    lines.append(f'- GPU: {gpu}')
    bc=OUT/'mustard_bc_v3b'
    if bc.exists():
        state=json.loads((bc/'pipeline_state.json').read_text()) if (bc/'pipeline_state.json').exists() else {}
        parts=[f"BC v3b stage={state.get('stage')} status={state.get('status')}"]
        for s in ('bc4000','bc_carry200'):
            r=tail(bc/s/'bc_progress.jsonl')
            if r:parts.append(f"{s} epoch {r['epoch']} mse {r['action_balanced_mse']:.2e}")
        if (bc/'pipeline_failed.json').exists():parts.append('FAILED: '+(bc/'pipeline_failed.json').read_text().strip())
        res=bc/'closed_loop_eval/results.json'
        if res.exists():
            for name,m in json.loads(res.read_text())['models'].items():
                parts.append(f"{name}: "+', '.join(f"{k} grasp {v['grasped']}/{v['cases']} reach {v['reached']}" for k,v in m['suites'].items()))
        lines.append('- '+' | '.join(parts))
    for run,label in [(OUT/'mustard_scratch_baseline_s42','scratch'),(OUT/'mustard_sapg_v5/joint','v5')]:
        r=tail(run/'sapg_progress.jsonl')
        if not r:continue
        goals=r.get('curriculum_mean_completed_goals');native=r.get('completed_leader_mean_goals_native')
        s=f"- {run.relative_to(OUT)}: epoch {r['epoch']}, frames {r['frames']/1e9:.3f}B, reward {r.get('mean_reward') or 0:.0f}, mean goals {goals if goals is None else round(goals,3)}"
        if 'pose_alpha_max' in r:s+=f", alpha {r['pose_alpha_max']}, last window {r.get('pose_last_window_rate')}"
        if native is not None:s+=f", native-kind goals {native:.3f}"
        s+=f", tol {r.get('success_tolerance')}, sigma {r.get('policy_sigma_mean',0):.3f}"
        if (run/'failed.json').exists():s+=' FAILED'
        lines.append(s)
        lines.extend('  - '+e for e in stage_evals(run,label))
    v5=OUT/'mustard_sapg_v5'
    if v5.exists() and (v5/'pipeline_state.json').exists():lines.append('- v5 pipeline: '+(v5/'pipeline_state.json').read_text().replace('\n',' '))
    with (OUT/'mustard_progress_reports.md').open('a') as f:f.write('\n'.join(lines)+'\n\n')
    print('\n'.join(lines))


if __name__=='__main__':main()
