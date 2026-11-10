"""Rebuild conservative acceleration evidence from recorded experiment artifacts."""
import argparse
import datetime
import json
from pathlib import Path


def production_cohorts(campaign):
    """Observational live evidence; never confuse selected successful examples with a cohort."""
    from examples.arm_hand.mustard_valid100 import valid
    buckets={}
    for runtime_path in sorted((campaign/'groups').glob('*/candidate_*/gpu/acceleration_runtime.json')):
        runtime=json.loads(runtime_path.read_text())
        if runtime.get('strategy')!='world_model_feedback':continue
        case=runtime_path.parent.parent;state_path=case/'state.json'
        if not state_path.exists():continue
        state=json.loads(state_path.read_text())
        if state.get('acceleration',{}).get('pilot_admission'):continue
        key=(case.parent.name,runtime.get('policy_sha256'))
        records=buckets.setdefault(key,[])
        terminal=state.get('status') in ['accepted','physics_rejected','physics_failed']
        accepted=bool(state.get('validated_success') and valid(state.get('selected_metrics',{})) and valid(state.get('replay_metrics',{})))
        records.append(dict(case=str(case),status=state.get('status'),terminal=terminal,accepted=accepted,
            physics_s=state.get('physics_elapsed_s'),validation_s=state.get('validation_elapsed_s'),ik_s=state.get('ik_elapsed_s'),
            started_at=state.get('physics_started_at_utc')))
    cohorts=[]
    for (group,policy),records in buckets.items():
        finished=[r for r in records if r['terminal']];physics=sum((r['physics_s'] or 0)+(r['validation_s'] or 0) for r in finished)
        accepted=sum(r['accepted'] for r in finished)
        cohorts.append(dict(group=group,policy_sha256=policy,completed=len(finished),accepted=accepted,rejected=len(finished)-accepted,
            unfinished=len(records)-len(finished),recorded_finished_physics_validation_s=physics,
            accepted_per_recorded_service_hour=accepted*3600/physics if physics else None,cases=records,
            causal_speedup_proven=False,limitations='Observational production cohort; includes every completed model attempt, but omits unfinished cost, preparation rejects and training setup. The service-time rate is not end-to-end throughput or a matched baseline comparison.'))
    return cohorts


def prospective_cohorts(root):
    summaries=[]
    for path in sorted(root.glob('prospective_*/plan.json')):
        plan=json.loads(path.read_text());modes={mode:dict(planned=0,terminal=0,accepted=0,execution_wall_s=0.,queue_s=0.) for mode in ['raw','bc','imagination']};cases=[]
        for job in plan['jobs']:
            mode=job['mode'];modes[mode]['planned']+=1
            execution=path.parent/f"{job['source_index']:03d}_{job['candidate_id']}_{mode}"/'execution.json'
            report=json.loads(execution.read_text()) if execution.exists() else {}
            terminal=report.get('status')=='complete'
            if terminal:
                modes[mode]['terminal']+=1;modes[mode]['accepted']+=bool(report.get('accepted'))
                modes[mode]['execution_wall_s']+=report.get('wall_s',0.)
                modes[mode]['queue_s']+=sum(t.get('queue_s',0.) for t in report.get('timing_breakdown',{}).values())
            cases.append(dict(source=job['source_index'],candidate=job['candidate_id'],mode=mode,execution=str(execution),status=report.get('status','not_started'),accepted=report.get('accepted',False)))
        complete=all(m['terminal']==m['planned'] for m in modes.values())
        summaries.append(dict(plan=str(path),complete=complete,modes=modes,cases=cases,models=plan['models'],mps_pipe_dir=plan['mps_pipe_dir'],
            end_to_end_acceleration_proven=False,limitations='Fixed six-case development cohort, 18 planned runs. Charge all failures, preparation and training setup. Incomplete arms are not zero-cost failures or finished comparisons; no cohort conclusion before all planned runs are terminal.'))
    return summaries


def build(root):
    experiments=[]
    for path in sorted(root.rglob('execution.json')):
        d=json.loads(path.read_text())
        experiments.append(dict(name=path.parent.relative_to(root).as_posix(),path=str(path),case=d.get('case'),
            controller=d.get('controller'),status=d.get('status'),accepted=d.get('accepted',False),
            wall_s=d.get('wall_s'),timing_breakdown=d.get('timing_breakdown',{})))
    for record in experiments:
        record['charged_increment_wall_s']=record['wall_s']
        launcher=root/(record['name']+'.launcher.json')
        if launcher.exists():
            launch=json.loads(launcher.read_text())
            if launch.get('shared_prefix_execution') and '--resume' in launch.get('args',[]) and record['wall_s'] is not None:
                prefix=json.loads(Path(launch['shared_prefix_execution']).read_text())
                record['charged_increment_wall_s']=max(0.,record['wall_s']-prefix.get('wall_s',0.))
                record['shared_prefix_charged_once_at']=launch['shared_prefix_execution']
    pairs=[]
    for source in [7,10,12,15,18]:
        for variant in ['model','contact']:
            names=[f'fresh_source{source:03d}_{suffix}' for suffix in [variant,'raw']]
            records=[next((e for e in experiments if e['name']==n),None) for n in names]
            model,raw=records
            if source==15 and raw is None:
                raw=next((e for e in experiments if e['name']=='balanced_ablation_source015/raw'),None);records=[model,raw]
            if variant=='contact' and model is None:continue
            comparable=bool(model and raw and model['case']==raw['case'] and
                all(e['status']=='complete' and e['accepted'] and e['wall_s'] for e in records))
            protocol_match=False
            if comparable:
                execution=[json.loads(Path(e['path']).read_text()) for e in records]
                results=[json.loads((Path(e['path']).parent/'gpu/result.json').read_text()) for e in records]
                fields=['seed','samples','max_iterations']
                protocol_match=(all(execution[0].get(k)==execution[1].get(k) for k in fields) and
                    results[0]['signature']==results[1]['signature'] and
                    all(results[0]['arguments'].get(k,default)==results[1]['arguments'].get(k,default) for k,default in [('strict_check_every',0),('strict_candidate_pool',0),('disable_target_stop',False)]))
            pairs.append(dict(source=source,variant=variant,model=model,raw=raw,both_accepted_same_case=comparable,
                matched_protocol=protocol_match,execution_speed_ratio=raw['wall_s']/model['wall_s'] if comparable and protocol_match else None))
    ablations=[]
    for path in sorted(root.glob('*/ablation.json')):
        a=json.loads(path.read_text());runs={}
        for run in a.get('runs',[]):
            execution=Path(run['execution'])
            if execution.exists():runs[run['mode']]=json.loads(execution.read_text())
        comparable=(set(runs)=={'raw','bc','imagination'} and len({r.get('case') for r in runs.values()})==1 and
            all(r.get('status')=='complete' and r.get('accepted') and r.get('wall_s') for r in runs.values()))
        service={mode:sum(t.get('process_s',0.) for t in report.get('timing_breakdown',{}).values()) for mode,report in runs.items()}
        ablations.append(dict(path=str(path),status=a['status'],all_three_accepted_same_case=comparable,
            recorded_service_s=service,observed_wall_ratio_including_queue_raw=runs['raw']['wall_s']/runs['imagination']['wall_s'] if comparable else None,
            observed_wall_ratio_including_queue_bc=runs['bc']['wall_s']/runs['imagination']['wall_s'] if comparable else None,
            service_ratio_excluding_queue_raw=service['raw']/service['imagination'] if comparable and service.get('imagination') else None,
            service_ratio_excluding_queue_bc=service['bc']/service['imagination'] if comparable and service.get('imagination') else None,
            causal_acceleration_proven=False,limitations='Sequential runs share load that changes over time. Queue differences are not a model-specific benefit; service rates still require matched prospective confirmation and full setup amortization.'))
    training=[]
    for path in sorted(root.glob('*/training.json')):
        d=json.loads(path.read_text())
        training.append(dict(path=str(path),elapsed_s=d.get('elapsed_s')))
    proposals=[]
    for path in sorted(root.rglob('proposal.json')):
        d=json.loads(path.read_text());proposals.append(dict(path=str(path),elapsed_s=d.get('elapsed_s')))
    collection=[]
    counterfactual=root/'counterfactual_v1/collection.json'
    if counterfactual.exists():
        d=json.loads(counterfactual.read_text());collection.append(dict(path=str(counterfactual),kind='counterfactual collection',measured_wall_s=d.get('wall_s')))
    for path in sorted((root/'exploration_v1').rglob('result.json')):
        d=json.loads(path.read_text());collection.append(dict(path=str(path),kind='exploration optimizer loop only; process setup not measured',measured_wall_s=d.get('optimization_elapsed_s')))
    ik=[]
    for path in sorted((root/'fresh_eval_cases').glob('*/candidate_*/state.json')):
        d=json.loads(path.read_text());ik.append(dict(path=str(path),source=d.get('source_index'),
            accepted=d.get('ik_pass',False),elapsed_s=d.get('ik_elapsed_s')))
    return dict(generated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),production_cohorts=production_cohorts(root.parent/'mustard_multisource_valid100'),prospective_cohorts=prospective_cohorts(root),pairs=pairs,ablations=ablations,physical_experiments=experiments,training_runs=training,proposal_runs=proposals,collection_runs=collection,ik_trials=ik,
        measured_costs=dict(training_s=sum(x['elapsed_s'] or 0 for x in training),proposal_s=sum(x['elapsed_s'] or 0 for x in proposals),incremental_collection_recorded_s=sum(x['measured_wall_s'] or 0 for x in collection),
            physical_experiment_wall_s=sum(x['charged_increment_wall_s'] or 0 for x in experiments),
            fresh_ik_wall_s=sum(x['elapsed_s'] or 0 for x in ik)),
        limitations=['All recorded training and physical experiments, including rejected attempts, are charged separately; wall sums are work accounting, not elapsed calendar time when runs overlap.',
            'Execution ratios include queue/setup/dual audits/video, but exclude shared IK and model preparation; they are not end-to-end speedup claims.',
            'Historical transition collection/import, unrecorded setup, interrupted attempts and training data generation costs remain incompletely measured. No end-to-end superiority claim is permitted from this report.',
            'Sources 010/018 occur in development data; 015 has been used for development diagnostics. Sources 007/012 are reserved for frozen-controller evaluation.'],
        end_to_end_acceleration_proven=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    report=build(a.root.resolve());a.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(pairs=[dict(source=x['source'],variant=x['variant'],both_accepted=x['both_accepted_same_case'],execution_speed_ratio=x['execution_speed_ratio']) for x in report['pairs']],measured_costs=report['measured_costs'],end_to_end_acceleration_proven=False),indent=2))


if __name__=='__main__':main()
