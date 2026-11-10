"""Account for full shared-seed comparison costs without claiming local tests prove speed."""
import argparse,json,time
from pathlib import Path
import torch

def main():
 p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);a=p.parse_args();o=a.experiment;plan=json.loads((o/'plan.json').read_text());rows=[];complete=True
 for case in plan['cases']:
  idx=case['source_index'];seedp=o/f'seed_{idx}/execution.json';seed=json.loads(seedp.read_text()) if seedp.exists() else {};prop=o/f'proposal_{idx}/proposal.json';pr=json.loads(prop.read_text()) if prop.exists() else {};common=seed.get('wall_s');proposal_wall=pr.get('elapsed_s');stagep=o/'status.json';status=json.loads(stagep.read_text()) if stagep.exists() else {}
  stage={s['stage']:s for s in status.get('steps',[])};common=stage.get(f'seed_{idx}',{}).get('wall_s',common);proposal_wall=stage.get(f'propose_{idx}',{}).get('wall_s',proposal_wall)
  for arm in ['raw','bc','world']:
   ep=o/f'{idx}_{arm}/execution.json';ex=json.loads(ep.read_text()) if ep.exists() else {};terminal=ex.get('status') in ['complete','failed'];complete &=terminal
   row=dict(source_index=idx,arm=arm,status=ex.get('status','pending'),accepted=ex.get('accepted',False),execution_wall_s=ex.get('wall_s'),shared_seed_charged_s=common,proposal_charged_s=proposal_wall if arm=='world' else 0,execution=str(ep),timing_breakdown=ex.get('timing_breakdown',{}))
   row['current_case_wall_s']=common+stage.get(f'{idx}_{arm}',{}).get('wall_s',ex['wall_s'])+(proposal_wall if arm=='world' else 0) if terminal and common is not None and (arm!='world' or proposal_wall is not None) else None
   row['total_compute_rounds']=None
   hp=o/f'{idx}_{arm}/gpu/history.json'
   if hp.exists():row['total_compute_rounds']=1+len(json.loads(hp.read_text()))
   if row['total_compute_rounds'] is not None and row['total_compute_rounds']>200:raise ValueError('Budget exceeded')
   rows.append(row)
  if (o/f'initial_{idx}.pt').exists() and (o/f'proposal_{idx}/proposal.pt').exists():
   x=torch.load(o/f'initial_{idx}.pt',map_location='cpu',weights_only=False);y=torch.load(o/f'proposal_{idx}/proposal.pt',map_location='cpu',weights_only=False);delta=(x['best']-y['best']).abs();assert x['signature']==y['signature'];assert float(delta[:,:7].max())<=.001501 and float(delta[:,7:].max())<=.005001
 result=dict(status='complete' if complete else 'pending',rows=rows,training_cost_accounted=False,end_to_end_advantage_proven=False,limitations=['Development cases, not independent held-out proof','Per-arm charges include full common seed acquisition and model-proposal preparation; historical training and preparation must additionally be amortized before any speed claim','Acceptance requires both real GPU reports; final admission additionally needs full provenance verification','Queues vary over time; wall-time differences alone do not identify model benefit'])
 (o/'comparison_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
