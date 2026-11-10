"""Continue the one failed probe; preserve all previous completed probes/costs."""
from pathlib import Path
import json,subprocess,sys,time


def save(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)


def main():
 r=Path(__file__).resolve().parents[2];pilot=r/'outputs/world_model_pilot';old=pilot/'contact_rescue_sources_v1';out=pilot/'contact_rescue_sources_guarded';out.mkdir(exist_ok=True);status=json.loads((out/'status.json').read_text());started=time.monotonic()
 case=old/'case_020';state=json.loads((case/'state.json').read_text());execution=out/'execution_020'
 cmd=[sys.executable,'-u','-m','examples.arm_hand.execute_contact_capture_guarded','--case',str(case),'--output',str(execution),'--baseline','--iterations','200','--strict-check-every','1','--strict-candidate-pool','16','--transition-dir',str(old/'transitions/20'),'--source-group',state['task'],'--resume']
 status['status']='running';save(out/'status.json',status)
 with (out/'execution_020.log').open('a') as log:result=subprocess.run(cmd,cwd=r,stdout=log,stderr=subprocess.STDOUT)
 d=json.loads((execution/'execution.json').read_text());status['records'].append(dict(source_index=20,execution=str(execution),returncode=result.returncode,accepted=d.get('accepted',False),new_execution_wall_s=d['wall_s'],note='Includes original failed-prefix wall once; continuation not a fresh candidate'))
 status.update(status='complete' if result.returncode==0 else 'infrastructure_failed',continuation_wall_s=time.monotonic()-started);save(out/'status.json',status)
 if result.returncode:return
 for name,cmd in [('refresh',['-m','examples.arm_hand.collect_warmup_candidates','--render','--output',r/'outputs/mustard_warmup_grasp_v2']),('verify',['-m','examples.arm_hand.verify_warmup_artifacts','--warmup',r/'outputs/mustard_warmup_grasp_v2'])]:
  with (out/(name+'.log')).open('w') as log:result=subprocess.run([sys.executable,'-u',*map(str,cmd)],cwd=r,stdout=log,stderr=subprocess.STDOUT)
  if result.returncode:status.update(status='infrastructure_failed',failed_stage=name);save(out/'status.json',status);return
 status['final_verification_complete']=True;save(out/'status.json',status)
if __name__=='__main__':main()
