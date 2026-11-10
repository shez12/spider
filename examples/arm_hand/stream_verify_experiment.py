"""Incremental isolated full verification; never changes the global dataset index."""
import argparse,fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
from examples.arm_hand.verify_warmup_artifacts import verify

def main():
 p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--launcher-pid',type=int,required=True);p.add_argument('--launcher-identity',required=True);a=p.parse_args();a.experiment=a.experiment.resolve();a.output.mkdir(exist_ok=True,parents=True);lock=(a.output/'worker.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);r=Path(__file__).resolve().parents[2];groups={g['source_index']:g for g in json.loads((r/'outputs/mustard_multisource_valid100/selected_sources.json').read_text())};seen={};rows=[];failures=[];s=time.time()
 while True:
  for ep in sorted(a.experiment.glob('execution_*/execution.json')):
   raw=ep.read_bytes()
   try:d=json.loads(raw)
   except json.JSONDecodeError:continue
   if d.get('status')!='complete' or not d.get('accepted'):continue
   key=hashlib.sha256(raw).hexdigest()
   if seen.get(str(ep))==key:continue
   seen[str(ep)]=key;dest=a.output/hashlib.sha256(str(ep.parent).encode()).hexdigest()[:16];dest.mkdir(exist_ok=True)
   with (dest/'collector.log').open('w') as f:rc=subprocess.run([os.sys.executable,'-u','-m','examples.arm_hand.collect_warmup_candidates','--campaign','/tmp/nonexistent_spider_campaign_incremental_verification','--experiments',str(ep.parent),'--output',str(dest)],cwd=r,stdout=f,stderr=subprocess.STDOUT).returncode
   try:
    if rc:raise ValueError('collector returncode '+str(rc))
    reports=list(dest.glob('*/validation.json'))
    if len(reports)!=1:raise ValueError('Expected one eligible dual-real report')
    row=verify(reports[0],groups);row.update(execution_report_sha256=key,individual_verified=True);(dest/'artifact_verification.json').write_text(json.dumps(row,indent=2));rows=[x for x in rows if x['execution']!=row['execution']]+[row]
   except Exception as e:failures.append(dict(execution=str(ep),error=str(e)))
  try:alive=a.launcher_identity.encode() in Path('/proc',str(a.launcher_pid),'cmdline').read_bytes()
  except FileNotFoundError:alive=False
  unique={(x['source_index'],x['augmentation_sha256']) for x in rows};result=dict(status='monitoring_live_launcher' if alive else 'terminal_launcher',launcher_pid=a.launcher_pid,verified_unique=len(unique),rows=rows,failures=failures,wall_s=time.time()-s,note='Individually verified isolated artifacts; global quota admission/dedup still required')
  temp=a.output/'manifest.tmp';temp.write_text(json.dumps(result,indent=2));temp.replace(a.output/'manifest.json')
  if not alive:break
  time.sleep(10)
if __name__=='__main__':main()
