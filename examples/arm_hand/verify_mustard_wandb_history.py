"""Require complete remote epoch1..5000 history before starting resumed training."""
import argparse,json
from pathlib import Path
from wandb_history_read import history_rows

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args()
root=a.run_dir.resolve();target=json.loads((root/'wandb_run.json').read_text())
rows=history_rows(target['url'],['_step','environment_frames','train/epoch','history_backfill/is_pre_resume'],samples=10000)
source=[r for r in rows if r.get('history_backfill/is_pre_resume')==1]
epochs=[int(r['train/epoch']) for r in source]
assert len(source)==5000 and len(set(epochs))==5000 and set(epochs)==set(range(1,5001)),dict(rows=len(source),unique_epochs=len(set(epochs)))
assert all(int(r['environment_frames'])==int(r['train/epoch'])*8184*16 for r in source)
proof=dict(complete=True,run_id=target['id'],verified_rows=len(source),min_epoch=min(epochs),max_epoch=max(epochs),
 first_frame=min(int(r['environment_frames']) for r in source),last_frame=max(int(r['environment_frames']) for r in source),
 training_started=False)
assert not (root/'sapg_progress.jsonl').exists(),'Verification must precede resumed training.'
(root/'history_remote_verification.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof),flush=True)
