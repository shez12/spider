"""Merge pre-resume scalar history into a paused destination W&B writer.

Local prepare and remote upload are separate. Training is never interrupted.
Use a dedicated receipt to make completed backfills idempotent.
"""
import argparse,json,math,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--through-epoch',type=int,default=5000)
    p.add_argument('--upload',action='store_true')
    p.add_argument('--fresh-run',action='store_true',help='Prepopulate a new corrected-protocol run before training starts.')
    a=p.parse_args();source=a.source.resolve();destination=a.destination.resolve()
    export=destination/f'history_before_resume_{a.through_epoch}.jsonl'
    receipt=destination/f'history_before_resume_{a.through_epoch}.receipt.json'
    if receipt.exists():
        print('Already completed:',receipt);return
    if not a.upload:
        from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader
        from tensorboard.util.tensor_util import make_ndarray
        progress=[json.loads(line) for line in (source/'sapg_progress.jsonl').read_text().splitlines()]
        progress=[row for row in progress if row['epoch']<=a.through_epoch]
        assert [row['epoch'] for row in progress]==list(range(1,a.through_epoch+1))
        cutoff=progress[-1]['frames'];records={}
        for path in source.glob('*/summaries/events.out.tfevents.*'):
            for event in LegacyEventFileLoader(str(path)).Load():
                if event.step>cutoff:continue
                for value in event.summary.value:
                    if value.HasField('simple_value'):v=float(value.simple_value)
                    elif value.HasField('tensor'):
                        arr=make_ndarray(value.tensor)
                        if arr.size!=1:continue
                        v=float(arr.reshape(-1)[0])
                    else:continue
                    if math.isfinite(v):records.setdefault(event.step,{})[value.tag]=v
        for row in progress:
            record=records.setdefault(row['frames'],{})
            record.update({'train/'+k:v for k,v in row.items() if v is not None and k!='frames'})
        for path in (source/'videos').glob('*.metrics.json'):
            metrics=json.loads(path.read_text());epoch=metrics.get('evaluated_training_epoch',0)
            if not 1<=epoch<=a.through_epoch:continue
            frame=next(row['frames'] for row in progress if row['epoch']==epoch)
            records.setdefault(frame,{}).update({'eval/'+k:v for k,v in metrics.items() if isinstance(v,(int,float)) and math.isfinite(v)})
        assert len(records)==a.through_epoch and max(records)==cutoff
        tmp=export.with_suffix('.tmp')
        with tmp.open('w') as f:
            for frame,record in sorted(records.items()):
                f.write(json.dumps(dict(environment_frames=frame,**record,
                    **{'history_backfill/is_pre_resume':1}))+'\n')
        tmp.replace(export)
        print(json.dumps(dict(rows=len(records),scalar_tags=len({key for row in records.values() for key in row}),max_frame=cutoff,export=str(export))))
        return
    assert export.is_file(),'Prepare the export before pausing the current W&B sidecar.'
    import wandb
    target=json.loads((destination/'wandb_run.json').read_text())
    existing=set()
    if not a.fresh_run:
        from wandb_history_read import history_rows
        for row in history_rows(target['url'],['history_backfill/is_pre_resume','environment_frames']):
            if row.get('history_backfill/is_pre_resume')==1:existing.add(int(row['environment_frames']))
    print('EXISTING_HISTORY_ROWS',len(existing),flush=True)
    run=wandb.init(project=target['project'],id=target['id'],resume='allow' if a.fresh_run else 'must',dir=str(destination),
        save_code=False,settings=wandb.Settings(disable_git=True,disable_code=True,x_disable_meta=True,x_disable_stats=True,
            x_save_requirements=False,x_file_stream_max_bytes=262144,x_file_stream_no_gzip=False,
            x_file_stream_transmit_interval=.5,
            x_file_stream_timeout_seconds=45,x_file_stream_retry_max=3,finish_timeout=180,finish_timeout_raises=True,console='off'))
    previous_summary={key:value for key,value in run.summary._as_dict().items() if not key.startswith('_')}
    run.define_metric('environment_frames')
    run.define_metric('*',step_metric='environment_frames')
    count=0;first=None;last=None
    try:
        with export.open() as f:
            for line in f:
                record=json.loads(line)
                first=record['environment_frames'] if first is None else first
                last=record['environment_frames']
                if record['environment_frames'] in existing:continue
                run.log(record);count+=1
        # Historical x values are custom metrics, not W&B's monotonically
        # increasing internal step. Preserve the current run's latest summary.
        run.summary.update(previous_summary)
        manifest=dict(source_run=source.name,through_epoch=a.through_epoch,rows=count+len(existing),
            new_rows=count,already_present_rows=len(existing),
            first_environment_frame=first,last_environment_frame=last,
            protocol='single-target warmstart v2, before native-curriculum resume',
            video_files_uploaded=False,model_or_data_uploaded=False)
        run.summary['history_backfill_manifest']=manifest
    finally:run.finish()
    assert count+len(existing)==a.through_epoch
    receipt.write_text(json.dumps(dict(manifest,destination_url=target['url'],completed_unix=time.time()),indent=2)+'\n')
    print('BACKFILL_COMPLETED',json.dumps(manifest),flush=True)


if __name__=='__main__':main()
