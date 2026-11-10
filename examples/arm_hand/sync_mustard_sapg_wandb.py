"""Attach W&B to an already-running SAPG run without restarting its trainer."""
import argparse
import json
import time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--project',default='simtoolreal-warmstart')
    parser.add_argument('--entity',default=None)
    parser.add_argument('--run-id',default=None,help='Explicit unique ID for nested experiment branches.')
    parser.add_argument('--poll-seconds',type=float,default=10.)
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--upload-videos',action='store_true',help='Opt-in only with explicit authorization to upload videos.')
    args=parser.parse_args()
    root=args.run_dir.resolve()
    import wandb
    from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
    from tensorboard.util.tensor_util import make_ndarray
    config={}
    for name in ('run_arguments','environment_config','sapg_config','sapg_contract'):
        path=root/(name+'.json')
        if path.exists():config[name]=json.loads(path.read_text())
    state_path=root/'wandb_sync_state.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {}
    state.setdefault('scalar_wall_times',{})
    state.setdefault('progress_lines',0)
    state.setdefault('videos',[])
    state.setdefault('evaluation_files',[])
    # Metrics-only by default: no local configuration files, paths, source code,
    # trajectories or model weights are attached to the remote run.
    run=wandb.init(project=args.project,entity=args.entity,id=args.run_id or root.name,name=args.run_id or root.name,
        resume='allow',dir=str(root),save_code=False,
        settings=wandb.Settings(disable_git=True,disable_code=True,x_disable_meta=True,x_disable_stats=True,
            x_save_requirements=False,x_file_stream_max_bytes=262144,x_file_stream_no_gzip=False,console='off'),
        tags=['SAPG','warm-start','target-pose-reaching','kuka-sharpa-right'])
    wandb.define_metric('environment_frames')
    wandb.define_metric('*',step_metric='environment_frames')
    wandb.define_metric('train/*',step_metric='environment_frames')
    wandb.define_metric('rollout/*',step_metric='environment_frames')
    (root/'wandb_run.json').write_text(json.dumps(dict(url=run.url,id=run.id,project=run.project),indent=2)+'\n')
    print('WANDB_RUN_URL',run.url,flush=True)
    loaders={}
    try:
        while True:
            records={}
            for path in root.glob('*/summaries/events.out.tfevents.*'):
                loader=loaders.setdefault(str(path),EventFileLoader(str(path)))
                for event in loader.Load():
                    for value in event.summary.value:
                        key=value.tag
                        if event.wall_time<=state['scalar_wall_times'].get(key,0):continue
                        if value.HasField('simple_value'):scalar=float(value.simple_value)
                        elif value.HasField('tensor'):
                            array=make_ndarray(value.tensor)
                            if array.size!=1:continue
                            scalar=float(array.reshape(-1)[0])
                        else:continue
                        records.setdefault(event.step,{})[key]=scalar
                        state['scalar_wall_times'][key]=event.wall_time
            for frame,record in sorted(records.items()):
                run.log(dict(environment_frames=frame,**record,**{'history_backfill/is_pre_resume':0}))
            progress=root/'sapg_progress.jsonl'
            if progress.exists():
                lines=progress.read_text().splitlines()
                for index in range(state['progress_lines'],len(lines)):
                    try:row=json.loads(lines[index])
                    except json.JSONDecodeError:break
                    run.log(dict(environment_frames=row['frames'],**{'history_backfill/is_pre_resume':0},**{
                        'train/'+k:v for k,v in row.items() if v is not None and k!='frames'}))
                    state['progress_lines']=index+1
            for path in sorted((root/'videos').glob('*.mp4')) if args.upload_videos else []:
                if path.name in state['videos'] or time.time()-path.stat().st_mtime<30:continue
                # RecordVideo closes the MP4 before it becomes stable. Upload
                # only files whose last write was at least thirty seconds ago.
                step=int(path.stem.rsplit('-',1)[-1])
                frames=step*config.get('run_arguments',{}).get('num_envs',120)
                run.log({'environment_frames':frames,'rollout/video':wandb.Video(str(path),format='mp4')})
                state['videos'].append(path.name)
                print('WANDB_VIDEO',path.name,flush=True)
            for path in sorted((root/'videos').glob('*.metrics.json')):
                if path.name in state['evaluation_files']:continue
                metrics=json.loads(path.read_text())
                epoch=metrics.get('evaluated_training_epoch',0)
                frames=epoch*config.get('run_arguments',{}).get('num_envs',120)*16
                run.log({'environment_frames':frames,**{'eval/'+k:v for k,v in metrics.items() if isinstance(v,(int,float))}})
                state['evaluation_files'].append(path.name)
            state_path.write_text(json.dumps(state,indent=2)+'\n')
            if args.once:break
            if (root/'completed.json').exists() or (root/'failed.json').exists():
                remaining=[p for p in (root/'videos').glob('*.mp4') if p.name not in state['videos']] if args.upload_videos else []
                if not remaining:break
            time.sleep(args.poll_seconds)
    finally:
        run.finish()


if __name__=='__main__':main()
