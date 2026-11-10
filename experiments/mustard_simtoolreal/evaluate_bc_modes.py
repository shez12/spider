"""Keep BC demo reproduction and native-reset transfer videos separate."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--waypoints',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--tag',required=True);p.add_argument('--frames',type=int,default=960)
    p.add_argument('--episode-seconds',type=float);p.add_argument('--reset-modes',default='demo,native')
    a=p.parse_args();modes=a.reset_modes.split(',')
    assert modes and len(set(modes))==len(modes) and set(modes)<=set(('demo','native'))
    status={}
    for mode in modes:
        destination=a.output_dir/mode/(a.tag+'.mp4');destination.parent.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,str(ROOT/'evaluate_actor_video.py'),'--headless','--checkpoint',str(a.checkpoint),
            '--dataset',str(a.dataset),'--waypoints',str(a.waypoints),'--output',str(destination),
            '--frames',str(a.frames),'--final-goal-only','--reset-mode',mode]
        if a.episode_seconds is not None:command+=['--episode-seconds',str(a.episode_seconds)]
        status[mode]=subprocess.run(command,check=False).returncode
    (a.output_dir/(a.tag+'.evaluation_status.json')).write_text(json.dumps(status,indent=2)+'\n')
    if any(status.values()):raise SystemExit(1)


if __name__=='__main__':main()
