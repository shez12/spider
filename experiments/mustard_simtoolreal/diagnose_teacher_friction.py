"""Replay a subset of native teacher trajectories in Isaac under different contact-friction settings.

Uses train_bc.collect unchanged except for cfg.assets friction overrides, so the hold proxy and
retiming are identical to the production teacher. Writes one collect output per setting.
"""
import argparse,json,random,sys,types
from pathlib import Path
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parents[1]
sys.path[:0]=[str(PROJECT/'third_party/simtoolreal'),str(PROJECT/'third_party/simtoolreal/rl_games'),str(ROOT)]


def main():
    from isaaclab.app import AppLauncher
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,default=PROJECT/'outputs/mustard_classified_native_20261102_inputs')
    p.add_argument('--sources',type=int,nargs='+',default=[5,16,13]);p.add_argument('--per-source',type=int,default=24)
    p.add_argument('--robot-friction',type=float,required=True);p.add_argument('--fingertip-friction',type=float,default=1.5)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--object-z-offset',type=float,default=0.)
    AppLauncher.add_app_launcher_args(p);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);app=AppLauncher(a).app
    try:
        import environment,train_bc
        items=json.loads((a.inputs/'source_manifest.json').read_text());rng=random.Random(a.seed);subset=[]
        for s in a.sources:
            pool=[x for x in items if x['source_index']==s];subset+=rng.sample(pool,min(a.per_source,len(pool)))
        (a.output/'subset_manifest.json').write_text(json.dumps(subset,indent=2))
        original=environment.create_env
        def create_env(num_envs,render_mode=None,configure=None):
            def wrapped(cfg):
                if configure:configure(cfg)
                cfg.assets.robot_friction=a.robot_friction;cfg.assets.finger_tip_friction=a.fingertip_friction
            return original(num_envs,render_mode=render_mode,configure=wrapped)
        environment.create_env=create_env
        n=len(subset);envs=max(6,((n+5)//6)*6)
        args=types.SimpleNamespace(source=a.output/'subset_manifest.json',waypoints=a.inputs/'waypoints.json',expected_trajectories=n,
            num_envs=envs,block_size=envs//6,retiming='continuous',path_reference='realized-arm',arm_smoothing=2.,
            teacher_control='native',teacher_video=False,smoke_steps=0,output=a.output/'collect',object_z_offset=a.object_z_offset)
        args.output.mkdir()
        train_bc.collect(args)
        g=json.loads((args.output/'grasp_validation.json').read_text());rows=[]
        for item,hold,lift in zip(subset,g['longest_hold_proxy_steps'],g['max_lift_m']):
            rows.append(dict(candidate_id=item['candidate_id'],source_index=item['source_index'],longest_hold=hold,max_lift_m=lift))
        summary={str(s):dict(n=sum(r['source_index']==s for r in rows),held_30=sum(r['source_index']==s and r['longest_hold']>=30 for r in rows))
            for s in a.sources}
        (a.output/'summary.json').write_text(json.dumps(dict(robot_friction=a.robot_friction,fingertip_friction=a.fingertip_friction,per_source=summary,rows=rows),indent=2))
        print('FRICTION_SUMMARY',json.dumps(dict(robot_friction=a.robot_friction,per_source=summary)),flush=True)
    except BaseException:
        import traceback;traceback.print_exc();sys.stdout.flush();raise
    finally:app.close()


if __name__=='__main__':main()
