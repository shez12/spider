"""Run the paired 512-sample / 200-round mustard pilot, then audit both outputs."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
MJ='/home/szc/.conda/envs/spider-repro/bin/python'
ISAAC='/home/szc/.conda/envs/simtoolreal_origin_isaacsim/bin/python'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/mustard_backend_comparison_v1')
    p.add_argument('--smoke-only',action='store_true')
    args=p.parse_args();out=args.output_dir.resolve();case=out/'case'
    env=os.environ.copy()
    runtime=out/'runtime'
    for d in ['cache','tmp','xdg_cache']:(runtime/d).mkdir(parents=True,exist_ok=True)
    env.update(LD_LIBRARY_PATH='/home/szc/.conda/envs/simtoolreal_origin_isaacsim/lib:'+env.get('LD_LIBRARY_PATH',''),
        OMNI_KIT_ACCEPT_EULA='YES',OMNI_KIT_CACHE_PATH=str(runtime/'cache'),TMPDIR=str(runtime/'tmp'),
        XDG_CACHE_HOME=str(runtime/'xdg_cache'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',
        PYTHONPATH=str(ROOT/'outputs/spider_isaac_benchmark_deps')+':'+str(ROOT))
    progress=dict(started_at_utc=datetime.now(timezone.utc).isoformat(),samples=512,iterations=200,seed=0,
        source_case=str(case),phase='isaac_smoke',results=[])
    def save():
        temp=out/'status.tmp';temp.write_text(json.dumps(progress,indent=2));temp.replace(out/'status.json')
    def run(command,log,execution_env=None):
        start=time.monotonic()
        with (out/log).open('a') as stream:
            subprocess.run(command,cwd=ROOT,env=execution_env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        return time.monotonic()-start
    save()
    try:
        smoke=out/'isaac_smoke'
        if not (smoke/'startup.json').exists():
            run([ISAAC,'-u',str(ROOT/'examples/arm_hand/benchmark_isaac_spider.py'),'--headless',
                '--input-dir',str(case),'--output-dir',str(smoke),'--smoke'],'isaac_smoke.log',env)
        progress['smoke']=json.loads((smoke/'startup.json').read_text());save()
        if args.smoke_only:
            progress['phase']='smoke_complete';save();return
        # Sequential execution avoids the two benchmark backends competing with
        # each other. Existing training is neither stopped nor reconfigured.
        for backend in ['mujoco_warp','isaacsim_physx']:
            progress['phase']=backend;save();target=out/backend
            if not (target/'result.json').exists():
                if backend=='mujoco_warp':
                    command=[MJ,'-u','-m','examples.arm_hand.optimize_kuka_gpu','--input-dir',str(case),
                        '--output-dir',str(target),'--samples','512','--max-iterations','200',
                        '--disable-target-stop','--disable-plateau-stop','--seed','0']
                    timing=run(command,backend+'.log')
                else:
                    command=[ISAAC,'-u',str(ROOT/'examples/arm_hand/benchmark_isaac_spider.py'),'--headless',
                        '--input-dir',str(case),'--output-dir',str(target),'--samples','512','--max-iterations','200','--seed','0']
                    timing=run(command,backend+'.log',env)
                (target/'process_timing.json').write_text(json.dumps(dict(wall_s=timing)))
            validation_start=time.monotonic()
            for trajectory,directory in [('trajectory.npz',target),('replay_trajectory.npz',target/'replay_audit')]:
                directory.mkdir(exist_ok=True)
                if directory!=target:
                    link=directory/'trajectory.npz'
                    if not link.exists():link.symlink_to(target/trajectory)
                run([MJ,'-m','examples.arm_hand.audit_right_trajectory','--case-dir',str(case),'--gpu-dir',str(directory)],backend+'_audit.log')
            reports=[json.loads((d/'tracking_collision_report.json').read_text()) for d in [target,target/'replay_audit']]
            from examples.arm_hand.mustard_valid100 import valid
            accepted=all(valid(r) for r in reports)
            wall=json.loads((target/'process_timing.json').read_text())['wall_s']+time.monotonic()-validation_start
            progress['results'].append(dict(backend=backend,validated_success=accepted,wall_s_including_startup_and_audit=wall,
                pilot_valid_per_hour=(3600/wall if accepted else 0),reports=reports))
            save()
        progress['phase']='complete';progress['completed_at_utc']=datetime.now(timezone.utc).isoformat()
        progress['limitations']='One paired trajectory/seed is a pilot, not a statistically reliable acceptance rate. Other GPU training continues, so shared-load timings are conditional.'
        save()
    except BaseException as error:
        progress.update(phase='failed',error=str(error));save();raise


if __name__=='__main__':main()
