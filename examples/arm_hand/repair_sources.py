"""Prepare only previously blocked source groups; preserve their published references."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time
from examples.arm_hand.resource_slots import resource_slot


def main():
    root=Path(__file__).resolve().parents[2];campaign=root/'outputs/mustard_multisource_valid100'
    groups=[g for g in json.loads((campaign/'selected_sources.json').read_text()) if g['source_index'] in [7,12,14,20,22]]
    def prepare(g):
        folder=Path(g['source_dir']);start=time.monotonic()
        if (folder/'preparation.json').exists():return dict(index=g['source_index'],status='already_prepared')
        command=[sys.executable,'-u','-m','examples.arm_hand.mustard_multisource','--prepare-one',str(folder/'source.json'),
                 '--dataset-dir',str(root/'example_datasets/retarget_full')]
        with resource_slot(campaign/'resource_slots/ik',8):
            with (folder/'prepare_repair.log').open('a') as f:result=subprocess.run(command,cwd=root,stdout=f,stderr=subprocess.STDOUT)
        record=dict(index=g['source_index'],status='prepared' if result.returncode==0 else 'failed',elapsed_s=time.monotonic()-start)
        (folder/'repair.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True);return record
    with ThreadPoolExecutor(max_workers=2) as pool:records=list(pool.map(prepare,groups))
    (campaign/'source_repairs.json').write_text(json.dumps(records,indent=2))
    if any(r['status']=='failed' for r in records):raise SystemExit(1)


if __name__=='__main__':main()
