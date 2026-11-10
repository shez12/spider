"""Concatenate already validated and packaged reset banks into one training bank.

Each input must be a packaged bank directory (reset_bank.npz + validation.json) built
for the current environment signature. States are not re-validated; the merged
validation.json keeps every accepted record, tagged with its source bank, in bank order.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from diverse_resets import FIELDS,environment_signature


def tilt_deg(root):
    w,x,y,z=root[:,3:7].T
    return np.degrees(np.arccos(np.clip(1-2*(x*x+y*y),-1,1)))


def merge(sources,output):
    assert not output.exists(),'Use a fresh output directory.'
    banks=[];records=[]
    for path in sources:
        with np.load(path/'reset_bank.npz') as z:data={k:z[k] for k in z.files}
        assert int(data['format_version'])==1 and str(data['environment_sha256'])==environment_signature(),f'{path}: stale bank'
        accepted=[r for r in json.loads((path/'validation.json').read_text())['records'] if r['accepted']]
        assert len(accepted)==len(data['kind']),f'{path}: records and bank disagree'
        records.extend(dict(r,source_bank=str(path.resolve())) for r in accepted);banks.append(data)
    keys=(*FIELDS,'table_z','kind')
    merged={k:np.concatenate([b[k] for b in banks]) for k in keys}
    merged.update(format_version=np.array(1),environment_sha256=np.array(environment_signature()))
    output.mkdir(parents=True);np.savez_compressed(output/'reset_bank.npz',**merged)
    lying=(tilt_deg(merged['root'])>60)&(tilt_deg(merged['root'])<120)
    counts={str(k):int((merged['kind']==k).sum()) for k in (1,2,3)}
    report=dict(merged_from=[str(p.resolve()) for p in sources],accepted=len(records),counts=counts,
        lying_60_120deg={str(k):int(((merged['kind']==k)&lying).sum()) for k in (1,2,3)},
        per_source_bank=[dict(bank=str(p.resolve()),states=len(b['kind'])) for p,b in zip(sources,banks)],
        revalidated=False,bank_sha256=hashlib.sha256((output/'reset_bank.npz').read_bytes()).hexdigest(),records=records)
    (output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'completed.json').write_text(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=merge(a.sources,a.output)
    print(json.dumps({k:v for k,v in r.items() if k!='records'},indent=2))
