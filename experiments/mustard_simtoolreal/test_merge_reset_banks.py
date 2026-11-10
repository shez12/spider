import json,tempfile,unittest
from pathlib import Path
import numpy as np
from diverse_resets import environment_signature
from merge_reset_banks import merge


def bank(path,kinds,signature=None,extra_rejected=1):
    path.mkdir();n=len(kinds)
    values={k:np.full((n,29),len(kinds),dtype='float32') for k in ('q','v','previous','current')}
    root=np.zeros((n,13),dtype='float32');root[:,3]=1
    values.update(root=root,goal=np.zeros((n,7),dtype='float32'),table=np.zeros((n,7),dtype='float32'),table_z=np.full(n,.38,dtype='float32'))
    np.savez(path/'reset_bank.npz',**values,kind=np.array(kinds),format_version=np.array(1),environment_sha256=np.array(signature or environment_signature()))
    records=[dict(candidate=i,accepted=True,kind=k) for i,k in enumerate(kinds)]+[dict(candidate=n,accepted=False,kind=1)]*extra_rejected
    (path/'validation.json').write_text(json.dumps(dict(records=records)))
    return path


class MergeResetBankTests(unittest.TestCase):
    def test_concatenates_in_order_with_tagged_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=bank(Path(tmp)/'a',[1,2]);b=bank(Path(tmp)/'b',[3,2,2]);out=Path(tmp)/'m'
            report=merge([a,b],out)
            with np.load(out/'reset_bank.npz') as z:
                self.assertEqual(z['kind'].tolist(),[1,2,3,2,2]);self.assertEqual(z['q'][:,0].tolist(),[2,2,3,3,3])
                self.assertEqual(str(z['environment_sha256']),environment_signature())
            self.assertEqual(report['counts'],{'1':1,'2':3,'3':1});self.assertEqual(len(report['records']),5)
            self.assertEqual([Path(r['source_bank']).name for r in report['records']],['a','a','b','b','b'])

    def test_rejects_stale_or_inconsistent_banks(self):
        with tempfile.TemporaryDirectory() as tmp:
            stale=bank(Path(tmp)/'s',[1],signature='old')
            with self.assertRaisesRegex(AssertionError,'stale'):merge([stale],Path(tmp)/'m1')
            bad=bank(Path(tmp)/'b',[1,2]);(bad/'validation.json').write_text(json.dumps(dict(records=[dict(accepted=True)])))
            with self.assertRaisesRegex(AssertionError,'disagree'):merge([bad],Path(tmp)/'m2')


if __name__=='__main__':unittest.main()
