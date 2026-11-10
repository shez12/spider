import hashlib
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from bc_balance import bounded_weights,load_weights,rotation


class BalanceTests(unittest.TestCase):
    def test_mass_and_bounded_ratio(self):
        lengths=np.array([10]*99+[1000]);groups=np.array([0]*99+[1])
        w=bounded_weights(groups,lengths)
        self.assertAlmostEqual(float(w@lengths/lengths.sum()),1.)
        self.assertLessEqual(w.max()/w.min(),12.+1e-6)
        self.assertTrue((w>0).all())

    def test_padding_and_identity_loss(self):
        error=torch.tensor([[1.,3.,1000.],[2.,1000.,1000.]])
        mask=torch.tensor([[True,True,False],[True,False,False]])
        original=(error*mask).sum()/mask.sum()
        weighted=(error*mask*torch.ones(2)[:,None]).sum()/mask.sum()
        self.assertEqual(float(original),2.)
        self.assertTrue(torch.equal(original,weighted))

    def test_reject_misaligned_data(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'cache';path.write_bytes(b'test')
            weights=Path(t)/'weights.npz';ids=np.array([1,7]);lengths=np.array([2,3])
            np.savez(weights,candidate_ids=ids,valid_lengths=lengths,weights=np.ones(2),dataset_sha256=np.array(hashlib.sha256(path.read_bytes()).hexdigest()))
            np.testing.assert_equal(load_weights(weights,ids,lengths,path),np.ones(2))
            with self.assertRaises(AssertionError):load_weights(weights,ids[::-1],lengths,path)
            path.write_bytes(b'changed')
            with self.assertRaises(AssertionError):load_weights(weights,ids,lengths,path)

    def test_quaternion_sign_invariance(self):
        q=np.array([[.2,.3,.1,.4]]);q/=np.linalg.norm(q)
        np.testing.assert_allclose(rotation(q),rotation(-q))
        np.testing.assert_allclose(rotation(q)[0].T@rotation(q)[0],np.eye(3),atol=1e-10)


if __name__=='__main__':unittest.main()
