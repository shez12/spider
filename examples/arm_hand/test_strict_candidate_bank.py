"""Constraint shortlist must preserve the exact sampled states and controls."""
import unittest
import numpy as np
import torch
from examples.arm_hand.strict_candidate_bank import StrictCandidateBank


class BankTests(unittest.TestCase):
    def test_feasible_samples_are_cloned_not_replayed_or_aliased(self):
        bank=StrictCandidateBank(2);q=torch.zeros(3,2,8);q[:,:,4]=1
        q[0,:,1]=.1;q[1,:,1]=.01;q[2,:,1]=.02
        v=torch.zeros(3,2,7);controls=torch.tensor([0.,.1,.2])[:,None,None].expand(3,2,1).clone()
        offsets=torch.arange(3,dtype=torch.float32)[:,None,None].expand(3,10,1).clone()
        costs=torch.tensor([-100.,2.,3.]);ref=np.zeros((3,7),np.float32);ref[:,3]=1
        original_q=q.clone();original_controls=controls.clone()
        bank.observe(offsets,costs,controls,q,v,ref);q.zero_();controls.zero_()
        rows=bank.take();self.assertEqual([row['cost'] for row in rows],[2.,3.])
        torch.testing.assert_close(rows[0]['rollout']['qpos'],original_q[1])
        torch.testing.assert_close(rows[0]['rollout']['ctrl'],original_controls[1])
        bank.observe(offsets,costs,original_controls,original_q,v,ref)
        self.assertEqual(bank.take(),[])


if __name__=='__main__':unittest.main()
