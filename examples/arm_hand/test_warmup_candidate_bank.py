import unittest,numpy as np,torch
from examples.arm_hand.warmup_candidate_bank import WarmupCandidateBank
from examples.arm_hand.strict_candidate_bank import StrictCandidateBank

class WarmupBankTests(unittest.TestCase):
 def values(self):
  q=torch.zeros(3,2,8);q[:,:,5]=1 #180-degree rotation relative to identity
  q[0,:,1]=.01;q[1,:,1]=.04;q[2,:,1]=.02
  v=torch.zeros(3,2,7);c=torch.arange(3,dtype=torch.float32)[:,None,None].expand(3,2,1).clone();o=c[:,:1].expand(3,10,1).clone();cost=torch.tensor([2.,-100.,3.]);ref=np.zeros((3,7),np.float32);ref[:,3]=1
  return o,cost,c,q,v,ref
 def test_orientation_diversity_preserves_exact_actual_samples(self):
  args=self.values();old=StrictCandidateBank(3);old.observe(*args);self.assertEqual(old.take(),[])
  bank=WarmupCandidateBank(3);original=args[3].clone();controls=args[2].clone();bank.observe(*args);args[3].zero_();args[2].zero_();rows=bank.take();self.assertEqual([r['cost'] for r in rows],[2.,3.]);torch.testing.assert_close(rows[0]['rollout']['qpos'],original[0]);torch.testing.assert_close(rows[0]['rollout']['ctrl'],controls[0])
 def test_nonfinite_rejected_and_capacity_enforced(self):
  args=self.values();args[4][0,0,0]=float('nan');bank=WarmupCandidateBank(1);bank.observe(*args);rows=bank.take();self.assertEqual(len(rows),1);self.assertEqual(rows[0]['cost'],3.)
 def test_repeated_control_not_recertified(self):
  args=self.values();bank=WarmupCandidateBank(3);bank.observe(*args);self.assertEqual(len(bank.take()),2);bank.observe(*args);self.assertEqual(bank.take(),[])
if __name__=='__main__':unittest.main()
