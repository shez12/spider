import unittest,torch
from examples.arm_hand.contact_sampling_cost import hand_self_penalty

class ContactCostTests(unittest.TestCase):
 def test_only_active_hand_self_pairs_count_per_world(self):
  g=torch.tensor([[1,2],[1,0],[1,3],[1,2],[1,2],[1,2]])
  d=torch.tensor([-.002,-.1,-.1,-.001,-.1,float('nan')]);w=torch.tensor([0,0,0,1,-1,0]);hand=torch.tensor([False,True,True,False])
  out=hand_self_penalty(g,d,w,torch.tensor(5),hand,2)
  torch.testing.assert_close(out,torch.tensor([.0015**2,.0005**2]));self.assertTrue(torch.isfinite(out).all())
 def test_tolerance_and_invalid_padding(self):
  g=torch.tensor([[1,2],[-1,100]]);d=torch.tensor([-.0004,float('nan')]);w=torch.tensor([0,100]);hand=torch.tensor([False,True,True])
  torch.testing.assert_close(hand_self_penalty(g,d,w,torch.tensor(2),hand,1),torch.zeros(1))
if __name__=='__main__':unittest.main()
