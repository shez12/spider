"""CPU correctness checks for state timing, quaternion continuity and action gradients."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from examples.arm_hand.world_model import Dynamics, canonical, save_shard, read_data


class WorldModelTests(unittest.TestCase):
    def test_state_action_alignment_and_finite_filter(self):
        spec=dict(nq=8,nv=7,nu=1,context_dim=2)
        q=np.zeros((2,3,8),np.float32);q[:,:,-4]=1;q[:,:,0]=[1,2,3]
        v=np.zeros((2,3,7),np.float32);u=np.ones((2,3,1),np.float32)
        q[1,1,0]=np.nan;initial=np.zeros(8);initial[-4]=1
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'transitions_test.npz'
            count=save_shard(path,q,v,u,initial,np.zeros(7),[1,2],spec,'demo',np.zeros(2))
            self.assertEqual(count,1)
            _,shards=read_data(root);s,a,g,c=shards[0]
            np.testing.assert_array_equal(s[0,:,0],[0,1,2,3])
            self.assertEqual(s.shape[1],a.shape[1]+1)
            self.assertEqual(s.dtype,np.float32)
            with self.assertRaises(FileExistsError):
                save_shard(path,q,v,u,initial,np.zeros(7),[1,2],spec,'demo',np.zeros(2))

    def test_quaternion_sign_continuity(self):
        q=np.zeros((1,3,8));q[0,:,-4]=[1,-1,1]
        np.testing.assert_array_equal(canonical(q)[0,:,-4],[1,1,1])

    def test_frozen_model_keeps_action_gradients(self):
        torch.manual_seed(7)
        net=Dynamics(dict(nq=8,nv=7,nu=1,context_dim=2),16)
        for p in net.parameters():p.requires_grad_(False)
        state=torch.zeros(2,15);state[:,4]=1
        action=torch.ones(2,1,requires_grad=True)
        pred=net(state,action,torch.zeros(2,2))
        pred=net(pred,action,torch.zeros(2,2));pred[:,0].sum().backward()
        self.assertTrue(torch.isfinite(action.grad).all());self.assertGreater(float(action.grad.abs().sum()),0)
        torch.testing.assert_close(pred[:,4:8].norm(dim=-1),torch.ones(2))
        self.assertTrue(all(p.grad is None for p in net.parameters()))


if __name__=='__main__':unittest.main()
