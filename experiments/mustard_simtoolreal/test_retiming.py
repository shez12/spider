"""Regression checks for independent clocks and native action feasibility."""
import unittest
import numpy as np
from experiments.mustard_simtoolreal.retiming import continuous_plan


class RetimingTests(unittest.TestCase):
    def test_stationary_trajectory_does_not_wait_for_moving_neighbor(self):
        previous=np.zeros((2,29));commands=np.zeros((2,20,29));commands[1,:,0]=np.linspace(0,1,20)
        lo=np.full(29,-2.);hi=-lo
        _,actions,_,_,audit=continuous_plan(previous,commands,lo,hi,.1,.1,.025,smoothing_sigma=0,hold_steps=0)
        self.assertEqual(audit['valid_lengths'][0],12)
        self.assertGreater(audit['valid_lengths'][1],400)
        np.testing.assert_array_equal(actions[12:,0,:7],0)

    def test_large_hand_jump_remains_bounded_without_stopping_arm(self):
        previous=np.zeros((1,29));commands=np.zeros((1,20,29))
        commands[0,:,0]=np.linspace(0,.1,20);commands[0,:,7:]=1.
        lo=np.full(29,-2.);hi=-lo
        initial,a,targets,_,audit=continuous_plan(previous,commands,lo,hi,.1,.1,.025,smoothing_sigma=0,hold_steps=0)
        self.assertLess(audit['valid_lengths'][0],50)
        self.assertLessEqual(np.abs(a).max(),1.)
        p=initial.copy()
        for action,target in zip(a,targets):
            nxt=np.empty_like(p)
            nxt[:,:7]=.9*p[:,:7]+.1*np.clip(p[:,:7]+.025*action[:,:7],lo[:7],hi[:7])
            nxt[:,7:]=.9*p[:,7:]+.1*(lo[7:]+.5*(action[:,7:]+1)*(hi[7:]-lo[7:]))
            np.testing.assert_allclose(nxt,target,atol=5e-7)
            p=nxt
        self.assertGreater(max(audit['hand_lag_max_rad_per_trajectory']),0.)


if __name__=='__main__':unittest.main()
