import unittest
from unittest.mock import patch
import numpy as np
from examples.arm_hand.warmup_validation_guard import audit_for_optimizer
from examples.arm_hand.warmup_acceptance import valid_warmup

class ValidationGuardTests(unittest.TestCase):
 def test_nonfinite_replay_is_rejection_without_calling_geometry(self):
  trajectory={'qpos':np.zeros((20,36))};trajectory['qpos'][3,29]=np.nan
  with patch('examples.arm_hand.warmup_validation_guard.original_audit',side_effect=AssertionError('Must not audit invalid physics')):report=audit_for_optimizer(None,None,trajectory)
  self.assertTrue(report['invalid_trajectory']);self.assertFalse(valid_warmup(report));self.assertFalse(report['grasp_retention']['retained'])
 def test_finite_audit_is_delegated_without_changes(self):
  t={'qpos':np.zeros((20,36))};expected={'original_report':'unchanged'}
  with patch('examples.arm_hand.warmup_validation_guard.original_audit',return_value=expected) as fn:
   actual=audit_for_optimizer(None,None,t);self.assertIs(actual,expected);fn.assert_called_once_with(None,None,t,'mujoco_warp')
 def test_nonfinite_velocity_and_controls_rejected(self):
  for name in ['qvel','ctrl']:
   t={'qpos':np.zeros((20,36)),name:np.array([[float('inf')]])};self.assertFalse(valid_warmup(audit_for_optimizer(None,None,t)))
if __name__=='__main__':unittest.main()
