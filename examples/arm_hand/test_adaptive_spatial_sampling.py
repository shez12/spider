"""Meaningful invariants for the isolated spatial sampling pilot."""
import unittest
import numpy as np
from examples.arm_hand.adaptive_spatial_sampling import proposals,coverage,SCALE,guidance


class SamplingTests(unittest.TestCase):
    def test_mixture_bounds_novelty_determinism(self):
        feedback=[dict(stage='physical_success',point=[0.,0.,0.])]
        a=proposals(feedback,16,7,True);self.assertEqual(a,proposals(feedback,16,7,True))
        self.assertEqual(sum(row['proposal_slot']=='global'for row in a),4)
        points=np.array([row['point']for row in a]);self.assertTrue((np.abs(points/SCALE)<=1).all())
        self.assertGreaterEqual(coverage(points)['min_normalized_distance'],.1)
        self.assertTrue((np.linalg.norm(points/SCALE,axis=1)>=.1).all())

    def test_ik_is_not_physical_success(self):
        rows,stage=guidance([dict(stage='ik_success',point=[0,0,0])]);self.assertEqual(stage,'ik_only')
        rows,stage=guidance([dict(stage='geometry_failure',point=[0,0,0],geometry_score=1)])
        self.assertEqual(stage,'geometry_near_miss_not_success')

    def test_uniform_control_has_same_diversity_guard(self):
        points=[row['point']for row in proposals([],16,8,False)]
        self.assertGreaterEqual(coverage(points)['min_normalized_distance'],.1)


if __name__=='__main__':unittest.main()
