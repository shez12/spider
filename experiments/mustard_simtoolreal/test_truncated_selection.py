import unittest
import numpy as np
from truncated_selection import cutoff,first_hold_end,hold_mask


def demo(T=200,lift_at=50,drop_at=None):
    obs=np.zeros((T,140),dtype=np.float32)
    obs[:,87:90]=[0.,0.,.6]                                 # palm
    obs[:,113:125]=np.tile([0.,0.,-.05],4)                  # object 5cm below palm
    obs[:,98:113]=np.tile([0.,0.,-.05],5)                   # fingertips on the object
    z=np.clip((np.arange(T)-lift_at)*.004,0,.10)            # slow lift, <2m/s
    obs[:,89]+=z
    if drop_at is not None:obs[drop_at:,113:125]=np.tile([0.,0.,-.5],4)
    return obs


class TruncatedSelectionTests(unittest.TestCase):
    def test_hold_starts_after_5cm_rise(self):
        h=hold_mask(demo())
        self.assertEqual(len(h),199)
        first=int(np.argmax(h));self.assertEqual(first+1,63)  # obs index with rise >=5cm (0.004*13=.052)
        self.assertTrue(h[first:].all())

    def test_fast_motion_and_far_object_break_hold(self):
        o=demo();o[100,89]+=.2;h=hold_mask(o)              # 12m/s jump in/out of obs 100
        self.assertFalse(h[98] and h[99])
        self.assertFalse(hold_mask(demo(drop_at=120))[119:].any())

    def test_first_hold_end_respects_valid_length(self):
        h=np.r_[np.zeros(10,bool),np.ones(40,bool)]
        self.assertEqual(first_hold_end(h,50),(39,40))
        self.assertEqual(first_hold_end(h,35),(None,25))

    def test_cutoff_policy(self):
        self.assertEqual(cutoff(900,100),900)              # fits: keep full demonstration
        self.assertEqual(cutoff(1200,300),361)             # truncate hold end + 1 + 60
        self.assertEqual(cutoff(1200,1000),None)           # hold completes too late
        self.assertEqual(cutoff(1200,None),None)           # never held
        self.assertEqual(cutoff(990,950,margin=60),None)   # 1011 > 960


if __name__=='__main__':unittest.main()
