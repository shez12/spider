"""Regression tests for the physical warmup acceptance gate."""
import unittest
import mujoco
import numpy as np
from examples.arm_hand.warmup_acceptance import grasp_retention


class GraspGateTest(unittest.TestCase):
    def make(self,thumb=True,floor_z=-.2):
        model=mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
          <geom name="floor" type="box" size="1 1 .02" pos="0 0 {floor_z}"/>
          <site name="sharpa_palm" pos="0 0 .1"/>
          <body name="right_{'thumb' if thumb else 'middle'}_DP" pos="-.06 0 .1"><geom name="collision_sharpa_0" type="box" size=".01 .02 .02"/></body>
          <body name="right_index_DP" pos=".06 0 .1"><geom name="collision_sharpa_1" type="box" size=".01 .02 .02"/></body>
          <body name="right_object" pos="0 0 .1"><freejoint/><geom name="right_object_0" type="sphere" size=".05" mass=".1"/></body>
        </worldbody></mujoco>''')
        return model,dict(qpos=np.tile(model.qpos0,(20,1)),control_dt=np.array(.01))
    def test_retained(self):
        m,t=self.make();self.assertTrue(grasp_retention(m,t)['retained'])
    def test_detached(self):
        m,t=self.make();t['qpos'][:,0]+=.2;self.assertFalse(grasp_retention(m,t)['retained'])
    def test_terminal_release(self):
        m,t=self.make();t['qpos'][-1,0]+=.2;self.assertFalse(grasp_retention(m,t)['retained'])
    def test_table_support_rejected(self):
        m,t=self.make(floor_z=.03);self.assertFalse(grasp_retention(m,t)['retained'])
    def test_missing_opposition_rejected(self):
        m,t=self.make(thumb=False)
        with self.assertRaises(ValueError):grasp_retention(m,t)
    def test_visual_mesh_does_not_establish_grasp(self):
        m,t=self.make()
        # A visual hull touches both fingers; the actual physical object does not.
        model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <geom name="floor" type="box" size="1 1 .02" pos="0 0 -.2"/>
          <site name="sharpa_palm" pos="0 0 .1"/>
          <body name="right_thumb_DP" pos="-.06 0 .1"><geom name="collision_sharpa_0" type="box" size=".01 .02 .02"/></body>
          <body name="right_index_DP" pos=".06 0 .1"><geom name="collision_sharpa_1" type="box" size=".01 .02 .02"/></body>
          <body name="right_object" pos="0 0 .1"><freejoint/>
            <geom name="right_object_0" type="sphere" size=".01" mass=".1"/>
            <geom name="right_object_visual" type="sphere" size=".05" contype="0" conaffinity="0" mass="0"/>
          </body></worldbody></mujoco>''')
        t['qpos']=np.tile(model.qpos0,(20,1))
        self.assertFalse(grasp_retention(model,t)['retained'])

    def test_timebase_and_nonfinite_rejected(self):
        m,t=self.make();t['control_dt']=np.array(.02)
        with self.assertRaises(ValueError):grasp_retention(m,t)
        t['control_dt']=np.array(.01);t['qpos'][-1,0]=np.nan
        with self.assertRaises(ValueError):grasp_retention(m,t)


if __name__=='__main__':unittest.main()
