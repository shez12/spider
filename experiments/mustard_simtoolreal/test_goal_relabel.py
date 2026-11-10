import math,unittest
import numpy as np
import torch
from goal_relabel import keypoint_offsets,object_keypoints,quat_apply,relabel,sample_goal_keypoints,stratified_split


class GoalRelabelTests(unittest.TestCase):
    def test_quat_apply_matches_rotation(self):
        q=torch.tensor([[math.cos(math.pi/4),0.,0.,math.sin(math.pi/4)]])  # +90 deg about z
        torch.testing.assert_close(quat_apply(q,torch.tensor([[1.,0.,0.]])),torch.tensor([[0.,1.,0.]]),atol=1e-6,rtol=0)

    def test_offsets_follow_native_fixed_size_formula(self):
        off=keypoint_offsets(1.5,(.2,.1,.04))
        torch.testing.assert_close(off[0],torch.tensor([.15,.075,.03]));torch.testing.assert_close(off[3],-off[0])

    def test_sampled_goals_are_rigid_and_inside_volume(self):
        off=keypoint_offsets(1.5,(.2,.1,.04));g=torch.Generator().manual_seed(0)
        kp=sample_goal_keypoints(2000,off,(-.35,-.2,.6),(.35,.2,.95),generator=g)
        center=kp.mean(1)  # corners are symmetric about the goal position
        self.assertTrue(bool(((center>=torch.tensor([-.35,-.2,.6])-1e-6)&(center<=torch.tensor([.35,.2,.95])+1e-6)).all()))
        pair=lambda k:torch.cdist(k,k)
        torch.testing.assert_close(pair(kp),pair(off[None]).expand(2000,-1,-1),atol=1e-5,rtol=0)

    def test_relabel_only_pregrasp_frames_of_active_trajectories(self):
        obs=torch.randn(2,6,140);goal=torch.randn(2,4,3)
        out=relabel(obs,goal,start=2,boundary=torch.tensor([5,5]),active=torch.tensor([True,False]))
        expected=(object_keypoints(obs)-goal[:,None]).reshape(2,6,12)
        torch.testing.assert_close(out[0,:3,125:137],expected[0,:3])     # absolute frames 2,3,4 < 5
        torch.testing.assert_close(out[0,3:],obs[0,3:]);torch.testing.assert_close(out[1],obs[1])
        torch.testing.assert_close(out[...,:125],obs[...,:125]);torch.testing.assert_close(out[...,137:],obs[...,137:])

    def test_stratified_split_is_deterministic_and_per_group(self):
        groups=[1]*20+[2]*5+[3]*30
        train,val=stratified_split(np.arange(55),groups,.1,seed=3)
        self.assertEqual(sorted(np.r_[train,val].tolist()),list(range(55)))
        counts={g:sum(groups[i]==g for i in val) for g in (1,2,3)};self.assertEqual(counts,{1:2,2:0,3:3})
        self.assertEqual(val.tolist(),stratified_split(np.arange(55),groups,.1,seed=3)[1].tolist())


if __name__=='__main__':unittest.main()
