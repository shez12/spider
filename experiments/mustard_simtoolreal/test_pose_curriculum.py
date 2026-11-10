import math,unittest
from types import SimpleNamespace
import torch
from pose_curriculum import PoseCurriculum,blend,slerp


def fake_env(n=4):
    lo=torch.full((29,),-2.);hi=torch.full((29,),2.)
    saved={}
    robot=SimpleNamespace(data=SimpleNamespace(joint_pos=torch.full((n,29),1.),joint_vel=torch.full((n,29),.4)),
        write_joint_state_to_sim=lambda q,v,env_ids:saved.update(q=q,v=v),
        set_joint_position_target=lambda q,env_ids:saved.update(target=q))
    native_quat=torch.tensor([[math.cos(math.pi/4),math.sin(math.pi/4),0.,0.]]).repeat(n,1)
    obj=SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.tensor([[.1,-.1,.63]]).repeat(n,1),root_quat_w=native_quat),
        write_root_state_to_sim=lambda s,env_ids:saved.update(root=s))
    goal=SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.tensor([[.3,.1,.9]]).repeat(n,1),root_quat_w=native_quat.clone()),
        write_root_pose_to_sim=lambda pose,env_ids:saved.update(goal=pose))
    env=SimpleNamespace(num_envs=n,device='cpu',robot=robot,object=obj,goal_viz=goal,_perm_canon_to_lab=torch.arange(29),
        _joint_lower_canon=lo,_joint_upper_canon=hi,scene=SimpleNamespace(env_origins=torch.zeros((n,3))),
        _table_z_per_env=torch.full((n,),.39),cfg=SimpleNamespace(reset=SimpleNamespace(table_reset_z=.38)),
        _prev_targets=torch.zeros((n,29)),_cur_targets=torch.zeros((n,29)),_object_init_z=torch.zeros(n))
    obs=torch.zeros((n,140));obs[:,:29]=0.;obs[:,29:58]=0.;obs[:,58:87]=.2
    obs[:,87:90]=torch.tensor([0.,0.,.5]);obs[:,113:125]=torch.tensor([0.,0.,.08]).repeat(4)
    obs[:,94:98]=torch.tensor([0.,0.,0.,1.]) # xyzw identity
    return env,(lambda:obs),saved


class PoseCurriculumTests(unittest.TestCase):
    def test_slerp_endpoints_and_midpoint(self):
        q0=torch.tensor([[1.,0.,0.,0.]]);q1=torch.tensor([[0.,1.,0.,0.]])
        torch.testing.assert_close(slerp(q0,q1,torch.zeros(1,1)),q0)
        torch.testing.assert_close(slerp(q0,q1,torch.ones(1,1)),q1)
        half=math.sqrt(.5);torch.testing.assert_close(slerp(q0,q1,torch.full((1,1),.5)),torch.tensor([[half,half,0.,0.]]))
        torch.testing.assert_close(slerp(q0,-q1,torch.ones(1,1)).abs(),q1) # shortest path, sign-invariant

    def test_blend_endpoints(self):
        demo=dict(q=torch.zeros(2,3),v=torch.zeros(2,3),previous=torch.full((2,3),.2),pos=torch.tensor([[0.,0.,.58]]*2),quat=torch.tensor([[1.,0.,0.,0.]]*2))
        native=dict(q=torch.ones(2,3),v=torch.ones(2,3),pos=torch.tensor([[.1,.1,.63]]*2),quat=torch.tensor([[0.,0.,0.,1.]]*2))
        out=blend(demo,native,torch.tensor([0.,1.]))
        for k in ('q','v','previous','pos','quat'):torch.testing.assert_close(out[k][0],demo[k][0])
        for k in ('q','v','pos','quat'):torch.testing.assert_close(out[k][1].abs(),native[k][1].abs())
        torch.testing.assert_close(out['previous'][1],native['q'][1])
        mid=blend(demo,native,torch.tensor([.5,.5]))
        torch.testing.assert_close(mid['pos'][:,2],torch.tensor([.63,.63])) # native drop height once alpha>0

    def test_apply_alpha_zero_restores_demo_and_alpha_one_is_native(self):
        env,reference,saved=fake_env()
        pose=PoseCurriculum(env,reference,alpha=0.,frontier_probability=1.)
        pose.apply(torch.arange(4))
        torch.testing.assert_close(saved['q'],torch.zeros(4,29));torch.testing.assert_close(saved['target'],torch.full((4,29),.2))
        torch.testing.assert_close(saved['root'][:,:3],torch.tensor([[0.,0.,.59]]).repeat(4,1)) # demo z follows table height
        torch.testing.assert_close(saved['root'][:,3:7],torch.tensor([[1.,0.,0.,0.]]).repeat(4,1))
        torch.testing.assert_close(env._object_init_z,torch.full((4,),.59))
        saved.clear();pose.alpha_max=1.;pose.apply(torch.arange(4))
        self.assertEqual(saved,{});self.assertTrue(bool((pose.current==1).all()))
        self.assertTrue(bool((pose.last_completed==0).all()))

    def test_first_goal_blends_from_demo_target_to_native_goal(self):
        env,reference,saved=fake_env()
        demo_goal=lambda ids:(torch.tensor([[0.,0.,.7]]).repeat(len(ids),1),torch.tensor([[1.,0.,0.,0.]]).repeat(len(ids),1))
        pose=PoseCurriculum(env,reference,alpha=0.,demo_goal=demo_goal)
        pose.apply(torch.arange(4))
        torch.testing.assert_close(saved['goal'],torch.tensor([[0.,0.,.7,1.,0.,0.,0.]]).repeat(4,1))
        pose.alpha_max=.5;pose.frontier_probability=1.;pose.apply(torch.arange(4))
        torch.testing.assert_close(saved['goal'][:,:3],torch.tensor([[.15,.05,.8]]).repeat(4,1))
        angle=math.pi/8;torch.testing.assert_close(saved['goal'][:,3:],torch.tensor([[math.cos(angle),math.sin(angle),0.,0.]]).repeat(4,1))
        saved.clear();pose.alpha_max=1.;pose.apply(torch.arange(4));self.assertNotIn('goal',saved) # native goal untouched

    def test_disabled_keeps_native_and_is_never_eligible(self):
        env,reference,saved=fake_env()
        pose=PoseCurriculum(env,reference);pose.enabled=False
        pose.apply(torch.arange(4));pose.apply(torch.arange(4))
        self.assertEqual(saved,{});self.assertFalse(bool(pose.eligible().any()))

    def test_sampling_has_frontier_and_easier_episodes(self):
        env,reference,_=fake_env(4000)
        pose=PoseCurriculum(env,reference,alpha=.4,frontier_probability=.5)
        torch.manual_seed(0);a=pose.sample(4000)
        frontier=torch.tensor(.4)
        self.assertLessEqual(float(a.max()),float(frontier));self.assertAlmostEqual(float((a==frontier).float().mean()),.5,delta=.05)
        self.assertGreater(float((a<.2).float().mean()),.15)

    def test_gate_is_monotonic_and_frontier_only(self):
        env,reference,_=fake_env()
        pose=PoseCurriculum(env,reference,alpha=.2,step=.1,threshold=.5,window=4)
        pose.record(torch.tensor([.1,.1,.1,.1]),torch.tensor([True]*4))
        self.assertEqual(len(pose.outcomes),0);self.assertIsNone(pose.maybe_advance())
        pose.record(torch.tensor([.2,.2,.2,.2]),torch.tensor([True,False,False,False]))
        self.assertIsNone(pose.maybe_advance());self.assertEqual(pose.alpha_max,.2) # poor results hold, never revert
        self.assertEqual(len(pose.outcomes),0);self.assertEqual(pose.last_window_rate,.25) # failed window is discarded
        pose.record(torch.tensor([.2,.2,.2]),torch.tensor([True,True,True]))
        self.assertIsNone(pose.maybe_advance()) # no decision on a partial window, no reuse of old episodes
        pose.record(torch.tensor([.2]),torch.tensor([False]))
        self.assertEqual(pose.maybe_advance(),dict(old_alpha_max=.2,new_alpha_max=.3,window_rate=.75));self.assertEqual(len(pose.outcomes),0)
        pose.alpha_max=.95;pose.record(torch.ones(4)*.95,torch.ones(4,dtype=torch.bool))
        self.assertEqual(pose.maybe_advance()['new_alpha_max'],1.);self.assertIsNone(pose.maybe_advance())

    def test_tolerance_eligibility_requires_fully_native_episode(self):
        env,reference,_=fake_env()
        pose=PoseCurriculum(env,reference);pose.last_completed=torch.tensor([-1.,.5,1.,.99])
        self.assertEqual(pose.eligible().tolist(),[False,False,True,False])


if __name__=='__main__':unittest.main()
