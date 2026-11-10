"""Cross-check differentiable helpers against MuJoCo, including derivatives."""
import unittest
import tempfile
from pathlib import Path
import mujoco
import numpy as np
import torch
from examples.arm_hand.world_model import ResidualDynamics, CompactResidualDynamics, usable_states, load_model, corrected_transition
from examples.arm_hand.world_model_kinematics import Kinematics
from examples.arm_hand.world_model_collision import ContactPenalty
from examples.arm_hand.world_model_policy import ReferencePolicy, source_frame_weights


class PhysicsTests(unittest.TestCase):
    def test_ballistic_integrator_matches_mujoco(self):
        model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body><joint type="hinge"/><geom size=".1"/></body><body pos="0 0 1"><freejoint/><geom size=".1"/></body></worldbody><actuator><position joint=""/></actuator></mujoco>'''.replace('<joint type="hinge"/>','<joint name="joint" type="hinge"/>').replace('joint=""','joint="joint"'))
        rng=np.random.default_rng(2);q=rng.normal(size=model.nq);q[-4:]/=np.linalg.norm(q[-4:]);v=rng.normal(size=model.nv)
        expected=q.copy();mujoco.mj_integratePos(model,expected,v,.01)
        net=ResidualDynamics(dict(nq=8,nv=7,nu=1,context_dim=1),16)
        state=torch.tensor(np.r_[q,v],dtype=torch.float32)[None]
        result=net.baseline(state)[0,:8].detach().numpy()
        np.testing.assert_allclose(result,expected,atol=1e-7)

    def test_original_contact_gradients_match_finite_difference(self):
        xml='<mujoco><compiler angle="radian"/><worldbody><geom name="floor" type="plane" pos="0 0 -1" size="1 1 .1"/><body><geom name="collision_sharpa_fixed" pos=".2 .17 .2" size=".1"/></body><body><joint name="hinge" axis="0 0 1"/><geom name="collision_sharpa_moving" pos=".2 0 .2" size=".1"/></body></worldbody><actuator><position joint="hinge"/></actuator></mujoco>'
        model=mujoco.MjModel.from_xml_string(xml);penalty=ContactPenalty(model,1)
        q=torch.tensor([[.3],[.4]],dtype=torch.float64,requires_grad=True)
        result=penalty(q);result.sum().backward();eps=1e-6
        finite=(penalty(q.detach()+eps)-penalty(q.detach()-eps))/(2*eps)
        self.assertTrue((result>0).all())
        torch.testing.assert_close(q.grad[:,0],finite,atol=1e-8,rtol=1e-6)
        # Contacts with the object are deliberately not part of the rejection geometry.
        self.assertIsNone(penalty.category(model.geom('floor').id,model.geom('floor').id))

    def test_source_sampling_is_balanced_across_repeated_trajectories(self):
        # Two source A trajectories and one source B trajectory, 110:1 frame imbalance.
        weights=source_frame_weights(['a','b','a'],[1000,10,100],'cpu')
        torch.testing.assert_close(weights[:1000].sum()+weights[1010:].sum(),weights[1000:1010].sum())
        torch.testing.assert_close(weights[0],weights[1010])
        with self.assertRaises(ValueError):source_frame_weights(['a'],[0],'cpu')

    def test_cached_policy_equals_full_scene_input(self):
        torch.manual_seed(113)
        spec=dict(nq=8,nv=7,nu=1,context_dim=3);policy=ReferencePolicy(spec,16)
        torch.nn.init.normal_(policy.net[-1].weight,std=.1)
        policy.mean.copy_(torch.randn_like(policy.mean));policy.scale.copy_(torch.rand_like(policy.scale)+.1)
        for parameter in policy.parameters():parameter.requires_grad_(False)
        state=torch.randn(4,15);nominal=torch.randn(4,1);next_goal=torch.randn(4,8);future_goal=torch.randn(4,8);context=torch.randn(3)
        low=torch.tensor([-10.]);high=torch.tensor([10.])
        expected=policy(state,nominal,next_goal,future_goal,context.expand(4,-1),low,high)
        actual=policy.with_context(state,nominal,next_goal,future_goal,policy.prepare_context(context),low,high)
        torch.testing.assert_close(actual,expected,atol=1e-7,rtol=1e-6)

    def test_real_transition_calibration_preserves_gradients(self):
        observed=torch.zeros(2,15);observed[:,4]=1;observed[:,0]=.3
        prediction=torch.randn(2,15,requires_grad=True)
        bias=observed-prediction.detach()
        corrected=corrected_transition(prediction,bias,8)
        torch.testing.assert_close(corrected,observed,atol=1e-7,rtol=1e-6)
        corrected[:,0].sum().backward()
        torch.testing.assert_close(prediction.grad[:,0],torch.ones(2))

    def test_compact_checkpoint_keeps_feature_selection(self):
        spec=dict(nq=8,nv=7,nu=1,context_dim=3)
        ids=list(range(16))+[17];net=CompactResidualDynamics(spec,16,ids)
        torch.nn.init.normal_(net.net[-1].weight,std=.01)
        state=torch.zeros(2,15);state[:,4]=1
        action=torch.ones(2,1);context=torch.randn(2,3)
        expected=net(state,action,context)
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'model.pt'
            torch.save(dict(spec=spec,width=16,architecture='compact',model=net.state_dict()),path)
            restored,_=load_model(path,'cpu')
            torch.testing.assert_close(restored(state,action,context),expected)
            self.assertEqual(restored.input_indices.tolist(),ids)

    def test_normal_failure_is_kept_but_explosion_is_rejected(self):
        spec=dict(nq=8,nv=7,nu=1)
        state=np.zeros((3,15));state[:,4]=1
        state[1,3]=-.2;state[1,11]=-4  # Ordinary falling object.
        state[2,1]=1e11;state[2,-1]=1e13
        np.testing.assert_array_equal(usable_states(state,spec),[True,True,False])

    def test_fk_and_gradient_match_mujoco(self):
        model=mujoco.MjModel.from_xml_string('''<mujoco><compiler angle="radian"/><worldbody>
        <body pos=".1 .2 .3" quat=".9238795 0 0 .3826834"><joint name="h" type="hinge" pos=".03 0 0" axis="0 1 0" ref=".2"/><geom size=".1"/>
        <body pos=".2 0 .1" quat=".9238795 .3826834 0 0"><joint name="s" type="slide" axis="0 1 0" ref=".05"/><geom size=".1"/><site name="tip" pos=".1 .1 .1" quat=".9238795 0 .3826834 0"/></body></body>
        </worldbody></mujoco>''')
        fk=Kinematics(model,['tip'],'cpu');data=mujoco.MjData(model)
        q=torch.tensor([[.4,.08]],requires_grad=True)
        pos,rot=fk(q);data.qpos[:]=q.detach().numpy()[0];mujoco.mj_forward(model,data)
        np.testing.assert_allclose(pos.detach().numpy()[0,0],data.site_xpos[0],atol=1e-6)
        np.testing.assert_allclose(rot.detach().numpy()[0,0],data.site_xmat[0].reshape(3,3),atol=1e-6)
        pos.sum().backward();finite=[];eps=1e-5
        for j in range(2):
            values=[]
            for sign in [-1,1]:
                data.qpos[:]=q.detach().numpy()[0];data.qpos[j]+=sign*eps;mujoco.mj_forward(model,data);values.append(data.site_xpos[0].sum())
            finite.append((values[1]-values[0])/(2*eps))
        np.testing.assert_allclose(q.grad.numpy()[0],finite,atol=1e-5)


if __name__=='__main__':unittest.main()
