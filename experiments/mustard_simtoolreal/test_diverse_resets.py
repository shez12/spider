import ast,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from diverse_resets import hold_actions,DiverseResetBank,environment_signature,table_top_z


class DiverseResetTests(unittest.TestCase):
    def test_tolerance_excludes_cached_and_unfinished_episodes(self):
        bank=object.__new__(DiverseResetBank)
        bank.last_completed_kind=torch.tensor([-1,0,1,2,3,0])
        self.assertEqual(bank.curriculum_eligible_mask().tolist(),[False,True,False,False,False,True])

    def test_table_top_is_center_plus_half_thickness(self):
        env=SimpleNamespace(table=SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.tensor([[0.,0.,.38],[2.,0.,1.38]]),
            root_quat_w=torch.tensor([[1.,0.,0.,0.],[1.,0.,0.,0.]]))),scene=SimpleNamespace(env_origins=torch.tensor([[0.,0.,0.],[2.,0.,1.]])))
        torch.testing.assert_close(table_top_z(env),torch.tensor([.53,.53]))

    def test_native_joint_reset_is_interpolation_not_radian_noise(self):
        path=Path(__file__).resolve().parents[2]/'third_party/simtoolreal/isaacsimenvs/tasks/simtoolreal/utils/reset_utils.py'
        node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_randomize_robot_dof_state')
        namespace={'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
        count=4000;lower=torch.linspace(-3.,-.3,29).expand(count,-1);upper=-lower
        default=torch.full((count,29),.1);limits=torch.stack([lower,upper],-1);saved={}
        robot=SimpleNamespace(data=SimpleNamespace(default_joint_pos=default,joint_pos_limits=limits),
            write_joint_state_to_sim=lambda q,v,env_ids:saved.update(q=q,v=v),set_joint_position_target=lambda q,env_ids:None)
        env=SimpleNamespace(robot=robot,cfg=SimpleNamespace(reset=SimpleNamespace(reset_dof_pos_random_interval_arm=.1,
            reset_dof_pos_random_interval_fingers=.1,reset_dof_vel_random_interval=.5)),_arm_joint_ids=list(range(7)),
            _hand_joint_ids=list(range(7,29)),_prev_targets=torch.zeros_like(default),_cur_targets=torch.zeros_like(default))
        torch.manual_seed(42);state=torch.get_rng_state();expected=torch.lerp(default,lower+(upper-lower)*torch.rand_like(default),.1)
        velocity=torch.empty_like(default).uniform_(-.5,.5);torch.set_rng_state(state)
        namespace['_randomize_robot_dof_state'](env,torch.arange(count))
        torch.testing.assert_close(saved['q'],expected);torch.testing.assert_close(saved['v'],velocity)
        self.assertGreater(float((saved['q'][:,:7]-default[:,:7]).abs().max()),.1)
        self.assertLessEqual(float(saved['v'].abs().max()),.5)

    def test_hold_actions_through_actual_vendor_pipeline(self):
        path=Path(__file__).resolve().parents[2]/'third_party/simtoolreal/isaacsimenvs/tasks/simtoolreal/utils/action_utils.py'
        node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='apply_action_pipeline')
        namespace={'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
        lo=torch.full((29,),-1.);hi=torch.full((29,),2.);previous=torch.rand((4,29))*2-1
        env=SimpleNamespace(device='cpu',_prev_targets=previous.clone(),_cur_targets=torch.zeros_like(previous),
            _perm_lab_to_canon=torch.arange(29),_perm_canon_to_lab=torch.arange(29),_joint_lower_canon=lo,_joint_upper_canon=hi,
            _arm_joint_ids=list(range(7)),_hand_joint_ids=list(range(7,29)),_arm_lower=lo[:7],_arm_upper=hi[:7],
            _hand_lower=lo[7:],_hand_upper=hi[7:],step_dt=1/60,domain_randomization=None,
            cfg=SimpleNamespace(domain_randomization=SimpleNamespace(use_action_delay=False),action=SimpleNamespace(
                dof_speed_scale=1.5,arm_moving_average=.1,hand_moving_average=.1)))
        namespace['apply_action_pipeline'](env,hold_actions(env))
        torch.testing.assert_close(env._cur_targets,previous,atol=1e-7,rtol=1e-7)

    def test_reject_wrong_environment_and_missing_kind(self):
        env=SimpleNamespace(device='cpu',num_envs=6,_reset_idx=lambda ids:None)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bank.npz'
            values={k:np.zeros((3,29),dtype='float32') for k in ('q','v','previous','current')}
            values.update(root=np.zeros((3,13),dtype='float32'),goal=np.zeros((3,7),dtype='float32'),table=np.zeros((3,7),dtype='float32'),table_z=np.ones(3,dtype='float32'))
            np.savez(path,**values,kind=np.array([1,2,3]),format_version=np.array(1),environment_sha256=np.array('wrong'))
            with self.assertRaisesRegex(AssertionError,'changed'):DiverseResetBank(env,path)
            np.savez(path,**values,kind=np.array([1,1,2]),format_version=np.array(1),environment_sha256=np.array(environment_signature()))
            with self.assertRaisesRegex(AssertionError,'Each reset'):DiverseResetBank(env,path)


if __name__=='__main__':unittest.main()
