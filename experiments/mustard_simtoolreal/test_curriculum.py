import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from curriculum import next_difficulty,mean_goal_threshold


class CurriculumTests(unittest.TestCase):
    def test_drop_holds_every_difficulty_component(self):
        for targets,level in ((1,0),(2,1),(5,2),(10,3),(50,4)):
            for seconds in (16.,14.,12.,10.):
                decision=next_difficulty(mean_goals=0.,max_targets=targets,target_level=level,
                    target_levels=5,position_level=5,position_levels=8,next_dimension='target',
                    seconds=seconds,restore_native=True,native_tolerance_enabled=False)
                self.assertIsNone(decision)
        self.assertEqual(mean_goal_threshold(1),.8)
        self.assertEqual(mean_goal_threshold(2),1.6)
        self.assertEqual(mean_goal_threshold(50),3.)

    def test_recovery_advances_one_component_at_a_time(self):
        state=dict(mean_goals=3.,max_targets=5,target_level=2,target_levels=5,
            position_level=3,position_levels=8,next_dimension='target',seconds=16.,
            restore_native=True,native_tolerance_enabled=False)
        self.assertEqual(next_difficulty(**state),'target')
        state.update(next_dimension='position')
        self.assertEqual(next_difficulty(**state),'position')
        state.update(target_level=4,position_level=7)
        self.assertEqual(next_difficulty(**state),'timeout')
        state.update(seconds=10.)
        self.assertEqual(next_difficulty(**state),'enable_native_tolerance')
        state.update(native_tolerance_enabled=True)
        self.assertIsNone(next_difficulty(**state))

    def test_original_tolerance_holds_on_drop_and_resumes_on_recovery(self):
        # Execute the original function itself without importing the Isaac app.
        path=Path(__file__).resolve().parents[2]/'third_party/simtoolreal/isaacsimenvs/tasks/simtoolreal/utils/termination_utils.py'
        tree=ast.parse(path.read_text())
        function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='update_tolerance_curriculum')
        namespace={'torch':torch};exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),namespace)
        update=namespace['update_tolerance_curriculum']
        term=SimpleNamespace(tolerance_curriculum_interval=3000,tolerance_curriculum_success_threshold=3.,
            tolerance_curriculum_increment=.9,success_tolerance=.075,target_success_tolerance=.01,eval_success_tolerance=None)
        env=SimpleNamespace(cfg=SimpleNamespace(termination=term),_frame_counter=2999,
            _last_curriculum_update=0,_prev_episode_successes=torch.tensor([2.,2.]),_current_success_tolerance=.075)
        update(env);self.assertEqual(env._current_success_tolerance,.075)
        env._prev_episode_successes.fill_(3.);update(env)
        self.assertAlmostEqual(env._current_success_tolerance,.0675)
        env._frame_counter+=4000;env._prev_episode_successes.zero_();last=env._last_curriculum_update
        update(env);self.assertAlmostEqual(env._current_success_tolerance,.0675)
        self.assertEqual(env._last_curriculum_update,last)
        env._prev_episode_successes.fill_(3.);update(env)
        self.assertAlmostEqual(env._current_success_tolerance,.06075)
        env._current_success_tolerance=.0101;env._frame_counter+=3000;update(env)
        self.assertEqual(env._current_success_tolerance,.01)


if __name__=='__main__':unittest.main()
