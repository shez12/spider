"""Production configuration must not contaminate experiments or change legacy resumes."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import torch
from examples.arm_hand.runtime_acceleration import apply_group_options


class RuntimeTests(unittest.TestCase):
    def setup_case(self,root):
        root=Path(root);group=root/'group';case=group/'candidate_00001';case.mkdir(parents=True)
        policies=root/'policies';policies.mkdir();content=b'policy-v1';sha=hashlib.sha256(content).hexdigest();policy=policies/(sha+'.pt');policy.write_bytes(content)
        config=dict(version=1,enabled=True,policy=str(policy),policy_sha256=sha,cache_policy_context=True,strict_check_every=10,transition_root=str(root/'transitions'))
        (group/'acceleration.json').write_text(json.dumps(config))
        args=SimpleNamespace(input_dir=case,output_dir=case/'gpu',ignore_group_acceleration=False,strict_check_every=0,
            reference_policy=None,warm_start=None,resume=False,cache_policy_context=False,transition_dir=None,source_group=None,
            transition_every=10,transition_worlds=8,transition_max_shards=20)
        return args,config

    def test_experiment_output_ignores_group_defaults(self):
        with tempfile.TemporaryDirectory() as root:
            args,_=self.setup_case(root);args.output_dir=Path(root)/'independent_experiment/gpu'
            self.assertIsNone(apply_group_options(args));self.assertIsNone(args.reference_policy);self.assertEqual(args.strict_check_every,0)

    def test_fresh_production_case_uses_frozen_policy(self):
        with tempfile.TemporaryDirectory() as root:
            args,c=self.setup_case(root);record=apply_group_options(args)
            self.assertEqual(args.reference_policy,Path(c['policy']));self.assertTrue(args.cache_policy_context)
            self.assertEqual(args.strict_check_every,10);self.assertEqual(record['policy_sha256'],c['policy_sha256'])

    def test_legacy_checkpoint_remains_original_sampling(self):
        with tempfile.TemporaryDirectory() as root:
            args,_=self.setup_case(root);args.resume=True;args.output_dir.mkdir()
            torch.save(dict(optimizer_settings={}),args.output_dir/'checkpoint.pt');record=apply_group_options(args)
            self.assertIsNone(args.reference_policy);self.assertTrue(record['legacy_resume_preserved']);self.assertEqual(args.strict_check_every,10)

    def test_resume_uses_historical_policy_and_execution_mode(self):
        with tempfile.TemporaryDirectory() as root:
            args,c=self.setup_case(root);old=b'old-policy';old_hash=hashlib.sha256(old).hexdigest();old_path=Path(c['policy']).parent/(old_hash+'.pt');old_path.write_bytes(old)
            args.resume=True;args.output_dir.mkdir();torch.save(dict(optimizer_settings=dict(reference_policy_sha256=old_hash)),args.output_dir/'checkpoint.pt')
            record=apply_group_options(args);self.assertEqual(args.reference_policy,old_path);self.assertFalse(args.cache_policy_context);self.assertEqual(record['policy_sha256'],old_hash)


if __name__=='__main__':unittest.main()
