"""Versioned per-group production options; independent experiment outputs are never affected."""
import hashlib
import json
from pathlib import Path


def apply_group_options(args):
    if args.ignore_group_acceleration:return None
    case=Path(args.input_dir).resolve();out=Path(args.output_dir).resolve()
    if out!=case/'gpu':return None
    path=case.parent/'acceleration.json'
    if not path.exists():return None
    config=json.loads(path.read_text())
    if not config.get('enabled'):return None
    if config.get('version')!=1:raise ValueError('Unsupported group acceleration version')
    record=dict(config_path=str(path),config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),strategy='original_spider')
    if args.strict_check_every==0:args.strict_check_every=int(config.get('strict_check_every',0))
    if args.reference_policy is None and args.warm_start is None:
        policy=config.get('policy');expected=config.get('policy_sha256');cache=bool(config.get('cache_policy_context',False))
        checkpoint=out/'checkpoint.pt'
        if args.resume and checkpoint.exists():
            import torch
            previous=torch.load(checkpoint,map_location='cpu',weights_only=False)['optimizer_settings']
            old_hash=previous.get('reference_policy_sha256')
            if old_hash:
                if not policy:raise ValueError('Cannot restore model checkpoint without immutable policy directory')
                policy=str(Path(policy).parent/(old_hash+'.pt'));expected=old_hash
                cache=previous.get('policy_execution')=='cached_scene_context_v1'
                if not Path(policy).exists() or hashlib.sha256(Path(policy).read_bytes()).hexdigest()!=expected:
                    raise ValueError('Historical feedback controller missing/corrupt; preserve checkpoint')
            else:policy=None;cache=False;record['legacy_resume_preserved']=True
        if policy:
            file=Path(policy)
            if file.is_file() and hashlib.sha256(file.read_bytes()).hexdigest()==expected:
                args.reference_policy=file;args.cache_policy_context=cache;record.update(strategy='world_model_feedback',policy_sha256=expected)
            else:record['controller_fallback']='configured policy missing or hash mismatch; retain original sampling'
    if args.reference_policy is not None:record['strategy']='world_model_feedback'
    capture=config.get('transition_root')
    if capture and args.transition_dir is None:
        args.transition_dir=Path(capture)/case.parent.name/case.name
        args.source_group=args.source_group or case.parent.name
        args.transition_every=int(config.get('transition_every',10));args.transition_worlds=int(config.get('transition_worlds',8))
        args.transition_max_shards=int(config.get('transition_max_shards',20))
    record['strict_check_every']=args.strict_check_every
    return record
