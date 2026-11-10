"""Full vendored SimToolReal SAPG with a demonstration-initialized actor.

Called after the aligned right KUKA/Sharpa Isaac environment is constructed.
"""
import json
import time
from pathlib import Path


def train_sapg(args,env,schema,save):
    import torch
    import yaml
    from rl_games.torch_runner import Runner
    from rl_games.common.algo_observer import AlgoObserver
    from isaacsimenvs.utils.rlgames_utils import register_rlgames_env,EnvStatsAlgoObserver,MultiObserver

    assert env.cfg.observation_space==140
    assert schema.get('residual_decoder',True)
    assert 0<=args.critic_warmup_epochs<args.sapg_epochs
    cfg=yaml.safe_load((args.simtoolreal/'isaacsimenvs/cfg/train/SimToolRealSAPG.yaml').read_text())
    config=cfg['params']['config']
    config.update(name='0_mustard_warmstart_sapg',full_experiment_name='0_mustard_warmstart_sapg',
        train_dir=str(args.output),device=args.sapg_device,device_name=args.sapg_device,
        num_actors=env.num_envs,max_epochs=args.sapg_epochs,save_frequency=10000,save_best_after=10,
        expl_coef_block_size=env.num_envs//6,clip_actions=False,mixed_precision=False,
        learning_rate=1e-5,lr_schedule='linear',e_clip=.05,mini_epochs=4,minibatch_size_per_env=16,
        use_experimental_cv=False)
    config['central_value_config'].update(minibatch_size_per_env=16,learning_rate=5e-5,mini_epochs=4)
    # The environment applies the trained residual decoder and clips final joint
    # targets. Do not clip raw Gaussian residuals or rescale them a second time.
    if args.capture_video:
        import gymnasium as gym
        env=gym.wrappers.RecordVideo(env,video_folder=str(args.output/'videos'),
            step_trigger=lambda step:step%args.video_interval==0,video_length=args.video_length,
            name_prefix='target_pose_reaching',disable_logger=True)
    register_rlgames_env(env,rl_device=args.sapg_device,clip_obs=10.,clip_actions=float('inf'))
    save('sapg_config.json',cfg)
    save('sapg_contract.json',dict(actor_observation_dim=140,algorithm='vendored SimToolReal SAPG',
        use_others_experience='lf',expl_type='mixed_expl_learn_param',exploration_groups=6,
        residual_scale=.1,action_contract='previous normalized joint target + 0.1 * raw policy action; final target clipped to joint limits',
        actor_normalizer='frozen at imitation checkpoint statistics for the whole run',
        critic_warmup_epochs=args.critic_warmup_epochs,total_epochs=args.sapg_epochs,
        initial_states='all twenty frozen spatial variants, repeated identically across six groups',
        actor_checkpoint=str(args.checkpoint),reward='native SimToolReal terms with mustard-size keypoints',
        task='target pose reaching, no demonstration labels or trajectory-following loss',
        termination='native success tolerance curriculum and 10 near-goal steps; terminate on success, fall below table -0.05m, hand far, or 600-step timeout',
        domain_randomization='native observation/action/object-state delays and noise, random wrench impulses',
        native_action_pipeline=False,video_interval_policy_steps=args.video_interval if args.capture_video else None))

    class WarmStartObserver(AlgoObserver):
        def after_init(self,agent):
            self.agent=agent;self.started=time.monotonic();self.actor_steps=0
            if not args.resume_sapg:
                old=torch.load(args.checkpoint,map_location=agent.ppo_device,weights_only=False)
                assert not old.get('schema',{}).get('phase_conditioned',False)
                agent.model.load_state_dict(old['model'],strict=True)
                with torch.no_grad():agent.model.a2c_network.sigma.fill_(-2.)
            normalizer=agent.model.running_mean_std
            original_train=normalizer.train
            normalizer.train=lambda mode=True:original_train(False)
            normalizer.eval()
            self.initial_actor={k:v.detach().cpu().clone() for k,v in agent.model.state_dict().items()}
            self.initial_critic={k:v.detach().cpu().clone() for k,v in agent.central_value_net.model.state_dict().items()}
            original_step=agent.optimizer.step
            def step(*a,**kw):
                if agent.epoch_num<=args.critic_warmup_epochs:return None
                self.actor_steps+=1
                return original_step(*a,**kw)
            agent.optimizer.step=step
            save('initialization_check.json',dict(strict_model_load=True,actor_obs_dim=140,
                model_parameter_count=sum(p.numel() for p in agent.model.parameters()),
                central_critic_parameter_count=sum(p.numel() for p in agent.central_value_net.model.parameters()),
                exploration_ids=agent.intr_reward_coef_embd[::agent.intr_coef_block_size,0].cpu().tolist()))

        def after_print_stats(self,frame,epoch_num,total_time):
            a=self.agent
            if epoch_num==args.critic_warmup_epochs:
                # Value normalization is unused by this asymmetric actor and is
                # allowed to change. Actor weights/input statistics must not.
                changed=[k for k,v in a.model.state_dict().items() if not k.startswith('value_mean_std.')
                    and not torch.equal(v.detach().cpu(),self.initial_actor[k])]
                critic_changed=any(not torch.equal(v.detach().cpu(),self.initial_critic[k])
                    for k,v in a.central_value_net.model.state_dict().items())
                save('critic_warmup_check.json',dict(actor_changed_keys=changed,critic_changed=critic_changed,
                    actor_optimizer_steps=self.actor_steps,epoch=epoch_num))
                assert not changed and critic_changed,'Critic-only warmup invariant failed.'
                a.save(str(args.output/'critic_warmup'))
            row=dict(epoch=epoch_num,frames=frame,phase='critic_only' if epoch_num<=args.critic_warmup_epochs else 'joint_sapg',
                actor_optimizer_steps=self.actor_steps,elapsed_s=time.monotonic()-self.started,
                mean_reward=float(a.game_rewards.get_mean()[0]) if a.game_rewards.current_size else None)
            with (args.output/'sapg_progress.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print('SAPG_PROGRESS',json.dumps(row),flush=True)
            if epoch_num%1000==0 or epoch_num==args.sapg_epochs:a.save(str(args.output/'sapg_latest'))

    observer=WarmStartObserver()
    runner=Runner(MultiObserver([EnvStatsAlgoObserver(),observer]))
    runner.load(cfg);runner.reset()
    runner.run(dict(train=True,play=False,checkpoint=str(args.resume_sapg) if args.resume_sapg else None))
    if args.capture_video:env.close()
    save('completed.json',dict(stage='full SAPG training',epochs=args.sapg_epochs,
        critic_warmup_epochs=args.critic_warmup_epochs,actor_optimizer_steps=observer.actor_steps,
        actor_fully_unfrozen=True,critic_trained=True,policy_gradient=True))
