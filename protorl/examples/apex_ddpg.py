import os
import numpy as np
import torch.multiprocessing as mp
from protorl.policies.noisy_deterministic import NoisyDeterministicPolicy
from protorl.wrappers.common import make_env
from protorl.utils.network_utils import make_apex_ddpg_networks
from protorl.runners.apex_actor import actor_fn
from protorl.runners.apex_learner import learner_fn
from protorl.actor.apex_ddpg import ApexActor as Actor
from protorl.learner.apex_ddpg import ApexLearner as Learner
from protorl.memory.server import memory_server_process
from protorl.config.general import Config
from protorl.config.policy import PolicyConfig


def main():
    config = Config(env_name='InvertedDoublePendulum-v4',
                    use_prioritization=True,
                    total_time=600,
                    load_checkpoint=False,
                    evaluate=False,
                    n_threads=4,
                    clip_reward=False,
                    memory_capacity=2_000_000,
                    batch_size=64,
                    alpha=0.6,
                    actor_lr=1e-4,
                    critic_lr=1e-4,
                    action_space='continuous',
                    warmup=10_000,
                    target_replace_interval=1000,
                    actor_dims=[400, 300],
                    critic_dims=[300, 200],
                    n_step=3,
                    # make sure batches to store is a multiple of n_step!
                    n_batches_to_store=90,
                    n_batches_to_sample=32)

    env = make_env(config.env_name, use_atari=config.use_atari)
    observation_shape = env.observation_space.shape
    actor_policy_config = [PolicyConfig(noise=0.001)]
    config.n_actions = env.action_space.shape[0]

    for _ in range(config.n_threads-1):
        actor_policy_config.append(PolicyConfig(noise=0.3))

    actor, critic, target_actor, target_critic = \
        make_apex_ddpg_networks(env, config=config, device=config.learner_device)

    apex_learner = Learner(actor, critic, target_actor, target_critic, config=config)

    env.close()
    del env

    mp.set_start_method("spawn")
    names = [str(i) for i in range(config.n_threads)]

    total_params = sum(p.numel() for p in apex_learner.actor.parameters()) +\
                   sum(p.numel() for p in apex_learner.critic.parameters())
    shared_params = mp.RawArray('f', total_params)

    global_idx = mp.Value('i', 0)

    active_threads = mp.Value('i', config.n_threads)

    update_event = mp.Event()

    request_queue = mp.Queue()
    response_queue = mp.Queue()

    extra_fields = ['gammas']
    extra_vals = [np.zeros(config.memory_capacity, dtype=np.float32)]
    ps = [mp.Process(target=memory_server_process,
                     args=(observation_shape, config,
                           request_queue, response_queue,
                           extra_fields, extra_vals))]

    ps.append(mp.Process(target=learner_fn, args=(apex_learner,
                                                  config,
                                                  shared_params,
                                                  update_event,
                                                  active_threads,
                                                  request_queue,
                                                  response_queue,
                                                  global_idx)))

    for idx, name in enumerate(names):
        ps.append(mp.Process(target=actor_fn, args=(name,
                                                    Actor,
                                                    make_apex_ddpg_networks,
                                                    NoisyDeterministicPolicy,
                                                    global_idx,
                                                    shared_params,
                                                    update_event,
                                                    active_threads,
                                                    request_queue,
                                                    response_queue,
                                                    config,
                                                    actor_policy_config[idx])))
    [p.start() for p in ps]
    [p.join() for p in ps]


if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '1'
    main()
