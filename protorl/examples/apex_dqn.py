import os
import numpy as np
import torch.multiprocessing as mp
from protorl.policies.epsilon_greedy import EpsilonGreedyPolicy
from protorl.wrappers.common import make_env
from protorl.utils.network_utils import make_dqn_networks
from protorl.runners.apex_actor import actor_fn
from protorl.runners.apex_learner import learner_fn
from protorl.actor.apex_dqn import ApexActor as Actor
from protorl.learner.apex_dqn import ApexLearner as Learner
from protorl.memory.server import memory_server_process
from protorl.config.general import Config
from protorl.config.policy import PolicyConfig


def main():
    config = Config(env_name='CartPole-v1',
                    use_prioritization=False,
                    use_atari=False,
                    total_time=600,
                    load_checkpoint=False,
                    evaluate=False,
                    n_threads=32,
                    use_double=True,
                    use_dueling=True,
                    clip_reward=False,
                    memory_capacity=1_000_000,
                    batch_size=64,
                    alpha=0.6,
                    beta=0.4,
                    learner_lr=1e-4,
                    action_space='discrete',
                    warmup=10_000,
                    n_step=3,
                    # make sure batches to store is a multiple of n_step!
                    n_batches_to_store=90,
                    n_batches_to_sample=32)

    env = make_env(config.env_name, use_atari=config.use_atari)
    observation_shape = env.observation_space.shape
    n_actions = env.action_space.n
    step_size = 0.5 / config.n_threads
    epsilons = [0.1 + i * step_size for i in range(config.n_threads)]
    actor_policy_config = [PolicyConfig(n_actions=n_actions,
                                        eps_dec=0,
                                        eps_start=0.1,
                                        eps_min=0.1)]

    for i in range(config.n_threads-1):
        actor_policy_config.append(PolicyConfig(n_actions=n_actions, 
                                        eps_dec=0,
                                        eps_start=epsilons[i+1],
                                        eps_min=epsilons[i+1]))

    q_eval, q_target = make_dqn_networks(env, config=config)

    apex_learner = Learner(q_eval, q_target, config=config)

    env.close()
    del env


    mp.set_start_method("spawn")
    names = [str(i) for i in range(config.n_threads)]

    total_params = sum(p.numel() for p in apex_learner.q_eval.parameters())

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
                                                    make_dqn_networks,
                                                    EpsilonGreedyPolicy,
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
