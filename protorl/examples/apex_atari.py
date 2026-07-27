import os
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"

import torch as T

import torch.multiprocessing as mp
from protorl.policies.epsilon_greedy import EpsilonGreedyPolicy
from protorl.wrappers.common import make_env
from protorl.utils.network_utils import make_dqn_networks
from protorl.runners.apex_atari_actor import actor_fn
from protorl.runners.apex_atari_learner import learner_fn
from protorl.actor.apex_dqn import ApexActor as Actor
from protorl.learner.apex_dqn import ApexLearner as Learner
from protorl.config.general import Config
from protorl.config.policy import PolicyConfig


def main():

    config = Config(env_name='PongNoFrameskip-v4',
                    use_prioritization=True,
                    use_atari=True,
                    total_time=36000,
                    load_checkpoint=False,
                    evaluate=False,
                    n_threads=28,
                    use_double=True,
                    use_dueling=True,
                    clip_reward=True,
                    memory_capacity=900_000,#524_288,# 1_048_576,#524_288,
                    batch_size=512,
                    alpha=0.6,
                    beta=0.4,
                    learner_lr=1e-4, # 6.25e-5,
                    action_space='discrete',
                    warmup=50_000,
                    n_step=3,
                    # make sure batches to store is a multiple of n_step!
                    n_batches_to_store=192,
                    n_batches_to_sample=32)

    env = make_env(config.env_name, use_atari=config.use_atari)
    observation_shape = env.observation_space.shape
    n_actions = env.action_space.n

    eps_min = 0.01
    eps_max = 0.5
    n = config.n_threads

    epsilons = [eps_min * (eps_max / eps_min) ** (i / (n - 1)) for i in range(n)]
    actor_policy_config = [PolicyConfig(n_actions=n_actions,
                                        eps_dec=0,
                                        eps_start=eps_min,
                                        eps_min=eps_min)]

    for i in range(config.n_threads-1):
        actor_policy_config.append(PolicyConfig(n_actions=n_actions,
                                        eps_dec=0,
                                        eps_start=epsilons[i+1],
                                        eps_min=epsilons[i+1]))

    q_eval, _ = make_dqn_networks(env, config=config)

    env.close()
    del env


    mp.set_start_method("spawn")
    names = [str(i) for i in range(config.n_threads)]

    total_params = sum(p.numel() for p in q_eval.parameters())

    del q_eval

    shared_params = mp.RawArray('f', total_params)

    global_idx = mp.Value('i', 0)

    global_memory_idx = mp.Value('i', 0)

    active_threads = mp.Value('i', config.n_threads)

    update_event = mp.Event()

    index_lock = mp.Lock()

    tree_lock = mp.Lock()

    param_update_lock = mp.Lock()

    fields = ['states', 'actions', 'rewards', 'states_', 'dones', 'gammas']
    max_mem_size = config.memory_capacity
    assert observation_shape is not None # linter complains because the observation shape is type tuple[int, ..] | None
    state_shape = (max_mem_size,*observation_shape)
    action_shape = done_shape = reward_shape = (max_mem_size)

    vals = [T.zeros(state_shape, dtype=T.uint8).pin_memory().share_memory_(),
            T.zeros(action_shape, dtype=T.int).pin_memory().share_memory_(),
            T.zeros(reward_shape, dtype=T.float).pin_memory().share_memory_(),
            T.zeros(state_shape, dtype=T.uint8).pin_memory().share_memory_(),
            T.zeros(done_shape, dtype=T.bool).pin_memory().share_memory_(),
            T.zeros(reward_shape, dtype=T.float).pin_memory().share_memory_()]

    shared_sum_tree = T.zeros(2*max_mem_size).share_memory_()
    ps = []
    ps.append(mp.Process(target=learner_fn, args=(Learner,
                                                  make_dqn_networks,
                                                  config,
                                                  shared_params,
                                                  update_event,
                                                  active_threads,
                                                  global_idx,
                                                  fields,
                                                  vals,
                                                  observation_shape,
                                                  global_memory_idx,
                                                  index_lock,
                                                  tree_lock,
                                                  shared_sum_tree,
                                                  param_update_lock)))

    for idx, name in enumerate(names):
        ps.append(mp.Process(target=actor_fn, args=(name,
                                                    Actor,
                                                    make_dqn_networks,
                                                    EpsilonGreedyPolicy,
                                                    global_idx,
                                                    shared_params,
                                                    update_event,
                                                    active_threads,
                                                    config,
                                                    actor_policy_config[idx],
                                                    fields, vals,
                                                    global_memory_idx,
                                                    index_lock,
                                                    tree_lock,
                                                    shared_sum_tree,
                                                    param_update_lock)))
    [p.start() for p in ps]
    [p.join() for p in ps]


if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '1'
    T.set_num_threads(1)
    T.set_num_interop_threads(1)
    main()
