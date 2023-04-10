import numpy as np
from protorl.agents.ppo import PPOAgent as Agent
from protorl.loops.ppo_episode import EpisodeLoop
from protorl.memory.generic import initialize_memory
from protorl.utils.network_utils import make_ppo_networks
from protorl.policies.discrete import DiscretePolicy
from protorl.wrappers.vec_env import make_vec_envs


def main():
    env_name = 'CartPole-v1'
    n_games = 4000
    bs = 64
    n_threads = 8
    n_epochs = 10
    horizon = 16384
    T = int(horizon // n_threads)

    env = make_vec_envs(env_name, n_threads=n_threads, seed=0)
    actor, critic = make_ppo_networks(env, action_space='discrete')

    fields = ['states', 'actions', 'rewards', 'states_',
              'mask', 'log_probs']
    state_shape = (T, n_threads, *env.observation_space.shape)
    action_shape = probs_shape = (T, n_threads)
    reward_shape = mask_shape = (T, n_threads)
    vals = [np.zeros(state_shape, dtype=np.float32),
            np.zeros(action_shape, dtype=np.float32),
            np.zeros(reward_shape, dtype=np.float32),
            np.zeros(state_shape, dtype=np.float32),
            np.zeros(mask_shape, dtype=np.float32),
            np.zeros(probs_shape, dtype=np.float32)]

    memory = initialize_memory(max_size=T,
                               obs_shape=env.observation_space.shape,
                               batch_size=bs,
                               n_actions=env.action_space.n,
                               action_space='discrete',
                               n_threads=n_threads,
                               fields=fields,
                               vals=vals
                               )

    policy = DiscretePolicy()

    agent = Agent(actor, critic, memory=memory, policy=policy,
                  action_type='discrete', N=T, n_epochs=n_epochs)

    ep_loop = EpisodeLoop(agent, env, n_threads=n_threads, clip_reward=True,
                          extra_functionality=[agent.anneal_policy_clip],
                          adapt_actions=True)

    scores, steps_array = ep_loop.run(n_games)


if __name__ == '__main__':
    main()
