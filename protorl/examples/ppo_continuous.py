import numpy as np
from protorl.agents.ppo import PPOAgent as Agent
from protorl.loops.ppo_episode import EpisodeLoop
from protorl.memory.generic import initialize_memory
from protorl.utils.network_utils import make_ppo_networks
from protorl.policies.beta import BetaPolicy
from protorl.wrappers.vec_env import make_vec_envs


def main():
    env_name = 'LunarLanderContinuous-v2'
    n_games = 4000
    bs = 64
    n_threads = 8
    n_epochs = 10
    horizon = 16384
    T = int(horizon // n_threads)

    env = make_vec_envs(env_name, n_threads=n_threads, seed=0)
    actor, critic = make_ppo_networks(env, action_space='continuous')

    fields = ['states', 'actions', 'rewards', 'states_',
              'mask', 'log_probs']
    state_shape = (T, n_threads, *env.observation_space.shape)
    action_shape = probs_shape = (T, n_threads, env.action_space.shape[0])
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
                               n_actions=env.action_space.shape[0],
                               action_space='continuous',
                               n_threads=n_threads,
                               fields=fields,
                               vals=vals
                               )

    policy = BetaPolicy(min_action=env.action_space.low[0],
                        max_action=env.action_space.high[0])

    agent = Agent(actor, critic, memory=memory, policy=policy,
                  action_type='continuous', N=T, n_epochs=n_epochs)

    ep_loop = EpisodeLoop(agent, env, n_threads=n_threads, clip_reward=True,
                          extra_functionality=[agent.anneal_policy_clip],
                          adapt_actions=True)

    scores, steps_array = ep_loop.run(n_games)


if __name__ == '__main__':
    main()
