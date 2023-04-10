from protorl.agents.dueling import DuelingDQNAgent as Agent
from protorl.loops.single import EpisodeLoop
from protorl.memory.generic import initialize_memory
from protorl.utils.network_utils import make_dqn_networks
from protorl.policies.epsilon_greedy import EpsilonGreedyPolicy
from protorl.wrappers.common import make_env


def main():
    env_name = 'PongNoFrameskip-v4'
    # env_name = 'CartPole-v1'
    use_double = True
    use_atari = True
    # use_atari = False
    use_dueling = True
    use_prioritization = True
    env = make_env(env_name, use_atari=use_atari)
    n_games = 1500
    bs = 256

    q_online, q_target = make_dqn_networks(env, use_double=use_double,
                                           use_atari=use_atari,
                                           use_dueling=use_dueling)

    memory = initialize_memory(max_size=100_000,
                               obs_shape=env.observation_space.shape,
                               batch_size=bs,
                               n_actions=env.action_space.n,
                               action_space='discrete',
                               prioritized=use_prioritization
                               )
    policy = EpsilonGreedyPolicy(n_actions=env.action_space.n, eps_dec=1e-4)
    agent = Agent(q_online, q_target, memory, policy, use_double=False)
    ep_loop = EpisodeLoop(agent, env)

    scores, steps_array = ep_loop.run(n_games)


if __name__ == '__main__':
    main()
