from protorl.agents.dueling import DuelingDQNAgent as Agent
from protorl.loops.single import EpisodeLoop
from protorl.memory.generic import initialize_memory
from protorl.utils.network_utils import make_dqn_networks
from protorl.policies.epsilon_greedy import EpsilonGreedyPolicy
from protorl.wrappers.common import make_env
from protorl.args import parse_args


def main():
    args = parse_args()
    # env_name = 'PongNoFrameskip-v4'
    env_name = 'CartPole-v1'
    use_double = True
    use_atari = False
    use_prioritization = True
    use_dueling = True
    env = make_env(env_name,  use_atari=use_atari)

    q_online, q_target = make_dqn_networks(env, use_dueling=use_dueling,
                                           use_double=use_double,
                                           use_atari=use_atari)

    memory = initialize_memory(max_size=100_000,
                               obs_shape=env.observation_space.shape,
                               batch_size=args.bs,
                               n_actions=env.action_space.n,
                               action_space='discrete',
                               prioritized=use_prioritization
                               )
    policy = EpsilonGreedyPolicy(n_actions=env.action_space.n,
                                 eps_dec=args.eps_dec)
    agent = Agent(q_online, q_target, memory, policy, use_double=True)
    ep_loop = EpisodeLoop(agent, env)

    scores, steps_array = ep_loop.run(args.n_games)


if __name__ == '__main__':
    main()
