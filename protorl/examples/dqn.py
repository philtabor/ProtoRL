from protorl.agents.dqn import DQNAgent as Agent
from protorl.loops.single import EpisodeLoop
from protorl.policies.epsilon_greedy import EpsilonGreedyPolicy
from protorl.utils.network_utils import make_dqn_networks
from protorl.wrappers.common import make_env
from protorl.memory.generic import initialize_memory
from protorl.args import parse_args


def main():
    args = parse_args()
    env = make_env(args.env, use_atari=args.use_atari)

    q_eval, q_target = make_dqn_networks(env, use_double=False,
                                         use_dueling=False,
                                         use_atari=args.use_atari)

    memory = initialize_memory(max_size=100_000,
                               obs_shape=env.observation_space.shape,
                               batch_size=args.bs,
                               n_actions=env.action_space.n,
                               action_space='discrete',
                               prioritized=args.prioritized
                               )

    policy = EpsilonGreedyPolicy(n_actions=env.action_space.n,
                                 eps_dec=args.eps_dec)
    agent = Agent(q_eval, q_target, memory, policy, use_double=args.use_double,
                  prioritized=args.prioritized)
    ep_loop = EpisodeLoop(agent, env)
    scores, steps_array = ep_loop.run(args.n_games)


if __name__ == '__main__':
    main()
