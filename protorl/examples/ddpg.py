from protorl.agents.ddpg import DDPGAgent as Agent
from protorl.loops.single import EpisodeLoop
from protorl.memory.generic import initialize_memory
from protorl.utils.network_utils import make_ddpg_networks
from protorl.policies.noisy_deterministic import NoisyDeterministicPolicy
from protorl.wrappers.common import make_env
from protorl.args import parse_args


def main():
    args = parse_args()
    env_name = 'LunarLanderContinuous-v2'

    env = make_env(env_name)

    actor, critic, target_actor, target_critic = make_ddpg_networks(env)

    memory = initialize_memory(max_size=100_000,
                               obs_shape=env.observation_space.shape,
                               batch_size=args.bs,
                               n_actions=env.action_space.shape[0],
                               action_space='continuous',
                               )
    policy = NoisyDeterministicPolicy(n_actions=env.action_space.shape[0],
                                      min_action=env.action_space.low[0],
                                      max_action=env.action_space.high[0])
    agent = Agent(actor, critic, target_actor, target_critic, memory, policy)
    ep_loop = EpisodeLoop(agent, env)

    scores, steps_array = ep_loop.run(args.n_games)


if __name__ == '__main__':
    main()
