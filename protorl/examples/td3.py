from protorl.agents.td3 import TD3Agent as Agent
from protorl.loops.single import EpisodeLoop
from protorl.memory.generic import initialize_memory
from protorl.utils.network_utils import make_td3_networks
from protorl.policies.noisy_deterministic import NoisyDeterministicPolicy
from protorl.wrappers.common import make_env


def main():
    env_name = 'LunarLanderContinuous-v2'
    n_games = 1500
    bs = 100

    env = make_env(env_name)

    actor, critic_1, critic_2,\
        target_actor, target_critic_1, target_critic_2 = make_td3_networks(env)

    memory = initialize_memory(max_size=1_000_000,
                               obs_shape=env.observation_space.shape,
                               batch_size=bs,
                               n_actions=env.action_space.shape[0],
                               action_space='continuous',
                               )

    policy = NoisyDeterministicPolicy(n_actions=env.action_space.shape[0],
                                      min_action=env.action_space.low[0],
                                      max_action=env.action_space.high[0])

    agent = Agent(actor, critic_1, critic_2, target_actor,
                  target_critic_1, target_critic_2, memory, policy)

    ep_loop = EpisodeLoop(agent, env)

    scores, steps_array = ep_loop.run(n_games)


if __name__ == '__main__':
    main()
