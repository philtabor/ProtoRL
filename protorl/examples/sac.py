import gym
from protorl.agents.sac import SACAgent as Agent
from protorl.loops.single import EpisodeLoop
from protorl.memory.generic import initialize_memory
from protorl.utils.network_utils import make_sac_networks
from protorl.policies.gaussian import GaussianPolicy


def main():
    env_name = 'LunarLanderContinuous-v2'
    env = gym.make(env_name)
    n_games = 1500
    bs = 256

    actor, critic_1, critic_2, value, target_value = \
        make_sac_networks(env)

    memory = initialize_memory(max_size=100_000,
                               obs_shape=env.observation_space.shape,
                               batch_size=bs,
                               n_actions=env.action_space.shape[0],
                               action_space='continuous',
                               )
    policy = GaussianPolicy(min_action=env.action_space.low,
                            max_action=env.action_space.high)
    agent = Agent(actor, critic_1, critic_2, value,
                  target_value, memory, policy)
    ep_loop = EpisodeLoop(agent, env)

    scores, steps_array = ep_loop.run(n_games)


if __name__ == '__main__':
    main()
