import gymnasium as gym
from protorl.agents.sac import SACAgent as Agent
from protorl.actor.sac import SACActor as Actor
from protorl.learner.sac import SACLearner as Learner
from protorl.loops.single import EpisodeLoop
from protorl.memory.generic import initialize_memory
from protorl.utils.network_utils import make_sac_networks
from protorl.policies.gaussian import GaussianPolicy


def main():
    # env_name = 'LunarLanderContinuous-v2'
    env_name = 'InvertedDoublePendulum-v4'
    env = gym.make(env_name)
    n_games = 1500
    bs = 256

    memory = initialize_memory(max_size=100_000,
                               obs_shape=env.observation_space.shape,
                               batch_size=bs,
                               n_actions=env.action_space.shape[0],
                               action_space='continuous',
                               )
    policy = GaussianPolicy(min_action=env.action_space.low,
                            max_action=env.action_space.high)

    actor_net, critic_1, critic_2, value, target_value = \
        make_sac_networks(env)

    actor = Actor(actor_net, critic_1, critic_2, value, target_value, policy)

    actor_net, critic_1, critic_2, value, target_value = \
        make_sac_networks(env)

    learner = Learner(actor_net, critic_1, critic_2,
                      value, target_value, policy)

    agent = Agent(actor, learner)
    ep_loop = EpisodeLoop(agent, env, memory)

    scores, steps_array = ep_loop.run(n_games)


if __name__ == '__main__':
    main()
