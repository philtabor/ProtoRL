from protorl.agents.ddpg import DDPGAgent as Agent
from protorl.actor.ddpg import DDPGActor as Actor
from protorl.learner.ddpg import DDPGLearner as Learner
from protorl.loops.single import EpisodeLoop
from protorl.memory.generic import initialize_memory
from protorl.utils.network_utils import make_ddpg_networks
from protorl.policies.noisy_deterministic import NoisyDeterministicPolicy
from protorl.wrappers.common import make_env


def main():
    env_name = 'LunarLanderContinuous-v2'
    n_games = 1500
    bs = 64

    env = make_env(env_name)

    memory = initialize_memory(max_size=100_000,
                               obs_shape=env.observation_space.shape,
                               batch_size=bs,
                               n_actions=env.action_space.shape[0],
                               action_space='continuous',
                               )
    policy = NoisyDeterministicPolicy(n_actions=env.action_space.shape[0],
                                      min_action=env.action_space.low[0],
                                      max_action=env.action_space.high[0])

    actor, critic, target_actor, target_critic = make_ddpg_networks(env)

    ddpg_actor = Actor(actor, critic, target_actor, target_critic, policy)

    actor, critic, target_actor, target_critic = make_ddpg_networks(env)

    ddpg_learner = Learner(actor, critic, target_actor, target_critic)

    agent = Agent(ddpg_actor, ddpg_learner)

    ep_loop = EpisodeLoop(agent, env, memory)

    scores, steps_array = ep_loop.run(n_games)


if __name__ == '__main__':
    main()
