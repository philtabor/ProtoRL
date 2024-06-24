from protorl.agents.dqn import DQNAgent as Agent
from protorl.actor.dqn import DQNActor as Actor
from protorl.learner.dqn import DQNLearner as Learner
from protorl.loops.single import EpisodeLoop
from protorl.policies.epsilon_greedy import EpsilonGreedyPolicy
from protorl.utils.network_utils import make_dqn_networks
from protorl.wrappers.common import make_env
from protorl.memory.generic import initialize_memory


def main():
    env_name = 'CartPole-v1'
    # env_name = 'PongNoFrameskip-v4'
    use_prioritization = True
    use_double = False
    use_dueling = False
    use_atari = False
    env = make_env(env_name, use_atari=use_atari)
    n_games = 1500
    bs = 64
    # 0.3, 0.5 works okay for cartpole
    # 0.25, 0.25 doesn't seem to work
    # 0.25, 0.75 doesn't work
    memory = initialize_memory(max_size=100_000,
                               obs_shape=env.observation_space.shape,
                               batch_size=bs,
                               n_actions=env.action_space.n,
                               action_space='discrete',
                               prioritized=use_prioritization,
                               alpha=0.3,
                               beta=0.5
                               )

    policy = EpsilonGreedyPolicy(n_actions=env.action_space.n, eps_dec=1e-4)

    q_eval, q_target = make_dqn_networks(env, use_double=use_double,
                                         use_dueling=use_dueling,
                                         use_atari=use_atari)
    dqn_actor = Actor(q_eval, q_target, policy)
    q_eval, q_target = make_dqn_networks(env, use_double=use_double,
                                         use_dueling=use_dueling,
                                         use_atari=use_atari)
    dqn_learner = Learner(q_eval, q_target,
                          prioritized=use_prioritization, lr=1e-4)

    agent = Agent(dqn_actor, dqn_learner, prioritized=use_prioritization)
    sample_mode = 'prioritized' if use_prioritization else 'uniform'
    ep_loop = EpisodeLoop(agent, env, memory, sample_mode=sample_mode,
                          prioritized=use_prioritization)
    scores, steps_array = ep_loop.run(n_games)


if __name__ == '__main__':
    main()
