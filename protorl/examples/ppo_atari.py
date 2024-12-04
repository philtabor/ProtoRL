import numpy as np
from protorl.loops.ppo_rollout import EpisodeLoop
from protorl.memory.generic import initialize_memory
from protorl.policies.discrete import DiscretePolicy
from protorl.wrappers.vec_env import make_vec_envs
from protorl.networks.full import PPOAtariNetwork
from protorl.agents.ppo_atari import PPOAtariAgent as Agent
from protorl.actor.ppo_atari import PPOAtariActor as Actor
from protorl.learner.ppo_atari import PPOAtariLearner as Learner


def main():
    env_name = "PongNoFrameskip-v4"
    bs = 64
    n_threads = 8
    n_epochs = 10
    horizon = 16384
    T = int(horizon // n_threads)
    n_batches = int(horizon // bs)

    env = make_vec_envs(env_name, n_threads=n_threads, seed=0, use_atari=True,
                        pixel_env=False, package_to_import='miniworld')

    fields = ['states', 'actions', 'rewards', 'states_',
              'mask', 'log_probs']
    state_shape = (T*n_threads, *env.observation_space.shape)
    action_shape = probs_shape = T*n_threads
    reward_shape = mask_shape = T*n_threads

    vals = [np.zeros(state_shape, dtype=np.float32),
            np.zeros(action_shape, dtype=np.float32),
            np.zeros(reward_shape, dtype=np.float32),
            np.zeros(state_shape, dtype=np.float32),
            np.zeros(mask_shape, dtype=np.float32),
            np.zeros(probs_shape, dtype=np.float32)]

    memory = initialize_memory(max_size= T*n_threads, #T,
                               obs_shape=env.observation_space.shape,
                               batch_size=bs,
                               n_actions=env.action_space.n,
                               action_space='discrete',
                               n_threads=n_threads,
                               fields=fields,
                               vals=vals
                               )

    policy = DiscretePolicy()

    actor_critic = PPOAtariNetwork(env.observation_space.shape, env.action_space.n)
    actor = Actor(actor_critic, policy)

    actor_critic = PPOAtariNetwork(env.observation_space.shape, env.action_space.n)
    learner = Learner(actor_critic, 'discrete', policy, lr=1e-4)

    agent = Agent(actor, learner)

    ep_loop = EpisodeLoop(agent, env, memory, n_epochs, T, n_batches,
                          n_threads=n_threads, clip_reward=False,
                          load_checkpoint=False, evaluate=False)

    scores, steps_array = ep_loop.run(max_steps=10_000_000)


if __name__ == '__main__':
    main()
