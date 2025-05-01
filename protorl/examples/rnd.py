import numpy as np
from protorl.wrappers.atari import MontezumaInfoWrapper
from protorl.memory.generic import initialize_memory
from protorl.policies.discrete import DiscretePolicy as Policy
from protorl.wrappers.vec_env import make_vec_envs
from protorl.loops.ppo_rollout import EpisodeLoop
from protorl.agents.rnd import RNDAgent as Agent
from protorl.actor.rnd import RNDActor as Actor
from protorl.learner.rnd import RNDLearner as Learner
from protorl.networks.rnd import PPONetwork, RandomCNN, PredictorCNN


def main():
    env_name = "MontezumaRevengeNoFrameskip-v0" # sticky action probability 0.25, v4 has prob=0
    n_threads = 128
    n_epochs = 4
    horizon = 16384
    T = int(horizon // n_threads)
    bs = 64# int(horizon // 4)
    n_batches = int(horizon // bs)

    env = make_vec_envs(env_name,
                        n_threads=n_threads,
                        seed=0,
                        pixel_env=True,
                        scale_obs=False,
                        new_img_shape=(84, 84, 1),
                        use_noops=False,
                        stack_frames=True,
                        terminal_on_life_loss=False,
                        fire_reset_env=False,
                        max_and_skip=True,
                        preprocess_frames=True,
                        additional_wrapper=MontezumaInfoWrapper
                        )
    fields = ['states', 'actions', 'rewards',
              'mask', 'log_probs', 'values', 'values_']
    state_shape = (T, n_threads, *env.observation_space.shape)
    action_shape = probs_shape = (T, n_threads)
    reward_shape = mask_shape = (T, n_threads)
    values_shape = (T, n_threads, 2)

    vals = [np.zeros(state_shape, dtype=np.float32),
            np.zeros(action_shape, dtype=np.float32),
            np.zeros(reward_shape, dtype=np.float32),
            np.zeros(mask_shape, dtype=np.float32),
            np.zeros(probs_shape, dtype=np.float32),
            np.zeros(values_shape, dtype=np.float32),
            np.zeros(values_shape, dtype=np.float32)]

    memory = initialize_memory(max_size=T,
                               obs_shape=env.observation_space.shape,
                               batch_size=bs,
                               n_actions=env.action_space.n,
                               action_space='discrete',
                               n_threads=n_threads,
                               fields=fields,
                               vals=vals
                               )

    actor_critic = PPONetwork(env.observation_space.shape, env.action_space.n)
    policy = Policy()
    actor = Actor(actor_critic, policy)
    random = RandomCNN(input_dims=env.observation_space.shape, input_channels=1)
    predictor = PredictorCNN(input_dims=env.observation_space.shape, input_channels=1)

    learner = Learner(actor_critic, random, predictor, policy, lr=1e-4, entropy_coeff=1e-3,
                      policy_clip=0.1, image_shape=(1, *env.observation_space.shape[1:]))

    agent = Agent(actor, learner)

    ep_loop = EpisodeLoop(agent, env, memory, n_epochs, T, n_batches,
                          n_threads=n_threads, clip_reward=True,
                          load_checkpoint=False, evaluate=False,
                          window=100)

    _, _ = ep_loop.run(max_steps=10_000_000)


if __name__ == '__main__':
    main()
