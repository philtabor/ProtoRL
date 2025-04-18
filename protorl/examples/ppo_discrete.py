import numpy as np
from protorl.agents.ppo import PPOAgent as Agent
from protorl.actor.ppo import PPOActor as Actor
from protorl.learner.ppo import PPOLearner as Learner
from protorl.loops.ppo_rollout import EpisodeLoop
from protorl.memory.generic import initialize_memory
from protorl.utils.network_utils import make_ppo_networks
from protorl.policies.discrete import DiscretePolicy
from protorl.wrappers.vec_env import make_vec_envs


def main():
    env_name = 'CartPole-v1'
    max_steps = 1_000_000
    bs = 64
    n_threads = 32
    n_epochs = 10
    horizon = 16384
    T = int(horizon // n_threads)
    n_batches = int(horizon // bs)

    env = make_vec_envs(env_name, n_threads=n_threads, seed=0)

    fields = ['states', 'actions', 'rewards',
              'mask', 'log_probs', 'values', 'values_']
    state_shape = (T, n_threads, *env.observation_space.shape)
    action_shape = probs_shape = reward_shape = mask_shape = (T, n_threads)
    reward_shape = mask_shape = (T, n_threads)
    values_shape = (T, n_threads, 1)
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

    policy = DiscretePolicy()

    actor_net, critic_net = make_ppo_networks(env, action_space='discrete', uniform_init=True)
    actor = Actor(actor_net, critic_net, 'discrete', policy)

    actor_net, critic_net = make_ppo_networks(env, action_space='discrete', uniform_init=True)
    learner = Learner(actor_net, critic_net, 'discrete', policy, entropy_coeff=0.0, lr=3e-4)

    agent = Agent(actor, learner)

    ep_loop = EpisodeLoop(agent, env, memory, n_epochs, T, n_batches,
                          n_threads=n_threads, clip_reward=True,
                          extra_functionality=[agent.anneal_policy_clip],
                          load_checkpoint=False, evaluate=False)

    scores, steps_array = ep_loop.run(max_steps=max_steps)


if __name__ == '__main__':
    main()
