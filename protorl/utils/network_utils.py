import math
import torch.nn as nn
from torch.nn import Sequential
from protorl.networks.base import AtariBase, CriticBase
from protorl.networks.base import LinearBase, LinearTanhBase
from protorl.networks.head import DuelingHead, QHead
from protorl.networks.head import BetaHead, SoftmaxHead
from protorl.networks.head import DeterministicHead, MeanAndSigmaHead
from protorl.networks.head import ValueHead
from protorl.utils.common import calculate_conv_output_dims


def make_dqn_networks(env, hidden_layers=[512],
                      use_atari=False, use_dueling=False,
                      config=None, *args, **kwargs):
    if config:
        use_dueling = config.use_dueling
        use_atari = config.use_atari

    head_fn = DuelingHead if use_dueling else QHead
    base_fn = AtariBase if use_atari else LinearBase

    input_dims = calculate_conv_output_dims() if use_atari else [256]
    base = base_fn(input_dims=env.observation_space.shape, *args, **kwargs)
    target_base = base_fn(input_dims=env.observation_space.shape,
                          *args, **kwargs)
    head = head_fn(n_actions=env.action_space.n,
                   input_dims=input_dims,
                   hidden_layers=hidden_layers,
                   *args, **kwargs)
    target_head = head_fn(n_actions=env.action_space.n,
                          input_dims=input_dims,
                          hidden_layers=hidden_layers,
                          *args, **kwargs)
    actor = Sequential(base, head)

    target_actor = Sequential(target_base, target_head)

    return actor, target_actor


def make_ddpg_networks(env,
                       actor_hidden_dims=[256, 256],
                       critic_hidden_dims=[256, 256],
                       config=None):
    if config:
        actor_hidden_dims = config.actor_dims
        critic_hidden_dims = config.critic_dims

    actor_base = LinearBase(input_dims=env.observation_space.shape,
                            hidden_dims=actor_hidden_dims)

    actor_head = DeterministicHead(n_actions=env.action_space.shape[0],
                                   input_dims=[actor_hidden_dims[-1]])

    actor = Sequential(actor_base, actor_head)

    target_actor_base = LinearBase(input_dims=env.observation_space.shape,
                                   hidden_dims=actor_hidden_dims)

    target_actor_head = DeterministicHead(n_actions=env.action_space.shape[0],
                                          input_dims=[actor_hidden_dims[-1]])

    target_actor = Sequential(target_actor_base, target_actor_head)

    input_dims = [env.observation_space.shape[0] + env.action_space.shape[0]]
    critic_base = CriticBase(input_dims=input_dims,
                             hidden_dims=critic_hidden_dims)

    critic_head = ValueHead(input_dims=[critic_hidden_dims[-1]])

    critic = Sequential(critic_base, critic_head)

    target_critic_base = CriticBase(input_dims=input_dims,
                                    hidden_dims=critic_hidden_dims)
    target_critic_head = ValueHead(input_dims=[critic_hidden_dims[-1]])

    target_critic = Sequential(target_critic_base, target_critic_head)

    return actor, critic, target_actor, target_critic


def make_td3_networks(env):
    actor_base = LinearBase(hidden_dims=[400, 300],
                            input_dims=env.observation_space.shape)

    actor_head = DeterministicHead(n_actions=env.action_space.shape[0],
                                   input_dims=[300])
    actor = Sequential(actor_base, actor_head)

    target_actor_base = LinearBase(hidden_dims=[400, 300],
                                   input_dims=env.observation_space.shape)

    target_actor_head = DeterministicHead(n_actions=env.action_space.shape[0],
                                          input_dims=[300])
    target_actor = Sequential(target_actor_base, target_actor_head)

    input_dims = [env.observation_space.shape[0] + env.action_space.shape[0]]
    critic_base_1 = CriticBase(hidden_dims=[400, 300],
                               input_dims=input_dims)
    critic_head_1 = ValueHead(input_dims=[300])

    critic_1 = Sequential(critic_base_1, critic_head_1)

    critic_base_2 = CriticBase(hidden_dims=[400, 300],
                               input_dims=input_dims)
    critic_head_2 = ValueHead(input_dims=[300])
    critic_2 = Sequential(critic_base_2, critic_head_2)

    target_critic_1_base = CriticBase(input_dims=input_dims,
                                      hidden_dims=[400, 300])
    target_critic_1_head = ValueHead(input_dims=[300])
    target_critic_1 = Sequential(target_critic_1_base, target_critic_1_head)

    target_critic_2_base = CriticBase(input_dims=input_dims,
                                      hidden_dims=[400, 300])
    target_critic_2_head = ValueHead(input_dims=[300])
    target_critic_2 = Sequential(target_critic_2_base, target_critic_2_head)

    return actor, critic_1, critic_2, target_actor,\
        target_critic_1, target_critic_2


def make_sac_networks(env):
    actor_base = LinearBase(input_dims=env.observation_space.shape)

    actor_head = MeanAndSigmaHead(n_actions=env.action_space.shape[0])
    actor = Sequential(actor_base, actor_head)

    input_dims = [env.observation_space.shape[0] + env.action_space.shape[0]]
    critic_base_1 = CriticBase(input_dims=input_dims)
    critic_head_1 = ValueHead()
    critic_1 = Sequential(critic_base_1, critic_head_1)

    critic_base_2 = CriticBase(input_dims=input_dims)
    critic_head_2 = ValueHead()
    critic_2 = Sequential(critic_base_2, critic_head_2)

    value_base = LinearBase(input_dims=env.observation_space.shape)
    value_head = ValueHead()
    value = Sequential(value_base, value_head)

    target_value_base = LinearBase(input_dims=env.observation_space.shape)
    target_value_head = ValueHead()
    target_value = Sequential(target_value_base, target_value_head)

    return actor, critic_1, critic_2, value, target_value


def make_ppo_networks(env, action_space, actor_hidden_dims=[128, 128],
                      critic_hidden_dims=[128, 128],
                      uniform_init=False, orthogonal_init=False):
    actor_base = LinearTanhBase(input_dims=env.observation_space.shape,
                                hidden_dims=actor_hidden_dims)
    critic_base = LinearTanhBase(hidden_dims=critic_hidden_dims,
                                 input_dims=env.observation_space.shape)
    critic_head = ValueHead(input_dims=[critic_hidden_dims[-1]])

    if action_space == 'discrete':
        actor_head = SoftmaxHead(n_actions=env.action_space.n,
                                 input_dims=[actor_hidden_dims[-1]])
    elif action_space == 'continuous':
        actor_head = BetaHead(input_dims=[actor_hidden_dims[-1]],
                              n_actions=env.action_space.shape[0])

    actor = Sequential(actor_base, actor_head)
    critic = Sequential(critic_base, critic_head)

    if uniform_init:
        for m in actor.modules():
            if isinstance(m, nn.Linear):
                stdv = 1. / math.sqrt(m.weight.size(1))
                nn.init.uniform_(m.weight, -stdv, stdv)
                if m.bias is not None:
                    m.bias.data.uniform_(-stdv, stdv)

        for m in critic.modules():
            if isinstance(m, nn.Linear):
                stdv = 1. / math.sqrt(m.weight.size(1))
                nn.init.uniform_(m.weight, -stdv, stdv)
                if m.bias is not None:
                    m.bias.data.uniform_(-stdv, stdv)

    if orthogonal_init:
        for m in actor.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

        for m in critic.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    return actor, critic

def make_apex_ddpg_networks(env,
                            actor_hidden_dims=[256, 256],
                            critic_hidden_dims=[256, 256],
                            config=None, *args, **kwargs):
    if config:
        actor_hidden_dims = config.actor_dims
        critic_hidden_dims = config.critic_dims

    actor_base = LinearTanhBase(input_dims=env.observation_space.shape,
                                hidden_dims=actor_hidden_dims,
                                *args, **kwargs)

    actor_head = DeterministicHead(n_actions=env.action_space.shape[0],
                                   input_dims=[actor_hidden_dims[-1]],
                                   *args, **kwargs)

    actor = Sequential(actor_base, actor_head)

    target_actor_base = LinearTanhBase(input_dims=env.observation_space.shape,
                                       hidden_dims=actor_hidden_dims,
                                       *args, **kwargs)

    target_actor_head = DeterministicHead(n_actions=env.action_space.shape[0],
                                          input_dims=[actor_hidden_dims[-1]],
                                          *args, **kwargs)

    target_actor = Sequential(target_actor_base, target_actor_head)

    input_dims = [env.observation_space.shape[0] + env.action_space.shape[0]]
    critic_base = CriticBase(input_dims=input_dims,
                             hidden_dims=critic_hidden_dims,
                             *args, **kwargs)

    critic_head = ValueHead(input_dims=[critic_hidden_dims[-1]],
                            *args, **kwargs)

    critic = Sequential(critic_base, critic_head)

    target_critic_base = CriticBase(input_dims=input_dims,
                                    hidden_dims=critic_hidden_dims,
                                    *args, **kwargs)
    target_critic_head = ValueHead(input_dims=[critic_hidden_dims[-1]],
                                   *args, **kwargs)

    target_critic = Sequential(target_critic_base, target_critic_head)

    return actor, critic, target_actor, target_critic

