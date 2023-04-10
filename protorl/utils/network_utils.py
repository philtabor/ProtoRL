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


def make_dqn_networks(env, use_double=False,
                      use_atari=False, use_dueling=False):
    algo = 'dueling_' if use_dueling else ''
    algo += 'ddqn' if use_double else 'dqn'

    head_fn = DuelingHead if use_dueling else QHead
    base_fn = AtariBase if use_atari else LinearBase

    input_dims = calculate_conv_output_dims() if use_atari else [256]

    base = base_fn(name=algo+'_q_base',
                   input_dims=env.observation_space.shape)
    target_base = base_fn(name='target_'+algo+'_q_base',
                          input_dims=env.observation_space.shape)
    head = head_fn(n_actions=env.action_space.n,
                   input_dims=input_dims,
                   name=algo+'_q_head')
    target_head = head_fn(n_actions=env.action_space.n,
                          input_dims=input_dims,
                          name='target_'+algo+'_q_head')
    actor = Sequential(base, head)

    target_actor = Sequential(target_base, target_head)

    return actor, target_actor


def make_ddpg_networks(env):
    actor_base = LinearBase(name='ddpg_actor_base',
                            input_dims=env.observation_space.shape)

    actor_head = DeterministicHead(n_actions=env.action_space.shape[0],
                                   name='ddpg_actor_head')

    actor = Sequential(actor_base, actor_head)

    target_actor_base = LinearBase(name='ddpg_target_actor_base',
                                   input_dims=env.observation_space.shape)

    target_actor_head = DeterministicHead(n_actions=env.action_space.shape[0],
                                          name='ddpg_target_actor_head')

    target_actor = Sequential(target_actor_base, target_actor_head)

    input_dims = [env.observation_space.shape[0] + env.action_space.shape[0]]
    critic_base = CriticBase(name='ddpg_critic_base',
                             input_dims=input_dims)

    critic_head = ValueHead(name='ddpg_critic_head')

    critic = Sequential(critic_base, critic_head)

    target_critic_base = CriticBase(name='ddpg_target_critic_base',
                                    input_dims=input_dims)
    target_critic_head = ValueHead(name='ddpg_target_critic_head')

    target_critic = Sequential(target_critic_base, target_critic_head)

    return actor, critic, target_actor, target_critic


def make_td3_networks(env):
    actor_base = LinearBase(name='td3_actor_base', hidden_dims=[400, 300],
                            input_dims=env.observation_space.shape)

    actor_head = DeterministicHead(n_actions=env.action_space.shape[0],
                                   input_dims=[300],
                                   name='td3_actor_head')
    actor = Sequential(actor_base, actor_head)

    target_actor_base = LinearBase(name='td3_target_actor_base',
                                   hidden_dims=[400, 300],
                                   input_dims=env.observation_space.shape)

    target_actor_head = DeterministicHead(n_actions=env.action_space.shape[0],
                                          input_dims=[300],
                                          name='td3_target_actor_head')
    target_actor = Sequential(target_actor_base, target_actor_head)

    input_dims = [env.observation_space.shape[0] + env.action_space.shape[0]]
    critic_base_1 = CriticBase(name='td3_critic_1_base',
                               hidden_dims=[400, 300],
                               input_dims=input_dims)
    critic_head_1 = ValueHead(name='td3_critic_1_head', input_dims=[300])

    critic_1 = Sequential(critic_base_1, critic_head_1)

    critic_base_2 = CriticBase(name='td3_critic_2_base',
                               hidden_dims=[400, 300],
                               input_dims=input_dims)
    critic_head_2 = ValueHead(name='td3_critic_2_head', input_dims=[300])
    critic_2 = Sequential(critic_base_2, critic_head_2)

    target_critic_1_base = CriticBase(name='td3_target_critic_1_base',
                                      input_dims=input_dims,
                                      hidden_dims=[400, 300])
    target_critic_1_head = ValueHead(name='td3_target_critic_1_head',
                                     input_dims=[300])
    target_critic_1 = Sequential(target_critic_1_base, target_critic_1_head)

    target_critic_2_base = CriticBase(name='td3_target_critic_2_base',
                                      input_dims=input_dims,
                                      hidden_dims=[400, 300])
    target_critic_2_head = ValueHead(name='td3_target_critic_2_head',
                                     input_dims=[300])
    target_critic_2 = Sequential(target_critic_2_base, target_critic_2_head)

    return actor, critic_1, critic_2, target_actor,\
        target_critic_1, target_critic_2


def make_sac_networks(env):
    actor_base = LinearBase(name='sac_actor_base',
                            input_dims=env.observation_space.shape)

    actor_head = MeanAndSigmaHead(n_actions=env.action_space.shape[0],
                                  name='sac_actor_head')
    actor = Sequential(actor_base, actor_head)

    input_dims = [env.observation_space.shape[0] + env.action_space.shape[0]]
    critic_base_1 = CriticBase(name='sac_critic_1_base',
                               input_dims=input_dims)
    critic_head_1 = ValueHead(name='sac_critic_1_head')
    critic_1 = Sequential(critic_base_1, critic_head_1)

    critic_base_2 = CriticBase(name='sac_critic_2_base',
                               input_dims=input_dims)
    critic_head_2 = ValueHead(name='sac_critic_2_head')
    critic_2 = Sequential(critic_base_2, critic_head_2)

    value_base = LinearBase(name='sac_value_base',
                            input_dims=env.observation_space.shape)
    value_head = ValueHead(name='sac_value_head')
    value = Sequential(value_base, value_head)

    target_value_base = LinearBase(name='sac_target_value_base',
                                   input_dims=env.observation_space.shape)
    target_value_head = ValueHead(name='sac_target_value_head')
    target_value = Sequential(target_value_base, target_value_head)

    return actor, critic_1, critic_2, value, target_value


def make_ppo_networks(env, action_space='discrete'):
    actor_base = LinearTanhBase(name='ppo_actor_base',
                                input_dims=env.observation_space.shape,
                                hidden_dims=[128, 128])
    critic_base = LinearTanhBase(name='ppo_critic_base',
                                 hidden_dims=[128, 128],
                                 input_dims=env.observation_space.shape)
    critic_head = ValueHead(name='ppo_critic_head', input_dims=[128])

    if action_space == 'discrete':
        actor_head = SoftmaxHead(n_actions=env.action_space.n,
                                 name='ppo_actor_head', input_dims=[128])
    elif action_space == 'continuous':
        actor_head = BetaHead(name='ppo_actor_head', input_dims=[128],
                              n_actions=env.action_space.shape[0])

    actor = Sequential(actor_base, actor_head)
    critic = Sequential(critic_base, critic_head)

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

    return actor, critic
