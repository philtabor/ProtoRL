import numpy as np
from protorl.learner.base import Learner
import torch as T
import torch.nn.functional as F


class ApexLearner(Learner):
    def __init__(self, actor_network, critic_network, target_actor_network,
                 target_critic_network, gamma=0.99, tau=0.001, config=None):
        super().__init__(gamma=gamma, tau=tau)
        self.actor = actor_network
        self.critic = critic_network
        self.target_actor = target_actor_network
        self.target_critic = target_critic_network

        if config:
            self.prioritized = config.use_prioritization
            actor_lr = config.actor_lr
            critic_lr = config.critic_lr
        else:
            self.prioritized = False
            actor_lr = 1e-4
            critic_lr = 1e-3

        self.actor_optimizer = T.optim.Adam(self.actor.parameters(),
                                            lr=actor_lr)
        self.critic_optimizer = T.optim.Adam(self.critic.parameters(),
                                             lr=critic_lr)

        self.update_network_parameters(self.actor, self.target_actor, tau=1.0)
        self.update_network_parameters(self.critic,
                                       self.target_critic, tau=1.0)

        self.networks_to_transmit = [self.actor, self.critic]

    def save_models(self, fname=None):
        fname = fname or 'models/ddpg_learner'
        checkpoint = {
            'actor_model_state_dict': self.actor.state_dict(),
            'critic_model_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }
        T.save(checkpoint, fname)

    def load_models(self, fname=None):
        fname = fname or 'models/ddpg_learner'
        checkpoint = T.load(fname)
        self.actor.load_state_dict(checkpoint['actor_model_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_model_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

        self.update_network_parameters(self.actor, self.target_actor, tau=1.0)
        self.update_network_parameters(self.critic,
                                       self.target_critic, tau=1.0)

    def update_networks(self):
        self.update_network_parameters(self.actor, self.target_actor)
        self.update_network_parameters(self.critic, self.target_critic)

    def update(self, transitions):
        if self.prioritized:
            sample_idx, states, actions, rewards, states_, dones, gammas, _ = transitions
        else:
            states, actions, rewards, states_, dones, gammas = transitions

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        states_ = states_.to(self.device)
        dones = dones.to(self.device)
        gammas = gammas.to(self.device)

        target_actions = self.target_actor(states_)
        critic_value_ = self.target_critic([states_, target_actions]).view(-1)
        critic_value = self.critic([states, actions]).view(-1)

        critic_value_[dones] = 0.0

        target = rewards + self.gamma*gammas * critic_value_

        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()

        if self.prioritized:
            td_error = 0.5 * ((target.detach().cpu().numpy() - \
                               critic_value.detach().cpu().numpy())**2)
            td_error = np.clip(td_error, 0.0, 1.0)

        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic([states, self.actor(states)]).view(-1)
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()

        with T.no_grad():
            for param in self.actor.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-1.0, 1.0)

        self.actor_optimizer.step()

        if self.prioritized:
            return sample_idx, td_error
