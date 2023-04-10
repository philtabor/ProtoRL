from protorl.agents.base import Agent
import torch as T
import torch.nn.functional as F


class DDPGAgent(Agent):
    def __init__(self, actor_network, critic_network, target_actor_network,
                 target_critic_network, memory, policy,
                 gamma=0.99, actor_lr=1e-4, critic_lr=1e-3, tau=0.001):
        super().__init__(memory, policy, gamma, tau)
        self.actor = actor_network
        self.critic = critic_network
        self.target_actor = target_actor_network
        self.target_critic = target_critic_network

        self.networks = [net for net in [self.actor, self.critic,
                                         self.target_actor, self.target_critic,
                                         ]]

        self.actor_optimizer = T.optim.Adam(self.actor.parameters(),
                                            lr=actor_lr)
        self.critic_optimizer = T.optim.Adam(self.critic.parameters(),
                                             lr=critic_lr)

        self.update_network_parameters(self.actor, self.target_actor, tau=1.0)
        self.update_network_parameters(self.critic,
                                       self.target_critic, tau=1.0)

    def choose_action(self, observation):
        state = T.tensor(observation, dtype=T.float, device=self.device)
        mu = self.actor(state)
        actions = self.policy(mu)

        return actions.cpu().detach().numpy()

    def update(self):
        if not self.memory.ready():
            return

        states, actions, rewards, states_, dones = self.sample_memory()

        target_actions = self.target_actor(states_)
        critic_value_ = self.target_critic([states_, target_actions]).view(-1)
        critic_value = self.critic([states, actions]).view(-1)

        critic_value_[dones] = 0.0

        target = rewards + self.gamma * critic_value_

        self.critic_optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss = -self.critic([states, self.actor(states)])
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor_optimizer.step()

        self.update_network_parameters(self.actor, self.target_actor)
        self.update_network_parameters(self.critic, self.target_critic)
