from protorl.learner.base import Learner
import torch as T
import torch.nn.functional as F


class TD3Learner(Learner):
    def __init__(self, actor_network, critic_network_1, critic_network_2,
                 target_actor_network, target_critic_1_network,
                 target_critic_2_network, policy, tau=0.005,
                 gamma=0.99, critic_lr=1e-3, actor_lr=1e-4,
                 actor_update_interval=1):
        super().__init__(gamma=gamma, tau=tau)
        self.actor_update_interval = actor_update_interval
        self.learn_step_counter = 0

        self.policy = policy

        self.actor = actor_network
        self.critic_1 = critic_network_1
        self.critic_2 = critic_network_2

        self.target_actor = target_actor_network
        self.target_critic_1 = target_critic_1_network
        self.target_critic_2 = target_critic_2_network

        self.networks = [net for net in [self.actor, self.critic_1,
                                         self.critic_2,
                                         self.target_actor,
                                         self.target_critic_1,
                                         self.target_critic_2]]

        self.actor_optimizer = T.optim.Adam(self.actor.parameters(),
                                            lr=actor_lr)
        self.critic_1_optimizer = T.optim.Adam(self.critic_1.parameters(),
                                               lr=critic_lr)
        self.critic_2_optimizer = T.optim.Adam(self.critic_2.parameters(),
                                               lr=critic_lr)

        self.update_network_parameters(self.actor, self.target_actor, tau=1.0)
        self.update_network_parameters(self.critic_1,
                                       self.target_critic_1, tau=1.0)
        self.update_network_parameters(self.critic_2,
                                       self.target_critic_2, tau=1.0)

    def save_models(self, fname=None):
        fname = fname or 'models/td3_learner'
        checkpoint = {
            'actor_model_state_dict': self.actor.state_dict(),
            'critic_1_model_state_dict': self.critic_1.state_dict(),
            'critic_2_model_state_dict': self.critic_2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_1_optimizer_state_dict': self.critic_1_optimizer.state_dict(),
            'critic_2_optimizer_state_dict': self.critic_2_optimizer.state_dict(),
        }
        T.save(checkpoint, fname)

    def load_models(self, fname=None):
        fname = fname or 'models/td3_learner'
        checkpoint = T.load(fname)
        self.actor.load_state_dict(checkpoint['actor_model_state_dict'])
        self.critic_1.load_state_dict(checkpoint['critic_1_model_state_dict'])
        self.critic_2.load_state_dict(checkpoint['critic_2_model_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_1_optimizer.load_state_dict(checkpoint['critic_1_optimizer_state_dict'])
        self.critic_2_optimizer.load_state_dict(checkpoint['critic_2_optimizer_state_dict'])

        self.update_network_parameters(self.actor, self.target_actor, tau=1.0)
        self.update_network_parameters(self.critic_1,
                                       self.target_critic_1, tau=1.0)
        self.update_network_parameters(self.critic_2,
                                       self.target_critic_2, tau=1.0)

    def update(self, transitions):
        states, actions, rewards, states_, dones = transitions

        target_mu = self.target_actor(states_)
        target_actions = self.policy(target_mu, scale=0.2,
                                     noise_bounds=[-0.5, 0.5])

        q1_ = self.target_critic_1([states_, target_actions]).squeeze()
        q2_ = self.target_critic_2([states_, target_actions]).squeeze()

        q1 = self.critic_1([states, actions]).squeeze()
        q2 = self.critic_2([states, actions]).squeeze()

        q1_[dones] = 0.0
        q2_[dones] = 0.0

        critic_value_ = T.min(q1_, q2_)

        target = rewards + self.gamma * critic_value_
        target = target.squeeze()

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()

        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()

        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        self.learn_step_counter += 1

        if self.learn_step_counter % self.actor_update_interval != 0:
            return

        self.actor_optimizer.zero_grad()
        actor_q1_loss = self.critic_1([states, self.actor(states)]).squeeze()
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor_optimizer.step()
