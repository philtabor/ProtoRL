from protorl.learner.base import Learner
import torch as T
import torch.nn.functional as F


class SACLearner(Learner):
    def __init__(self, actor_network, critic_network_1, critic_network_2,
                 value_network, target_value_network, policy,
                 reward_scale=2, gamma=0.99, actor_lr=3e-4, critic_lr=3e-4,
                 value_lr=3e-4, tau=0.005):
        super().__init__(gamma=gamma, tau=tau)
        self.reward_scale = reward_scale
        self.actor = actor_network
        self.critic_1 = critic_network_1
        self.critic_2 = critic_network_2
        self.value = value_network
        self.target_value = target_value_network

        self.policy = policy

        self.networks = [net for net in [self.actor, self.critic_1,
                                         self.critic_2, self.value,
                                         self.target_value]]

        self.actor_optimizer = T.optim.Adam(self.actor.parameters(),
                                            lr=actor_lr)
        self.critic_1_optimizer = T.optim.Adam(self.critic_1.parameters(),
                                               lr=critic_lr)
        self.critic_2_optimizer = T.optim.Adam(self.critic_2.parameters(),
                                               lr=critic_lr)
        self.value_optimizer = T.optim.Adam(self.value.parameters(),
                                            lr=value_lr)

        self.update_network_parameters(self.value, self.target_value, tau=1.0)

    def save_models(self, fname=None):
        fname = fname or 'models/sac_learner'
        checkpoint = {
            'actor_model_state_dict': self.actor.state_dict(),
            'critic_1_model_state_dict': self.critic_1.state_dict(),
            'critic_2_model_state_dict': self.critic_2.state_dict(),
            'value_model_state_dict': self.value.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_1_optimizer_state_dict': self.critic_1_optimizer.state_dict(),
            'critic_2_optimizer_state_dict': self.critic_2_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
        }
        T.save(checkpoint, fname)

    def load_models(self, fname=None):
        fname = fname or 'models/sac_learner'
        checkpoint = T.load(fname)
        self.actor.load_state_dict(checkpoint['actor_model_state_dict'])
        self.critic_1.load_state_dict(checkpoint['critic_1_model_state_dict'])
        self.critic_2.load_state_dict(checkpoint['critic_2_model_state_dict'])
        self.value.load_state_dict(checkpoint['value_model_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_1_optimizer.load_state_dict(checkpoint['critic_1_optimizer_state_dict'])
        self.critic_2_optimizer.load_state_dict(checkpoint['critic_2_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        self.update_network_parameters(self.value, self.target_value, tau=1.0)

    def update(self, transitions):
        states, actions, rewards, states_, dones = transitions

        value = self.value(states).view(-1)
        value_ = self.target_value(states_).view(-1)
        value_[dones] = 0.0

        # CALCULATE VALUE LOSS #
        mu, sigma = self.actor(states)
        new_actions, log_probs = self.policy(mu, sigma, False)
        log_probs -= T.log(1 - new_actions.pow(2) + 1e-6)
        log_probs = log_probs.sum(1, keepdim=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1([states, new_actions])
        q2_new_policy = self.critic_2([states, new_actions])
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value_optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * (F.mse_loss(value, value_target))
        # value_loss.backward(retain_graph=True)
        value_loss.backward()
        self.value_optimizer.step()

        # CACULATE ACTOR LOSS #
        mu, sigma = self.actor(states)
        new_actions, log_probs = self.policy(mu, sigma, True)
        log_probs -= T.log(1 - new_actions.pow(2) + 1e-6)
        log_probs = log_probs.sum(1, keepdim=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1([states, new_actions])
        q2_new_policy = self.critic_2([states, new_actions])
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor_optimizer.zero_grad()
        # actor_loss.backward(retain_graph=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        # CALCULATE CRITIC LOSS #
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()

        q_hat = self.reward_scale * rewards + self.gamma * value_
        q1_old_policy = self.critic_1([states, actions]).view(-1)
        q2_old_policy = self.critic_2([states, actions]).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)
        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()
