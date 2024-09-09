import torch as T
from protorl.learner.base import Learner
from protorl.utils.common import convert_arrays_to_tensors
from protorl.utils.common import calc_adv_and_returns


class PPOLearner(Learner):
    def __init__(self, actor_net, critic_net, action_type, policy,
                 gamma=0.99, lr=1E-4, gae_lambda=0.95, entropy_coeff=0,
                 policy_clip=0.2):
        super().__init__(gamma=gamma)
        self.policy_clip = policy_clip
        self.gae_lambda = gae_lambda
        self.step_counter = 0
        self.entropy_coefficient = entropy_coeff
        self.action_type = action_type
        self.policy_clip_start = policy_clip
        self.policy = policy
        self.actor = actor_net
        self.critic = critic_net
        self.networks = [net for net in [self.actor, self.critic]]

        self.actor_optimizer = T.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = T.optim.Adam(self.critic.parameters(), lr=lr)

    def save_models(self, fname=None):
        fname = fname or 'models/ppo_' + self.action_type + '_learner'
        checkpoint = {
            'actor_model_state_dict': self.actor.state_dict(),
            'critic_model_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }
        T.save(checkpoint, fname)

    def load_models(self, fname=None):
        fname = fname or 'models/ppo_' + self.action_type + '_learner'
        checkpoint = T.load(fname)
        self.actor.load_state_dict(checkpoint['actor_model_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_model_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

    def update(self, transitions, batches):
        s, a, r, s_, d, _ = transitions

        with T.no_grad():
            values = self.critic(s).squeeze()
            values_ = self.critic(s_).squeeze()

        adv, returns = calc_adv_and_returns(values, values_, r, d)

        for batch in batches:
            indices, states, actions, _, _, _, old_probs = batch
            if self.action_type == 'continuous':
                alpha, beta = self.actor(states)
                _, new_probs, entropy = self.policy(alpha, beta,
                                                    old_action=actions,
                                                    with_entropy=True)
                last_dim = int(len(new_probs.shape) - 1)
                prob_ratio = T.exp(
                    new_probs.sum(last_dim, keepdims=True) -
                    old_probs.sum(last_dim, keepdims=True))

                entropy = entropy.sum(last_dim, keepdims=True)

            elif self.action_type == 'discrete':
                probs = self.actor(states)
                _, new_probs, entropy = self.policy(probs,
                                                    old_action=actions,
                                                    with_entropy=True)
                prob_ratio = T.exp(new_probs - old_probs)
            a = adv[indices].view(prob_ratio.shape)
            weighted_probs = a * prob_ratio
            weighted_clipped_probs = T.clamp(
                    prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * a

            actor_loss = -T.min(weighted_probs,
                                weighted_clipped_probs)

            actor_loss -= self.entropy_coefficient * entropy

            self.actor_optimizer.zero_grad()
            actor_loss.mean().backward()
            T.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()

            critic_value = self.critic(states).squeeze()
            critic_loss = (critic_value - returns[indices].squeeze()).\
                pow(2).mean()
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

    def anneal_policy_clip(self, n_ep, max_ep):
        self.policy_clip = self.policy_clip_start * (1 - n_ep / max_ep)
