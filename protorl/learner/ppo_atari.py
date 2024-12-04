import torch as T
from protorl.learner.base import Learner
from protorl.utils.common import calc_adv_and_returns


class PPOAtariLearner(Learner):
    def __init__(self, actor_critic, action_type, policy,
                 gamma=0.99, lr=1E-4,
                 gae_lambda=0.95, entropy_coeff=0,
                 policy_clip=0.2):
        super().__init__(gamma=gamma)
        self.policy_clip = policy_clip
        self.gae_lambda = gae_lambda
        self.entropy_coefficient = entropy_coeff
        self.action_type = action_type
        self.policy_clip_start = policy_clip
        self.policy = policy
        self.actor_critic = actor_critic
        self.optimizer = T.optim.Adam(self.actor_critic.parameters(), lr=lr)

    def save_models(self, fname=None):
        fname = fname or 'models/ppo_atari_' + self.action_type + '_learner'
        checkpoint = {
            'actor_critic_model_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        T.save(checkpoint, fname)

    def load_models(self, fname=None):
        fname = fname or 'models/ppo_atari_' + self.action_type + '_learner'
        checkpoint = T.load(fname)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def update(self, transitions, batches):
        s, a, r, s_, d, _ = transitions

        with T.no_grad():
            _, extrinsic_values = self.actor_critic(s)
            _, extrinsic_values_ = self.actor_critic(s_)

        extrinsic_adv, extrinsic_returns = \
            calc_adv_and_returns(extrinsic_values.squeeze(), extrinsic_values_.squeeze(), r, d)

        for batch in batches:
            indices, states, actions, _, _, _, old_probs = batch
            probs, ext_critic_value = self.actor_critic(states)
            _, new_probs, entropy = self.policy(probs,
                                                old_action=actions,
                                                with_entropy=True)
            prob_ratio = T.exp(new_probs - old_probs)
            adv = extrinsic_adv.squeeze()
            a = adv[indices].view(prob_ratio.shape)
            weighted_probs = a * prob_ratio
            weighted_clipped_probs = T.clamp(
                    prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * a

            actor_loss = -T.min(weighted_probs,
                                weighted_clipped_probs)

            actor_loss -= self.entropy_coefficient * entropy
            ext_critic_loss = (ext_critic_value.squeeze() - extrinsic_returns[indices].squeeze()).\
               pow(2).mean()
            critic_loss = ext_critic_loss
            total_loss = actor_loss.mean() + 0.5 * critic_loss
            self.optimizer.zero_grad()
            total_loss.backward()
            T.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
            self.optimizer.step()


    def anneal_policy_clip(self, n_ep, max_ep):
        self.policy_clip = self.policy_clip_start * (1 - n_ep / max_ep)
