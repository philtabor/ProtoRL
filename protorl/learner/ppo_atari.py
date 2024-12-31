import numpy as np
import torch as T
import torch.nn.functional as F
from protorl.learner.base import Learner
from protorl.utils.common import calc_adv_and_returns, swap_and_flatten


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
        self.optimizer = T.optim.Adam(self.actor_critic.parameters(), lr=lr, eps=1e-5)

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

    def update(self, transitions, batches, ext_adv, ext_ret):
        s, a, r, d, lp, v, v_ = transitions
        if ext_adv is None and ext_ret is None:
            ext_adv, ext_ret = \
                calc_adv_and_returns(v, v_, r, d)
            ext_adv = swap_and_flatten(ext_adv).squeeze()
            ext_ret = swap_and_flatten(ext_ret).squeeze()
        s = swap_and_flatten(s)
        a = swap_and_flatten(a).squeeze()
        lp = swap_and_flatten(lp).squeeze()
        a_loss, c_loss = 0, 0
        for batch in batches:
            indices = batch[0].to(T.int)
            states = s[indices]
            actions = a[indices]
            old_probs = lp[indices]
            probs, ext_critic_value = self.actor_critic(states)
            _, new_probs, entropy = self.policy(probs,
                                                old_action=actions,
                                                with_entropy=True)
            prob_ratio = T.exp(new_probs - old_probs)
            adv = ext_adv[indices]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            weighted_probs = adv * prob_ratio
            weighted_clipped_probs = T.clamp(
                    prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * adv
            actor_loss = -T.min(weighted_probs,
                                weighted_clipped_probs)

            actor_loss -= self.entropy_coefficient * entropy
            ext_critic_loss = F.mse_loss(ext_ret[indices], ext_critic_value.squeeze())
            critic_loss = ext_critic_loss
            a_loss += actor_loss.mean().item()
            c_loss += critic_loss.item()
            total_loss = actor_loss.mean() + 0.5 * critic_loss
            self.optimizer.zero_grad()
            total_loss.backward()
            T.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
            self.optimizer.step()
        # print(f"actor loss: {a_loss:.1f} critic loss {c_loss:.1f}")
        return ext_adv, ext_ret


    def anneal_policy_clip(self, n_ep, max_ep):
        self.policy_clip = self.policy_clip_start * (1 - n_ep / max_ep)
