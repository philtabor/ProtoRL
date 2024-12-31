import torch as T
from protorl.learner.base import Learner
from protorl.utils.common import calc_adv_and_returns, swap_and_flatten


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

    def update(self, transitions, batches, ext_adv, ext_ret):
        s, a, r, d, lp, v, v_ = transitions

        if ext_adv is None and ext_ret is None:
            ext_adv, ext_ret = calc_adv_and_returns(v, v_, r, d)
            ext_adv = swap_and_flatten(ext_adv).squeeze()
            ext_ret = swap_and_flatten(ext_ret).squeeze()

        s = swap_and_flatten(s)
        a = swap_and_flatten(a)
        lp = swap_and_flatten(lp)

        for batch in batches:
            indices = batch[0].to(T.int)
            states = s[indices]
            actions = a[indices]
            old_probs = lp[indices]

            if self.action_type == 'continuous':
                alpha, beta = self.actor(states)
                _, new_probs, entropy = self.policy(alpha, beta,
                                                    old_action=actions,
                                                    with_entropy=True)
                last_dim = int(len(new_probs.shape) - 1)
                prob_ratio = T.exp(
                    new_probs.sum(last_dim, keepdims=True) -
                    old_probs.sum(last_dim, keepdims=True)).squeeze()
                entropy = entropy.sum(last_dim, keepdims=True).squeeze()

            elif self.action_type == 'discrete':
                probs = self.actor(states)
                _, new_probs, entropy = self.policy(probs,
                                                    old_action=actions.squeeze(),
                                                    with_entropy=True)
                prob_ratio = T.exp(new_probs - old_probs.squeeze())
            adv = ext_adv[indices]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            weighted_probs = adv * prob_ratio
            weighted_clipped_probs = T.clamp(
                    prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * adv

            actor_loss = -T.min(weighted_probs,
                                weighted_clipped_probs)

            actor_loss -= self.entropy_coefficient * entropy
            self.actor_optimizer.zero_grad()
            actor_loss.mean().backward()
            T.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()

            critic_value = self.critic(states).squeeze()
            critic_loss = (critic_value - ext_ret[indices].squeeze()).\
                pow(2).mean()
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
        return ext_adv, ext_ret

    def anneal_policy_clip(self, n_ep, max_ep):
        self.policy_clip = self.policy_clip_start * (1 - n_ep / max_ep)
