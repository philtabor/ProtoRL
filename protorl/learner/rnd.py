import torch as T
import torch.nn.functional as F
from protorl.learner.base import Learner
from protorl.utils.common import calc_adv_and_returns, swap_and_flatten
from protorl.utils.rnd_normalizer import ObsNormalizer, RNormalizer


class RNDLearner(Learner):
    def __init__(self, actor_critic, target_cnn, predictor_cnn,
                 policy,
                 gamma=0.99, lr=1E-4,
                 gae_lambda=0.95, entropy_coeff=0,
                 policy_clip=0.2, image_shape=(1, 84, 84)):
        super().__init__(gamma=gamma)
        self.policy_clip = policy_clip
        self.gae_lambda = gae_lambda
        self.entropy_coefficient = entropy_coeff
        self.policy_clip_start = policy_clip
        self.policy = policy
        self.actor_critic = actor_critic
        self.target_cnn = target_cnn
        self.predictor_cnn = predictor_cnn
        self.optimizer = T.optim.Adam(
                list(self.actor_critic.parameters()) + list(self.predictor_cnn.parameters()), lr=lr)
        self.obs_normalizer = ObsNormalizer(shape=image_shape)
        self.r_normalizer = RNormalizer(shape=(1,))

    def save_models(self, fname=None):
        fname = fname or 'models/rnd_learner'
        checkpoint = {
            'actor_critic_model_state_dict': self.actor_critic.state_dict(),
            'target_model_state_dict': self.target_cnn.state_dict(),
            'predictor_model_state_dict': self.predictor_cnn.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        T.save(checkpoint, fname)

    def load_models(self, fname=None):
        fname = fname or 'models/rnd_learner'
        checkpoint = T.load(fname)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.target_cnn.load_state_dict(checkpoint['target_model_state_dict'])
        self.predictor_cnn.load_state_dict(checkpoint['predictor_model_state_dict'])

    def update(self, transitions, batches, adv, ret):
        s, a, r_e, d, lp, v, v_ = transitions
        s = swap_and_flatten(s)
        a = swap_and_flatten(a).squeeze()
        lp = swap_and_flatten(lp).squeeze()

        predictor_states = s[:, ::4, :, :]

        for state in predictor_states:
            self.obs_normalizer.update(state)
        normalized_s = T.zeros_like(predictor_states, dtype=T.float)
        for idx, state in enumerate(predictor_states):
            normalized_s[idx] = self.obs_normalizer.normalize(state)
        normalized_states = T.clamp(normalized_s, -5.0, 5.0)

        if adv is None and ret is None:
            v_e, v_i = v[:, :, 0], v[:, :, 1]
            v_e_, v_i_ = v_[:, :, 0], v_[:, :, 1]
            v_e = swap_and_flatten(v_e)
            v_e_ = swap_and_flatten(v_e_)
            v_i = swap_and_flatten(v_i)
            v_i_ = swap_and_flatten(v_i_)
            d = swap_and_flatten(d)
            r_e = swap_and_flatten(r_e)

            ext_adv, ext_ret = \
                calc_adv_and_returns(v_e, v_e_, r_e, d, gamma=0.999)
            ext_adv = ext_adv.squeeze()
            ext_ret = ext_ret.squeeze()

            with T.no_grad():
                target_f = self.target_cnn(normalized_states)
                predicted_f = self.predictor_cnn(normalized_states)
                r_i = 0.5 * (target_f - predicted_f).pow(2).mean(1)
            self.r_normalizer.update(r_i.cpu().numpy())
            r_i = T.tensor(self.r_normalizer.normalize(r_i.data.cpu().numpy()),
                           device=self.device).unsqueeze(1)
            int_adv, int_ret = calc_adv_and_returns(v_i, v_i_, r_i, T.ones_like(d), gamma=0.99)
            int_adv = int_adv.squeeze()
            int_ret = int_ret.squeeze()

        else:
            ext_adv, int_adv = adv
            ext_ret, int_ret = ret

        for batch in batches:
            indices = batch[0].to(T.int)
            states = s[indices]
            actions = a[indices]
            old_probs = lp[indices]
            ppo_states = states / 255.0

            probs, ext_critic_value, int_critic_value = self.actor_critic(ppo_states)
            _, new_probs, entropy = self.policy(probs,
                                                old_action=actions,
                                                with_entropy=True)
            prob_ratio = T.exp(new_probs - old_probs)
            adv = 2 * ((ext_adv[indices] - ext_adv[indices].mean()) / \
                    (ext_adv[indices].std() + 1e-4)) + int_adv[indices] # suspect?
            weighted_probs = adv * prob_ratio
            weighted_clipped_probs = T.clamp(
                    prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * adv
            actor_loss = -T.min(weighted_probs,
                                weighted_clipped_probs)

            actor_loss -= self.entropy_coefficient * entropy
            ext_critic_loss = F.mse_loss(ext_ret[indices], ext_critic_value.squeeze())
            int_critic_loss = F.mse_loss(int_ret[indices], int_critic_value.squeeze())
            critic_loss = ext_critic_loss + int_critic_loss

            batch_size = len(states)
            sub_batch_size = int(0.25 * batch_size)
            samples = indices[T.randperm(batch_size)[:sub_batch_size]]
            predictor_loss = ((self.target_cnn(normalized_states[samples]).detach() - \
                               self.predictor_cnn(normalized_states[samples]))**2).mean()

            total_loss = actor_loss.mean() + 0.5 * critic_loss + predictor_loss
            self.optimizer.zero_grad()
            total_loss.backward()
            T.nn.utils.clip_grad_norm_(
                    list(self.actor_critic.parameters()) + list(self.predictor_cnn.parameters()),
                    0.5)
            self.optimizer.step()
        return (ext_adv, int_adv), (ext_ret, int_ret)


    def anneal_policy_clip(self, n_ep, max_ep):
        self.policy_clip = self.policy_clip_start * (1 - n_ep / max_ep)
