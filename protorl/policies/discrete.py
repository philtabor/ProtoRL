import torch as T


class DiscretePolicy:
    def __call__(self, probs, with_entropy=False, old_action=None):
        dist = T.distributions.Categorical(probs)
        actions = old_action if old_action is not None else dist.sample()
        log_probs = dist.log_prob(actions)
        if with_entropy:
            entropy = dist.entropy()
            return actions, log_probs, entropy

        return actions, log_probs
