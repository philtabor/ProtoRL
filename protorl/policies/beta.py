import torch as T


class BetaPolicy:
    def __init__(self, min_action=-1, max_action=1):
        self.min_action = min_action
        self.max_action = max_action

    def __call__(self, alpha, beta, with_entropy=False, old_action=None):
        probs = T.distributions.Beta(alpha, beta)
        actions = probs.sample()
        if old_action is None:
            a = actions
        else:
            a = old_action
        log_probs = probs.log_prob(a)

        if with_entropy:
            entropy = probs.entropy()
            return actions, log_probs, entropy

        return actions, log_probs
