import torch as T


class GaussianPolicy:
    def __init__(self, min_action=-1, max_action=1):
        self.min_action = min_action
        self.max_action = max_action

    def __call__(self, mu, sigma,
                 reparam=False, with_entropy=False, old_action=None):
        probs = T.distributions.Normal(mu, sigma)
        actions = probs.rsample() if reparam else probs.sample()
        if old_action is None:
            a = actions
        else:
            a = old_action
        log_probs = probs.log_prob(a)
        actions = T.tanh(actions)*T.tensor(self.max_action).to(actions.device)

        if with_entropy:
            entropy = probs.entropy()
            return actions, log_probs, entropy

        return actions, log_probs
