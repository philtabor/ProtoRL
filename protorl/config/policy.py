class PolicyConfig:
    def __init__(self,
                 n_actions=1,
                 eps_start=1.0,
                 eps_dec=1e-5,
                 eps_min=0.01,
                 min_action=-1,
                 max_action=1,
                 noise=0.1):
        self.n_actions = n_actions
        self.eps_start = eps_start
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.min_action = min_action
        self.max_action = max_action
        self.noise = noise
