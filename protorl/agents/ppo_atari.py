from protorl.agents.base import Agent


class PPOAtariAgent(Agent):
    def __init__(self, actor, learner):
        super().__init__(actor=actor, learner=learner)

    def choose_action(self, observation):
        action, log_probs = self.actor.choose_action(observation)

        return action, log_probs

    def evaluate_state(self, observation):
        value = self.actor.evaluate_state(observation)

        return value

    def update_networks(self):
        src = self.learner.actor_critic
        dest = self.actor.actor_critic
        self.actor.update_network_parameters(src, dest, tau=1.0)

    def anneal_policy_clip(self, n_ep, max_ep):
        self.learner.anneal_policy_clip(n_ep, max_ep)

    def update(self, transitions, batches, adv, ret):
        adv, ret = self.learner.update(transitions, batches, adv, ret)
        self.update_networks()
        return adv, ret
