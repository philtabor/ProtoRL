from protorl.actor.base import Actor
import torch as T


class DQNActor(Actor):
    def __init__(self, eval_net, target_net, policy, tau=1.0, device=None):
        super().__init__(policy=policy, tau=tau, device=device)
        self.q_eval = eval_net
        self.q_next = target_net
        self.networks = [net for net in [self.q_eval, self.q_next]]

    def choose_action(self, observation):
        state = T.tensor(observation, dtype=T.float).to(self.device)
        q_values = self.q_eval(state)
        action = self.policy(q_values)
        return action
