from protorl.learner.base import Learner
import numpy as np
import torch as T


class DQNLearner(Learner):
    def __init__(self, eval_net, target_net, use_double=False,
                 tau=1.0, gamma=0.99, lr=1e-4, prioritized=False):
        super().__init__(gamma=gamma, tau=tau)
        self.use_double = use_double
        self.prioritized = prioritized

        self.q_eval = eval_net
        self.q_next = target_net
        self.networks = [net for net in [self.q_eval, self.q_next]]

        self.optimizer = T.optim.Adam(self.q_eval.parameters(), lr=lr)
        self.loss = T.nn.MSELoss()

    def update(self, transitions):
        self.optimizer.zero_grad()

        if self.prioritized:
            sample_idx, states, actions, rewards, states_, dones, weights =\
                    transitions
        else:
            states, actions, rewards, states_, dones = transitions
        indices = np.arange(len(states))
        q_pred = self.q_eval.forward(states)[indices, actions]

        q_next = self.q_next(states_)
        q_next[dones] = 0.0

        if self.use_double:
            q_eval = self.q_eval(states_)

            max_actions = T.argmax(q_eval, dim=1)
            q_next = q_next[indices, max_actions]
        else:
            q_next = q_next.max(dim=1)[0]

        q_target = rewards + self.gamma * q_next

        if self.prioritized:
            td_error = np.abs((q_target.detach().cpu().numpy() -
                               q_pred.detach().cpu().numpy()))
            td_error = np.clip(td_error, 0., 1.)

            q_target *= weights
            q_pred *= weights

        loss = self.loss(q_target, q_pred).to(self.device)
        loss.backward()
        self.optimizer.step()

        if self.prioritized:
            return sample_idx, td_error
