from protorl.learner.base import Learner
import numpy as np
import torch as T


class DuelingDQNLearner(Learner):
    def __init__(self, online_net, target_net,
                 use_double=False, prioritized=False, use_atari=False,
                 gamma=0.99, lr=1e-4, tau=1.0):
        super().__init__(tau=tau, gamma=gamma)
        self.use_double = use_double
        self.prioritized = prioritized
        self.use_atari = use_atari

        self.q_eval = online_net
        self.q_next = target_net

        self.optimizer = T.optim.Adam(self.q_eval.parameters(), lr=lr)
        self.loss = T.nn.MSELoss()

    def save_models(self, fname=None):
        fname = fname or 'models/dueling_dqn_learner'
        checkpoint = {
            'q_eval_model_state_dict': self.q_eval.state_dict(),
            'q_next_model_state_dict': self.q_next.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        T.save(checkpoint, fname)

    def load_models(self, fname=None):
        fname = fname or 'models/dueling_dqn_learner'
        checkpoint = T.load(fname)
        self.q_eval.load_state_dict(checkpoint['q_eval_model_state_dict'])
        self.q_next.load_state_dict(checkpoint['q_next_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def update(self, transitions):
        self.optimizer.zero_grad()

        if self.prioritized:
            sample_idx, states, actions, rewards, states_, dones, weights =\
                transitions
        else:
            states, actions, rewards, states_, dones = transitions

        if self.use_atari:
            states /= 255.0
            states_ /= 255.0

        indices = np.arange(len(states))

        V_s, A_s = self.q_eval(states)
        V_s_, A_s_ = self.q_next(states_)


        q_pred = T.add(V_s,
                       (A_s - A_s.mean(dim=1,
                                       keepdim=True)))[indices, actions]
        q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True)))
        q_next[dones] = 0.0

        if self.use_double:
            V_s_eval, A_s_eval = self.q_eval(states_)
            q_eval = T.add(V_s_eval,
                           (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))
            max_actions = T.argmax(q_eval, dim=1)
            q_next = q_next[indices, max_actions]
        else:
            q_next = q_next.max(dim=1)[0]

        q_target = rewards + self.gamma * q_next

        if self.prioritized:
            td_error = T.abs((q_target.detach() -
                               q_pred.detach()))
            td_error = T.clamp(td_error, 0., 1.)
            q_target *= weights
            q_pred *= weights

        loss = self.loss(q_target, q_pred).to(self.device)
        loss.backward()
        self.optimizer.step()

        if self.prioritized:
            return sample_idx.cpu(), td_error.cpu()
