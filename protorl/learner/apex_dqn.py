from protorl.learner.base import Learner
import numpy as np
import torch as T


class ApexLearner(Learner):
    def __init__(self, online_net, target_net, use_double=False,
                 use_prioritization=False, lr=1e-4,
                 gamma=0.99, tau=1.0, config=None,
                 target_replace_interval=1000):
        super().__init__(tau=tau, gamma=gamma)
        if config:
            self.use_double = config.use_double
            self.n = config.n_step
            self.lr = config.learner_lr
            self.prioritized = config.use_prioritization
            self.replace_target_cnt = config.target_replace_interval
        else:
            self.use_double = use_double
            self.prioritized = use_prioritization
            self.n = 1
            self.lr = lr
            self.replace_target_cnt = target_replace_interval

        self.learn_step_counter = 0
        self.q_eval = online_net
        self.q_next = target_net

        self.optimizer = T.optim.RMSprop(self.q_eval.parameters(), lr=lr)
        self.loss = T.nn.MSELoss()

        self.networks_to_transmit = [self.q_eval]

    def save_models(self, fname=None):
        fname = fname or 'models/apex_dqn_learner'
        checkpoint = {
            'q_eval_model_state_dict': self.q_eval.state_dict(),
            'q_next_model_state_dict': self.q_next.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        T.save(checkpoint, fname)

    def load_models(self, fname=None):
        fname = fname or 'models/apex_dqn_learner'
        checkpoint = T.load(fname)
        self.q_eval.load_state_dict(checkpoint['q_eval_model_state_dict'])
        self.q_next.load_state_dict(checkpoint['q_next_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def update_networks(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            src = self.q_eval
            dest = self.q_next
            self.update_network_parameters(src, dest, tau=1.0)

    def update(self, transitions):
        self.optimizer.zero_grad()

        if self.prioritized:
            sample_idx, states, actions, rewards, states_, dones, gammas, weights =\
                transitions
        else:
            states, actions, rewards, states_, dones, gammas = transitions

        indices = np.arange(len(states))

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        states_ = states_.to(self.device)
        dones = dones.to(self.device)
        gammas = gammas.to(self.device)

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

        q_target = rewards + self.gamma*gammas * q_next

        if self.prioritized:
            td_error = np.abs((q_target.detach().cpu().numpy() -
                               q_pred.detach().cpu().numpy()))
            td_error = np.clip(td_error, 0., 1.)
            q_target *= weights.to(self.device)
            q_pred *= weights.to(self.device)

        loss = self.loss(q_target, q_pred).to(self.device)
        loss.backward()
        T.nn.utils.clip_grad_norm_(self.q_eval.parameters(), 1)
        self.optimizer.step()
        self.learn_step_counter += 1
        if self.prioritized:
            return sample_idx, td_error
