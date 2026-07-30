from protorl.learner.base import Learner
import numpy as np
import torch as T
import torch.nn.functional as F


class ApexLearner(Learner):
    def __init__(self, online_net, target_net, use_double=False,
                 use_prioritization=False, lr=1e-4,
                 gamma=0.99, tau=1.0, config=None,
                 target_replace_interval=2500):
        super().__init__(tau=tau, gamma=gamma)
        if config:
            self.use_double = config.use_double
            self.n = config.n_step
            self.lr = config.learner_lr
            self.prioritized = config.use_prioritization
            self.replace_target_cnt = config.target_replace_interval
            self.batch_size = config.batch_size
        else:
            self.use_double = use_double
            self.prioritized = use_prioritization
            self.n = 1
            self.lr = lr
            self.replace_target_cnt = target_replace_interval

        self.learn_step_counter = 0
        self.q_eval = online_net
        self.q_next = target_net

        # self.optimizer = T.optim.RMSprop(self.q_eval.parameters(), lr=lr)
        self.optimizer = T.optim.Adam(self.q_eval.parameters(), lr=lr)
        # self.loss = T.nn.MSELoss()
        self.networks_to_transmit = self.q_eval

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
        weights = None
        sample_idx = None
        td_error = None
        # be defensive, we can't guarantee that self.prioritized won't get changed due to concurrence
        prioritized = self.prioritized

        self.optimizer.zero_grad()
        if prioritized:
            sample_idx, states, actions, rewards, states_, dones, gammas, weights =\
                transitions
        else:
            states, actions, rewards, states_, dones, gammas = transitions

        indices = T.arange(self.batch_size, device=self.device)# np.arange(len(states))
        states = states.to(device=self.device, dtype=T.float).squeeze(1)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        states_ = states_.to(device=self.device, dtype=T.float).squeeze(1)
        dones = dones.to(self.device)
        gammas = gammas.to(self.device)

        states /= 255.0
        states_ /= 255.0

        V_s, A_s = self.q_eval(states)
        V_s_, A_s_ = self.q_next(states_)

        q_pred = T.add(V_s,
                       (A_s - A_s.mean(dim=1,
                                       keepdim=True)))[indices, actions]#.to(T.long)]
        q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True)))
        q_next[dones.bool()] = 0.0

        if self.use_double:
            V_s_eval, A_s_eval = self.q_eval(states_)
            q_eval = T.add(V_s_eval,
                           (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))
            max_actions = T.argmax(q_eval, dim=1)
            q_next = q_next[indices, max_actions]
        else:
            q_next = q_next.max(dim=1)[0]

        q_target = rewards + gammas * q_next # the gamma coefficients are calculated by the nstep boostrap
        td = q_target - q_pred
        per_sample_loss = td.square()

        if self.prioritized:
            td_error = T.abs(q_target - q_pred)
            td_error = T.clamp(td_error, 1e-5, 1.)
            assert weights is not None
            per_sample_loss *= weights.to(self.device)
            loss = per_sample_loss.mean()
            td_error = td.mean()
            td_error = T.clamp(td_error, 0., 1.)
        else:
            loss = per_sample_loss.mean()

        loss.backward()
        T.nn.utils.clip_grad_norm_(self.q_eval.parameters(), 10)
        self.optimizer.step()
        self.learn_step_counter += 1
        if not prioritized:
            return None
        assert td_error is not None
        assert sample_idx is not None
        return sample_idx, td_error.detach().cpu()
