from protorl.learner.base import Learner
import torch as T


class R2D2Learner(Learner):
    def __init__(self, online_net, target_net, use_double=False,
                 use_prioritization=False, lr=1e-4,
                 gamma=0.997, tau=1.0, config=None,
                 target_replace_interval=2500):
        super().__init__(tau=tau, gamma=gamma, device=config.learner_device)
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

    def rescale_q(self, q, eps=1e-3):
        h = T.sign(q) * (T.sqrt(T.abs(q) + 1) - T.ones_like(q))\
                     + eps * q
        return h

    def inverse_rescale_q(self, q):
        # neglect the small epsilon term!
        sign_q = T.sign(q)
        abs_q = T.abs(q)
        abs_h = (abs_q + 1)**2 - 1
        inverse_h = sign_q * abs_h
        return inverse_h

    def update(self, transitions):
        weights = None
        sample_idx = None
        td_error = None

        self.optimizer.zero_grad()
        sample_idx, states, actions, rewards, states_, dones, gammas, hx, cx, weights =\
            transitions

        states = states.to(device=self.device, dtype=T.float).squeeze(2)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        states_ = states_.to(device=self.device, dtype=T.float).squeeze(2)
        dones = dones.to(self.device)
        gammas = gammas.to(self.device)

        states /= 255.0
        states_ /= 255.0

        hx = hx.transpose(0, 1)
        cx = cx.transpose(0, 1)

        # M1 fix: the acting path feeds the LSTM the PREVIOUS action/reward
        # (choose_action gets (a_{t-1}, r_{t-1}) with s_t), so training must pair
        # s_t with (a_{t-1}, r_{t-1}) too. Training on (s_t, a_t, r_t) while acting
        # on (s_t, a_{t-1}, r_{t-1}) makes the greedy argmax out-of-context.
        # prev_a[t] = a_{t-1} (zero for t=0: no predecessor inside the stored row,
        # matching the fresh-episode convention of the stored pre-hiddens).
        prev_a = T.cat([T.zeros_like(actions[:, :1]), actions[:, :-1]], dim=1)
        prev_r = T.cat([T.zeros_like(rewards[:, :1]), rewards[:, :-1]], dim=1)

        burn_in_length = 40
        batch_size = states[:, :burn_in_length, ...].shape[0]
        seq_burn = states[:, :burn_in_length, ...].shape[1]
        seq_learn = states[:, burn_in_length:, ...].shape[1]

        # Successor states s_{t+n} pair with a_{t+n-1}. a_120..a_123 were never
        # stored (row holds a_0..a_119), so the last n-1 positions get zeros;
        # those tail positions are already approximate pending the H1 rework.
        succ_a = T.cat([actions[:, burn_in_length + self.n - 1:],
                        T.zeros(batch_size, self.n - 1, dtype=actions.dtype,
                                device=actions.device)], dim=1)
        succ_r = T.cat([rewards[:, burn_in_length + self.n - 1:],
                        T.zeros(batch_size, self.n - 1, dtype=rewards.dtype,
                                device=rewards.device)], dim=1)

        with T.no_grad():
            _, _, (hx_burn, cx_burn) = self.q_eval(states[:, :burn_in_length, ...].reshape((batch_size * seq_burn, 4, 84, 84)),
                                                   hx, cx,
                                                   # M1: was (a_t, r_t); burn-in must replay the
                                                   # same (s_t, a_{t-1}, r_{t-1}) inputs as acting
                                                   # actions[:, :burn_in_length],
                                                   # rewards[:, :burn_in_length].unsqueeze(2))
                                                   prev_a[:, :burn_in_length],
                                                   prev_r[:, :burn_in_length].unsqueeze(2))

            V_s_, A_s_, _ = self.q_next(states_[:, burn_in_length:, ...].reshape((batch_size * seq_learn, 4, 84, 84)),
                                        hx_burn, cx_burn,
                                        # M1: s_{t+n} pairs with a_{t+n-1}, not a_t
                                        # actions[:, burn_in_length:],
                                        # rewards[:, burn_in_length:].unsqueeze(2))
                                        succ_a, succ_r.unsqueeze(2))

            V_s_eval, A_s_eval, _ = self.q_eval(states_[:, burn_in_length:, ...].reshape((batch_size*seq_learn, 4,84,84)),
                                                hx_burn, cx_burn,
                                                # actions[:, burn_in_length:],
                                                # rewards[:, burn_in_length:].unsqueeze(2))
                                                succ_a, succ_r.unsqueeze(2))


        V_s, A_s, _ = self.q_eval(states[:, burn_in_length:, ...].reshape((batch_size * seq_learn, 4, 84, 84)),
                                  hx_burn, cx_burn,
                                  # M1: learn window s_t pairs with a_{t-1} — same inputs
                                  # the policy saw when it acted on s_t
                                  # actions[:, burn_in_length:],
                                  # rewards[:, burn_in_length:].unsqueeze(2))
                                  prev_a[:, burn_in_length:],
                                  prev_r[:, burn_in_length:].unsqueeze(2))


        # print(f"V shape {V_s.shape} A shape {A_s.shape} A mean shape {A_s.mean(dim=1, keepdim=True).shape}")
        q_pred = T.add(V_s,
                       (A_s - A_s.mean(dim=2,
                                       keepdim=True)))#[indices, actions]#.to(T.long)]
        q_pred = q_pred.gather(-1, actions.long()[:, burn_in_length:].unsqueeze(-1)).squeeze(-1)
        q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=2, keepdim=True)))

        q_next[dones.bool()[:, burn_in_length:]] = 0.0



        q_eval = T.add(V_s_eval,
                       (A_s_eval - A_s_eval.mean(dim=2, keepdim=True)))
        max_actions = T.argmax(q_eval, dim=-1)
        q_next = q_next.gather(dim=-1, index=max_actions.unsqueeze(-1)).squeeze(-1)

        # the gamma coefficients are calculated by the nstep boostrap
        td_target = rewards[:, burn_in_length:] + gammas[:, burn_in_length:]*self.inverse_rescale_q(q_next)

        q_target = self.rescale_q(td_target)
        td = q_target - q_pred
        per_sample_loss = td.square()

        # C3 fix: the old code returned td.mean() — a single signed, clamped scalar
        # broadcast to ALL sampled rows, which collapsed prioritization to ~uniform.
        # Paper: p = eta*max_i|delta_i| + (1-eta)*mean_i|delta_i| per sequence, eta=0.9.
        # td is in rescaled space, which is what the paper prioritizes on.
        # OLD (dead): td_error = T.abs(q_target - q_pred); T.clamp(td_error, 1e-5, 1.)
        # OLD (dead): td_error = td.mean(); T.clamp(td_error, 0., 1.)
        assert weights is not None
        # weights shape is [batch_size] per_sample_loss size is [batch_size, seq_length]
        per_sample_loss *= weights.to(self.device).unsqueeze(-1)
        loss = per_sample_loss.mean()
        td_error = 0.9 * td.abs().max(dim=1).values + 0.1 * td.abs().mean(dim=1)
        # floor only: the old upper clamp at 1.0 saturated max|delta| as the Q-value
        # range grows, making the most informative sequences indistinguishable from
        # each other in the tree (PER / paper use raw |delta|)
        # OLD: T.clamp(td_error, 1e-5, 1.)
        td_error = T.clamp(td_error, min=1e-5)

        loss.backward()
        T.nn.utils.clip_grad_norm_(self.q_eval.parameters(), 10)
        self.optimizer.step()
        self.learn_step_counter += 1
        assert td_error is not None
        assert sample_idx is not None
        return sample_idx, td_error.detach().cpu()
