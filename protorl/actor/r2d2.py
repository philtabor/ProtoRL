import numpy as np
import torch as T
from protorl.memory.nstep import NStepBuffer
from protorl.actor.base import Actor
from protorl.utils.common import convert_arrays_to_tensors


class R2D2Actor(Actor):
    def __init__(self, online_net, target_net, policy,
                 gamma=0.99, tau=1.0, device='cpu', config=None,
                 name=None):
        super().__init__(policy=policy, tau=tau, device=device)
        self.q_eval = online_net
        self.q_next = target_net
        self.name = name

        if config:
            self.n = config.n_step
            # self.n_batches_to_store = config.n_batches_to_store
            self.use_double = config.use_double

        self.gamma = gamma
        self.local_memory = NStepBuffer(self.n, self.gamma)

    def save_models(self, fname=None):
        fname = fname or 'models/apex_dqn_actor'
        checkpoint = {
            'q_eval_model_state_dict': self.q_eval.state_dict(),
            'q_next_model_state_dict': self.q_next.state_dict(),
            'epsilon': self.policy.epsilon,
            'epsilon_dec': self.policy.eps_dec,
            'epsilon_min': self.policy.eps_min,
        }
        T.save(checkpoint, fname)

    def load_models(self, fname=None):
        fname = fname or 'models/apex_dqn_actor'
        checkpoint = T.load(fname)
        self.q_eval.load_state_dict(checkpoint['q_eval_model_state_dict'])
        self.q_next.load_state_dict(checkpoint['q_next_model_state_dict'])
        self.policy.epsilon = checkpoint['epsilon']
        self.policy.eps_dec = checkpoint['epsilon_dec']
        self.policy.eps_min = checkpoint['epsilon_min']

    def choose_action(self, observation, hx, cx, last_a, last_r):
        state = T.tensor(observation, dtype=T.float, device=self.device)
        state /= 255.0
        last_a = T.tensor(last_a, dtype=T.long, device=self.device)
        last_r = T.tensor(last_r, dtype=T.float, device=self.device)
        _, advantage, hidden_states = self.q_eval(state, hx, cx, last_a, last_r)
        action = self.policy(advantage).item()
        return action, hidden_states

    def store_transition(self, transition):
        self.local_memory.store_transition(transition)

    def update_networks(self):
        self.update_network_parameters(self.q_eval, self.q_next)

    def download_learner_params(self, flat_params):
        pointer = 0
        for param in self.q_eval.parameters():
            num_params = param.numel()
            param.data = flat_params[pointer:pointer + num_params].view_as(param).clone()
            pointer += num_params

    def get_n_step_returns(self, end_of_episode=False):
        transitions = self.local_memory.compute_n_step_returns(end_of_episode)
        return transitions

    @T.no_grad()
    def calculate_priorities(self, transitions):
        # exp = convert_arrays_to_tensors(transitions, self.device)
        states, actions, R, states_, dones, gammas, hx, cx = transitions

        hx = T.transpose(hx, 0, 1)
        cx = T.transpose(cx, 0, 1)

        hx_current = hx[:, :120, :]
        hx_next = hx[:, 5:125, :]
        cx_current = cx[:, :120, :]
        cx_next = cx[:, 5:125, :]

        states = states.squeeze(1).to(dtype=T.float) / 255.0
        states_ = states_.squeeze(1).to(dtype=T.float) / 250.0

        indices = np.arange(len(states))
        # print(f"r unsqueeze(0): {R.unsqueeze(0).shape} r unsqueeze(1): {R.unsqueeze(1).shape}")
        V_s, A_s, _ = self.q_eval(states, hx_current, cx_current, actions, R.unsqueeze(1))
        V_s_, A_s_, _ = self.q_next(states_, hx_next, cx_next, actions, R.unsqueeze(1))


        q_pred = T.add(V_s.squeeze(1),
                       (A_s.squeeze(1) - A_s.mean(dim=2)))
        q_pred = q_pred[indices, actions]
        q_next = T.add(V_s_.squeeze(1), (A_s_.squeeze(1) - A_s_.mean(dim=2)))
        q_next[dones] = 0.0

        if self.use_double:
            V_s_eval, A_s_eval, _ = self.q_eval(states_, hx_next, cx_next, actions, R.unsqueeze(1))
            q_eval = T.add(V_s_eval.squeeze(1),
                           (A_s_eval.squeeze(1) - A_s_eval.mean(dim=2)))
            max_actions = T.argmax(q_eval, dim=1)
            q_next = q_next[indices, max_actions]
        else:
            q_next = q_next.max(dim=1)[0]

        q_target = R + gammas*q_next

        td_error = np.abs((q_target.detach().cpu().numpy() -
                            q_pred.detach().cpu().numpy()))
        td_error = np.clip(td_error, 0., 1.)
        return td_error
