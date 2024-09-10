import numpy as np
import torch as T
from protorl.memory.nstep import NStepBuffer
from protorl.actor.base import Actor
from protorl.utils.common import convert_arrays_to_tensors


class ApexActor(Actor):
    def __init__(self, online_net, target_net, policy,
                 gamma=0.99, tau=1.0, device='cpu', config=None,
                 name=None):
        super().__init__(policy=policy, tau=tau, device=device)
        self.q_eval = online_net
        self.q_next = target_net
        self.name = name

        if config:
            self.n = config.n_step
            self.n_batches_to_store = config.n_batches_to_store
            self.use_double = config.use_double

        self.gamma = gamma
        self.local_memory = NStepBuffer(self.n, self.gamma, self.n_batches_to_store)

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

    def choose_action(self, observation):
        state = T.tensor(observation, dtype=T.float, device=self.device)
        _, advantage = self.q_eval(state)
        action = self.policy(advantage).item()
        return action

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

    def get_n_step_returns(self):
        transitions = self.local_memory.compute_n_step_returns()
        return transitions

    @T.no_grad()
    def calculate_priorities(self, transitions):
        exp = convert_arrays_to_tensors(transitions, self.device)
        states, actions, R, states_, dones, gammas = exp

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

        q_target = R + self.gamma*gammas*q_next

        td_error = np.abs((q_target.detach().cpu().numpy() -
                            q_pred.detach().cpu().numpy()))
        return td_error
