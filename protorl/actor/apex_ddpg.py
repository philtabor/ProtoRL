
import torch as T
from protorl.actor.base import Actor
from protorl.memory.nstep import NStepBuffer
from protorl.utils.common import convert_arrays_to_tensors


class ApexActor(Actor):
    def __init__(self, actor_network, critic_network, target_actor_network,
                 target_critic_network, policy, tau=0.001, device='cpu',
                 gamma=0.99, config=None, name=None):
        super().__init__(policy=policy, tau=tau, device=device)
        self.actor = actor_network
        self.critic = critic_network
        self.target_actor = target_actor_network
        self.target_critic = target_critic_network
        self.name = name

        self.update_network_parameters(self.actor, self.target_actor, tau=1.0)
        self.update_network_parameters(self.critic,
                                       self.target_critic, tau=1.0)

        self.gamma = gamma
        self.local_memory = NStepBuffer(config.n_step, gamma, config.n_batches_to_store)

    def save_models(self, fname=None):
        fname = fname or 'models/ddpg_actor_' + self.name
        checkpoint = {
            'actor_model_state_dict': self.actor.state_dict(),
            'critic_model_state_dict': self.critic.state_dict(),
            'noise': self.policy.noise,
        }
        T.save(checkpoint, fname)

    def load_models(self, fname=None):
        fname = fname or 'models/ddpg_actor_' + self.name
        checkpoint = T.load(fname)
        self.actor.load_state_dict(checkpoint['actor_model_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_model_state_dict'])
        self.policy.noise = checkpoint['noise']

        self.update_network_parameters(self.actor, self.target_actor, tau=1.0)
        self.update_network_parameters(self.critic,
                                       self.target_critic, tau=1.0)

    def choose_action(self, observation):
        state = T.tensor(observation, dtype=T.float, device=self.device)
        mu = self.actor(state)
        actions = self.policy(mu)
        return actions.cpu().detach().numpy()

    def update_networks(self):
        self.update_network_parameters(self.actor, self.target_actor)
        self.update_network_parameters(self.critic, self.target_critic)

    def download_learner_params(self, flat_params):
        pointer = 0
        for param in self.actor.parameters():
            num_param = param.numel()
            param.data = flat_params[pointer:pointer + num_param].view_as(param).clone()
            pointer += num_param

        for param in self.critic.parameters():
            num_param = param.numel()
            param.data = flat_params[pointer:pointer + num_param].view_as(param).clone()
            pointer += num_param

    def store_transition(self, transition):
        self.local_memory.store_transition(transition)

    def get_n_step_returns(self):
        transitions = self.local_memory.compute_n_step_returns()
        return transitions

    @T.no_grad()
    def calculate_priorities(self, transitions):
        exp = convert_arrays_to_tensors(transitions, self.device)

        states, actions, R, states_, dones, gammas = exp

        target_actions = self.target_actor(states_.type(T.float))
        critic_value_ = self.target_critic([states_.type(T.float), target_actions]).view(-1)
        critic_value = self.critic([states.type(T.float), actions]).view(-1)

        critic_value_[dones] = 0.0

        target = R + self.gamma*gammas * critic_value_

        td_errors = 0.5*((target - critic_value)**2).numpy()

        return td_errors
