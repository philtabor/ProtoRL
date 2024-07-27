from protorl.actor.base import Actor
import torch as T


class DDPGActor(Actor):
    def __init__(self, actor_network, critic_network, target_actor_network,
                 target_critic_network, policy, tau=0.001):
        super().__init__(policy=policy, tau=tau)
        self.actor = actor_network
        self.critic = critic_network
        self.target_actor = target_actor_network
        self.target_critic = target_critic_network

        self.networks = [net for net in [self.actor, self.critic,
                                         self.target_actor, self.target_critic,
                                         ]]

        self.update_network_parameters(self.actor, self.target_actor, tau=1.0)
        self.update_network_parameters(self.critic,
                                       self.target_critic, tau=1.0)

    def choose_action(self, observation):
        state = T.tensor(observation, dtype=T.float, device=self.device)
        mu = self.actor(state)
        actions = self.policy(mu)
        return actions.cpu().detach().numpy()
