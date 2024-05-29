from protorl.actor.base import Actor
import torch as T


class TD3Actor(Actor):
    def __init__(self, actor_network, critic_network_1, critic_network_2,
                 target_actor_network, target_critic_1_network,
                 target_critic_2_network, policy, tau=0.005,
                 warmup=1000):
        super().__init__(policy=policy, tau=tau)
        self.warmup = warmup
        self.n_steps = 0

        self.actor = actor_network
        self.critic_1 = critic_network_1
        self.critic_2 = critic_network_2

        self.target_actor = target_actor_network
        self.target_critic_1 = target_critic_1_network
        self.target_critic_2 = target_critic_2_network

        self.networks = [net for net in [self.actor, self.critic_1,
                                         self.critic_2,
                                         self.target_actor,
                                         self.target_critic_1,
                                         self.target_critic_2]]

        self.update_network_parameters(self.actor, self.target_actor, tau=1.0)
        self.update_network_parameters(self.critic_1,
                                       self.target_critic_1, tau=1.0)
        self.update_network_parameters(self.critic_2,
                                       self.target_critic_2, tau=1.0)

    def choose_action(self, observation):
        state = T.tensor(observation, dtype=T.float, device=self.device)
        mu = self.actor(state)
        if self.n_steps < self.warmup:
            mu = T.zeros(size=mu.shape)
        mu_prime = self.policy(mu)

        self.n_steps += 1

        return mu_prime.cpu().detach().numpy()
