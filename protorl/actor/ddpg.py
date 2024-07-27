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

        self.update_network_parameters(self.actor, self.target_actor, tau=1.0)
        self.update_network_parameters(self.critic,
                                       self.target_critic, tau=1.0)

    def save_models(self, fname=None):
        fname = fname or 'models/ddpg_actor'
        checkpoint = {
            'actor_model_state_dict': self.actor.state_dict(),
            'critic_model_state_dict': self.critic.state_dict(),
            'noise': self.policy.noise,
        }
        T.save(checkpoint, fname)

    def load_models(self, fname=None):
        fname = fname or 'models/ddpg_actor'
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
