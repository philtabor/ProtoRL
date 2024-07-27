from protorl.actor.base import Actor
import torch as T


class SACActor(Actor):
    def __init__(self, actor_network,  policy, tau=0.005):
        super().__init__(policy=policy, tau=tau)
        self.actor = actor_network

    def save_models(self, fname=None):
        fname = fname or 'models/sac_actor'
        checkpoint = {
            'actor_model_state_dict': self.actor.state_dict(),
        }
        T.save(checkpoint, fname)

    def load_models(self, fname=None):
        fname = fname or 'models/sac_actor'
        checkpoint = T.load(fname)
        self.actor.load_state_dict(checkpoint['actor_model_state_dict'])

    def choose_action(self, observation):
        state = T.tensor(observation, dtype=T.float).to(self.device)
        mu, sigma = self.actor(state)
        actions, _ = self.policy(mu, sigma)
        return actions.cpu().detach().numpy()
