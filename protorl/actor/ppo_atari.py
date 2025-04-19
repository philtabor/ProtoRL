import torch as T
from protorl.actor.base import Actor

class PPOAtariActor(Actor):
    def __init__(self, actor_critic, policy):
        super().__init__(policy=policy)
        self.action_type = 'discrete'
        self.actor_critic = actor_critic

    def save_models(self, fname=None):
        fname = fname or 'models/ppo_atari_actor'
        checkpoint = {
            'actor_critic_model_state_dict': self.actor_critic.state_dict(),
        }
        T.save(checkpoint, fname)

    def load_models(self, fname=None):
        fname = fname or 'models/ppo_atari_actor'
        checkpoint = T.load(fname)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_model_state_dict'])

    def choose_action(self, observation):
        state = T.tensor(observation, dtype=T.float, device=self.device)
        with T.no_grad():
            probs, _ = self.actor_critic(state)
            action, log_probs = self.policy(probs)

        return action.cpu().numpy(), log_probs.cpu().numpy()

    def evaluate_state(self, observation):
        state = T.tensor(observation, dtype=T.float, device=self.device)
        with T.no_grad():
            _, value = self.actor_critic(state)
        return value.cpu().numpy()
