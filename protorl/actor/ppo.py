import torch as T
from protorl.actor.base import Actor


class PPOActor(Actor):
    def __init__(self, actor_net, critic_net, action_type, policy):
        super().__init__(policy=policy)
        self.action_type = action_type

        self.actor = actor_net
        self.critic = critic_net

    def save_models(self, fname=None):
        fname = fname or 'models/ppo_' + self.action_type + '_actor'
        checkpoint = {
            'actor_model_state_dict': self.actor.state_dict(),
            'critic_model_state_dict': self.critic.state_dict(),
        }
        T.save(checkpoint, fname)

    def load_models(self, fname=None):
        fname = fname or 'models/ppo_' + self.action_type + '_actor'
        checkpoint = T.load(fname)
        self.actor.load_state_dict(checkpoint['actor_model_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_model_state_dict'])

    def evaluate_state(self, observation):
        state = T.tensor(observation, dtype=T.float, device=self.device)
        with T.no_grad():
            value = self.critic(state)
        return value.cpu().numpy()

    def choose_action(self, observation):
        state = T.tensor(observation, dtype=T.float, device=self.device)
        with T.no_grad():
            if self.action_type == 'continuous':
                alpha, beta = self.actor(state)
                action, log_probs = self.policy(alpha, beta)

            elif self.action_type == 'discrete':
                probs = self.actor(state)
                action, log_probs = self.policy(probs)

        return action.cpu().numpy(), log_probs.cpu().numpy()
