import torch as T
from protorl.actor.base import Actor


class PPOActor(Actor):
    def __init__(self, actor_net, critic_net, action_type, policy):
        super().__init__(policy=policy)
        self.action_type = action_type

        self.actor = actor_net
        self.critic = critic_net
        self.networks = [net for net in [self.actor, self.critic]]

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
