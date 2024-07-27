from protorl.actor.base import Actor
import torch as T


class DuelingDQNActor(Actor):
    def __init__(self, online_net, target_net, policy, tau=1.0, device=None):
        super().__init__(policy=policy, tau=tau, device=device)
        self.q_eval = online_net
        self.q_next = target_net

    def save_models(self, fname=None):
        fname = fname or 'models/dueling_dqn_actor'
        checkpoint = {
            'q_eval_model_state_dict': self.q_eval.state_dict(),
            'q_next_model_state_dict': self.q_next.state_dict(),
            'epsilon': self.policy.epsilon,
            'epsilon_dec': self.policy.eps_dec,
            'epsilon_min': self.policy.eps_min,
        }
        T.save(checkpoint, fname)

    def load_models(self, fname=None):
        fname = fname or 'models/dueling_dqn_actor'
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
