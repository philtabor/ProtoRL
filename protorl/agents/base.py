import torch as T


class Agent:
    def __init__(self, actor, learner):
        self.actor = actor
        self.learner = learner
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    def save_models(self):
        self.actor.save_models()
        self.learner.save_models()

    def load_models(self):
        self.actor.load_models()
        self.learner.load_models()

    def update_networks(self):
        raise NotImplementedError

    def update(self, transitions):
        raise NotImplementedError

    def choose_action(self, observation):
        raise NotImplementedError
