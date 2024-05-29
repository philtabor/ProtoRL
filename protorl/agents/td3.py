from protorl.agents.base import Agent


class TD3Agent(Agent):
    def __init__(self, actor, learner, update_actor_interval=1):
        super().__init__(actor=actor, learner=learner)
        self.learn_step_counter = 0
        self.update_actor_interval = update_actor_interval

    def choose_action(self, observation):
        action = self.actor.choose_action(observation)
        return action

    def update_networks(self):
        src = self.learner.actor
        dest = self.learner.target_actor
        self.learner.update_network_parameters(src, dest, tau=1.0)

        src = self.learner.critic_1
        dest = self.learner.target_critic_1
        self.learner.update_network_parameters(src, dest, tau=1.0)

        src = self.learner.critic_2
        dest = self.learner.target_critic_2
        self.learner.update_network_parameters(src, dest, tau=1.0)

        if self.learn_step_counter % self.update_actor_interval == 0:
            src = self.learner.actor
            dest = self.actor.actor
            self.actor.update_network_parameters(src, dest, tau=1.0)

    def update(self, transitions):
        self.learner.update(transitions)
        self.learn_step_counter += 1
        self.update_networks()
