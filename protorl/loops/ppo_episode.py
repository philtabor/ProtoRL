import numpy as np
from protorl.utils.common import action_adapter, clip_reward


class EpisodeLoop:
    def __init__(self, agent, env, memory, n_epochs, T, n_batches,
                 n_threads=1, print_every=1, adapt_actions=False,
                 load_checkpoint=False, clip_reward=False,
                 evaluate=False, extra_functionality=None, seed=None):
        self.agent = agent
        self.seed = seed
        self.env = env
        self.memory = memory
        self.n_epochs = n_epochs
        self.T = T
        self.print_every = print_every
        self.n_batches = n_batches
        self.load_checkpoint = load_checkpoint
        self.evaluate = evaluate
        self.clip_reward = clip_reward
        self.n_threads = n_threads
        self.adapt_actions = adapt_actions
        self.step_counter = 0

        self.functions = extra_functionality or []

    def run(self, n_episodes=1):
        if self.load_checkpoint:
            self.agent.load_models()
        n_steps = 0
        best_score = self.env.reward_range[0]
        scores, steps = [], []
        if self.adapt_actions:
            max_a = self.env.action_space.high[0]

        for i in range(n_episodes):
            done = [False] * self.n_threads
            trunc = [False] * self.n_threads
            if self.seed is not None:
                observation, _ = self.env.reset(self.seed)
                self.seed = None
            else:
                observation, _ = self.env.reset()
            score = [0] * self.n_threads
            while not (any(done) or any(trunc)):
                action, log_prob = self.agent.choose_action(observation)
                if self.adapt_actions:
                    act = action_adapter(action, max_a).reshape(
                        action.shape)
                else:
                    act = action
                observation_, reward, done, trunc, _ = self.env.step(act)
                self.step_counter += 1
                score += reward
                r = clip_reward(reward) if self.clip_reward else reward
                mask = [0.0 if d or t else 1.0 for d, t in zip(done, trunc)]
                if not self.evaluate:
                    self.memory.store_transition([observation, action,
                                                  r, observation_, mask,
                                                  log_prob])
                    if self.step_counter % self.T == 0:
                        transitions = self.memory.sample_buffer(mode='all')
                        for _ in range(self.n_epochs):
                            batches = [self.memory.sample_buffer(mode='batch')
                                       for _ in range(self.n_batches)]
                            self.agent.update(transitions, batches)
                        self.step_counter = 0
                observation = observation_
                n_steps += 1
            scores.append(np.mean(score))
            steps.append(n_steps)

            avg_score = np.mean(scores[-100:])
            if i % self.print_every == 0:
                print(f'episode {i} average score {avg_score:.1f} n steps {n_steps}')
            if avg_score > best_score:
                if not self.evaluate:
                    self.agent.save_models()
                best_score = avg_score
            # TODO - more general way of handling the parameters
            self.handle_extra_functionality(i, n_episodes)
        self.env.close()
        return scores, steps

    def handle_extra_functionality(self, *args):
        for func in self.functions:
            func(*args)
