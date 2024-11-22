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
        self.epoch_counter = 0

        self.functions = extra_functionality or []

    def run(self, max_steps=1):
        if self.load_checkpoint:
            self.agent.load_models()
        n_steps = 0
        # best_score = -np.inf
        scores, steps = [], []

        if self.adapt_actions:
            max_a = self.env.action_space.high[0]

        if self.seed is not None:
            observation, info = self.env.reset(self.seed)
            self.seed = None
        else:
            observation, info = self.env.reset()
        rew = []
        while n_steps < max_steps:
            action, log_prob = self.agent.choose_action(observation)
            if self.adapt_actions:
                act = action_adapter(action, max_a).reshape(
                    action.shape)
            else:
                act = action
            observation_, reward, done, trunc, info = self.env.step(act)
            n_steps += 1
            self.step_counter += self.n_threads
            r = clip_reward(reward) if self.clip_reward else reward
            if any(done) or any(trunc):
                for d in info:
                    if 'ep_r' in d.keys():
                        score = d['ep_r']
                rew.append(score)
            mask = [0.0 if d or t else 1.0 for d, t in zip(done, trunc)]
            if not self.evaluate:
                for o, a, r, o_, m, lp in zip(observation, action, r,
                                              observation_, mask, log_prob):
                    self.memory.store_transition([o, a, r, o_, m, lp])
                if n_steps % self.T == 0:
                    transitions = self.memory.sample_buffer(mode='all')
                    for _ in range(self.n_epochs):
                        batches = [self.memory.sample_buffer(mode='batch')
                                   for _ in range(self.n_batches)]
                        self.agent.update(transitions, batches)
                    self.epoch_counter += self.n_epochs
                    print(f"Epoch: {self.epoch_counter} avg rollout score: {np.mean(rew):.1f}"
                          f" n_steps {self.step_counter}")
                    rew = []
            observation = observation_
            # scores.append(np.mean(score))
            # steps.append(n_steps)

            # avg_score = np.mean(scores[-100:])
            # if i % self.print_every == 0:
            #    print(f'episode {i} score {np.sum(score):.1f} average score {avg_score:.1f} n steps {n_steps}')
            # if avg_score > best_score:
            #    if not self.evaluate:
            #        self.agent.save_models()
            #    best_score = avg_score
            # TODO - more general way of handling the parameters
            # self.handle_extra_functionality(i, n_episodes)
        self.env.close()
        return scores, steps

    def handle_extra_functionality(self, *args):
        for func in self.functions:
            func(*args)
