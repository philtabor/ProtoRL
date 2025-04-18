import numpy as np
from protorl.utils.common import clip_reward


class EpisodeLoop:
    def __init__(self, agent, env, memory,
                 sample_mode='uniform', print_every=1,
                 load_checkpoint=False, clip_reward=False,
                 prioritized=False,
                 evaluate=False):
        self.agent = agent
        self.env = env
        self.memory = memory
        self.sample_mode = sample_mode
        self.print_every = print_every
        self.prioritized = prioritized
        self.load_checkpoint = load_checkpoint
        self.evaluate = evaluate
        self.clip_reward = clip_reward

        self.functions = []

    def run(self, n_episodes=1):
        if self.load_checkpoint:
            self.agent.load_models()
        n_steps = 0
        best_score = -np.inf
        scores, steps = [], []
        for i in range(n_episodes):
            done, trunc = False, False
            observation, info = self.env.reset()
            score = 0
            while not (done or trunc):
                action = self.agent.choose_action(observation)
                observation_, reward, done, trunc, info = self.env.step(action)
                score += reward
                r = clip_reward(reward) if self.clip_reward else reward
                if not self.evaluate:
                    ep_end = done or trunc
                    self.memory.store_transition([observation, action,
                                                  r, observation_, ep_end])
                    if self.memory.ready():
                        transitions = \
                            self.memory.sample_buffer(self.sample_mode)
                        if self.prioritized:
                            s_idx, td_errors = self.agent.update(transitions)
                            self.memory.update_priorities(s_idx, td_errors)
                        else:
                            self.agent.update(transitions)
                observation = observation_
                n_steps += 1
            if 'ep_r' in info.keys():
                score = info['ep_r']
            # score = np.mean(score)
            scores.append(score)
            steps.append(n_steps)

            avg_score = np.mean(scores[-100:])
            if i % self.print_every == 0 and i > 0:
                print(f"episode {i} ep score {score:.1f} average score {avg_score:.1f}"
                      f" n steps {n_steps}")
            if avg_score > best_score:
                if not self.evaluate:
                    self.agent.save_models()
                best_score = avg_score

            self.handle_extra_functionality()

        return scores, steps

    def handle_extra_functionality(self):
        for func in self.functions:
            continue
