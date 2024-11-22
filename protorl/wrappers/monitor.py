import gymnasium as gym


class Monitor(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env=env)
        self.rewards = []
        self.env = env

    def reset(self, **kwargs):
        self.rewards = []
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.rewards.append(reward)
        if terminated or truncated:
            info['ep_r'] = sum(self.rewards)
        return obs, reward, terminated, truncated, info
