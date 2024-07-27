import gymnasium as gym
from gymnasium.wrappers import FrameStack
from protorl.wrappers.single_threaded import BatchDimensionWrapper


def make_env(env_name, use_atari=False, repeat=4, no_ops=0, **kwargs):
    env = gym.make(env_name, **kwargs)

    if use_atari:
        env = gym.wrappers.AtariPreprocessing(env, noop_max=no_ops, scale_obs=True)
        env = FrameStack(env, num_stack=repeat)
        env = BatchDimensionWrapper(env)

    return env
