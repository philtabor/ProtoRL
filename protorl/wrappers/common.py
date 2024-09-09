import importlib
import gymnasium as gym
from gymnasium.wrappers import FrameStack
from protorl.wrappers.single_threaded import BatchDimensionWrapper


def make_env(env_name, use_atari=False, repeat=4,
             no_ops=0, package_to_import=None, **kwargs):

    try:
        env = gym.make(env_name, **kwargs)
    except (gym.error.Error, ImportError) as e:
        if package_to_import:
            try:
                importlib.import_module(package_to_import)
                env = gym.make(env_name, **kwargs)
            except ImportError:
                raise ImportError(f"Could not import {package_to_import}. Please ensure it's installed.")

    if use_atari:
        env = gym.wrappers.AtariPreprocessing(env, noop_max=no_ops, scale_obs=True)
        env = FrameStack(env, num_stack=repeat)
        env = BatchDimensionWrapper(env)

    return env
