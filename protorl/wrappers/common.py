import importlib
import gymnasium as gym
import ale_py
from gymnasium.wrappers import FrameStackObservation
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
        else: # there was some error with an installed package
            print(f'Error encountered {e}')
            exit()

    if use_atari:
        env = gym.wrappers.AtariPreprocessing(env, noop_max=no_ops, scale_obs=True)
        env = FrameStackObservation(env, stack_size=repeat)
        env = BatchDimensionWrapper(env)

    return env
