import importlib
import gymnasium as gym
# from gymnasium.wrappers import FrameStackObservation
from protorl.wrappers.single_threaded import BatchDimensionWrapper
from protorl.wrappers.atari import PreprocessFrame, RepeatActionAndMaxFrame, StackFrames, EpisodicLifeEnv, NoopResetEnv, FireResetEnv
from protorl.wrappers.monitor import Monitor


def make_env(env_name, use_atari=False, repeat=4,
             no_ops=0, package_to_import=None, **kwargs):
    try:
        if use_atari:
            import ale_py
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
        env = Monitor(env)
        env = NoopResetEnv(env, 30)
        env = RepeatActionAndMaxFrame(env)
        env = EpisodicLifeEnv(env)
        env = FireResetEnv(env)
        env = PreprocessFrame(shape=(84, 84, 1), env=env, scale_obs=True)
        env = StackFrames(repeat=4, env=env) 
        # env = gym.wrappers.AtariPreprocessing(env, noop_max=no_ops, scale_obs=True)
        # env = FrameStackObservation(env, stack_size=repeat)
        env = BatchDimensionWrapper(env)

    return env

class PyTorchObsWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
                self.observation_space.low[0, 0, 0],
                self.observation_space.high[0, 0, 0],
                [obs_shape[2], obs_shape[0], obs_shape[1]],
                dtype=self.observation_space.dtype
                )

    def observation(self, observation):
        return observation.transpose(2, 0, 1)
