import importlib
import gymnasium as gym
# from gymnasium.wrappers import FrameStackObservation
from protorl.wrappers.single_threaded import BatchDimensionWrapper
from protorl.wrappers.atari import PreprocessFrame, RepeatActionAndMaxFrame, StackFrames, EpisodicLifeEnv, NoopResetEnv, FireResetEnv
from protorl.wrappers.monitor import Monitor


def make_env(env_name, use_atari=False, repeat=4,
             no_ops=30, package_to_import=None, 
             no_op=True, repeat_and_max=True,
             episodic_life=False, fire_reset=True,
             preprocess=True, stack=True, add_batch=True,
             **kwargs):
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
        if no_op:
            env = NoopResetEnv(env, no_ops)
        if repeat_and_max:
            env = RepeatActionAndMaxFrame(env)
        if episodic_life:
            env = EpisodicLifeEnv(env)
        if fire_reset:
            env = FireResetEnv(env)
        if preprocess:
            env = PreprocessFrame(shape=(84, 84, 1), env=env, scale_obs=True)
        if stack:
            env = StackFrames(repeat=4, env=env)
        # env = gym.wrappers.AtariPreprocessing(env, noop_max=no_ops, scale_obs=True)
        # env = FrameStackObservation(env, stack_size=repeat)
        if add_batch:
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
