import gym
from protorl.wrappers.atari import PreprocessFrame
from protorl.wrappers.atari import RepeatActionAndMaxFrame, StackFrames
from protorl.wrappers.single_threaded import SingleThreadedWrapper
from protorl.wrappers.single_threaded import BatchDimensionWrapper


def make_env(env_name, use_atari=False, shape=(84, 84, 1),
             repeat=4, clip_rewards=False, no_ops=0,
             fire_first=False, n_threads=1):
    env = gym.make(env_name)

    if use_atari:
        env = RepeatActionAndMaxFrame(env, repeat, clip_rewards,
                                      no_ops, fire_first)
        env = PreprocessFrame(shape, env)
        env = StackFrames(env, repeat)
        if n_threads == 1:
            env = BatchDimensionWrapper(env)
            env = SingleThreadedWrapper(env)
            return env

    if n_threads == 1:
        env = SingleThreadedWrapper(env)

    return env
