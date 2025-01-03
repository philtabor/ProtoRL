import importlib
import multiprocessing as mp
import ale_py
import gymnasium as gym
import numpy as np
import torch as T
# from miniworld import wrappers
from protorl.wrappers.common import PyTorchObsWrapper
from protorl.wrappers.atari import PreprocessFrame, RepeatActionAndMaxFrame, StackFrames, EpisodicLifeEnv, NoopResetEnv, FireResetEnv
from protorl.wrappers.monitor import Monitor


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, trunc, info = env.step(data)
            if done:
                ob, _ = env.reset()
            remote.send((ob, reward, done, trunc, info))
        elif cmd == 'reset':
            ob, info = env.reset()
            remote.send([ob, info])
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError


class SubprocVecEnv:
    def __init__(self, env_fns, spaces=None):
        self.closed = False
        nenvs = len(env_fns)
        mp.set_start_method('forkserver')
        self.remotes, self.work_remotes = zip(*[mp.Pipe()
                                                for _ in range(nenvs)])
        self.ps = [mp.Process(target=worker, args=(work_remote, remote,
                              CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in
                   zip(self.work_remotes, self.remotes, env_fns)]

        for p in self.ps:
            p.daemon = True
            p.start()

        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.observation_space = observation_space
        self.action_space = action_space

    def step_async(self, actions):
        assert not self.closed, "trying to operate after calling close()"
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, trunc, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), \
            np.stack(trunc), infos

    def reset(self):
        assert not self.closed, "trying to operate after calling close()"
        for remote in self.remotes:
            remote.send(('reset', None))
        obs, info = [], []
        for remote in self.remotes:
            o, i = remote.recv()
            obs.append(o)
            info.append(i)
        obs = np.array(obs)
        info = np.array(info)
        return obs, info

    def close_extras(self):
        if self.closed:
            return
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def close(self):
        if self.closed:
            return
        self.close_extras()
        self.closed = True

    def step(self, actions):
        obs, reward, dones, trunc, info = self.step_async(actions)
        return obs, reward, dones, trunc, info

    def __del__(self):
        if not self.closed:
            self.close()


class CloudpickleWrapper:
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)



def make_single_env(env_id, seed=None, rank=0,
                    use_atari=False, pixel_env=False, repeat=4,
                    no_ops=0, package_to_import=None, **kwargs):
    def _thunk():
        try:
            env = gym.make(env_id, **kwargs)
        except (gym.error.Error, ImportError) as e:
            if package_to_import:
                try:
                    importlib.import_module(package_to_import)
                    env = gym.make(env_id, **kwargs)
                except ImportError:
                    raise ImportError(f"Could not import {package_to_import}. Please ensure it's installed.")
            else:
                print(f"Error {e} encountered.")

        # TODO - need a better way of applying wrappers
        if use_atari:
            env = NoopResetEnv(env, 30)
            env = RepeatActionAndMaxFrame(env)
            env = EpisodicLifeEnv(env)
            env = FireResetEnv(env)
            env = PreprocessFrame(shape=(84, 84, 1), env=env, scale_obs=True)
            env = StackFrames(repeat=4, env=env)

        if pixel_env:
            # env = wrappers.GreyscaleWrapper(env)
            env = PreprocessFrame(shape=(60, 80, 1), env=env, scale_obs=False)
            # env = PyTorchObsWrapper(env)
            env = StackFrames(repeat=4, env=env)

        env = Monitor(env)

        env.action_space.seed(seed+rank)
        return env

    return _thunk


def make_vec_envs(env_name, use_atari=False, seed=None, n_threads=2,
                  package_to_import=None, **kwargs):
    # mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    # seed = seed + 10000 * mpi_rank if seed is not None else None
    seed = seed or 1
    set_global_seeds(seed)
    envs = [make_single_env(env_id=env_name, use_atari=use_atari, rank=i, seed=seed,
                            package_to_import=package_to_import, **kwargs)
            for i in range(n_threads)]
    envs = SubprocVecEnv(envs, seed)

    return envs


def set_global_seeds(seed):
    if seed is not None:
        import random
        np.random.seed(seed)
        random.seed(seed)
        T.manual_seed(seed)
