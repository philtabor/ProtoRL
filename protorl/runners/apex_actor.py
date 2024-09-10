import threading
import time
import numpy as np
import torch as T
from protorl.utils.common import clip_reward
from protorl.wrappers.common import make_env
from protorl.memory.client import MemoryClient


def actor_fn(name, actor_creator, network_creator, policy_creator,
             global_idx, shared_params, update_event,
             active_threads, request_queue, response_queue,
             config, policy_config):
    memory_client = MemoryClient(request_queue, response_queue)

    env = make_env(config.env_name, use_atari=config.use_atari)

    policy = policy_creator(config=policy_config)

    networks = network_creator(env, config=config, device='cpu')

    actor = actor_creator(*networks, config=config, policy=policy, name=name)

    if config.load_checkpoint or config.evaluate:
        actor.load_models()

    n_steps = 0
    best_score = env.reward_range[0]
    scores, steps = [], []
    t_start = time.time()
    i = 0
    elapsed_time = time.time() - t_start
    last_update = 0

    while elapsed_time < config.total_time:
        done, trunc = False, False
        observation, _ = env.reset()
        score = 0
        while not (done or trunc):
            action = actor.choose_action(observation)
            observation_, reward, done, trunc, _ = env.step(action)
            score += reward
            r = clip_reward(reward) if config.clip_reward else reward
            n_steps += 1
            if not config.evaluate:
                ep_end = done or trunc
                actor.store_transition([observation, action, r, observation_, ep_end])
                if n_steps % (config.n_batches_to_store) == 0 and n_steps > 0:
                    states, actions, rewards, states_, dones, gammas = \
                        actor.get_n_step_returns()
                    transitions = [states, actions, rewards, states_, dones, gammas]
                    if config.use_prioritization:
                        td_errors = actor.calculate_priorities(transitions)
                    else:
                        td_errors = [None] * config.n_batches_to_store
                    memory_client.add(
                        zip(states, actions, rewards, states_, dones, gammas),
                        vals=td_errors)
                if n_steps % config.target_replace_interval == 0:
                    actor.update_networks()

                if update_event.is_set() and (n_steps - last_update) >= 100:
                    last_update = n_steps
                    flat_params = T.from_numpy(np.frombuffer(shared_params, dtype=np.float32))
                    actor.download_learner_params(flat_params)
            observation = observation_
        i += 1
        elapsed_time = time.time() - t_start
        scores.append(score)
        steps.append(elapsed_time)

        avg_score = np.mean(scores[-100:])

        if name == '0':
            print('thread {} episode {} ep score {:.1f} average score {:.1f} n steps {} learner steps {} time {:.1f}'.
                format(name, i, score, avg_score,  n_steps, global_idx.value, elapsed_time))

        if avg_score > best_score:
            if not config.evaluate:
                actor.save_models()
            best_score = avg_score

    print(f'exiting thread {name}')
    with threading.Lock():
        active_threads.value -= 1

    memory_client.exit()
