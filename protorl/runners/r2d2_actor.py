import os
# import signal
import time
import numpy as np
import torch as T
from protorl.utils.common import clip_reward
from protorl.wrappers.common import make_env
from protorl.memory.generic import initialize_memory

SAMPLER_CPUS = {0, 32}
LEARNER_CPUS = {1, 2, 3, 4, 33, 34, 35, 36}
ACTOR_CPUS = set(range(64)) - SAMPLER_CPUS - LEARNER_CPUS


def actor_fn(name, actor_creator, network_creator, policy_creator,
             global_idx, shared_params, update_event,
             active_threads, config, policy_config, fields, vals, global_memory_idx,
             index_lock, tree_lock, shared_sum_tree, param_update_lock,):# stop_event):
    #signal.signal(signal.SIGINT, signal.SIG_IGN)

    #while not stop_event.is_set():
    os.sched_setaffinity(0, ACTOR_CPUS)
    os.nice(5)
    T.set_num_threads(1)
    T.set_num_interop_threads(1)
    env = make_env(config.env_name, use_atari=config.use_atari, episodic_life=False, scale_obs=False)

    policy = policy_creator(config=policy_config)

    networks = network_creator(env, config=config, device='cpu')
    actor = actor_creator(*networks, config=config, policy=policy, name=name)

    replay_buffer = initialize_memory(obs_shape=env.observation_space.shape,
                                      n_actions=config.n_actions,
                                      max_size=config.memory_capacity,
                                      prioritized=config.use_prioritization,
                                      alpha=config.alpha,
                                      beta=config.beta,
                                      device=config.memory_device,
                                      action_space=config.action_space,
                                      batch_size=config.batch_size,
                                      warmup=config.warmup,
                                      fields=fields,
                                      vals=vals,
                                      global_memory_idx=global_memory_idx,
                                      index_lock=index_lock,
                                      tree_lock=tree_lock,
                                      shared_sum_tree=shared_sum_tree)

    if config.load_checkpoint or config.evaluate:
        actor.load_models()

    n_steps = 0
    best_score = -np.inf
    scores, steps = [], []
    t_start = time.time()
    i = 0
    elapsed_time = time.time() - t_start
    last_update = 0

    while elapsed_time < config.total_time:
        done, trunc = False, False
        observation, _ = env.reset()
        score = 0.0
        hx = T.ones((1, 128)).to('cpu')
        cx = T.ones((1, 128)).to('cpu')
        action = env.action_space.sample()
        observation, reward, done, trunc, _ = env.step(action)
        r = clip_reward(reward) if config.clip_reward else reward
        hx_seq, cx_seq = [], []
        ep_length = 0

        offset = config.n_batches_to_store
        while not (done or trunc):
            action, (hx_new, cx_new) = actor.choose_action(observation, hx, cx, action, reward)
            hx_seq.append(hx_new.detach().numpy())
            cx_seq.append(cx_new.detach().numpy())
            hx = hx_new
            cx = cx_new
            observation_, reward, done, trunc, _ = env.step(action)
            ep_length += 1
            reward = float(reward) # Gymnasium returns type "SupportsFloat" which makes my linter complain
            score += reward
            r = clip_reward(reward) if config.clip_reward else reward
            n_steps += 1
            if not config.evaluate:
                ep_end = done or trunc
                actor.store_transition([observation, action, r, observation_, ep_end])
                time_to_store = n_steps % (config.n_step + offset) == 0

                if time_to_store or ep_end:
                    # we produce so much data, we can afford to throw away short episodes
                    if ep_length < config.n_batches_to_store:
                        continue
                    states, actions, rewards, states_, dones, gammas = \
                        actor.get_n_step_returns(end_of_episode=ep_end)

                    if len(states) > 0:
                        transitions = [
                                T.tensor(states[-offset:],  dtype=vals[0].dtype),
                                T.tensor(actions[-offset:], dtype=vals[1].dtype),
                                T.tensor(rewards[-offset:], dtype=vals[2].dtype),
                                T.tensor(states_[-offset:], dtype=vals[3].dtype),
                                T.tensor(dones[-offset:],   dtype=vals[4].dtype),
                                T.tensor(gammas[-offset:],  dtype=vals[5].dtype),
                                T.tensor(np.array(hx_seq),  dtype=hx_new.dtype),
                                T.tensor(np.array(cx_seq),  dtype=cx_new.dtype)
                        ]

                        if config.use_prioritization:
                            td_errors = actor.calculate_priorities(transitions)
                        else:
                            td_errors = None
                        transitions = [
                                T.tensor(states[:offset],  dtype=vals[0].dtype),
                                T.tensor(actions[:offset], dtype=vals[1].dtype),
                                T.tensor(rewards[:offset], dtype=vals[2].dtype),
                                T.tensor(states_[:offset], dtype=vals[3].dtype),
                                T.tensor(dones[:offset],   dtype=vals[4].dtype),
                                T.tensor(gammas[:offset],  dtype=vals[5].dtype),
                                T.tensor(np.array(hx_seq[:offset]),  dtype=hx_new.dtype),
                                T.tensor(np.array(cx_seq[:offset]),  dtype=cx_new.dtype)
                        ]


                        replay_buffer.store_batch_transition(transitions, td_errors)

                if n_steps % config.target_replace_interval == 0:
                    actor.update_networks()

                if update_event.is_set() and (n_steps - last_update) >= 1000:
                    last_update = n_steps
                    with param_update_lock:
                        flat_params = T.from_numpy(np.frombuffer(shared_params, dtype=np.float32)).clone()
                    actor.download_learner_params(flat_params)
            observation = observation_
            # hx = hx_new
            # cx = cx_new
        i += 1
        elapsed_time = time.time() - t_start
        scores.append(score)
        steps.append(elapsed_time)

        avg_score = np.mean(scores[-100:])

        if name == '0':
            global_steps = global_memory_idx.value
            global_sps = global_steps / elapsed_time
            updates_per_second = global_idx.value / elapsed_time
            replay_ratio = config.batch_size * global_idx.value / global_steps
            # print(f'episode {i} ep score {score:.1f} average score {avg_score:.1f} '
            #      f'n steps {n_steps} learner steps {global_idx.value} time {elapsed_time:.1f}')
            print(f"episode {i} ep score {score:.1f} avg score {avg_score:.1f} "
                  f"global steps {global_steps:.1f} steps/s {global_sps:.2f} learner steps/s {updates_per_second:.2f} "
                  f"replay ratio {replay_ratio:.2f} wall time {elapsed_time:.1f}")
        if avg_score > best_score:
            if not config.evaluate:
                actor.save_models()
            best_score = avg_score

    print(f'exiting actor process {name}')
    active_threads.value -= 1
