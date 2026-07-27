from queue import Empty
import torch as T
import torch.multiprocessing as mp
from protorl.memory.generic import initialize_memory
from protorl.wrappers.common import make_env

def sampler_worker(config, obs_shape, fields, vals, global_memory_idx, 
                   index_lock, tree_lock, shared_sum_tree,
                   batch_queue, priority_queue, active_threads, sample_mode):
    # Re-initialize the buffer 'shell' inside the worker process.
    # Because you pass the same 'vals', 'shared_sum_tree', and 'locks', 
    # this instance points to the EXACT same shared memory as the Learner.
    # from protorl.memory.generic import initialize_memory
    T.set_num_threads(1)
    T.set_num_interop_threads(1)
    replay_buffer = initialize_memory(
        obs_shape=obs_shape,
        n_actions=config.n_actions,
        max_size=config.memory_capacity,
        prioritized=config.use_prioritization,
        alpha=config.alpha,
        beta=config.beta,
        # device=config.memory_device,
        batch_size=config.batch_size,
        fields=fields,
        vals=vals,
        index_lock=index_lock,
        tree_lock=tree_lock,
        global_memory_idx=global_memory_idx,
        shared_sum_tree=shared_sum_tree
    )

    while active_threads.value > 1:
        if replay_buffer.ready() and batch_queue.qsize() < 32:
            payload = replay_buffer.sample_indices(sample_mode)
            batch_queue.put(payload)
        
        while True:
            try:
                indices, errors = priority_queue.get_nowait()
                replay_buffer.update_priority(indices, errors)
            except Empty:
                break
    print("exiting sampler process")


def learner_fn(learner_fn, network_fn, config, shared_params, update_event,
               active_threads, global_idx, fields, vals, obs_shape, global_memory_idx,
               index_lock, tree_lock, shared_sum_tree, param_update_lock):
    T.set_num_threads(4)
    T.set_num_interop_threads(1)
    replay_buffer = initialize_memory(obs_shape=obs_shape,
                                      n_actions=config.n_actions,
                                      max_size=config.memory_capacity,
                                      prioritized=config.use_prioritization,
                                      alpha=config.alpha,
                                      beta=config.beta,
                                      # device=config.memory_device,
                                      action_space=config.action_space,
                                      batch_size=config.batch_size,
                                      warmup=config.warmup,
                                      fields=fields,
                                      vals=vals,
                                      index_lock=index_lock,
                                      tree_lock=tree_lock,
                                      global_memory_idx=global_memory_idx,
                                      shared_sum_tree=shared_sum_tree)

    sample_mode = 'prioritized' if config.use_prioritization else 'uniform'
    
    batch_queue = mp.Queue(maxsize=32) 
    priority_queue = mp.Queue()
    sampler = mp.Process(target=sampler_worker, 
                        args=(config, obs_shape, fields, vals, global_memory_idx,
                              index_lock, tree_lock, shared_sum_tree,
                              batch_queue, priority_queue, active_threads, sample_mode)) 
    sampler.start()
    env = make_env(config.env_name, use_atari=config.use_atari)
    networks = network_fn(env, config=config, device=config.learner_device)

    global_learner = learner_fn(*networks, config=config)
    while active_threads.value > 1:
        try:
            payload = batch_queue.get(timeout=1)

            transition = replay_buffer.sample_buffer(mode=sample_mode, payload=payload) 

            if config.use_prioritization:
                s_idx, td_errors = global_learner.update(transition)
            else:
                _ = global_learner.update(transition)
            if config.use_prioritization:
                priority_queue.put((s_idx.detach().cpu().numpy(), td_errors.detach().cpu().numpy()))
            
            global_learner.update_networks() 
            global_idx.value += 1
            
            if global_idx.value % 128 == 0:
                with param_update_lock:
                    flat_params = T.cat([param.data.view(-1)
                        for network in global_learner.networks_to_transmit
                        for param in network.parameters()])
                    shared_params[:] = flat_params.cpu().numpy()
                    update_event.set()

            if global_idx.value % 100 == 0:
                replay_buffer.update_beta(current_step=global_idx.value)

        except Empty:
            continue

    sampler.join()
    print("exiting learner process")
