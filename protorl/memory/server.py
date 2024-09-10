import threading
from protorl.memory.generic import initialize_memory

class MemoryServer:
    def __init__(self, observation_shape, config,
                 extra_fields=None, extra_vals=None):
        self.replay_buffer = initialize_memory(obs_shape=observation_shape,
                                               n_actions=config.n_actions,
                                               max_size=config.memory_capacity,
                                               prioritized=config.use_prioritization,
                                               alpha=config.alpha,
                                               beta=config.beta,
                                               device=config.memory_device,
                                               action_space=config.action_space,
                                               batch_size=config.batch_size,
                                               warmup=config.warmup,
                                               extra_fields=extra_fields,
                                               extra_vals=extra_vals)
        self.prioritized = config.use_prioritization
        self.n_batches = config.n_batches_to_sample

    def add(self, transitions, vals=None):
        with threading.Lock():
            for idx, transition in enumerate(transitions):
                # val = vals[idx] if vals is not None else None
                self.replay_buffer.store_transition(transition, vals[idx])

    def sample(self):
        mode = 'prioritized' if self.prioritized else 'uniform'
        batch = []
        with threading.Lock():
            for _ in range(self.n_batches):
                batch.append(self.replay_buffer.sample_buffer(mode))
        return batch

    def update_priorities(self, indices, priorities):
        with threading.Lock():
            self.replay_buffer.update_priorities(indices, priorities)

    def ready(self):
        with threading.Lock():
            return self.replay_buffer.ready()


def memory_server_process(observation_shape, config,
                          request_queue, response_queue,
                          extra_fields=None, extra_vals=None):
    # swapped extra_fields and extra_vals
    memory = MemoryServer(observation_shape, config, extra_fields, extra_vals)
    active_threads = config.n_threads + 1
    while active_threads > 0:
        request_id, request = request_queue.get()
        if request[0] == "add":
            memory.add(request[1], request[2])
            response_queue.put((request_id, 'added'))
        elif request[0] == "sample":
            samples = memory.sample()
            response_queue.put((request_id, samples,))
        elif request[0] == "update":
            memory.update_priorities(request[1], request[2])
            response_queue.put((request_id, 'updated'))
        elif request[0] == "ready":
            ready = memory.ready()
            response_queue.put((request_id, ready))
        elif request[0] == "exit":
            active_threads -= 1
            response_queue.put((request_id, 'exit'))
