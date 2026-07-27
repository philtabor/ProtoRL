import torch as T
from protorl.memory.client import MemoryClient


def learner_fn(global_learner, config,
               shared_params, update_event, active_threads,
               request_queue, response_queue, global_idx):
    memory_client = MemoryClient(request_queue, response_queue)
    if config.load_checkpoint:
        global_learner.load_models()
    while active_threads.value > 1:
        if memory_client.ready() and not config.evaluate:
            batch_of_transitions = memory_client.sample()
            for transition in batch_of_transitions:
                if config.use_prioritization:
                    s_idx, td_errors = global_learner.update(transition)
                    _ = memory_client.update_priorities(s_idx, td_errors)
                else:
                    global_learner.update(transition)
                global_learner.update_networks()
                global_idx.value += 1
                if global_idx.value % 1000 == 0:
                    global_learner.save_models()

            flat_params = T.cat([param.data.view(-1)
                for network in global_learner.networks_to_transmit
                for param in network.parameters()])
            shared_params[:] = flat_params.cpu().numpy()
            update_event.set()
    memory_client.exit()
