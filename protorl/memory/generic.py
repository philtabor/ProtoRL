from contextlib import nullcontext
from types import SimpleNamespace
import numpy as np
import torch as T
# from protorl.memory.sum_tree import SumTree
from protorl.memory.sum_tree_shared import SumTree
from protorl.utils.common import convert_arrays_to_tensors


class GenericBuffer:
    def __init__(self, max_size, batch_size, fields,
                 prioritized=False, alpha=0.5, beta=0.5,
                 device=None, warmup=0,
                 n_threads=1, global_memory_idx=None,
                 index_lock=None, tree_lock=None,
                 shared_tree_tensor=None):
        self.mem_size = max_size
        self.mem_cntr = global_memory_idx or SimpleNamespace(value=0)
        self.batch_size = batch_size
        self.fields = fields
        self.prioritized = prioritized
        self.device = device
        self.warmup = warmup
        self.n_threads = n_threads
        self.index_lock = index_lock if index_lock is not None else nullcontext()
        self.tree_lock = tree_lock if tree_lock is not None else nullcontext()
        if prioritized:
            tree_tensor = shared_tree_tensor if shared_tree_tensor is not None else T.zeros(2 * max_size, dtype=T.float32)
            self.sum_tree = SumTree(max_size, batch_size,
                                    shared_tree_tensor=tree_tensor,
                                    alpha=alpha,
                                    beta=beta)

    def store_transition(self, items, vals=None):
        index = self.mem_cntr.value % self.mem_size
        items = convert_arrays_to_tensors(items, device='cpu')
        for item, field in zip(items, self.fields):
            getattr(self, field)[index] = item
        self.mem_cntr.value += 1
        if self.prioritized:
            indices = np.arange(index, index+1)
            priority = vals or 0.1
            self.sum_tree.update_priorities(indices, priority)

    def store_batch_transition(self, items, vals=None):
        # Ensure items is a list/tuple of tensors
        n_to_store = items[0].shape[0] # Use the first dimension of the tensor (192)
        if n_to_store > self.mem_size:
            raise ValueError("A single replay write cannot exceed replay capacity.")

        # if any(item[0] != n_to_store for item in items):
        #    raise ValueError("All replay fields must have equal batch lengths")

        with self.index_lock:
            start_idx = self.mem_cntr.value % self.mem_size
            # Store indices before we increment for the priority update later
            raw_indices = np.arange(start_idx, start_idx + n_to_store)
            end_idx = start_idx + n_to_store

            if end_idx <= self.mem_size:
                # Case 1: No wrap around
                for batch_t, f in zip(items, self.fields):
                    getattr(self, f)[start_idx:end_idx] = batch_t
                indices = raw_indices
            else:
                # Case 2: Wrap around logic
                overflow = end_idx - self.mem_size
                chunk_1_size = self.mem_size - start_idx
                for batch_t, f in zip(items, self.fields):
                    target = getattr(self, f)
                    target[start_idx:self.mem_size] = batch_t[:chunk_1_size]
                    target[0:overflow] = batch_t[chunk_1_size:]
                
                indices = np.concatenate([
                    np.arange(start_idx, self.mem_size), 
                    np.arange(0, overflow)
                ])
        self.mem_cntr.value += n_to_store
        if self.prioritized and vals is not None:
            # Don't recalculate indices here; use the ones from the lock block
            self.update_priority(indices, vals)

    def update_priority(self, indices, values):
        with self.tree_lock:
            self.sum_tree.update_priorities(indices, values)

    def update_beta(self, current_step, beta_start=0.4, target_beta=1.0, k=0.000005):
        # Moves beta from beta_start toward target_beta asymptotically.
        # Formula: beta(t) = target_beta - (target_beta - beta_start) * exp(-k * t)
        
        if self.prioritized:
            current_beta = target_beta - (target_beta - beta_start) * np.exp(-k * current_step)
            
            with self.tree_lock:
                self.sum_tree.beta = current_beta

    def sample_buffer(self, mode='uniform', payload=None):

        max_mem = min(self.mem_cntr.value, self.mem_size)
        if mode == 'uniform':
            batch = payload if payload is not None else np.random.choice(max_mem, self.batch_size, replace=False)
            arr = []
            for field in self.fields:
                arr.append(getattr(self, field)[batch])

        elif mode == 'batch':
            batch = np.random.choice(self.mem_size * self.n_threads, self.batch_size, replace=False)
            arr = [batch]

        elif mode == 'all':
            arr = [getattr(self, field)[:max_mem] for field in self.fields]

        elif mode == 'prioritized':
            if not payload:
                with self.tree_lock:
                    indices, weights = self.sum_tree.sample(max_mem)
            else:
                indices, weights = payload
            arr = [indices]
            for field in self.fields:
                source_tensor = getattr(self, field)
                arr.append(T.index_select(source_tensor, 0, T.tensor(indices)))
            arr.append(weights)
        else:
            raise ValueError("mode must be one of 'uniform', 'batch', 'all', 'prioritized'")

        device = self.device or \
            T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        arr = convert_arrays_to_tensors(arr, device, non_blocking=True)
        return arr

    def sample_indices(self, sample_mode='uniform'):
        max_mem = min(self.mem_cntr.value, self.mem_size)
        if sample_mode == 'uniform':
            indices = np.random.choice(max_mem, self.batch_size, replace=False)
            arr = [indices]

        elif sample_mode == 'prioritized':
            indices, weights = self.sum_tree.sample(max_mem)
            arr = [indices, weights]
        else:
            raise ValueError("sample_mode must be one of 'uniform', 'prioritized'")
        return arr 

    def ready(self):
        return self.mem_cntr.value >= self.batch_size and self.mem_cntr.value >= self.warmup


def initialize_memory(obs_shape, n_actions, max_size, batch_size,
                      n_threads=1, extra_fields=None, extra_vals=None,
                      action_space='discrete', fields=None, vals=None,
                      prioritized=False, alpha=0.5, beta=0.5, device=None,
                      warmup=0, global_memory_idx=None, index_lock=None,
                      tree_lock=None, shared_sum_tree=None):

    a_dtype = T.float if action_space == 'continuous' else T.long
    state_shape = [max_size, n_threads, *obs_shape] if n_threads > 1 else \
                  [max_size, *obs_shape]
    reward_shape = [max_size, n_threads] if n_threads > 1 else max_size
    done_shape = [max_size, n_threads] if n_threads > 1 else max_size

    if action_space == 'continuous':
        action_shape = [max_size, n_threads, n_actions] if n_threads > 1 else \
                       [max_size, n_actions]
    elif action_space == 'discrete':
        action_shape = [max_size, n_threads] if n_threads > 1 else max_size

    else:
        raise ValueError("action_space must be one of 'continuous', 'discrete'")

    fields = fields or ['states', 'actions', 'rewards', 'states_', 'dones']
    vals = vals or [T.zeros(state_shape, dtype=T.float),
                    T.zeros(action_shape, dtype=a_dtype),
                    T.zeros(reward_shape, dtype=T.float),
                    T.zeros(state_shape, dtype=T.float),
                    T.zeros(done_shape, dtype=T.bool)]
    if extra_fields is not None:
        fields += extra_fields
        vals += extra_vals # type: ignore

    Memory = type('ReplayBuffer', (GenericBuffer,),
                  {field: value for field, value in zip(fields, vals)})
    memory_buffer = Memory(max_size, batch_size, fields,
                           prioritized, alpha, beta, device, warmup, n_threads,
                           global_memory_idx, index_lock, tree_lock,
                           shared_tree_tensor=shared_sum_tree)

    return memory_buffer
