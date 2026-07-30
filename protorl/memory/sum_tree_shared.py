import numpy as np
import torch as T


class SumTree:
    def __init__(self, max_size: int, batch_size: int, 
                 shared_tree_tensor: T.Tensor,
                 alpha: float = 0.5, beta: float = 0.5):
        
        if max_size <= 0:
            raise ValueError(f"max_size must be strictly positive. Received {max_size}.")
            
        self.max_size = max_size
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        
        # Array of size 2 * max_size. 
        # Index 0 is empty. Root sits at index 1.
        # Leaves sit at indices [max_size] to [2 * max_size - 1].
        if shared_tree_tensor is not None:
            if shared_tree_tensor.shape[0] != 2 * max_size:
                raise ValueError("Shared tensor must be exactly size 2 * max_size.")
            self.tree_tensor = shared_tree_tensor

        self.tree = self.tree_tensor.numpy()

    def update_priorities(self, buffer_indices, priorities):
        buffer_indices = np.array(buffer_indices, dtype=np.int32)
        priorities = np.array(priorities, dtype=np.float32)
        
        p_alpha = priorities ** self.alpha
        tree_indices = buffer_indices + self.max_size
        self.tree[tree_indices] = p_alpha
        
        ancestors = set()
        for idx in tree_indices.tolist():
            while idx > 1:
                idx //= 2
                ancestors.add(idx)
                
        for parent in sorted(ancestors, reverse=True):
            self.tree[parent] = self.tree[2 * parent] + self.tree[2 * parent + 1]

    def get_leaf(self, target: float):
        """Traverses the tree to find the buffer index for a target sum."""
        tree_idx = 1 # Start at root
        
        while tree_idx < self.max_size:
            left = 2 * tree_idx
            right = left + 1
            
            if target <= self.tree[left]:
                tree_idx = left
            else:
                target -= self.tree[left]
                tree_idx = right
                
        buffer_idx = tree_idx - self.max_size
        
        if buffer_idx >= self.max_size:
            buffer_idx = self.max_size - 1
            tree_idx = buffer_idx + self.max_size
            
        return buffer_idx, self.tree[tree_idx]

    def sample(self, current_mem_size: int):
        """Samples a batch of indices based on priority."""
        total_weight = self.tree[1].item()
        
        if total_weight <= 0.0:
            samples = np.random.choice(current_mem_size, self.batch_size, replace=False)
            weights = np.ones(self.batch_size)
            return samples, weights

        samples = np.zeros(self.batch_size, dtype=np.int32)
        priorities = np.zeros(self.batch_size, dtype=np.float32)
        segment = total_weight / self.batch_size
        
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            target = np.random.uniform(a, b)
            
            buffer_idx, priority = self.get_leaf(target)
            samples[i] = buffer_idx
            priorities[i] = priority
            
        probs = priorities / total_weight
        probs = np.maximum(probs, 1e-8)
        weights = (current_mem_size * probs) ** -self.beta
        weights /= weights.max()
        
        return samples.tolist(), weights.tolist()
