# from collections import deque
import numpy as np


class NStepBuffer:
    def __init__(self, n, gamma):
        self.n = n
        self.gamma = gamma
        self.buffer = [] # Use a list to allow slicing

    def store_transition(self, experience):
        self.buffer.append(experience)

    def compute_n_step_returns(self, end_of_episode=False):
        states, actions, returns, states_, dones, gammas = [], [], [], [], [], []
        
        # If the episode isn't over, we can only safely process transitions
        # that have n-1 steps of future data available.
        if end_of_episode:
            processing_limit = len(self.buffer)
        else:
            processing_limit = max(0, len(self.buffer) - self.n + 1)

        for i in range(processing_limit):
            s, a, r, s_, d = self.buffer[i]
            
            R = r
            power = 1
            current_done = d
            last_s_ = s_
            
            if not current_done:
                for k in range(i + 1, i + self.n):
                    if k >= len(self.buffer):
                        break
                    
                    R += self.buffer[k][2] * (self.gamma ** power)
                    power += 1
                    last_s_ = self.buffer[k][3]
                    
                    if self.buffer[k][4]: # If lookahead hits a 'done'
                        current_done = True
                        break
            
            states.append(s)
            actions.append(a)
            returns.append(R)
            states_.append(last_s_)
            dones.append(current_done)
            gammas.append(self.gamma ** power)
        
        # Crucial: Remove processed transitions, keep the 'tail' for the next batch
        self.buffer = self.buffer[processing_limit:]
            
        return [np.array(states), np.array(actions),
                np.array(returns), np.array(states_),
                np.array(dones), np.array(gammas)]

"""
class NStepBuffer:
    def __init__(self, n, gamma, n_batches_to_store=30):
        self.n = n
        self.gamma = gamma
        self.length = n_batches_to_store

        self.buffer = deque(maxlen=self.length)

    def store_transition(self, experience):
        self.buffer.append(experience)

    def compute_n_step_returns(self):
            states, actions, returns, states_, dones, gammas = [], [], [], [], [], []
            
            # We can only compute a full n-step return if we have n-1 transitions ahead
            # or we hit a terminal state.
            for i in range(len(self.buffer)):
                s, a, r, s_, d = self.buffer[i]
                
                # Initialize with the immediate reward (R_t+1)
                R = r
                power = 1
                current_done = d
                last_s_ = s_
                
                # Look ahead up to n-1 more steps (total n steps)
                if not current_done:
                    for k in range(i + 1, i + self.n):
                        if k >= len(self.buffer):
                            break
                        
                        # Sum the discounted future rewards
                        # R_t+2 * gamma^1, R_t+3 * gamma^2...
                        R += self.buffer[k][2] * (self.gamma ** power)
                        power += 1
                        last_s_ = self.buffer[k][3]
                        
                        # If we hit a death during the lookahead, mark transition as done
                        if self.buffer[k][4]:
                            current_done = True
                            break
                
                states.append(s)
                actions.append(a)
                returns.append(R)
                states_.append(last_s_)
                dones.append(current_done)
                gammas.append(self.gamma ** power)
                
            return [np.array(states), np.array(actions),
                    np.array(returns), np.array(states_),
                    np.array(dones), np.array(gammas)]

    def compute_n_step_returns(self):
        # OLD FUNCTION
        states, actions, returns, states_, dones, gammas = [], [], [], [], [], []
        for i, exp in enumerate(self.buffer):
            s, a, _, _, d = exp
            R, power = 0, 0
            states.append(s)
            actions.append(a)
            dones.append(d)
            if not d:
                for k in range(i+1, min(len(self.buffer), i+self.n+1)):
                    index = k
                    R += self.buffer[index][2] * self.gamma**power
                    power += 1
                    if self.buffer[k][4]:
                        break
            else:
                index = min(i, len(self.buffer)-1)
            states_.append(self.buffer[index][3])
            returns.append(R)
            gammas.append(self.gamma**power)
        return [np.array(states), np.array(actions),
                np.array(returns), np.array(states_),
                np.array(dones), np.array(gammas)]
"""
