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
            # processing_limit = max(0, len(self.buffer) - self.n + 1)
            processing_limit=120
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
