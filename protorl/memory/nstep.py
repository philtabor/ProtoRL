from collections import deque
import numpy as np

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
