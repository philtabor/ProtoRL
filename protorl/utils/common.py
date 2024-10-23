import numpy as np
import torch as T
import matplotlib.pyplot as plt


def plot_learning_curve(x, scores, figure_file=None, window=100):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-window):(i+1)])
    plt.plot(x, running_avg)
    plt.title(f'Running average of previous {window} scores')
    plt.xlabel('Time Steps')
    plt.ylabel('Average Score')
    if figure_file:
        plt.savefig(figure_file)
    else:
        plt.show()


def calculate_conv_output_dims(input_dims=(4, 84, 84),
                               channels=(32, 64, 64),
                               kernels=(8, 4, 3)):
    # assume input_dims is [channels, height, width]
    # assume channels is (x, y, z)
    # assume kernels is (a, b, c)
    _, _, w = input_dims
    _, _, ch3 = channels
    _, k2, k3 = kernels
    output_dims = ch3 * w // (k2 * k3) * w // (k2 * k3)
    return [output_dims]


def convert_arrays_to_tensors(array, device):
    tensors = []
    for arr in array:
        tensors.append(T.tensor(np.array(arr)).to(device))
    return tensors


def action_adapter(a, max_a):
    return 2 * (a - 0.5) * max_a


def clip_reward(x):
    if type(x) is np.ndarray:
        rewards = []
        for r in x:
            if r < -1:
                rewards.append(-1)
            elif r > 1:
                rewards.append(1)
            else:
                rewards.append(r)
        return np.array(rewards)
    else:
        if r > 1:
            return 1
        elif r < -1:
            return -1
        else:
            return r


def calc_adv_and_returns(values, values_, rewards, dones,
                         gamma=0.99, gae_lambda=0.95):
    # TODO - multi gpu support
    device = 'cuda:0' if T.cuda.is_available() else 'cpu'
    deltas = rewards + gamma * values_ - values
    deltas = deltas.cpu().numpy()
    dones = dones.cpu().numpy()
    adv = [0]
    for step in reversed(range(deltas.shape[0])):
        advantage = deltas[step] +\
            gamma * gae_lambda * adv[-1] * dones[step]
        adv.append(advantage)
    adv.reverse()
    adv = adv[:-1]
    last_dim = len(deltas.shape)
    adv = T.tensor(np.array(adv), dtype=T.double,
                   device=device, requires_grad=False).unsqueeze(last_dim)
    returns = adv + values.unsqueeze(last_dim)
    adv = (adv - adv.mean()) / (adv.std() + 1e-4)
    return adv, returns
