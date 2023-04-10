import numpy as np
import torch as T
import matplotlib.pyplot as plt


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)


def calculate_conv_output_dims(channels=(32, 64, 64),
                               kernels=(8, 4, 3),
                               input_dims=(4, 84, 84)):
    # assume input_dims is [channels, height, width]
    # assume channels is (x, y, z)
    # assume kernels is (a, b, c)
    _, h, w = input_dims
    ch1, ch2, ch3 = channels
    k1, k2, k3 = kernels
    output_dims = ch3 * w // (k2 * k3) * w // (k2 * k3)
    return [output_dims]


def convert_arrays_to_tensors(array, device):
    tensors = []
    for arr in array:
        tensors.append(T.tensor(arr).to(device))
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
    # TODO - support for different vector shapes
    device = 'cuda:0' if T.cuda.is_available() else 'cpu'
    deltas = rewards + gamma * values_ - values
    deltas = deltas.cpu().numpy()
    adv = [0]
    for step in reversed(range(deltas.shape[0])):
        advantage = deltas[step] +\
            gamma * gae_lambda * adv[-1] * dones[step]
        adv.append(advantage)
    adv.reverse()
    adv = adv[:-1]
    adv = T.tensor(adv, dtype=T.double,
                   device=device, requires_grad=False).unsqueeze(2)
    returns = adv + values.unsqueeze(2)
    adv = (adv - adv.mean()) / (adv.std() + 1e-4)
    return adv, returns
