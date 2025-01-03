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
                               kernels=(8, 4, 3),
                               strides=(4, 2, 1)):
    curr_channels, curr_height, curr_width = input_dims

    layer_outputs = []

    for layer_idx, (out_channels, kernel, stride) in enumerate(zip(channels, kernels, strides)):
        new_height = ((curr_height - kernel) // stride) + 1
        new_width = ((curr_width - kernel) // stride) + 1

        layer_outputs.append({
            'layer': f'conv{layer_idx + 1}',
            'shape': (out_channels, new_height, new_width)
        })
        curr_channels = out_channels
        curr_height = new_height
        curr_width = new_width
    final_output_size = curr_channels * curr_height * curr_width
    return [final_output_size]


def convert_arrays_to_tensors(array, device):
    tensors = []
    for arr in array:
        tensors.append(T.tensor(np.array(arr), device=device))
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
    adv = T.tensor(np.array(adv), dtype=T.float,
                   device=device, requires_grad=False).unsqueeze(last_dim)
    returns = adv + values.unsqueeze(last_dim)

    return adv, returns

def swap_and_flatten(arr):
    shape = arr.shape
    if len(shape) < 3:
        shape = (*shape, 1)
    return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])
