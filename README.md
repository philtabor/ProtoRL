# ProtoRL: A Torch Based RL Framework for Rapid Prototyping of Research Papers

ProtoRL is developed for students and academics that want to quickly reproduce algorithms
found in research papers. It is designed to be used on a single machine with a multithreaded
CPU and single GPU. 

Out of the box, ProtoRL implements the following algorithms:

- **DQN Double DQN, D3QN, PPO** for single agents with a discrete action space
- **DDPG, TD3, SAC, PPO** for single agents with a continuous action space
- **Prioritized Experience Replay** for any off policy RL algorithm

Note that this is a v0.1 release, and more agents are coming. I am working on developing
open source versions of:
- **Distributed Prioritized Experience Replay (APE-X)**
- **Random Network Distillation (RND)**
- **Recurrent Replay Distributed DQN (R2D2)**
- **Never Give Up (NGU)**

## Requirements
ProtoRL is built around Python3.8 and a minimal set of dependencies. 

We utilize the OpenAI Gym (v0.26), PyTorch (v1.11) and Numpy (v1.21). Support
for the Atari environments comes from atari-py (v0.2.6). Support for some of the
control environments comes from the Box2D dependency.

## Recommended Setup
It's recommended that you install ProtoRL to a virtual environment, due to the stringent
nature of the dependencies. 
```
python3.8 -m venv <your virtual env directory>
source <your virtual env directory>/bin/activate

git clone https://github.com/philtabor/protorl

pip install --upgrade pip

pip install -e .

python

import protorl as prl
```
If you don't receive any errors at this point, ProtoRL has installed correctly.

## Quick Start

Next, you should check out the provided examples.

```
cd protorl/examples
mkdir models
python dqn.py
```
This will train a deep Q agent on the CartPole environment. If you want to try
out other environments, please feel free to edit the dqn.py file.

## How ProtoRL Works

Agents are comprised of a number of bolt-on modules that perform the basic functionality needed.
There is the agent class, that encapsulates all the functionality we would associate with a
deep reinforcement learning agent, network classes, memory classes, policy classes,
episode loop classes, and wrapper classes for various environments.

### Agent
The base agent class has functionality for choosing actions, updating target networks,
interfacing with the memory, learning from its experience, and saving the models. 

Algorithms, such as deep Q learning, deep deterministic policy gradients, etc. are implemented
by deriving from this base agent class and implementing the update and choose_action functions.

### Replay Memory
The replay buffer is designed to be as generic as possible. Some defaults are provided, however
the end user has complete control over what types of data are stored, and as what data types.
Support for prioritized replay is provided out of the box, as well as a number of sampling
modes (sample all memories, batch sampling, uniform random sampling, prioritized sampling).

### Networks
The neural networks are also meant to be used as lego pieces for rapid implementation. 
The networks are generally divided into a base and a head, which are combined with a PyTorch
sequential. The base handles interfacing with the environment, while the head supplies output
to the agent's policy for action selection. ProtoRL implements a broad variety of bases and
heads, ranging from convolutional networks to process screen images, to beta distributions
for action selection. All of the built-in agents have functions that combine the appropriate base
and head for easy use.

### Policies
Policies are treated as separate modules. Each takes output from a neural network, and
performs the appropriate action selection for the task at hand.
Policies supported out of the box include:
- **Epsilon-greedy (discrete actions)**
- **Categorical distribution (discrete actions)**
- **Noisy Deterministic (continuous actions)**
- **Gaussian distribution (continuous actions)**
- **Beta distribution (continuous actions)**

### Episode Loops
The episode loop handles the interface between the agent and the environment. Included is
a loop to handle single as well as multi-threaded agents, as well as to handle PPO. The
inclusion of a PPO specific loop is due to the nature of data stored for replay in PPO.

Episode loops are  built around the latest version of gym, where the step function returns
5 variables instead of 4. Attempting to use ProtoRL with earlier versions of gym will 
break due to this.

### Wrappers
ProtoRL includes wrappers for stacking frames in environments that return screen images.
Additional functionality required to reproduce results in the Atari library is provided.
Since ProtoRL is multithreaded by default, we use a wrapper to handle passing actions to the
step function for single threaded agents (this should probably change).

## TODO
- Buried within the code is functionality for creating networks using a factory. This is more
elegant than the current procedural implementation.
- Rethink how the framework handles single threaded vs. multithreaded agents.
- Implement an actor learner type architecture to support newer Deep Mind algorithms.
- Incorporate multi agent algorithms and environments.
- Implement an environment spec to handle environment contingent processing.
- Implement basic command line interface.
- Do away with having user make a model directory

## Release Notes
v0.1: First stable release
